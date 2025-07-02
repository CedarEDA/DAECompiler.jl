const CompositeIndex = Vector{Int}

struct ArgumentMap
    variables::Vector{CompositeIndex} # index into argument tuple type
    equations::Vector{CompositeIndex} # index into argument tuple type
end
ArgumentMap() = ArgumentMap(CompositeIndex[], CompositeIndex[])

function ArgumentMap(argtypes::Vector{Any})
    map = ArgumentMap()
    index = CompositeIndex()
    fill_argument_map!(map, index, argtypes)
    return map
end

function fill_argument_map!(map::ArgumentMap, index::CompositeIndex, types::Vector{Any})
    for (i, type) in enumerate(types)
        push!(index, i)
        fill_argument_map!(map, index, type)
        pop!(index)
    end
end

function fill_argument_map!(map::ArgumentMap, index::CompositeIndex, @nospecialize(type))
    if isprimitivetype(type) || isa(type, Incidence)
        push!(map.variables, copy(index))
    elseif type === equation
        push!(map.equations, copy(index))
    elseif isa(type, PartialStruct) || isstructtype(type)
        fields = isa(type, PartialStruct) ? type.fields : collect(Any, fieldtypes(type))
        fill_argument_map!(map, index, fields)
    end
end

struct FlatteningState
    compact::IncrementalCompact
    settings::Settings
    map::ArgumentMap
    nvariables::Int
    nequations::Int
end

function FlatteningState(compact::IncrementalCompact, settings::Settings, map::ArgumentMap)
    FlatteningState(compact, settings, deepcopy(map), length(map.variables), length(map.equations))
end

function next_variable!(state::FlatteningState)
    popfirst!(state.map.variables)
    return state.nvariables - length(state.map.variables)
end

function next_equation!(state::FlatteningState)
    popfirst!(state.map.equations)
    return state.nequations - length(state.map.equations)
end

function flatten_arguments!(state::FlatteningState)
    argtypes = copy(state.compact.ir.argtypes)
    empty!(state.compact.ir.argtypes) # will be recomputed during flattening
    args = flatten_arguments!(state, argtypes)
    if args !== nothing
        @assert isempty(state.map.variables)
        @assert isempty(state.map.equations)
    end
    return args
end

function flatten_arguments!(state::FlatteningState, argtypes::Vector{Any})
    args = Any[]
    for argt in argtypes
        arg = flatten_argument!(state, argt)
        arg === nothing && return nothing
        push!(args, arg)
    end
    return args
end

function flatten_argument!(state::FlatteningState, @nospecialize(argt))
    @assert !isa(argt, Incidence) && !isa(argt, Eq)
    (; compact, settings) = state
    if isa(argt, Const)
        return argt.val
    elseif Base.issingletontype(argt)
        return argt.instance
    elseif isprimitivetype(argt)
        push!(state.compact.ir.argtypes, argt)
        return Argument(next_variable!(state))
    elseif argt === equation
        eq = next_equation!(state)
        line = compact[Compiler.OldSSAValue(1)][:line]
        ssa = @insert_instruction_here(compact, line, settings, (:invoke)(nothing, InternalIntrinsics.external_equation)::Eq(eq))
        return ssa
    elseif argt <: Type
        return argt.parameters[1]
    elseif isabstracttype(argt) || ismutabletype(argt) || (!isa(argt, DataType) && !isa(argt, PartialStruct))
        line = compact[Compiler.OldSSAValue(1)][:line]
        ssa = @insert_instruction_here(compact, line, settings, error("Cannot IPO model arg type $argt")::Union{})
        return nothing
    else
        if !isa(argt, PartialStruct) && Base.datatype_fieldcount(argt) === nothing
            line = compact[Compiler.OldSSAValue(1)][:line]
            ssa = @insert_instruction_here(compact, line, settings, error("Cannot IPO model arg type $argt")::Union{})
            return nothing
        end
        fields = isa(argt, PartialStruct) ? argt.fields : collect(Any, fieldtypes(argt))
        args = flatten_arguments!(state, fields)
        args === nothing && return nothing
        this = Expr(:new, isa(argt, PartialStruct) ? argt.typ : argt, args...)
        line = compact[Compiler.OldSSAValue(1)][:line]
        ssa = @insert_instruction_here(compact, line, settings, this::argt)
        return ssa
    end
end

function flatten_arguments_for_callee!(compact::IncrementalCompact, map::ArgumentMap, argtypes, args, line, settings, 𝕃 = Compiler.fallback_lattice)
    list = Any[]
    this = nothing
    last_index = CompositeIndex()
    for index in map.variables
        from = findfirst(j -> get(last_index, j, -1) !== index[j], eachindex(index))::Int
        for i in from:length(index)
            field = index[i]
            if i == 1
                this = args[field]
            else
                thistype = argextype(this, compact)
                fieldtype = Compiler.getfield_tfunc(𝕃, thistype, Const(field))
                this = @insert_instruction_here(compact, line, settings, getfield(this, field)::fieldtype)
            end
        end
        push!(list, this)
    end
    return list
end

remove_variable_and_equation_annotations(argtypes) = Any[widenconst(T) for T in argtypes]

function annotate_variables_and_equations(argtypes::Vector{Any}, map::ArgumentMap)
    argtypes_annotated = Any[]
    pstructs = Dict{CompositeIndex,PartialStruct}()
    for (i, arg) in enumerate(argtypes)
        if arg !== equation && arg !== Incidence && isstructtype(arg) && (any(==(i) ∘ first, map.variables) || any(==(i) ∘ first, map.equations))
            arg = init_partialstruct(arg)
            pstructs[[i]] = arg
        end
        push!(argtypes_annotated, arg)
    end

    function fields_for_index(index)
        length(index) > 1 || return argtypes_annotated
        # Find the parent `PartialStruct` that holds the variable field,
        # creating any further `PartialStruct` going down if necessary.
        i, base = find_base(pstructs, index)
        local fields = base.fields
        for j in @view index[(i + 1):(end - 1)]
            pstruct = init_partialstruct(fields[j])
            fields[j] = pstruct
            fields = pstruct.fields
        end
        return fields
    end

    # Populate `PartialStruct` variable fields with an `Incidence` lattice element.
    for (variable, index) in enumerate(map.variables)
        fields = fields_for_index(index)
        type = get_fieldtype(argtypes, index)
        fields[index[end]] = Incidence(type, variable)
    end

    # Do the same for equations with an `Eq` lattice element.
    for (equation, index) in enumerate(map.equations)
        fields = fields_for_index(index)
        fields[index[end]] = Eq(equation)
    end

    return argtypes_annotated
end

init_partialstruct(@nospecialize(T)) = PartialStruct(T, collect(Any, fieldtypes(T)))
init_partialstruct(pstruct::PartialStruct) = pstruct

function find_base(dict::Dict{CompositeIndex}, index::CompositeIndex)
    for i in reverse(eachindex(index))
        base = get(dict, @view(index[1:i]), nothing)
        base !== nothing && return i, base
    end
end
