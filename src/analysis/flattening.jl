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
    new_argtypes::Vector{Any}
end

function FlatteningState(compact::IncrementalCompact, settings::Settings, map::ArgumentMap)
    FlatteningState(compact, settings, deepcopy(map), length(map.variables), length(map.equations), Any[])
end

function next_variable!(state::FlatteningState)
    popfirst!(state.map.variables)
    return state.nvariables - length(state.map.variables)
end

function next_equation!(state::FlatteningState)
    popfirst!(state.map.equations)
    return state.nequations - length(state.map.equations)
end

function flatten_arguments!(state::FlatteningState, argtypes::Vector{Any})
    args = Any[]
    # push!(state.new_argtypes, argtypes[1])
    for argt in argtypes
        arg = flatten_argument!(state, argt)
        arg === nothing && return nothing
        push!(args, arg)
    end
    @assert isempty(state.map.variables)
    @assert isempty(state.map.equations)
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
        push!(state.new_argtypes, argt)
        return Argument(next_variable!(state))
    elseif argt === equation
        eq = next_equation!(state)
        line = compact[Compiler.OldSSAValue(1)][:line]
        ssa = @insert_instruction_here(compact, line, settings, (:invoke)(nothing, InternalIntrinsics.external_equation)::Eq(eq))
        return ssa
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

function flatten_arguments_for_callee!(compact::IncrementalCompact, map::ArgumentMap, argtypes, 𝕃, line, settings)
    list = Any[]
    this = nothing
    last_index = Int[]
    for index in map.variables
        from = findfirst(j -> get(last_index, j, -1) !== index[j], eachindex(index))::Int
        for i in from:length(index)
            field = index[i]
            if i == 1
                this = Argument(2 + field)
            else
                thistype = argextype(this, compact)
                fieldtype = Compiler.getfield_tfunc(𝕃, Const(field))
                this = @insert_instruction_here(compact, line, settings, getfield(this, field)::fieldtype)
            end
        end
        push!(list, this)
    end
    return list
end

function _flatten_parameter!(𝕃, compact, argtypes, ntharg, line, settings)
    list = Any[]
    for (argn, argt) in enumerate(argtypes)
        if isa(argt, Const)
            continue
        elseif Base.issingletontype(argt)
            continue
        elseif Base.isprimitivetype(argt) || isa(argt, Incidence)
            push!(list, ntharg(argn))
        elseif argt === equation || isa(argt, Eq)
            continue
        elseif isa(argt, Type) && argt <: Intrinsics.AbstractScope
            continue
        elseif isabstracttype(argt) || ismutabletype(argt) || (!isa(argt, DataType) && !isa(argt, PartialStruct))
            continue
        else
            if !isa(argt, PartialStruct) && Base.datatype_fieldcount(argt) === nothing
                continue
            end
            this = ntharg(argn)
            nthfield(i) = @insert_instruction_here(compact, line, settings, getfield(this, i)::Compiler.getfield_tfunc(𝕃, argextype(this, compact), Const(i)))
            if isa(argt, PartialStruct)
                fields = _flatten_parameter!(𝕃, compact, argt.fields, nthfield, line, settings)
            else
                fields = _flatten_parameter!(𝕃, compact, fieldtypes(argt), nthfield, line, settings)
            end
            append!(list, fields)
        end
    end
    return list
end

function flatten_parameter!(𝕃, compact, argtypes, ntharg, line, settings)
    return @insert_instruction_here(compact, line, settings, tuple(_flatten_parameter!(𝕃, compact, argtypes, ntharg, line, settings)...)::Tuple)
end

# Needs to match flatten_arguments!
function process_template_arg!(𝕃, coeffs, eq_mapping, applied_scopes, argt, template_argt, offset=0, eqoffset=0)::Pair{Int, Int}
    if isa(template_argt, Const)
        @assert isa(argt, Const) && argt.val === template_argt.val
        return Pair{Int, Int}(offset, eqoffset)
    elseif Base.issingletontype(template_argt)
        @assert isa(template_argt, Type) && argt.instance === template_argt.instance
        return Pair{Int, Int}(offset, eqoffset)
    elseif Base.isprimitivetype(template_argt)
        coeffs[offset+1] = argt
        return Pair{Int, Int}(offset + 1, eqoffset)
    elseif template_argt === equation
        eq_mapping[eqoffset+1] = argt.id
        return Pair{Int, Int}(offset, eqoffset + 1)
    elseif isabstracttype(template_argt) || ismutabletype(template_argt) || (!isa(template_argt, DataType) && !isa(template_argt, PartialStruct))
        return Pair{Int, Int}(offset, eqoffset)
    else
        if !isa(template_argt, PartialStruct) && Base.datatype_fieldcount(template_argt) === nothing
            return Pair{Int, Int}(offset, eqoffset)
        end
        template_fields = isa(template_argt, PartialStruct) ? template_argt.fields : collect(fieldtypes(template_argt))
        return process_template!(𝕃, coeffs, eq_mapping, applied_scopes, Any[Compiler.getfield_tfunc(𝕃, argt, Const(i)) for i = 1:length(template_fields)], template_fields, offset)
    end
end

function process_template!(𝕃, coeffs, eq_mapping, applied_scopes, argtypes, template_argtypes, offset=0, eqoffset=0)
    @assert length(argtypes) == length(template_argtypes)
    for (i, template_arg) in enumerate(template_argtypes)
        (offset, eqoffset) = process_template_arg!(𝕃, coeffs, eq_mapping, applied_scopes, argtypes[i], template_arg, offset)
    end
    return Pair{Int, Int}(offset, eqoffset)
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

function find_base(dict::Dict{CompositeIndex}, index::CompositeIndex)
    for i in reverse(eachindex(index))
        base = get(dict, @view(index[1:i]), nothing)
        base !== nothing && return i, base
    end
end
