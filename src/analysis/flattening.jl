# TODO: We should unify this function with _make_argument_lattice_elem to ensure consistency
function _flatten_parameter!(𝕃, compact, argtypes, ntharg, line)
    list = Any[]
    for (argn, argt) in enumerate(argtypes)
        if isa(argt, Const)
            continue
        elseif Base.issingletontype(argt)
            continue
        elseif Base.isprimitivetype(argt) || isa(argt, Incidence)
            push!(list, ntharg(argn))
        elseif isa(argt, Type) && argt <: Intrinsics.AbstractScope
            continue
        elseif isabstracttype(argt) || ismutabletype(argt) || (!isa(argt, DataType) && !isa(argt, PartialStruct))
            continue
        else
            if !isa(argt, PartialStruct) && Base.datatype_fieldcount(argt) === nothing
                continue
            end
            this = ntharg(argn)
            nthfield(i) = @insert_node_here compact line getfield(this, i)::Compiler.getfield_tfunc(𝕃, argextype(this, compact), Const(i))
            if isa(argt, PartialStruct)
                fields = _flatten_parameter!(𝕃, compact, argt.fields, nthfield, line)
            else
                fields = _flatten_parameter!(𝕃, compact, fieldtypes(argt), nthfield, line)
            end
            append!(list, fields)
        end
    end
    return list
end

function flatten_parameter!(𝕃, compact, argtypes, ntharg, line)
    return @insert_node_here compact line tuple(_flatten_parameter!(𝕃, compact, argtypes, ntharg, line)...)::Tuple
end

# Needs to match flatten_arguments!
function process_template_arg!(𝕃, coeffs, eq_mapping, applied_scopes, argt, template_argt, offset=0)
    if isa(template_argt, Const)
        @assert isa(argt, Const) && argt.val === template_argt.val
        return offset
    elseif Base.issingletontype(template_argt)
        @assert isa(template_argt, Type) && argt.instance === template_argt.instance
        return offset
    elseif Base.isprimitivetype(template_argt)
        coeffs[offset+1] = argt
        return offset + 1
    elseif isabstracttype(template_argt) || ismutabletype(template_argt) || (!isa(template_argt, DataType) && !isa(template_argt, PartialStruct))
        return offset
    else
        if !isa(template_argt, PartialStruct) && Base.datatype_fieldcount(template_argt) === nothing
            return offset
        end
        template_fields = isa(template_argt, PartialStruct) ? template_argt.fields : collect(fieldtypes(template_argt))
        return process_template!(𝕃, coeffs, eq_mapping, applied_scopes, Any[Compiler.getfield_tfunc(𝕃, argt, Const(i)) for i = 1:length(template_fields)], template_fields, offset)
    end
end

function process_template!(𝕃, coeffs, eq_mapping, applied_scopes, argtypes, template_argtypes, offset=0)
    @assert length(argtypes) == length(template_argtypes)
    for (i, template_arg) in enumerate(template_argtypes)
        offset = process_template_arg!(𝕃, coeffs, eq_mapping, applied_scopes, argtypes[i], template_arg, offset)
    end
    return offset
end

function flatten_argument!(compact::Compiler.IncrementalCompact, argt, offset, argtypes::Vector{Any})::Pair{Any, Int}
    @assert !isa(argt, Incidence)
    if isa(argt, Const)
        return Pair{Any, Int}(argt.val, offset)
    elseif Base.issingletontype(argt)
        return Pair{Any, Int}(argt.instance, offset)
    elseif Base.isprimitivetype(argt)
        push!(argtypes, argt)
        return Pair{Any, Int}(Argument(offset+1), offset+1)
    elseif isabstracttype(argt) || ismutabletype(argt) || (!isa(argt, DataType) && !isa(argt, PartialStruct))
        ssa = @insert_node_here compact compact[Compiler.OldSSAValue(1)][:line] error("Cannot IPO model arg type $argt")::Union{}
        return Pair{Any, Int}(ssa, offset)
    else
        if !isa(argt, PartialStruct) && Base.datatype_fieldcount(argt) === nothing
            ssa = @insert_node_here compact compact[Compiler.OldSSAValue(1)][:line] error("Cannot IPO model arg type $argt")::Union{}
            return Pair{Any, Int}(ssa, offset)
        end
        (args, _, offset) = flatten_arguments!(compact, isa(argt, PartialStruct) ? argt.fields : fieldtypes(argt), offset, argtypes)
        this = Expr(:new, isa(argt, PartialStruct) ? argt.typ : argt, args...)
        ssa = @insert_node_here compact compact[Compiler.OldSSAValue(1)][:line] this::argt
        return Pair{Any, Int}(ssa, offset)
    end
end

function flatten_arguments!(compact::Compiler.IncrementalCompact, argtypes, offset=0, new_argtypes::Vector{Any} = Any[])
    args = Any[]
    for argt in argtypes
        (ssa, offset) = flatten_argument!(compact, argt, offset, new_argtypes)
        push!(args, ssa)
    end
    return (args, new_argtypes, offset)
end
