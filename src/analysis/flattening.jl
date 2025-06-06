function _flatten_parameter!(𝕃, compact, argtypes, ntharg, line)
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
            nthfield(i) = insert_node_here!(compact,
                NewInstruction(Expr(:call, getfield, this, i), Compiler.getfield_tfunc(𝕃, argextype(this, compact), Const(i)), line))
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
    return insert_node_here!(compact,
        NewInstruction(Expr(:call, tuple, _flatten_parameter!(𝕃, compact, argtypes, ntharg, line)...), Tuple, line))
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

struct TransformedArg
    ssa::Any
    offset::Int
    eqoffset::Int
    TransformedArg(@nospecialize(arg), new_offset::Int, new_eqoffset::Int) = new(arg, new_offset, new_eqoffset)
end

function flatten_argument!(compact::Compiler.IncrementalCompact, @nospecialize(argt), offset::Int, eqoffset::Int, argtypes::Vector{Any})::TransformedArg
    @assert !isa(argt, Incidence) && !isa(argt, Eq)
    if isa(argt, Const)
        return TransformedArg(argt.val, offset, eqoffset)
    elseif Base.issingletontype(argt)
        return TransformedArg(argt.instance, offset, eqoffset)
    elseif Base.isprimitivetype(argt)
        push!(argtypes, argt)
        return TransformedArg(Argument(offset+1), offset+1, eqoffset)
    elseif argt === equation
        ssa = insert_node_here!(compact, NewInstruction(Expr(:invoke, nothing, InternalIntrinsics.external_equation), Eq(eqoffset+1), compact[Compiler.OldSSAValue(1)][:line]))
        return TransformedArg(ssa, offset, eqoffset+1)
    elseif isabstracttype(argt) || ismutabletype(argt) || (!isa(argt, DataType) && !isa(argt, PartialStruct))
        ssa = insert_node_here!(compact, NewInstruction(Expr(:call, error, "Cannot IPO model arg type $argt"), Union{}, compact[Compiler.OldSSAValue(1)][:line]))
        return TransformedArg(ssa, -1, eqoffset)
    else
        if !isa(argt, PartialStruct) && Base.datatype_fieldcount(argt) === nothing
            ssa = insert_node_here!(compact, NewInstruction(Expr(:call, error, "Cannot IPO model arg type $argt"), Union{}, compact[Compiler.OldSSAValue(1)][:line]))
            return TransformedArg(ssa, -1, eqoffset)
        end
        (args, _, offset) = flatten_arguments!(compact, isa(argt, PartialStruct) ? argt.fields : collect(Any, fieldtypes(argt)), offset, eqoffset, argtypes)
        offset == -1 && return TransformedArg(ssa, -1, eqoffset)
        this = Expr(:new, isa(argt, PartialStruct) ? argt.typ : argt, args...)
        ssa = insert_node_here!(compact, NewInstruction(this, argt, compact[Compiler.OldSSAValue(1)][:line]))
        return TransformedArg(ssa, offset, eqoffset)
    end
end

function flatten_arguments!(compact::Compiler.IncrementalCompact, argtypes::Vector{Any}, offset::Int=0, eqoffset::Int=0, new_argtypes::Vector{Any} = Any[])
    args = Any[]
    for argt in argtypes
        (; ssa, offset, eqoffset) = flatten_argument!(compact, argt, offset, eqoffset, new_argtypes)
        offset == -1 && break
        push!(args, ssa)
    end
    return (args, new_argtypes, offset, eqoffset)
end
