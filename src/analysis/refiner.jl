"""
    struct StructuralRefiner

This abstract interpreter consumes results of the ADAnalyzer and computes the
code's structural properties. It in particular performs interprocedural propagation
of structural incidence information.
"""
struct StructuralRefiner <: Compiler.AbstractInterpreter
    world::UInt
    var_to_diff::DiffGraph
    varkinds::Vector{Intrinsics.VarKind}
    varclassification::Vector{VarEqClassification}
end

struct StructureCache; end

Compiler.optimizer_lattice(interp::StructuralRefiner) = Compiler.PartialsLattice(EqStructureLattice())
Compiler.typeinf_lattice(interp::StructuralRefiner) = Compiler.PartialsLattice(EqStructureLattice())
Compiler.ipo_lattice(interp::StructuralRefiner) = Compiler.PartialsLattice(EqStructureLattice())

Compiler.InferenceParams(interp::StructuralRefiner) = Compiler.InferenceParams()
Compiler.OptimizationParams(interp::StructuralRefiner) = Compiler.OptimizationParams()
Compiler.get_inference_world(interp::StructuralRefiner) = interp.world
Compiler.cache_owner(::StructuralRefiner) = StructureCache()

# This is the main logic. We visit an :invoke instruction and either apply the known transfer function for one of our
# DAECompiler intrinsics or lookup the structural incidence matrix in the cache, applying it as appropriate.
@override function Compiler.abstract_eval_invoke_inst(interp::StructuralRefiner, inst::Compiler.Instruction, irsv::Compiler.IRInterpretationState)
    stmt = inst[:stmt]
    invokee = stmt.args[1]
    RT = Pair{Any, Tuple{Bool, Bool}}
    good_effects = (true, true)
    if isa(invokee, Core.CodeInstance)
        m = invokee.def.def
    else
        m = invokee.def
    end

    if m === Intrinsics.ddt_method
        argtypes = Compiler.collect_argtypes(interp, stmt.args, Compiler.StatementState(nothing, false), irsv)
        argtypes === nothing && return RT(Union{}, (false, true))
        # First arg is invoke mi
        if length(argtypes) == 3 && isa(argtypes[3], Union{Incidence, Const})
            return RT(structural_inc_ddt(interp.var_to_diff, interp.varclassification, interp.varkinds, argtypes[3]), (false, true))
        end
        return RT(nothing, (false, true))
    elseif m === Intrinsics.equation_method
        return RT(nothing, (false, true))
    end

    callee_codeinst = invokee
    callee_result = structural_analysis!(callee_codeinst, Compiler.get_inference_world(interp))

    if isa(callee_result, UncompilableIPOResult) || isa(callee_result.extended_rt, Const) || isa(callee_result.extended_rt, Type)
        # If this is uncompilable, we will be notfiying the user in the outer loop - here we just ignore it
        return RT(nothing, (false, false))
    end

    argtypes = Compiler.collect_argtypes(interp, stmt.args, Compiler.StatementState(nothing, false), irsv)[2:end]
    mapping = CalleeMapping(Compiler.optimizer_lattice(interp), argtypes, callee_result)
    new_rt = apply_linear_incidence(Compiler.optimizer_lattice(interp), callee_result.extended_rt, callee_result, interp.var_to_diff, interp.varclassification, VarEqClassification[], mapping)

    # Remember this mapping, both for performance of not having to recompute it
    # and because we may have assigned caller variables to internal variables
    # that we now need to remember.
    inst[:info] = MappingInfo(inst[:info], callee_result, mapping)
    if new_rt === nothing
        return RT(nothing, (false, false))
    end
    return RT(new_rt, (false, false))

end

#==================== DAECompiler Intrinsic Refinement Models ===========================#

function structural_inc_ddt(var_to_diff::DiffGraph, varclassification::Union{Vector{VarEqClassification}, Nothing}, varkinds::Union{Vector{Intrinsics.VarKind}, Nothing}, inc::Union{Incidence, Const})
    isa(inc, Const) && return Const(zero(inc.val))
    r = _zero_row()
    function get_or_make_diff(v_offset::Int)
        v = v_offset - 1
        var_to_diff[v] !== nothing && return var_to_diff[v] + 1
        dv = add_vertex!(var_to_diff)
        if varclassification !== nothing
            push!(varclassification, varclassification[v])
        end
        if varkinds !== nothing
            push!(varkinds, Intrinsics.Continuous)
        end
        add_edge!(var_to_diff, v, dv)
        return dv + 1
    end
    base = isa(inc.typ, Const) ? Const(zero(inc.typ.val)) : inc.typ
    for (v_offset, coeff) in zip(rowvals(inc.row), nonzeros(inc.row))
        if v_offset == 1
            # t
            if isa(base, Const)
                if isa(coeff, Float64)
                    base = Const(base.val + coeff)
                    # Do not set r[v_offset]; d/dt t = 1
                else
                    r[v_offset] = nonlinear
                end
            elseif !isa(coeff, Const)
                r[v_offset] = nonlinear
            end
            continue
        end
        if isa(coeff, Float64)
            # Linear, just add to the derivative
            r[get_or_make_diff(v_offset)] += coeff
        else
            # nonlinear
            r[v_offset] = nonlinear
            r[get_or_make_diff(v_offset)] = nonlinear
        end
    end
    return Incidence(base, r)
end

#==================== Base Math Intrinsic Refinement Models ===========================#
function tfunc(::Val{Core.Intrinsics.neg_float}, @nospecialize(a::Union{Const, Incidence}))
    if isa(a, Incidence)
        arow = copy(a.row)
        for (i, v) in zip(rowvals(a.row), nonzeros(a.row))
            arow[i] = -v
        end
        eps = copy(a.eps)
    else
        arow = _zero_row()
        eps = BitSet()
    end
    return Incidence(builtin_math_tfunc(Core.Intrinsics.neg_float, isa(a, Incidence) ? a.typ : a), arow, eps)
end

get_eps(inc::Incidence) = inc.eps
get_eps(c::Const) = BitSet()
get_eps(::Type) = error()

function tfunc(::Val{Core.Intrinsics.add_float}, @nospecialize(a::Union{Const, Type{Float64}, Incidence}), @nospecialize(b::Union{Const, Type{Float64}, Incidence}))
    if a === Float64 || b === Float64
        return Float64
    end
    isa(a, Const) && isa(b, Const) && return builtin_math_tfunc(Core.Intrinsics.add_float, a, b)
    arow = isa(a, Incidence) ? a.row : _ZERO_ROW
    brow = isa(b, Incidence) ? b.row : _ZERO_ROW
    rrow = copy(arow) .+= brow
    const_val = builtin_math_tfunc(Core.Intrinsics.add_float, isa(a, Incidence) ? a.typ : a, isa(b, Incidence) ? b.typ : b)
    if isa(const_val, Const) && !any(!iszero, rrow)
        return const_val
    end
    return Incidence(const_val, rrow)
end

function tfunc(::Val{Core.Intrinsics.sub_float}, @nospecialize(a::Union{Const, Incidence}), @nospecialize(b::Union{Const, Incidence}))
    isa(a, Const) && isa(b, Const) && return builtin_math_tfunc(Core.Intrinsics.sub_float, a, b)
    arow = isa(a, Incidence) ? a.row : _ZERO_ROW
    brow = isa(b, Incidence) ? b.row : _ZERO_ROW
    # return Incidence(a.row + b.row), but see https://github.com/JuliaArrays/OffsetArrays.jl/issues/299
    # and https://github.com/JuliaSparse/SparseArrays.jl/issues/101
    rrow = copy(arow) .-= brow
    const_val = builtin_math_tfunc(Core.Intrinsics.sub_float, isa(a, Incidence) ? a.typ : a, isa(b, Incidence) ? b.typ : b)
    if isa(const_val, Const) && !any(!iszero, rrow)
        return const_val
    end
    return Incidence(const_val, rrow)
end

function tfunc(::Val{Core.Intrinsics.mul_float}, @nospecialize(a::Union{Const, Type{Float64}, Incidence}), @nospecialize(b::Union{Const, Type{Float64}, Incidence}))
    if a === Float64 || b === Float64
        return Float64
    end
    if isa(a, Const)
        isa(b, Const) && return builtin_math_tfunc(Core.Intrinsics.mul_float, a, b)
        iszero(a.val) && return a
        return Incidence(builtin_math_tfunc(Core.Intrinsics.mul_float, a, b.typ), a.val * b.row)
    elseif isa(b, Const)
        iszero(b.val) && return b
        return Incidence(builtin_math_tfunc(Core.Intrinsics.mul_float, a.typ, b), a.row * b.val)
    end
    rrow = _zero_row()
    for i in Iterators.flatten((rowvals(a.row), rowvals(b.row)))
        rrow[i] = nonlinear
    end
    return Incidence(builtin_math_tfunc(Core.Intrinsics.mul_float, a.typ, b.typ), rrow)
end

function tfunc(::Val{Core.Intrinsics.div_float}, @nospecialize(a::Union{Const, Type{Float64}, Incidence}), @nospecialize(b::Union{Const, Type{Float64}, Incidence}))
    if isa(a, Const) && isa(b, Const)
        return builtin_math_tfunc(Core.Intrinsics.div_float, a, b)
    elseif isa(a, Incidence) && isa(b, Const)
        return Incidence(builtin_math_tfunc(Core.Intrinsics.div_float, a.typ, b), a.row / b.val)
    elseif isa(a, Const) && iszero(a.val)
        # TODO: Sign preservation?
        return a
    end
    if a === Float64 || b === Float64
        return Float64
    end
    rrow = copy(b.row)
    if isa(a, Incidence)
        rrow .+= a.row
    end
    rrow .*= nonlinear
    return Incidence(builtin_math_tfunc(Core.Intrinsics.div_float, isa(a, Incidence) ? a.typ : a, widenconst(b.typ)), rrow)
end

function builtin_math_tfunc(@nospecialize(f), @nospecialize(a), @nospecialize(b), @nospecialize(c))
    if isa(a, Const) && isa(b, Const) && isa(c, Const)
        return Const(f(a.val::Float64, b.val::Float64, c.val::Float64))
    end
    return widenconst(a)
end

function builtin_math_tfunc(@nospecialize(f), @nospecialize(a), @nospecialize(b))
    if isa(a, Const) && isa(b, Const)
        return Const(f(a.val::Float64, b.val::Float64))
    end
    return widenconst(a)
end

function builtin_math_tfunc(@nospecialize(f), @nospecialize(a))
    if isa(a, Const)
        return Const(f(a.val::Float64))
    end
    return widenconst(a)
end

function is_elem_inc_ir_const(@nospecialize(x))
    x === Union{} && return true
    isa(x, Incidence) && return true
    isa(x, Eq) && return true
    Compiler.isalreadyconst(x) && return true
    if isa(x, PartialStruct)
        return is_all_inc_or_const(x.fields)
    end
    if isa(x, Compiler.Conditional)
        return is_elem_inc_ir_const(x.thentype) && is_elem_inc_ir_const(x.elsetype)
    end
    return false
end

is_all_inc_or_const(args) = all(is_elem_inc_ir_const, args)
is_any_incidence(@nospecialize args...) = any(@nospecialize(x)->isa(x, Incidence), args) && is_all_inc_or_const(args)

@override function Compiler.builtin_tfunction(interp::StructuralRefiner,
        @nospecialize(f), argtypes::Vector{Any}, sv::Union{Compiler.AbsIntState,Nothing})

    bargtypes = argtypes
    if length(argtypes) == 1
        a = argtypes[1]
        if is_any_incidence(a)
            if f == Core.Intrinsics.neg_float
                return tfunc(Val(f), a)
            end
        end
    elseif length(argtypes) == 2
        a = argtypes[1]
        b = argtypes[2]
        if is_any_incidence(a, b)
            if (f == Core.Intrinsics.add_float || f == Core.Intrinsics.sub_float) ||
                (f == Core.Intrinsics.mul_float || f == Core.Intrinsics.div_float)
                return tfunc(Val(f), a, b)
            elseif f == Core.Intrinsics.lt_float
                r = tmerge(typeinf_lattice(interp), argtypes[1], argtypes[2])
                @assert isa(r, Incidence)
                return Incidence(Bool, r.row, r.eps)
            elseif f === Core.getfield && isa(a, Incidence)
                a = argtypes[1]
                fT = getfield_tfunc(typeinf_lattice(interp), widenconst(a), argtypes[2])
                fT === Union{} && return Union{}
                Base.issingletontype(fT) && return fT
                return Incidence(fT, copy(a.row), copy(a.eps))
            end
        end
    elseif length(argtypes) == 3
        a = argtypes[1]
        b = argtypes[2]
        c = argtypes[3]
        if is_any_incidence(a, b, c)
            if f === Core.Intrinsics.muladd_float
                # TODO: muladd vs fma here?
                if is_any_incidence(a, b)
                    x = tfunc(Val(Core.Intrinsics.mul_float), a, b)
                else
                    x = builtin_math_tfunc(Core.Intrinsics.mul_float, a, b)
                end
                if is_any_incidence(x, c)
                    return tfunc(Val(Core.Intrinsics.add_float), x, c)
                else
                    return builtin_math_tfunc(Core.Intrinsics.add_float, x, c)
                end
            elseif f === Core.ifelse
                if isa(a, Const)
                    if a.val === true
                        return b
                    elseif a.val === false
                        return c
                    end
                end
                # TODO: tmergea
            end
        end
    end
    if !(f in (Core.tuple, Core.getfield))
        bargtypes = Any[isa(a, Incidence) ? widenconst(a) : a for a in argtypes]
    end

    rt = @invoke Compiler.builtin_tfunction(interp::AbstractInterpreter,
        f::Any, bargtypes::Vector{Any}, sv::Union{AbsIntState,Nothing})

    return rt
end