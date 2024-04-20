using ForwardDiff
using Base.Meta
using Graphs
using Core.IR
using Core.Compiler: InferenceState, bbidxiter, dominates, tmerge, typeinf_lattice

@breadcrumb "ir_levels" function run_dae_passes(
    interp::DAEInterpreter, ir::IRCode, debug_config::DebugConfig = DebugConfig())
    inlining = CC.InliningState(interp)
    ir = cfg_simplify!(ir)
    record_ir!(debug_config, "cfg_simplify!", ir)

    ir = compact!(ir, true)
    record_ir!(debug_config, "compact!.1", ir)

    # We need aggressive SROA for at least two cases: The finalizers and the sls.
    # Finalizers need at least three SROA passes - the first to get rid of `finalizer!`,
    # the second for any mutables that may now be SROA eligible, the third to simplify
    # now-exposed immutable SROA opportunities.
    for idx = 1:3
        ir = CC.sroa_pass!(ir, inlining)
        record_ir!(debug_config, "sroa_pass!.$(idx)", ir)
    end
    ir = compact!(ir)
    record_ir!(debug_config, "compact!.2", ir)
    return ir
end

function is_known_invoke_or_call(@nospecialize(x), @nospecialize(func), ir::Union{IRCode,IncrementalCompact})
    return is_known_invoke(x, func, ir) || is_known_call(x, func, ir)
end

function is_known_invoke(@nospecialize(x), @nospecialize(func), ir::Union{IRCode,IncrementalCompact})
    isexpr(x, :invoke) || return false
    ft = argextype(x.args[2], ir)
    return singleton_type(ft) === func
end

using .Intrinsics: solved_variable
is_solved_variable(stmt) = isexpr(stmt, :call) && stmt.args[1] == solved_variable ||
    isexpr(stmt, :invoke) && stmt.args[2] == solved_variable

function is_equation_call(@nospecialize(x), ir::Union{IRCode,IncrementalCompact},
                          allow_call::Bool=true)
    isexpr(x, :invoke) || (allow_call && isexpr(x, :call)) || return false
    ft = argextype(_eq_function_arg(x), ir)
    return widenconst(ft) === equation
end

function _eq_function_arg(stmt::Expr)
    ft_ind = 1 + isexpr(stmt, :invoke)
    return stmt.args[ft_ind]
end
function _eq_val_arg(stmt::Expr)
    ft_ind = 1 + isexpr(stmt, :invoke)
    return stmt.args[ft_ind + 1]
end

function isscalar(inc::Incidence)
    return length(rowvals(inc.row)) <= 1 &&
         (isempty(rowvals(inc.row)) || rowvals(inc.row)[] == 1) &&
         inc.row[1] !== nonlinear
end

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
    return Incidence(const_val, rrow, union(get_eps(a), get_eps(b)))
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
    return Incidence(const_val, rrow, union(get_eps(a), get_eps(b)))
end

function tfunc(::Val{Core.Intrinsics.mul_float}, @nospecialize(a::Union{Const, Type{Float64}, Incidence}), @nospecialize(b::Union{Const, Type{Float64}, Incidence}))
    if a === Float64 || b === Float64
        return Float64
    end
    if isa(a, Const)
        isa(b, Const) && return builtin_math_tfunc(Core.Intrinsics.mul_float, a, b, union(get_eps(a), get_eps(b)))
        iszero(a.val) && return a
        return Incidence(builtin_math_tfunc(Core.Intrinsics.mul_float, a, b.typ), a.val * b.row, union(get_eps(a), get_eps(b)))
    elseif isa(b, Const)
        iszero(b.val) && return b
        return Incidence(builtin_math_tfunc(Core.Intrinsics.mul_float, a.typ, b), a.row * b.val, union(get_eps(a), get_eps(b)))
    end
    rrow = _zero_row()
    for i in Iterators.flatten((rowvals(a.row), rowvals(b.row)))
        rrow[i] = nonlinear
    end
    return Incidence(builtin_math_tfunc(Core.Intrinsics.mul_float, a.typ, b.typ), rrow, union(get_eps(a), get_eps(b)))
end

function tfunc(::Val{Core.Intrinsics.div_float}, @nospecialize(a::Union{Const, Type{Float64}, Incidence}), @nospecialize(b::Union{Const, Type{Float64}, Incidence}))
    if isa(a, Const) && isa(b, Const)
        return builtin_math_tfunc(Core.Intrinsics.div_float, a, b)
    elseif isa(a, Incidence) && isa(b, Const)
        return Incidence(builtin_math_tfunc(Core.Intrinsics.div_float, a.typ, b), a.row / b.val, union(get_eps(a), get_eps(b)))
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
    return Incidence(builtin_math_tfunc(Core.Intrinsics.div_float, isa(a, Incidence) ? a.typ : a, widenconst(b.typ)), rrow, union(get_eps(a), get_eps(b)))
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
    CC.isalreadyconst(x) && return true
    if isa(x, PartialStruct)
        return is_all_inc_or_const(x.fields)
    end
    if isa(x, CC.Conditional)
        return is_elem_inc_ir_const(x.thentype) && is_elem_inc_ir_const(x.elsetype)
    end
    return false
end

is_all_inc_or_const(args) = all(is_elem_inc_ir_const, args)
is_any_incidence(@nospecialize args...) = any(@nospecialize(x)->isa(x, Incidence), args) && is_all_inc_or_const(args)

@override function CC.builtin_tfunction(interp::DAEInterpreter,
    @nospecialize(f), argtypes::Vector{Any}, sv::Union{AbsIntState,Nothing})

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

    rt = @invoke CC.builtin_tfunction(interp::AbstractInterpreter,
        f::Any, bargtypes::Vector{Any}, sv::Union{AbsIntState,Nothing})

    return rt
end

"""
    fallback_incidence(rt, argtypes)

Determine incidence for an unknown operation with return type `rt` and input types `argtypes`.
As we do not know what the operation does, we generate the broadest possible incidence.
That is to say one which is `nonlinear` in all the places any of the inputs are nonzero,
and with the union of all the epsilon incidences.

See also [`DAECompiler.aggressive_incidence_join`](@ref)
"""
function fallback_incidence(@nospecialize(rt), argtypes::Vector{Any})
    rt === Union{} && return rt
    Base.issingletontype(rt) && return rt
    inci = _fallback_incidence(argtypes)
    inci === nothing && return rt
    return inci
end

function _fallback_incidence!(inci, argtypes::Vector{Any})
    found_any = false
    for a in argtypes
        if isa(a, PartialStruct)
            found_any |= _fallback_incidence!(inci, a.fields)
            continue
        end
        if isa(a, Const)
            found_any = true
        elseif isa(a, Incidence)
            found_any = true
            for i in rowvals(a.row)
                inci.row[i] = nonlinear
            end
            union!(inci.eps, a.eps)
        end
    end
    return found_any
end

function _fallback_incidence(argtypes::Vector{Any})
    inci = Incidence(Float64)
    _fallback_incidence!(inci, argtypes) || return nothing
    return inci
end

function isretblock(ir::IRCode, bbidx::Int)
    bb = ir.cfg.blocks[bbidx]
    length(bb.succs) != 0 && return false
    terminator = ir.stmts[last(bb.stmts)][:inst]
    return isa(terminator, ReturnNode) && isdefined(terminator, :val)
end


function Base.show(io::IO, e::UnsupportedIRException)
    println(io, "Unsupported IR: ", e.msg)
    show(io, e.ir)
end

has_any_genscope(sc::GenScope) = true
has_any_genscope(sc::Scope) = isdefined(sc, :parent) && has_any_genscope(sc.parent)
has_any_genscope(sc::PartialScope) = false
has_any_genscope(sc::PartialStruct) = false # TODO

function _make_argument_lattice_elem(which::Argument, @nospecialize(argt), add_variable!, add_equation!, add_scope!)
    if isa(argt, Const)
        @assert !isa(argt.val, Scope) # Shouldn't have been forwarded
        return argt
    elseif isa(argt, Type) && argt <: Intrinsics.AbstractScope
        return PartialScope(add_scope!(which))
    elseif isa(argt, Type) && argt == equation
        return Eq(add_equation!(which))
    elseif is_non_incidence_type(argt)
        return argt
    elseif CC.isprimitivetype(argt)
        inc = Incidence(add_variable!(which))
        return Incidence(argt, inc.row, inc.eps)
    elseif isa(argt, PartialStruct)
        return PartialStruct(argt.typ, Any[make_argument_lattice_elem(which, f, add_variable!, add_equation!, add_scope!) for f in argt.fields])
    elseif isabstracttype(argt) || ismutabletype(argt) || isa(argt, Union)
        return nothing
    else
        fields =  Any[]
        any = false
        # TODO: This doesn't handle recursion
        for i = 1:length(fieldtypes(argt))
            # TODO: Can we make this lazy?
            ft = fieldtype(argt, i)
            mft = _make_argument_lattice_elem(which, ft, add_variable!, add_equation!, add_scope!)
            if mft === nothing
                push!(fields, Incidence(ft))
            else
                any = true
                push!(fields, mft)
            end
        end
        return any ? PartialStruct(argt, fields) : nothing
    end
end

function make_argument_lattice_elem(which::Argument, @nospecialize(argt), add_variable!, add_equation!, add_scope!)
    mft = _make_argument_lattice_elem(which, argt, add_variable!, add_equation!, add_scope!)
    mft === nothing ? Incidence(argt) : mft
end

function resolve_genscopes(names)
    new_names = OrderedDict{LevelKey, NameLevel}()
    for (key, val) in collect(names)
        if val.children !== nothing
            @reset val.children = resolve_genscopes(val.children)
        end
        if isa(key, Gen)
            idx = 1
            while true
                proposed_name = Symbol(string(key.name, idx))
                if !haskey(names, proposed_name) && !haskey(new_names, proposed_name)
                    new_names[proposed_name] = val
                    break
                end
                idx += 1
            end
        else
            new_names[key] = val
        end
    end
    return new_names
end

"""
    compute_structure(interp::DAEInterpreter,
        mi::MethodInstance, ir::IRCode, debug_config::DebugConfig) -> result::StructuralAnalysisResult

Perform the structural analysis on optimized code of `mi` and return `structure::StructuralAnalysisResult`.
"""
@breadcrumb "ir_levels" function compute_structure(interp::DAEInterpreter,
    mi::MethodInstance, debug_config::DebugConfig)

    if interp.ipo_analysis_mode
        codeinst = CC.get(CC.code_cache(interp), mi, nothing)
        codeinst === nothing && error()
        callee_result = CC.traverse_analysis_results(codeinst) do @nospecialize result
            return result isa Union{DAEIPOResult, UncompilableIPOResult} ? result : nothing
        end
        isa(callee_result, UncompilableIPOResult) && return callee_result
        names = resolve_genscopes(callee_result.names)
        return @set callee_result.names = names
    else
        codeinst = CC.getindex(CC.code_cache(interp), mi)
        cache = (@atomic :monotonic codeinst.inferred)::DAECache
        ir = copy(cache.ir::IRCode)
        record_ir!(debug_config, "unoptimized", ir)
        ir = copy(ir)
        infer_ir!(ir, interp, mi)
        record_ir!(debug_config, "initial_infer", ir)
        ir = run_dae_passes(interp, ir, debug_config)
        infer_ir!(ir, interp, mi)
        ir = run_dae_passes(interp, ir, debug_config)

        result = ipo_dae_analysis!(interp, ir, mi, nothing; debug_config)
        isa(result, UncompilableIPOResult) && return result

        names = resolve_genscopes(result.names)

        return @set result.names = names
    end
end

function refresh_identities(names::OrderedDict{LevelKey, NameLevel})
    new_names = OrderedDict{LevelKey, NameLevel}()
    for (key, val) in names
        if isa(key, Gen)
            key = Gen(Intrinsics.ScopeIdentity(), key.name)
        end
        if val.children !== nothing
            @reset val.children = refresh_identities(val.children)
        end
        new_names[key] = val
    end
    return new_names
end

@noinline function ipo_dae_analysis!(interp::DAEInterpreter, ir::IRCode, mi::MethodInstance, caller::Union{InferenceResult, Nothing};
        debug_config = nothing)
    ir = copy(ir)

    if !interp.ipo_analysis_mode
        for (bb, i) in bbidxiter(ir)
            stmt = ir[SSAValue(i)][:stmt]
            if isa(stmt, EnterNode)
                # We don't model try/catch and our tearing code does not
                # respect the ordering invariants, so just delete these.
                if stmt.catch_dest != 0
                    CC.kill_edge!(ir, bb, stmt.catch_dest)
                end
                ir[SSAValue(i)] = nothing
            end
        end
    end

    domtree = CC.construct_domtree(ir.cfg.blocks)

    var_to_diff = interp.var_to_diff

    eqcallssas = SSAValue[]
    eqssas = Pair{Union{SSAValue, Argument}, Vector{SSAValue}}[]
    epsssas = SSAValue[]
    varssa = Union{SSAValue, Argument}[]
    obsssa = SSAValue[]
    singularity_root_ssas = SSAValue[]
    time_periodic_ssas = SSAValue[]
    var_to_diff = DiffGraph(0)
    warnings = UnsupportedIRException[]

    names = OrderedDict{LevelKey, NameLevel}()

    nsysmscopes = 0

    function add_variable!(i::Union{SSAValue, Argument})
        v = add_vertex!(var_to_diff)
        push!(varssa, i)
        return v
    end

    function add_equation!(i::Union{SSAValue, Argument})
        push!(eqssas, i=>SSAValue[])
        return length(eqssas)
    end

    function add_scope!(i::Union{SSAValue, Argument})
        nsysmscopes += 1
        return nsysmscopes
    end

    # Allocate variables for all arguments f
    nexternalvars = 0 # number of variables that we expect to come in
    nexternaleqs = 0 # number of equation references that we expect to come in
    if caller !== nothing
        argtypes = Any[make_argument_lattice_elem(Argument(i), argt, add_variable!, add_equation!, add_scope!) for (i, argt) in enumerate(ir.argtypes)]
        nexternalvars = length(var_to_diff)
    else
        arg1type = isa(ir.argtypes[1], Const) ? ir.argtypes[1] : Incidence(ir.argtypes[1])
        argtypes = Any[arg1type]
    end

    nobserved = 0
    function add_observed!(i::Int)
        nobserved += 1
        push!(obsssa, SSAValue(i))
        return nobserved
    end

    function add_epsilon!(i::Int)
        push!(epsssas, SSAValue(i))
        return length(epsssas)
    end

    if caller === nothing
        exit_blocks = Int[bb for bb in 1:length(ir.cfg.blocks) if isretblock(ir, bb)]
        if isempty(exit_blocks)
            record_ir!(debug_config, "compute_structure_error", ir)
            return UncompilableIPOResult(warnings, UnsupportedIRException("nonempty exit_blocks", ir))
        end

        function check_dynamic_state!(bb, i)
            if !all(retbb -> dominates(domtree, bb, retbb), exit_blocks)
                ir.stmts[i][:inst] = Expr(:call, throw, DynamicStateError())
                ir.stmts[i][:type] = Union{}
                ir.stmts[i][:flag] = 0x00
                ir.stmts[i][:info] = CC.NoCallInfo()
                return false
            end
            return true
        end
    end

    # Scan the IR, computing equations, variables, diffgraph, etc.
    externally_refined = CC.BitSet()
    nthisvars = 0 # Number of variables declared directly in this function
    for (bb, i) in bbidxiter(ir)
        stmt = ir.stmts[i][:inst]
        if isexpr(stmt, :invoke)
            if is_known_invoke(stmt, variable, ir)
                @assert length(stmt.args) >= 2
                v = add_variable!(SSAValue(i))
                ir.stmts[i][:type] = Incidence(v)
                nthisvars += 1
                CC.push!(externally_refined, i)
            elseif is_known_invoke(stmt, epsilon, ir)
                @assert length(stmt.args) >= 2
                ε = add_epsilon!(i)
                ir.stmts[i][:type] = Incidence(Const(0.0), _zero_row(), BitSet(ε))
                CC.push!(externally_refined, i)
            elseif is_known_invoke(stmt, equation, ir)
                @assert length(stmt.args) == 3
                ir.stmts[i][:type] = Eq(add_equation!(SSAValue(i)))
                CC.push!(externally_refined, i)
            elseif is_equation_call(stmt, ir, #=allow_call=#false)
                push!(eqcallssas, SSAValue(i))
            elseif is_known_invoke(stmt, observed!, ir)
                #if check_dynamic_state!(bb, i)
                    obsnum = add_observed!(i)
                    push!(stmt.args, obsnum)
                #end
            elseif is_known_invoke(stmt, singularity_root!, ir)
                if length(stmt.args) != 3
                    return UncompilableIPOResult(warnings, UnsupportedIRException("singularity_root!() requires a single argument!"))
                end
                #if check_dynamic_state!(bb, i)
                    push!(singularity_root_ssas, SSAValue(i))
                #end
            elseif is_known_invoke(stmt, time_periodic_singularity!, ir)
                if length(stmt.args) != 5
                    return UncompilableIPOResult(warnings, UnsupportedIRException("time_periodic_singularity!() requires three values!"))
                end
                #if check_dynamic_state!(bb, i)
                    push!(time_periodic_ssas, SSAValue(i))
                #end
            elseif is_known_invoke_or_call(stmt, state_ddt, ir)
                @assert length(stmt.args) == 3
                dv = add_variable!(SSAValue(i))
                nthisvars += 1
                ir.stmts[i][:type] = Incidence(dv)
                CC.push!(externally_refined, i)
            elseif is_known_invoke(stmt, sim_time, ir)
                ir.stmts[i][:type] = Incidence(0)
                CC.push!(externally_refined, i)
            else
                # TODO: This shouldn't be required. Base should have done this.
                is_all_const = true
                for i = 3:length(stmt.args)
                    if !isa(argextype(stmt.args[i], ir), Const)
                        is_all_const = false
                        break
                    end
                end
                if is_all_const
                    ir.stmts[i][:flag] |= CC.IR_FLAG_REFINED
                end
            end
        elseif isexpr(stmt, :call)
            if stmt.args[1] === Core.current_scope
                # N.B.: We make the assumption here that all current_scope that
                # was inside EnterScope within the same function has already
                # been folded by SROA, so the only thing left are those that
                # refer to the function's entry scope.
                ir.stmts[i][:type] = PartialStruct(Base.ScopedValues.Scope,
                    Any[PartialKeyValue(Incidence(Base.PersistentDict{Base.ScopedValues.ScopedValue, Any}))])
                ir.stmts[i][:flag] |= CC.IR_FLAG_REFINED
                CC.push!(externally_refined, i)
                continue
            end
            # TODO: This shouldn't be required. Base should have done this.
            is_all_const = true
            for i = 2:length(stmt.args)
                if !isa(argextype(stmt.args[i], ir), Const)
                    is_all_const = false
                    break
                end
            end
            if is_all_const
                ir.stmts[i][:flag] |= CC.IR_FLAG_REFINED
            end
            continue
        elseif isexpr(stmt, :code_coverage_effect) || isexpr(stmt, :throw_undef_if_not)
            continue
        elseif isa(stmt, ReturnNode)
            # Nothing to do for the moment
            continue
        elseif isexpr(stmt, :foreigncall)
            # Could be bad, but we'll just ignore it for now.
            # One of the testcases wants to allocate an array
            continue
        elseif isa(stmt, GlobalRef)
            continue
        elseif isexpr(stmt, :new)
            newT = argextype(stmt.args[1], ir)
            if isa(newT, Const) && newT.val === Intrinsics.ScopeIdentity
                # Allocate the identity now. After inlining, we're guaranteed that
                # every Expr(:new) uniquely corresponds to a scope identity, so this
                # is legal here (bug not before)
                ir.stmts[i][:stmt] = Intrinsics.ScopeIdentity()
                ir.stmts[i][:flag] |= CC.IR_FLAG_REFINED
            end
            continue
        elseif isexpr(stmt, :splatnew)
            continue
        elseif isa(stmt, PhiNode)
            # Take into account control-dependent taint
            ir.stmts[i][:flag] |= CC.IR_FLAG_REFINED
        elseif isa(stmt, PiNode)
            continue
        elseif isa(stmt, GotoIfNot) || isa(stmt, GotoNode)
            continue
        elseif stmt === nothing
            continue
        end
    end

    # TODO better work here?
    method_info = CC.MethodInfo(#=propagate_inbounds=#true, nothing)
    min_world = world = get_inference_world(interp)
    max_world = get_world_counter()
    if caller !== nothing
        @assert interp.ipo_analysis_mode
    end
    analysis_interp = DAEInterpreter(interp; var_to_diff, in_analysis=interp.ipo_analysis_mode)
    irsv = CC.IRInterpretationState(analysis_interp, method_info, ir, mi, argtypes,
                                    world, min_world, max_world)
    ultimate_rt, _ = CC._ir_abstract_constant_propagation(analysis_interp, irsv; externally_refined)
    record_ir!(debug_config, "incidence_propagation", ir)

    # Encountering a `ddt` during abstract interpretation can add variables,
    # count them here
    var_to_diff = complete(var_to_diff)
    diff_to_var = invview(var_to_diff)
    nimplicitexternalvars = 0
    nimplicitinternalvars = 0
    for i = (nexternalvars + nthisvars + 1):length(var_to_diff)
        # If this is a derivative of an external var, consider it external
        while diff_to_var[i] !== nothing
            i = diff_to_var[i]
        end
        if i <= nexternalvars
            nimplicitexternalvars += 1
        else
            nimplicitinternalvars += 1
        end
    end

    if caller === nothing &&  ultimate_rt === Union{}
        # TODO: Can we find out what this error is? It's a bit tricky, because
        # we don't necessarily know what the parameterization is, but since we
        # have proven that it unconditionally errors, we should be able to do
        # something here.
        return UncompilableIPOResult(warnings, UnsupportedIRException("Function was discovered to unconditionally error", ir))
    end

    # recalculate domtree (inference could have changed the cfg)
    domtree = CC.construct_domtree(ir.cfg.blocks)

    # We use the _ir_abstract_constant_propagation pass for three things:
    # 1. To establish incidence
    # 2. To constant propagate scope information that may not have been
    #    available at inference time
    # 3. To constant propagate the relationship between variables and
    #    state_ddt.
    function record_this_scope!(scope::Union{PartialScope, PartialStruct, Intrinsics.AbstractScope}, args...)
        record_scope!(ir, names, scope, args...)
    end
    record_this_scope!(scope::Const, args...) = record_this_scope!(scope.val, args...)

    # First record all the scope information
    for ssa in varssa
        isa(ssa, SSAValue) || continue
        inst = ir[ssa][:inst]::Union{Nothing, Expr}
        if inst !== nothing # && check_dynamic_state!(block_for_inst(ir, ssa.id), ssa.id)
            type = ir[ssa][:type]
            if is_known_invoke(inst, variable, ir)
                var_num = idnum(type)::Int
                scope = argextype(inst.args[3], ir)
                isa(scope, Incidence) && (scope = scope.typ)
                if (!isa(scope, Const) || !isa(scope.val, Intrinsics.AbstractScope)) && !is_valid_partial_scope(scope)
                    push!(warnings,
                        UnsupportedIRException(
                            "Saw non-constant name (`$scope`) for variable $(var_num) (SSA $ssa)",
                            ir))
                elseif isa(scope, Const) && scope.val === Scope()
                    # Explicitly unnamed
                else
                    record_this_scope!(scope, varssa, var_num, @o _.var)
                end
                # Delete the scope. It's only used for debug and we
                # expect it to be constant anyway. However, if it's
                # an SSAValue (with Const type), we don't want to worry
                # about having to schedule it later.
                inst.args[3] = nothing
            elseif is_known_invoke_or_call(inst, state_ddt, ir)
                vtyp = argextype(inst.args[3], ir)
                if !isa(vtyp, Incidence)
                    return UncompilableIPOResult(warnings, UnsupportedIRException("Failed to find matching variable definition for state_ddt at $ssa", ir))
                end
                vp = vtyp.row
                if length(collect(rowvals(vp))) != 1
                    return UncompilableIPOResult(warnings, UnsupportedIRException("Expected single variable incidence for state_ddt, got $vtyp at $ssa", ir))
                end
                v = only(rowvals(vp))
                if !isone(vp[v])
                    vname = get_variable_name(names, var_to_diff, v-1)
                    return UncompilableIPOResult(warnings, UnsupportedIRException("Expected incidence of variable `$vname` to state_ddt to be 1.0, got $vtyp at $ssa", ir))
                end
                if var_to_diff[v-1] !== nothing
                    vname = get_variable_name(names, var_to_diff, v-1)
                    return UncompilableIPOResult(warnings, UnsupportedIRException("Duplicated state_ddt for variable `$vname` at $ssa", ir))
                end
                add_edge!(var_to_diff, v-1, idnum(type))
                inst.args[3] = nothing
            end
        end
    end

    for ssa in obsssa
        inst = ir[ssa][:inst]
        if inst !== nothing && length(inst.args) == 5
            # If inst is nothing, there was an error path, allow this for now.
            inst = inst::Expr
            scope = argextype(inst.args[4], ir)
            isa(scope, Incidence) && (scope = scope.typ)
            if !isa(scope, Const) && !is_valid_partial_scope(scope)
                record_ir!(debug_config, "compute_structure_error", ir)
                return UncompilableIPOResult(warnings, UnsupportedIRException("Expected scope ($(scope), SSA %$(ssa.id)) to be a const", ir))
            elseif isa(scope, Const) && scope.val === Scope()
                # Explicitly unnamed
            else
                var_num = inst.args[end]::Int
                record_this_scope!(scope, obsssa, var_num, @o _.obs)
            end
            # Delete the scope (see above)
            inst.args[4] = nothing
        end
    end

    for ssa in epsssas
        inst = ir[ssa][:inst]
        if inst !== nothing
            scope = argextype(inst.args[end], ir)
            isa(scope, Incidence) && (scope = scope.typ)
            if !isa(scope, Const) && !is_valid_partial_scope(scope)
                record_ir!(debug_config, "compute_structure_error", ir)
                return UncompilableIPOResult(warnings, UnsupportedIRException("Expected scope ($(scope), SSA %$(ssa.id)) to be a const", ir))
            elseif isa(scope, Const) && scope.val === Scope()
                # Explicitly unnamed
            else
                record_this_scope!(scope, epsssas, epsnum(ir[ssa][:type]), @o _.eps)
            end
            # Delete the scope
            inst.args[end] = nothing
        end
    end

    ic_nzc = 0
    vcc_nzc = 0
    for ssa in singularity_root_ssas
        inst = ir[ssa][:inst]::Expr
        incidence = argextype(inst.args[3], ir)
        isa(incidence, Incidence) || continue

        if !has_simple_incidence_info(incidence) || !has_dependence(incidence)
            return UncompilableIPOResult(warnings, UnsupportedIRException("singularity_root!() requires one value with incidence on time or state!", ir))
        end

        vcc_nzc += 1
        push!(inst.args, vcc_nzc)
    end

    unwrap_const(x::Const) = x.val
    unwrap_const(x) = x
    for ssa in time_periodic_ssas
        inst = ir[ssa][:inst]::Expr
        offset, period, count = unwrap_const.(inst.args[3:5])

        argts = argextype.([offset, period, count], Ref(ir))
        if !all(has_simple_incidence_info.(argts)) || any(has_dependence.(argts))
            return UncompilableIPOResult(warnings, UnsupportedIRException("time_periodic_singularity!() requires three values with no incidence on time or state!", ir))
        end

        # We will slurp this up into an `IterativeCallback`.  Once again, we assign a
        # unique identifier to each invoke, but this is more for debugging than anything else.
        ic_nzc += 1
        push!(inst.args, ic_nzc)
    end

    neqs = length(eqssas)
    graph = BipartiteGraph(neqs, length(var_to_diff))
    solvable_graph = BipartiteGraph(neqs, length(var_to_diff))
    eq_to_diff = DiffGraph(neqs)
    total_incidence = Vector{Any}(undef, neqs)

    eq_callee_mapping = Vector{Union{Nothing, Pair{SSAValue, Int}}}(nothing, neqs)
    var_callee_mapping = Vector{Union{Nothing, Pair{SSAValue, Int}}}(nothing, length(varssa))

    for (ieq, (ssa, _)) in enumerate(eqssas)
        isa(ssa, Argument) && continue
        eqinst = ir[ssa][:inst]
        if eqinst === nothing
            continue
        end
        eqinst = eqinst::Expr
        @assert ieq == ir[ssa][:type].id
        scope = argextype(eqinst.args[3], ir)
        eqnum = idnum(ir[ssa][:type])::Int

        if !isa(scope, Const) && !is_valid_partial_scope(scope)
            push!(warnings, UnsupportedIRException("Saw non-constant name (`$scope`) for equation $(eqnum) (SSA $ssa)", ir))
        elseif isa(scope, Const) && scope.val === Scope()
            # Explicitly unnamed
        else
            record_this_scope!(scope, varssa, eqnum, @o _.eq)
        end
        # Delete scope (see above)
        eqinst.args[3] = nothing
    end

    for ssa in eqcallssas
        eqcall = ir[ssa][:inst]
        if eqcall === nothing
            continue
        end
        eqcall = eqcall::Expr

        # check_dynamic_state!(block_for_inst(ir, ssa.id), ssa.id) || continue

        eqeq = argextype(eqcall.args[2], ir)
        if !isa(eqeq, Eq)
            return UncompilableIPOResult(warnings, UnsupportedIRException("Equation call at $ssa has unknown equation reference.", ir))
        end
        ieq = eqeq.id

        eqssaval = eqcall.args[3]
        if !isa(eqssaval, SSAValue)
            isa(eqssaval, Argument) && continue
            if !iszero(eqssaval)
                return UncompilableIPOResult(warnings, UnsupportedIRException(
                    "Equation call for $ieq at $ssa is set to $eqssaval. The system is unsolvable.", ir))
            end
            continue
        end
        inc = ir[eqssaval][:type]
        if !isa(inc, Incidence)
            if caller === nothing
                record_ir!(debug_config, "compute_structure_error", ir)
                return UncompilableIPOResult(warnings, UnsupportedIRException("Expected incidence analysis to produce result for $eqssaval, got $inc", ir))
            else
                total_incidence[ieq] = isassigned(total_incidence, ieq) ? tmerge(inc, total_incidence[ieq]) : inc
                continue
            end
        end
        for (v, coeff) in zip(rowvals(inc.row), nonzeros(inc.row))
            v == 1 && continue
            add_edge!(graph, ieq, v-1)
            coeff !== nonlinear && add_edge!(solvable_graph, ieq, v-1)
        end
        if isassigned(total_incidence, ieq)
            total_incidence[ieq] += inc
        else
            total_incidence[ieq] = inc
        end

        push!(eqssas[ieq][2], ssa)
    end

    # For easier debugging, delete all the statements that are dead, but don't renumber things
    for (i, bb) in enumerate(ir.cfg.blocks)
        (length(bb.preds) == 0 && i != 1) || continue
        for i in bb.stmts
            ir[SSAValue(i)] = nothing
            ir[SSAValue(i)][:type] = Union{}
        end
    end

    structure = SystemStructure(complete(var_to_diff), complete(eq_to_diff), graph, solvable_graph, nothing, false)

    ninternalvars = 0
    if caller !== nothing
        handler_at, handlers = CC.compute_trycatch(ir, Core.Compiler.BitSet())
        for i = 1:length(ir.stmts)
            inst = ir[SSAValue(i)]
            stmt = inst[:stmt]
            if isexpr(stmt, :invoke)
                if is_known_invoke(stmt, observed!, ir)
                    continue
                end

                # TODO: Factor this out into base
                info = inst[:info]
                mi = stmt.args[1]
                if isa(info, CC.ConstCallInfo)
                    if length(info.results) != 1
                        # TODO: When does this happen? Union split?
                        continue
                    end
                    cpr = info.results[1]
                    # ConcreteResult, we don't need to bother handling, because
                    # we know that anything with interesting intrinsics in it is
                    # not concrete eval eligible
                    isa(cpr, CC.ConcreteResult) && continue
                    cpr::Union{ConstPropResult, CC.VolatileInferenceResult}
                    infr = isa(cpr, ConstPropResult) ? cpr.result : cpr.inf_result
                    result = CC.traverse_analysis_results(infr) do @nospecialize result
                        return result isa Union{DAEIPOResult, UncompilableIPOResult} ? result : nothing
                    end
                else
                    codeinst = CC.get(CC.code_cache(interp), mi, nothing)
                    codeinst === nothing && continue
                    result = CC.traverse_analysis_results(codeinst) do @nospecialize result
                        return result isa Union{DAEIPOResult, UncompilableIPOResult} ? result : nothing
                    end
                    result === nothing && continue
                end

                if isa(result, UncompilableIPOResult)
                    # TODO: stack these
                    return result
                end
                append!(warnings, result.warnings)

                nvars = nexternalvars+nthisvars+nimplicitexternalvars+nimplicitinternalvars+ninternalvars
                @assert nvars == length(var_to_diff)
                ninternalvars += result.ntotalvars - result.nexternalvars

                # The first `n` vars are external from the perspective of our
                # callee, so will get replaced by some linear combination of
                # other variables. The remainder get added as internal vars
                # at the end of ours.
                offset = nvars - result.nexternalvars
                for i = (result.nexternalvars+1):result.ntotalvars
                    nv = add_vertex!(var_to_diff)
                    push!(var_callee_mapping, SSAValue(i)=>i)
                    @assert nv == i+offset
                end
                for i = (result.nexternalvars+1):result.ntotalvars
                    if result.var_to_diff[i] !== nothing
                        var_to_diff[i+offset] = result.var_to_diff[i]+offset
                    end
                end

                callee_argtypes = CC.va_process_argtypes(CC.optimizer_lattice(analysis_interp),
                    CC.collect_argtypes(analysis_interp, stmt.args[2:end], nothing, irsv), mi)
                eq_offset = length(total_incidence)

                if isempty(result.total_incidence) && isempty(result.names)
                    continue
                end

                mapping = CalleeMapping(CC.optimizer_lattice(analysis_interp), callee_argtypes, result, nvars, eq_offset)

                for i = 1:result.nexternaleqs
                    mapped_eq = mapping.eq_mapping[i]
                    if isassigned(total_incidence, mapped_eq)
                        total_incidence[mapped_eq] += result.total_incidence[i]
                    else
                        total_incidence[mapped_eq] = result.total_incidence[i]
                    end
                end

                for (ieq, inc) in enumerate(result.total_incidence[(result.nexternaleqs+1):end])
                    push!(total_incidence, apply_linear_incidence(inc, mapping))
                    push!(eq_callee_mapping, SSAValue(i)=>ieq)
                end

                function resolve_ipo_scope(key::PartialScope)
                    local newscope
                    if isa(key.id, Base.ScopedValues.ScopedValue)
                        # Scope from ScopedValue
                        scope_idx = i
                        while true
                            handler_idx = handler_at[scope_idx][1]
                            if handler_idx == 0
                                # Inherits scope from outside
                                newscope = key
                                break
                            end
                            scope_idx = handlers[handler_idx].enter_idx
                            en = ir[SSAValue(scope_idx)][:stmt]::EnterNode
                            if isdefined(en, :scope)
                                scopet = argextype(en.scope, ir)
                                newscope = nothing
                                if isa(scopet, PartialStruct) &&
                                scopet.typ === Base.ScopedValues.Scope &&
                                isa(scopet.fields[1], PartialKeyValue)
                                    𝕃 = CC.typeinf_lattice(analysis_interp)
                                    val = getkeyvalue_tfunc(𝕃,
                                        scopet.fields[1], Const(key.id))
                                    val = CC.getfield_tfunc(𝕃, val, Const(1))
                                    if isa(val, Const)
                                        newscope = val.val
                                    else
                                        newscope = val
                                    end
                                end
                                break
                            end
                        end
                    else
                        newscope = mapping.applied_scopes[idnum(key)]
                    end
                    return newscope
                end

                if !isempty(result.names)
                    for (key, val) in refresh_identities(result.names)
                        if isa(key, PartialScope)
                            newscope = resolve_ipo_scope(key)
                            if !isa(newscope, Union{Intrinsics.AbstractScope, PartialScope}) && !is_valid_partial_scope(newscope)
                                push!(warnings, UnsupportedIRException(
                                    "Saw non-constant name (`$newscope`) for name with symbolic scope $key (SSA $i)",
                                    ir))
                                continue
                            end
                            merge_scopes!(names, newscope, val, offset, -typemax(Int), eq_offset, -typemax(Int))
                        else
                            merge_scopes!(names, key, val, offset, -typemax(Int), eq_offset, -typemax(Int))
                        end
                    end
                end
            end
        end
    end

    for i = 1:length(total_incidence)
        if !isassigned(total_incidence, i)
            # TODO: Remove the equations altogether?
            total_incidence[i] = Const(0.)
        end
    end

    return DAEIPOResult(ir, argtypes,
        nexternalvars+nimplicitexternalvars,
        nexternalvars+nthisvars+nimplicitexternalvars+nimplicitinternalvars+ninternalvars,
        nsysmscopes,
        nexternaleqs,
        var_to_diff,
        ultimate_rt, total_incidence, eq_callee_mapping, var_callee_mapping,
        names, nobserved, length(epsssas), ic_nzc, vcc_nzc,
        warnings)
end

function get_variable_name(names::OrderedDict{LevelKey, NameLevel}, var_to_diff, var_idx)
    var_names = build_var_names(names, var_to_diff)
    return var_names[var_idx]
end

function maybe_realpath(path::AbstractString)
    try
        return realpath(path)
    catch
        # If the path doesn't exist (e.g. `runtime.jl` doesn't exist in our JuliaHub builds)
        # just return the normal path.
        return path
    end
end

function get_inline_backtrace(ir::IRCode, v::SSAValue)
    frames = Base.StackTrace();
    runtime_jl_path = maybe_realpath(joinpath(dirname(pathof(@__MODULE__)), "runtime.jl"))

    frames = Base.StackTrace();
    for lineinfo in Base.IRShow.buildLineInfoNode(ir.debuginfo, nothing, v.id)
        btpath = maybe_realpath(expanduser(string(lineinfo.file)))
        if btpath != runtime_jl_path
            frame = Base.StackFrame(lineinfo.method, lineinfo.file, lineinfo.line)
            push!(frames, frame)
        end
    end
    return frames
end

function walk_dict(names::OrderedDict{LevelKey, NameLevel}, stack::Vector{<:LevelKey})
    for i = length(stack):-1:2
        s = stack[i]
        if !haskey(names, s)
            names[s] = NameLevel(OrderedDict{LevelKey, NameLevel}())
        end
        names = names[s].children
    end
    return names
end

is_valid_partial_scope(_) = false
is_valid_partial_scope(ps::PartialScope) = true
function is_valid_partial_scope(ps::PartialStruct)
    if ps.typ === Scope
        isa(ps.fields[2], Const) || return false
        isa(ps.fields[2].val, Symbol) || return false
        return is_valid_partial_scope(ps.fields[1])
    elseif ps.typ === GenScope
        isa(ps.fields[1], Const) || return false
        return is_valid_partial_scope(ps.fields[2])
    else
        return false
    end
end

function sym_stack(ps::PartialStruct)
    if ps.typ === Scope
        sym = (ps.fields[2]::Const).val::Symbol
        return pushfirst!(sym_stack(ps.fields[1]), sym)
    else
        @assert ps.typ === GenScope
        stack = sym_stack(ps.fields[2])
        scope_identity = ((ps.fields[1]::Const).val)::Intrinsics.ScopeIdentity
        stack[1] = Gen(scope_identity, stack[1])
        return stack
    end
end

sym_stack(ps::PartialScope) = LevelKey[ps]
function record_scope!(ir::IRCode, names::OrderedDict{LevelKey, NameLevel}, scope::Union{Scope, GenScope, PartialStruct, PartialScope},
                       varssa::Vector, idx::Int, lens)

    stack = sym_stack(scope)
    name_dict = walk_dict(names, stack)
    existing = get!(name_dict, stack[1], NameLevel())
    if lens(existing) !== nothing
        new = get_inline_backtrace(ir, varssa[idx])
        existing = get_inline_backtrace(ir, varssa[lens(names[stack[1]])])

        io = IOBuffer()
        Base.show_backtrace(io, new)
        print(io, "\n")
        Base.show_backtrace(io, existing)
        @warn "Duplicate $(lens) definition for scope $scope" * String(take!(io))
    else
        name_dict[stack[1]] = set(existing, lens, idx)
    end
end

function merge_scopes!(names::OrderedDict{LevelKey, NameLevel}, key::LevelKey, val::NameLevel,
        varoffset::Int, obsoffset::Int, eqoffset::Int, epsoffset::Int)

    haskey(names, key) || (names[key] = NameLevel())
    existing = names[key]
    for (offset, lens) in ((varoffset, @o _.var), (obsoffset, @o _.obs),
                           (eqoffset, @o _.eq), (epsoffset, @o _.eps))
        if lens(val) !== nothing
            if lens(existing) !== nothing
                @warn "Duplicate $(lens) for $key"
            else
                existing = set(existing, lens, lens(val)+offset)
            end
        end
    end

    if val.children !== nothing
        if existing.children === nothing
            @reset existing.children = OrderedDict{LevelKey, NameLevel}()
        end
        child_dict = existing.children
        for (val_key, val_val) in val.children
            merge_scopes!(child_dict, val_key, val_val,
                varoffset, obsoffset, eqoffset, epsoffset)
        end
    end
    names[key] = existing
end

function merge_scopes!(names::OrderedDict{LevelKey, NameLevel}, key::Union{Scope, PartialStruct}, val::NameLevel,
        varoffset::Int, obsoffset::Int, eqoffset::Int, epsoffset::Int)

    stack = sym_stack(key)
    if isempty(stack)
        @assert val.var == val.obs == val.eps == val.eq == nothing
        for (k, v) in pairs(val.children)
            merge_scopes!(names, k, v, varoffset, obsoffset, eqoffset, epsoffset)
        end
        return
    end
    while length(stack) > 1
        val = NameLevel(OrderedDict{LevelKey, NameLevel}(pop!(stack) => val))
    end
    merge_scopes!(names, only(stack), val, varoffset, obsoffset, eqoffset, epsoffset)
end

function peephole_pass!(ir::IRCode)
    compact = IncrementalCompact(ir)
    for i in compact
        oldinst = compact.result[compact.result_idx-1][:inst]
        newinst = check_inst(oldinst, compact)
        compact[SSAValue(compact.result_idx-1)] = newinst
    end
    compact!(finish(compact))
end

function check_inst(@nospecialize(inst), compact::IncrementalCompact)
    isexpr(inst, :call) || return inst
    if is_known_call(inst, Base.add_int, compact)
        if inst.args[2] isa Integer
            arg = inst.args[2]
            arg == zero(arg) && return inst.args[3]
        elseif inst.args[3] isa Integer
            arg = inst.args[3]
            arg == zero(arg) && return inst.args[2]
        end
    elseif is_known_call(inst, Base.sub_int, compact)
        if inst.args[2] isa Integer
            arg = inst.args[2]
            arg == zero(arg) && return Expr(:call, Base.neg_int, inst.args[3])
        elseif inst.args[3] isa Integer
            arg = inst.args[3]
            arg == zero(arg) && return inst.args[2]
        end
    elseif is_known_call(inst, Base.mul_int, compact)
        if inst.args[2] isa Integer
            arg = inst.args[2]
            arg == zero(arg) && return zero(arg)
            arg == one(arg) && return inst.args[3]
            arg == -one(arg) && return Expr(:call, Base.neg_int, inst.args[3])
        elseif inst.args[3] isa Integer
            arg = inst.args[3]
            arg == zero(arg) && return zero(arg)
            arg == one(arg) && return inst.args[2]
            arg == -one(arg) && return Expr(:call, Base.neg_int, inst.args[2])
        end
    elseif is_known_call(inst, Base.add_float, compact)
        if inst.args[2] isa AbstractFloat
            arg = inst.args[2]
            arg == zero(arg) && return inst.args[3]
        elseif inst.args[3] isa AbstractFloat
            arg = inst.args[3]
            arg == zero(arg) && return inst.args[2]
        end
    elseif is_known_call(inst, Base.sub_float, compact)
        if inst.args[2] isa AbstractFloat
            arg = inst.args[2]
            if iszero(arg)
                if inst.args[3] isa AbstractFloat
                    return -inst.args[3]
                end
                return Expr(:call, Base.neg_float, inst.args[3])
            end
        elseif inst.args[3] isa AbstractFloat
            arg = inst.args[3]
            arg == zero(arg) && return inst.args[2]
        end
    elseif is_known_call(inst, Base.neg_float, compact)
        if inst.args[2] isa AbstractFloat
            arg = inst.args[2]
            return -arg
        elseif inst.args[2] isa SSAValue
            inst′ = compact[inst.args[2]][:inst]
            if is_known_call(inst′, Base.neg_float, compact)
                return inst′.args[2]
            end
        end
    elseif is_known_call(inst, Base.mul_float, compact)
        if inst.args[2] isa AbstractFloat
            arg = inst.args[2]
            arg == zero(arg) && return zero(arg)
            arg == one(arg) && return inst.args[3]
            arg == -one(arg) && return Expr(:call, Base.neg_float, inst.args[3])
        elseif inst.args[3] isa AbstractFloat
            arg = inst.args[3]
            arg == zero(arg) && return zero(arg)
            arg == one(arg) && return inst.args[2]
            arg == -one(arg) && return Expr(:call, Base.neg_float, inst.args[2])
        end
    elseif is_known_call(inst, Base.div_float, compact)
        if inst.args[2] isa AbstractFloat
            arg = inst.args[2]
            # arg == zero(arg) && return zero(arg) maybe inf?
        elseif inst.args[3] isa AbstractFloat
            arg = inst.args[3]
            # arg == zero(arg) && return zero(arg)
            arg == one(arg) && return inst.args[2]
            arg == -one(arg) && return Expr(:call, Base.neg_float, inst.args[2])
        end
    end
    return inst
end

"""
    compact_cfg(ir; max_iterations = 10)

Run `cfg_simplify!()` and `compact!()` in a loop, until we either hit the maximum number
of allowed iterations or hit a fixed point.
"""
function compact_cfg(ir; max_iterations::Int = 10)
    last_length = length(ir.stmts)
    last_num_bbs = length(ir.cfg.blocks)
    for idx in 1:max_iterations
        ir = compact!(cfg_simplify!(ir), true)
        # Good debugging info
        #@info("compact", idx, length(ir.stmts))
        if length(ir.stmts) == last_length && length(ir.cfg.blocks) == last_num_bbs
            break
        end
        last_length = length(ir.stmts)
        last_num_bbs = length(ir.cfg.blocks)
    end
    return ir
end
