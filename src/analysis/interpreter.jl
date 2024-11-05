# AbstractInterpreter integration
# ===============================

using Core: CodeInfo, MethodInstance, CodeInstance, SimpleVector, MethodMatch, MethodTable
using .CC: AbstractInterpreter, NativeInterpreter, InferenceParams, OptimizationParams,
    InferenceResult, InferenceState, OptimizationState, WorldRange, WorldView, ArgInfo,
    StmtInfo, MethodCallResult, ConstCallResult, ConstPropResult, MethodTableView,
    CachedMethodTable, InternalMethodTable, OverlayMethodTable, CallMeta, CallInfo,
    IRCode, LazyDomtree, IRInterpretationState, set_inlineable!, block_for_inst,
    BitSetBoundedMinPrioritySet, AbsIntState, Future
using Base: IdSet
using StateSelection: DiffGraph

using Cthulhu
using ChainRulesCore

# similer to `Core.Compiler.Effects` but for DAE-specific information
struct DAEInfo
    has_dae_intrinsics::Bool
    has_scoperead::Bool
    function DAEInfo(
        has_dae_intrinsics::Bool,
        has_scoperead::Bool)
        return new(
            has_dae_intrinsics,
            has_scoperead)
    end
end
function DAEInfo(
    info::DAEInfo = DAEInfo(
        #=has_dae_intrinsics::Bool=# false,
        #=has_scoperead::Bool=# false);
    has_dae_intrinsics::Bool = info.has_dae_intrinsics,
    has_scoperead::Bool = info.has_scoperead)
    return DAEInfo(has_dae_intrinsics, has_scoperead)
end
function merge_daeinfo(old::DAEInfo, new::DAEInfo)
    return DAEInfo(
        old.has_dae_intrinsics | new.has_dae_intrinsics,
        old.has_scoperead | new.has_scoperead)
end

struct DAEGlobalCache
    cache::IdDict{MethodInstance,CodeInstance}
end
DAEGlobalCache() = @new_cache DAEGlobalCache(IdDict{MethodInstance,CodeInstance}())
Base.empty!(cache::DAEGlobalCache) = empty!(cache.cache)
CC.get(wvc::CC.WorldView{DAEGlobalCache}, key, default) =
    Base.get(wvc.cache.cache, key, default)
CC.getindex(wvc::CC.WorldView{DAEGlobalCache}, key) =
    Base.getindex(wvc.cache.cache, key)
CC.get(cache::DAEGlobalCache, key, default) =
    Base.get(cache.cache, key, default)
Base.get(cache::DAEGlobalCache, key, default) =
    Base.get(cache.cache, key, default)
CC.haskey(wvc::CC.WorldView{DAEGlobalCache}, key) =
    Base.haskey(wvc.cache.cache, key)
CC.setindex!(cache::DAEGlobalCache, val, key) =
    Base.setindex!(cache.cache, val, key)
CC.getindex(cache::DAEGlobalCache, key) =
    Base.getindex(cache.cache, key)

const GLOBAL_CODE_CACHE = @new_cache Dict{UInt, DAEGlobalCache}()
function get_code_cache(args...)
    @static if !(VERSION ≥ v"1.11.0-DEV.1255")
        # XXX Avoid storing analysis results into a cache that persists across the
        #     precompilation, as pkgimage currently doesn't support serializing externally
        #     created `CodeInstance`. Otherwise, `CodeInstance`s created by DAEInterpreter,
        #     containing DAECompiler-specific data structures, will leak into the native
        #     code cache, likely causing segfaults or undefined behavior.
        #     (see https://github.com/JuliaLang/julia/issues/48453).
        if !iszero(@ccall jl_generating_output()::Cint)
            return DAEGlobalCache()
        end
    end
    return get!(DAEGlobalCache, GLOBAL_CODE_CACHE, compute_hash(args...))
end

function compute_hash(objs...)
    @assert length(objs) ≠ 0 "given no objects to be hashed"
    return _compute_hash(objs...)
end
_compute_hash(o, objs...) = hash(o, _compute_hash(objs...))
let hash_seed = rand(UInt)
    global _compute_hash() = hash_seed
end

# AbstractInterpreter interface
# -----------------------------

struct DAEInterpreter <: AbstractInterpreter
    world::UInt64
    method_table::MethodTableView
    inf_cache::Vector{InferenceResult}
    code_cache::DAEGlobalCache
    dae_cache::IdDict{InferenceResult,DAEInfo}

    var_to_diff::DiffGraph
    var_kind::Vector{VarEqKind}
    eq_kind::Vector{VarEqKind}

    # For debugging/Cthulhu integration only (TODO: Only collect this optionally)
    unopt::Dict{Union{MethodInstance,InferenceResult}, Cthulhu.InferredSource}
    remarks::Dict{Union{MethodInstance,InferenceResult}, Cthulhu.PC2Remarks}

    ipo_analysis_mode::Bool
    in_analysis::Bool

    function DAEInterpreter(world::UInt = get_world_counter();
        method_table::Union{Nothing,MethodTable} = nothing,
        inf_cache::Vector{InferenceResult} = InferenceResult[],
        code_cache::Union{DAEGlobalCache, Nothing} = nothing,
        dae_cache::IdDict{InferenceResult,DAEInfo} = IdDict{InferenceResult,DAEInfo}(), # we intentionally don't track this with `new_cache` as its not used again after inference is done and is kept only for debugging
        var_to_diff::DiffGraph = DiffGraph(0),
        var_kind::Vector{VarEqKind} = VarEqKind[],
        eq_kind::Vector{VarEqKind} = VarEqKind[],
        ipo_analysis_mode::Bool = false,
        in_analysis::Bool = false)
        if code_cache === nothing
            code_cache = get_code_cache(world, method_table, ipo_analysis_mode)
        end
        if method_table !== nothing
            method_table = CachedMethodTable(OverlayMethodTable(world, method_table))
        else
            method_table = CachedMethodTable(InternalMethodTable(world))
        end
        return new(world, method_table, inf_cache, code_cache, dae_cache,
            var_to_diff, var_kind, eq_kind,
            Dict{Union{MethodInstance,InferenceResult}, Cthulhu.InferredSource}(),
            Dict{Union{MethodInstance,InferenceResult}, Cthulhu.PC2Remarks}(),
            ipo_analysis_mode, in_analysis)
    end
    function DAEInterpreter(interp::DAEInterpreter;
        world::UInt = interp.world,
        method_table::MethodTableView = interp.method_table,
        inf_cache::Vector{InferenceResult} = interp.inf_cache,
        code_cache::DAEGlobalCache = interp.code_cache,
        dae_cache::IdDict{InferenceResult,DAEInfo} = interp.dae_cache,
        var_to_diff::DiffGraph = DiffGraph(0),
        var_kind::Vector{VarEqKind} = VarEqKind[],
        eq_kind::Vector{VarEqKind} = VarEqKind[],
        ipo_analysis_mode = interp.ipo_analysis_mode,
        in_analysis = interp.in_analysis)
        return new(world, method_table, inf_cache, code_cache, dae_cache,
            var_to_diff, var_kind, eq_kind,
            interp.unopt,
            interp.remarks,
            ipo_analysis_mode,
            in_analysis)
    end
end

Diffractor.disable_forward(interp::DAEInterpreter) = CC.NativeInterpreter(interp.world)

function CC.InferenceParams(::DAEInterpreter)
    args = (;
    assume_bindings_static=true,
    ignore_recursion_hardlimit=true)
    if VERSION < v"1.12.0-DEV.1017"
        args = (; unoptimize_throw_blocks=false, args...)
    end
    return CC.InferenceParams(; args...)
end
function CC.OptimizationParams(::DAEInterpreter)
    opt_params = CC.OptimizationParams(;
        inline_cost_threshold=205,
        compilesig_invokes=false,
        assume_fatal_throw=true,
        preserve_local_sources=true)
    return opt_params
end
#=CC.=#get_inference_world(interp::DAEInterpreter) = interp.world
CC.get_inference_cache(interp::DAEInterpreter) = interp.inf_cache

if Base.__has_internal_change(v"1.12-alpha", :methodspecialization)
    """
        struct AnalysisSpec

    The cache partition for DAECompiler analysis results. This is essentially
    equivalent to a regular type inference, except that optimization ir prohibited
    from inlining any functions that have frules and we perform the DAE analysis
    on every ir after optimization.
    """
    struct AnalysisSpec; end

    """
        struct RHSSpec

    Cache partition for the RHS
    """
    struct RHSSpec
        key::TornCacheKey
        ordinal::Int
    end

    Base.show(io::IO, ms::MethodSpecialization{RHSSpec}) = print(io, "RHS Spec#$(ms.data.ordinal) for ", ms.def)


    """
        struct SICMSpec

    Cache partition for the state-invariant prologue
    """
    struct SICMSpec
        key::TornCacheKey
    end

    Base.show(io::IO, ms::MethodSpecialization{SICMSpec}) = print(io, "SICM Spec for ", ms.def)

    function CC.code_cache(interp::DAEInterpreter)
        if interp.ipo_analysis_mode
            return CC.WorldView(
                CC.InternalCodeCache(Core.MethodSpecialization{AnalysisSpec}),
                CC.WorldRange(CC.get_inference_world(interp)))
        else
            return interp.code_cache
        end
    end
else
    CC.cache_owner(interp::DAEInterpreter) = interp.code_cache
    CC.method_table(interp::DAEInterpreter) = interp.method_table
end

# abstract interpretation
# -----------------------

# TODO override `DAEInfo` propagation with const-prop' callsite

function merge_daeinfo!(interp::DAEInterpreter, result::InferenceResult, info::DAEInfo)
    return interp.dae_cache[result] = merge_daeinfo(
        get(interp.dae_cache, result, DAEInfo()), info)
end

@override function CC.InferenceState(result::InferenceResult,
    src::CodeInfo, cache_mode::UInt8, interp::DAEInterpreter)
    frame = @invoke CC.InferenceState(result::InferenceResult,
        src::CodeInfo, cache_mode::UInt8, interp::AbstractInterpreter)
    if cache_mode !== CC.CACHE_MODE_NULL
        interp.dae_cache[result] = DAEInfo()
    end
    return frame
end

struct IncompleteInferenceException end
function Base.showerror(io::IO, e::IncompleteInferenceException)
    print(io, "Inference failed to discover a DAECompiler intrinsic during its initial scan.")
end

function structural_inc_ddt(var_to_diff::DiffGraph, var_kind::Vector{VarEqKind}, inc::Union{Incidence, Const})
    isa(inc, Const) && return Const(zero(inc.val))
    r = _zero_row()
    function get_or_make_diff(v_offset::Int)
        v = v_offset - 1
        var_to_diff[v] !== nothing && return var_to_diff[v] + 1
        dv = add_vertex!(var_to_diff)
        push!(var_kind, var_kind[v])
        add_edge!(var_to_diff, v, dv)
        return dv + 1
    end
    for (v_offset, coeff) in zip(rowvals(inc.row), nonzeros(inc.row))
        if isa(coeff, Float64)
            # Linear, just add to the derivative
            r[get_or_make_diff(v_offset)] += coeff
        else
            # nonlinear
            r[v_offset] = nonlinear
            r[get_or_make_diff(v_offset)] = nonlinear
        end
    end
    return Incidence(isa(inc.typ, Const) ? Const(zero(inc.typ.val)) : inc.typ, r, inc.eps)
end

widenincidence(inc::Incidence) = inc.typ
widenincidence(p::PartialStruct) = PartialStruct(p.typ, Any[widenincidence(f) for f in  p.fields])
widenincidence(@nospecialize(x)) = x

@override function CC.abstract_call_known(interp::DAEInterpreter, @nospecialize(f),
    arginfo::ArgInfo, si::StmtInfo, sv::AbsIntState, max_methods::Int)
    if f === ddt
        merge_daeinfo!(interp, sv.result, DAEInfo(; has_dae_intrinsics=true))
        (;argtypes) = arginfo
        if length(argtypes) == 2
            xarg = argtypes[2]
            if isa(xarg, Union{Incidence, Const})
                return Future{CallMeta}(structural_inc_ddt(interp.var_to_diff, interp.var_kind, xarg))
            end
        end
    end
    if interp.in_analysis && !isa(f, Core.Builtin) && !isa(f, Core.IntrinsicFunction)
        # We don't want to do new inference here
        return Future{CallMeta}(CallMeta(Any, Any, CC.Effects(), CC.NoCallInfo()))
    end
    ret = @invoke CC.abstract_call_known(interp::AbstractInterpreter, f::Any,
        arginfo::ArgInfo, si::StmtInfo, sv::AbsIntState, max_methods::Int)
    if isa(sv, InferenceState)
        if f === sim_time
            merge_daeinfo!(interp, sv.result, DAEInfo(; has_dae_intrinsics=true))
            #return CallMeta(Incidence(0), ret.effects, ret.info)
        elseif f === variable || f === state_ddt
            merge_daeinfo!(interp, sv.result, DAEInfo(; has_dae_intrinsics=true))
        elseif f === equation || f === observed! || f === singularity_root! || f === time_periodic_singularity!
            merge_daeinfo!(interp, sv.result, DAEInfo(; has_dae_intrinsics=true))
        elseif f === Core.current_scope
            merge_daeinfo!(interp, sv.result, DAEInfo(; has_scoperead=true))
        end
    end
    arginfo = ArgInfo(arginfo.fargs, map(widenincidence, arginfo.argtypes))
    return Diffractor.fwd_abstract_call_gf_by_type(interp, f, arginfo, si, sv, ret)
end

@override function CC.abstract_call_method(interp::DAEInterpreter,
    method::Method, @nospecialize(sig), sparams::SimpleVector, hardlimit::Bool, si::StmtInfo, sv::InferenceState)
    mret = @invoke CC.abstract_call_method(interp::AbstractInterpreter,
        method::Method, sig::Any, sparams::SimpleVector, hardlimit::Bool, si::StmtInfo, sv::InferenceState)
    return Future{MethodCallResult}(mret, interp, sv) do ret, interp, sv
        edge = ret.edge
        if edge !== nothing
            src = @atomic :monotonic edge.inferred
            if isa(src, DAECache)
                info = src.info
                merge_daeinfo!(interp, sv.result, info)
            end
        end
        return ret
    end
end

@override function CC.const_prop_call(interp::DAEInterpreter,
    mi::MethodInstance, result::MethodCallResult, arginfo::ArgInfo,
    sv::InferenceState, concrete_eval_result::Union{Nothing,ConstCallResult})
    ret = @invoke CC.const_prop_call(interp::AbstractInterpreter,
        mi::MethodInstance, result::MethodCallResult, arginfo::ArgInfo,
        sv::InferenceState, concrete_eval_result::Union{Nothing,ConstCallResult})
    if isa(ret, ConstCallResult)
        const_result = ret.const_result::ConstPropResult
        info = interp.dae_cache[const_result.result]
        merge_daeinfo!(interp, sv.result, info)
    end
    return ret
end

# cache integration
# -----------------

struct DAECache
    inferred
    ir::Union{Nothing,IRCode}
    info::DAEInfo
    DAECache(@nospecialize(inferred), ir::Union{Nothing,IRCode}, info::DAEInfo) =
        new(inferred, ir, info)
end

function CC.transform_result_for_cache(interp::DAEInterpreter, result::InferenceResult)
    src = result.src
    if isa(src, DAECache)
        return src
    end
    inferred = @invoke CC.transform_result_for_cache(interp::AbstractInterpreter, result)
    return DAECache(inferred, nothing, interp.dae_cache[result])
end

# inlining
# --------

function dae_inlining_policy(@nospecialize(src), @nospecialize(info::CallInfo), raise::Bool=true)
    if isa(info, Diffractor.FRuleCallInfo) && info.frule_call.rt !== Const(nothing)
        return nothing
    end
    osrc = src
    if src isa DAECache
        ir = src.ir
        if ir !== nothing
            return CC.is_inlineable(src.inferred::CodeInfo) ? ir : nothing
        end
        src = src.inferred
    end
    if isa(src, IRCode)
        return src
    end
    src === nothing && return nothing
    if raise
        # What happened here?
        @show (typeof(osrc), src)
        error()
    else
        return src::CodeInfo
    end
end

@override function CC.src_inlining_policy(interp::DAEInterpreter,
    @nospecialize(src), @nospecialize(info::CallInfo), stmt_flag::UInt32)
    ret = dae_inlining_policy(src, info, #=raise=#false)
    if ret isa CodeInfo
        return @invoke CC.src_inlining_policy(interp::AbstractInterpreter,
            ret::Any, info::CallInfo, stmt_flag::UInt32)
    end
    return ret !== nothing
end
CC.retrieve_ir_for_inlining(codeinst::CodeInstance, src::DAECache) =
    CC.retrieve_ir_for_inlining(codeinst.def, src, true)
function CC.retrieve_ir_for_inlining(mi::MethodInstance, src::DAECache, preserve_local_sources::Bool)
    ir = src.ir
    ir isa IRCode && return CC.retrieve_ir_for_inlining(mi, ir, preserve_local_sources)
    return CC.retrieve_ir_for_inlining(mi, src.inferred, preserve_local_sources)
end

@override function CC.finish(interp::DAEInterpreter,
    opt::OptimizationState, ir::IRCode, caller::InferenceResult)
    ret = @invoke CC.finish(interp::AbstractInterpreter,
        opt::OptimizationState, ir::IRCode, caller::InferenceResult)
    src = opt.src
    info = interp.dae_cache[caller]
    if !interp.ipo_analysis_mode && (info.has_dae_intrinsics || info.has_scoperead)
        set_inlineable!(src, true)
    end
    return ret
end

if VERSION ≥ v"1.12.0-DEV.734"
@override function CC.finishinfer!(state::InferenceState, interp::DAEInterpreter)
    res = @invoke CC.finishinfer!(state::InferenceState, interp::AbstractInterpreter)
    key = CC.is_constproped(state) ? state.result : state.linfo
    interp.unopt[key] = Cthulhu.InferredSource(state)
    return res
end
else
@override function CC.finish(state::InferenceState, interp::DAEInterpreter)
    res = @invoke CC.finish(state::InferenceState, interp::AbstractInterpreter)
    key = CC.is_constproped(state) ? state.result : state.linfo
    interp.unopt[key] = Cthulhu.InferredSource(state)
    return res
end
end

# TODO propagate debug configurations here
@override function CC.optimize(interp::DAEInterpreter, opt::OptimizationState, caller::InferenceResult)
    ir = CC.run_passes_ipo_safe(opt.src, opt)
    ir = run_dae_passes(interp, ir)
    CC.ipo_dataflow_analysis!(interp, opt, ir, caller)
    if interp.ipo_analysis_mode
        result = ipo_dae_analysis!(interp, ir, caller.linfo, caller)
        if result !== nothing
            CC.stack_analysis_result!(caller, result)
        end
    end
    return CC.finish(interp, opt, ir, caller)
end

if VERSION ≥ v"1.12.0-DEV.734"
@override function CC.finish!(interp::DAEInterpreter, caller::InferenceState; can_discard_trees::Bool=CC.may_discard_trees(interp))
    result = caller.result
    opt = result.src
    valid_worlds = result.valid_worlds
    if opt isa OptimizationState # implies `may_optimize(interp) === true`
        ir = opt.ir
        if ir !== nothing
            if isa(opt.src, CodeInfo)
                opt.src.min_world = CC.first(valid_worlds)
                opt.src.max_world = CC.last(valid_worlds)
            end
            result.src = DAECache(opt.src, copy(ir), interp.dae_cache[result])
        end
    end
    return @invoke CC.finish!(interp::AbstractInterpreter, caller::InferenceState; can_discard_trees)
end
else
@override function CC.finish!(interp::DAEInterpreter, caller::InferenceState)
    result = caller.result
    opt = result.src
    valid_worlds = result.valid_worlds
    if CC.last(valid_worlds) >= get_world_counter()
        # if we aren't cached, we don't need this edge
        # but our caller might, so let's just make it anyways
        CC.store_backedges(result, caller.stmt_edges[1])
    end
    if opt isa OptimizationState # implies `may_optimize(interp) === true`
        ir = opt.ir
        if ir !== nothing
            #=
            if isa(opt.src, CodeInfo)
                opt.src.min_world = CC.first(valid_worlds)
                opt.src.max_world = CC.last(valid_worlds)
            end
            =#
            result.src = DAECache(opt.src, copy(ir), interp.dae_cache[result])
        end
    end
    return nothing
end
end

@override function CC.const_prop_rettype_heuristic(interp::DAEInterpreter,
    result::MethodCallResult, si::StmtInfo, sv::InferenceState, force::Bool)
    edge = result.edge
    if edge !== nothing
        src = @atomic :monotonic edge.inferred
        if isa(src, DAECache)
            src.info.has_dae_intrinsics && return true
            src.info.has_scoperead && return true
        end
    end
    return @invoke CC.const_prop_rettype_heuristic(interp::AbstractInterpreter,
        result::MethodCallResult, si::StmtInfo, sv::InferenceState, force::Bool)
end

# semi-concrete interpretation
# ----------------------------

@override function CC.IRInterpretationState(interp::DAEInterpreter,
    code::CodeInstance, mi::MethodInstance, argtypes::Vector{Any}, world::UInt)
    src = @atomic :monotonic code.inferred
    src === nothing && return nothing
    (; inferred, ir) = src::DAECache
    (isa(inferred, CodeInfo) && isa(ir, IRCode)) || return nothing
    method_info = CC.SpecInfo(inferred)
    ir = copy(ir)
    (; min_world, max_world) = inferred
    argtypes = CC.va_process_argtypes(CC.optimizer_lattice(interp), argtypes, inferred.nargs, inferred.isva)
    return IRInterpretationState(interp, method_info, ir, mi, argtypes,
                                 world, min_world, max_world)
end

const DAE_LATTICE = CC.PartialsLattice(DAELattice())
CC.optimizer_lattice(interp::DAEInterpreter) =
    (!interp.ipo_analysis_mode || interp.in_analysis) ?
    DAE_LATTICE :
    CC.SimpleInferenceLattice.instance
CC.typeinf_lattice(interp::DAEInterpreter) =
    (!interp.ipo_analysis_mode || interp.in_analysis) ?
        CC.InferenceLattice(CC.ConditionalsLattice(DAE_LATTICE)) :
        CC.InferenceLattice(CC.BaseInferenceLattice.instance)
CC.ipo_lattice(interp::DAEInterpreter) =
    (!interp.ipo_analysis_mode || interp.in_analysis) ?
        CC.InferenceLattice(CC.InterConditionalsLattice(DAE_LATTICE)) :
        CC.InferenceLattice(CC.IPOResultLattice.instance)

function dominator_bb_set(dt::CC.DomTree, ir, bb; bbfilter = nothing)
    bbs = ir.cfg.blocks[bb].preds
    if bbfilter !== nothing
        bbs = filter(bbfilter, bbs)
    end
    if length(bbs) == 1
        # Could return the one predecessor, but this makes it easier to detect
        # this case and retains the invariant that all the returned bbs have
        # conditional branch terminators.
        return Int[]
    elseif length(bbs) == 2
        return Int[dt.idoms_bb[bb]]
    else
        list = Pair{Int, Int}[dt.nodes[bb].level => bb for bb in bbs]
        sort!(list, by = x->x[1])
        dom_set = Int[]
        seen = BitSet()
        while length(list) > 1
            cur = list[end]
            i = length(list)
            while i >= 1 && list[i][1] == cur[1]
                if list[i][2] == 0
                    # TODO: This should really not happen, but sometimes our pipeline leaves
                    # dead code that doesn't ascend properly. We should fix that.
                    deleteat!(list, i)
                elseif list[i][2] in seen
                    push!(dom_set, list[i][2])
                    deleteat!(list, i)
                else
                    push!(seen, list[i][2])
                    list[i] = (list[i][1]-1)=>dt.idoms_bb[list[i][2]]
                end
                i -= 1
            end
        end
        return dom_set
    end
end

@override function CC.abstract_eval_phi_stmt(interp::DAEInterpreter, phi::PhiNode, idx::Int, irsv::IRInterpretationState)
    ir = irsv.ir
    valr = CC.abstract_eval_phi(interp, phi, nothing, irsv)
    if interp.ipo_analysis_mode && !interp.in_analysis
        return valr
    end

    # If the phi node is const, all reachable branches returned the same result,
    # so we don't need to introduced additional control-dependent taint.
    # If the result is a Type, we don't have any incidence information anyway,
    # so there's nothing to add.
    if !isa(valr, Incidence) && !isa(valr, PartialStruct)
        return valr
    end

    # Find the branch condition of the idom
    bb = block_for_inst(ir, idx)
    domtree = if hasfield(typeof(irsv), :lazydomtree)
        CC.get!(irsv.lazydomtree)
    else
        CC.get!(irsv.lazyreachability).domtree
    end
    idombbs = dominator_bb_set(domtree, ir, bb;
        bbfilter = bb->let idx = findfirst(==(bb), phi.edges)
            idx !== nothing && isassigned(phi.values, idx)
        end)

    isempty(idombbs) && return valr

    function update_incidence(i::Incidence)
        retinc = copy(i)
        for idombb in idombbs
            idomterm = ir[SSAValue(ir.cfg.blocks[idombb].stmts[end])][:inst]
            if !isa(idomterm, GotoIfNot)
                # TODO: This really should not happen, but currently it's possible
                # for the IR to be updated to remove a control flow edge without
                # that being reflected in the CFG.
                continue
            end
            condT = argextype(idomterm.cond, ir)
            if isa(condT, Const)
                @assert isa(condT.val, Bool)
                continue
            end
            if !isa(condT, Incidence)
                return widenconst(retinc)
            end
            for i in rowvals(condT.row)
                retinc.row[i] = nonlinear
            end
        end
        return retinc
    end
    update_incidence(@nospecialize(a)) = a
    function update_incidence(i::PartialStruct)
        return PartialStruct(i.typ, Any[update_incidence(f) for f in i.fields])
    end

    return update_incidence(valr)
end

const equation_method = only(methods(equation, (Union{Nothing, Intrinsics.AbstractScope},)))
const equation_call_method = only(methods(equation(), (Any,)))
const variable_method0 = only(methods(variable, ()))
const variable_method1 = only(methods(variable, (Union{Nothing, Intrinsics.AbstractScope},)))
const epsilon_method0 = only(methods(epsilon, ()))
const epsilon_method1 = only(methods(epsilon, (Union{Nothing, Intrinsics.AbstractScope},)))
const observed!_method1 = only(methods(observed!, (Float64,)))
const observed!_method2 = only(methods(observed!, (Float64, Union{Nothing, Intrinsics.AbstractScope})))
const singularity_root!_method = only(methods(singularity_root!))
const time_periodic_singularity!_method = only(methods(time_periodic_singularity!))
const state_ddt_method = only(methods(state_ddt))
const sim_time_method = only(methods(sim_time))
# hack to avoid infinite recursion, remove once julia#48913 is merged
using Random, Distributions
const rand_method = only(methods(rand, (Random.Xoshiro, Distributions.Normal)))
const ddt_method = only(methods(ddt, (Float64,)))

function process_template!(𝕃, coeffs, eq_mapping, applied_scopes, argtypes, template_argtypes)
    for (arg, template) in zip(argtypes, template_argtypes)
        if isa(template, Incidence)
            if isempty(template)
                # @assert iszero(arg)
                continue
            end
            (idxs, vals) = findnz(template.row)
            @assert only(vals) == 1.0
            @assert !isassigned(coeffs, only(idxs)-1)
            coeffs[only(idxs)-1] = arg
        elseif isa(template, Eq)
            @assert isa(arg, Eq)
            eq_mapping[idnum(template)] = idnum(arg)
        elseif CC.is_const_argtype(template)
            #@Core.Compiler.show (arg, template)
            #@assert CC.is_lattice_equal(DAE_LATTICE, arg, template)
        elseif isa(template, PartialScope)
            id = idnum(template)
            (id > length(applied_scopes)) && resize!(applied_scopes, id)
            if isa(arg, Const)
                @assert isa(arg.val, Union{Scope, Nothing})
                applied_scopes[id] = arg.val
            elseif isa(arg, PartialScope)
                applied_scopes[id] = arg
            else
                applied_scopes[id] = arg
            end
        elseif isa(template, PartialStruct)
            if isa(arg, PartialStruct)
                fields = arg.fields
            else
                fields = Any[getfield_tfunc(𝕃, arg, Const(i)) for i = 1:length(template.fields)]
            end
            process_template!(𝕃, coeffs, eq_mapping, applied_scopes, fields, template.fields)
        else
            @show (arg, template)
            error()
        end
    end
end

struct CalleeMapping
    var_coeffs::Vector{Any}
    eqs::Vector{Int}
    applied_scopes::Vector{Any}
end

function CalleeMapping(𝕃, argtypes::Vector{Any}, callee_result::DAEIPOResult)
    applied_scopes = Any[]
    coeffs = Vector{Any}(undef, length(callee_result.var_to_diff))
    eq_mapping = fill(0, length(callee_result.total_incidence))

    process_template!(𝕃, coeffs, eq_mapping, applied_scopes, argtypes, callee_result.argtypes)

    return CalleeMapping(coeffs, eq_mapping, applied_scopes)
end

function compute_missing_coeff!(coeffs, result, caller_var_to_diff, caller_var_kind, v)
    # First find the rootvar, and if we already have a coeff for it
    # apply the derivatives.
    ndiffs = 0
    calle_inv = invview(result.var_to_diff)
    while calle_inv[v] !== nothing && !isassigned(coeffs, v)
        ndiffs += 1
        v = calle_inv[v]
    end

    if !isassigned(coeffs, v)
        @assert v > result.nexternalvars
        # Reached the root and it's an internal variable. We need to allocate
        # it in the caller now
        coeffs[v] = Incidence(add_vertex!(caller_var_to_diff))
        push!(caller_var_kind, result.var_kind[v] == External ? Owned : CalleeInternal)
    end
    thisinc = coeffs[v]

    for _ = 1:ndiffs
        dv = result.var_to_diff[v]
        coeffs[dv] = structural_inc_ddt(caller_var_to_diff, caller_var_kind, thisinc)
        v = dv
    end

    return nothing
end

apply_linear_incidence(ret::Type, result::DAEIPOResult, caller_var_to_diff::DiffGraph, caller_var_kind::Vector{VarEqKind}, caller_eq_kind::Vector{VarEqKind}, mapping::CalleeMapping) = ret
apply_linear_incidence(ret::Const, result::DAEIPOResult, caller_var_to_diff::DiffGraph, caller_var_kind::Vector{VarEqKind}, caller_eq_kind::Vector{VarEqKind}, mapping::CalleeMapping) = ret
function apply_linear_incidence(ret::Incidence, result::DAEIPOResult, caller_var_to_diff::DiffGraph, caller_var_kind::Vector{VarEqKind}, caller_eq_kind::Vector{VarEqKind}, mapping::CalleeMapping)
    coeffs = mapping.var_coeffs

    const_val = ret.typ
    new_row = _zero_row()

    for (v_offset, coeff) in zip(rowvals(ret.row), nonzeros(ret.row))
        v = v_offset - 1

        # Time dependence persists as itself
        if v == 0
            new_row[v_offset] += coeff
            continue
        end

        if !isassigned(coeffs, v)
            compute_missing_coeff!(coeffs, result, caller_var_to_diff, caller_var_kind, v)
        end

        replacement = coeffs[v]
        if isa(replacement, Incidence)
            new_row .+= replacement.row .* coeff
        else
            if isa(replacement, Const)
                if isa(const_val, Const)
                    new_const_val = const_val.val + replacement.val * coeff
                    if isa(new_const_val, Float64)
                        const_val = Const(new_const_val)
                    else
                        const_val = widenconst(const_val)
                    end
                else
                    const_val = widenconst(const_val)
                end
            else
                # The replacement has some unknown type - we need to widen
                # all the way here.
                return widenconst(const_val)
            end
        end
    end

    return Incidence(const_val, new_row, BitSet())
end

function apply_linear_incidence(ret::Eq, result::DAEIPOResult, caller_var_to_diff::DiffGraph, caller_var_kind::Vector{VarEqKind}, caller_eq_kind::Vector{VarEqKind}, mapping::CalleeMapping)
    eq_mapping = mapping.eqs[ret.id]
    if eq_mapping == 0
        push!(caller_eq_kind, Owned)
        mapping.eqs[ret.id] = eq_mapping = length(caller_eq_kind)
    end
    return Eq(eq_mapping)
end

function apply_linear_incidence(ret::PartialStruct, result::DAEIPOResult, caller_var_to_diff::DiffGraph, caller_var_kind::Vector{VarEqKind}, caller_eq_kind::Vector{VarEqKind}, mapping::CalleeMapping)
    return PartialStruct(ret.typ, Any[apply_linear_incidence(f, result, caller_var_to_diff, caller_var_kind, caller_eq_kind, mapping) for f in ret.fields])
end

if isdefined(CC, :abstract_eval_invoke_inst)
    @override function CC.abstract_eval_invoke_inst(interp::DAEInterpreter, inst::CC.Instruction, irsv::IRInterpretationState)
        stmt = inst[:stmt]
        _abstract_eval_invoke_inst(interp, inst, inst[:stmt], irsv)
    end
else
    @override function CC.concrete_eval_invoke(interp::DAEInterpreter, stmt::Expr, mi::MethodInstance, irsv::IRInterpretationState)
        _abstract_eval_invoke_inst(interp, nothing, stmt, irsv)
    end
end

struct MappingInfo <: CC.CallInfo
    info::Any
    result::DAEIPOResult
    mapping::CalleeMapping
end

function _abstract_eval_invoke_inst(interp::DAEInterpreter, inst::Union{CC.Instruction, Nothing}, @nospecialize(stmt), irsv::IRInterpretationState)
    mi = stmt.args[1]
    RT = Pair{Any, Tuple{Bool, Bool}}
    good_effects = (true, true)
    m = mi.def
    if m === variable_method0 || m === variable_method1
        # Nothing to do - we'll read the incidence out of the ssavaluetypes
        return RT(nothing, good_effects)
    elseif m === epsilon_method0 || m === epsilon_method1
        return RT(nothing, good_effects)
    elseif m === equation_method
        return RT(nothing, good_effects)
    elseif m === equation_call_method
        @assert length(stmt.args) == 3
        # Nothing to do - we'll read the incidence out of the ssavaluetypes
        return RT(nothing, good_effects)
    elseif m === observed!_method1 || m === observed!_method2
        # Nothing to do - we'll read the incidence out of the ssavaluetypes
        return RT(nothing, good_effects)
    elseif m === singularity_root!_method || m === time_periodic_singularity!_method
        # Nothing to do - we'll read the incidence out of the ssavaluetypes
        return RT(nothing, good_effects)
    elseif m === state_ddt_method
        # Nothing to do - we'll read the incidence out of the ssavaluetypes
        return RT(nothing, good_effects)
    elseif m === sim_time_method
        return RT(Incidence(0), good_effects)
    elseif m === rand_method
        return RT(Incidence(Float64), good_effects)
    elseif m === ddt_method
        argtypes = CC.collect_argtypes(interp, stmt.args, nothing, irsv)
        argtypes === nothing && return RT(Union{}, (false, true))
        # First arg is invoke mi
        if length(argtypes) == 3 && isa(argtypes[3], Union{Incidence, Const})
            return RT(structural_inc_ddt(interp.var_to_diff, interp.var_kind, argtypes[3]), (false, true))
        end
        return RT(nothing, (false, true))
    end

    if interp.in_analysis
        @assert interp.ipo_analysis_mode
        info = inst[:info]

        argtypes = CC.collect_argtypes(interp, stmt.args[2:end], nothing, irsv)
        argtypes === nothing && return Pair{Any,Tuple{Bool,Bool}}(Bottom, (false, false))

        if isa(argtypes[1], Const)
            if argtypes[1].val === Core.OptimizedGenerics.KeyValue.get
                pkv = argtypes[2]
                if isa(pkv, PartialKeyValue)
                    if isa(argtypes[3], Const)
                        index = argtypes[3].val
                        if haskey(pkv.vals, index)
                            return RT(CC.tuple_tfunc(CC.optimizer_lattice(interp), Any[pkv.vals[index]]), good_effects)
                        end
                        # Not something we asked about before
                        if isa(index, Base.ScopedValues.ScopedValue)
                            # Special assumption on the correctness of ScopedValue
                            sct = eltype(index)
                            if sct === Intrinsics.AbstractScope
                                # If this is a scoped value for our debug scopes,
                                # provide a special placeholder PartialScope.
                                # HACK: This currently assumes there's only one scope
                                # being passed through ScopedValue
                                return RT(PartialStruct(Tuple{Intrinsics.AbstractScope}, Any[PartialScope(index)]), good_effects)
                            end
                        end
                        return RT(Union{Nothing, widenconst(CC.tuple_tfunc(CC.optimizer_lattice(interp), Any[sct]))}, good_effects)
                    end
                end
                return RT(nothing, good_effects)
            elseif argtypes[1].val === Core.OptimizedGenerics.KeyValue.set
                if isa(argtypes[3], Const)
                    new_rt = PartialKeyValue(inst[:type], argtypes[2], Dict(argtypes[3].val => argtypes[4]))
                    return RT(new_rt, good_effects)
                end
            end
        end

        codeinst = CC.get(CC.code_cache(interp), mi, nothing)
        if codeinst === nothing
            return RT(nothing, (false, false))
        end

        @assert !isa(info, MappingInfo)

        if isa(info, Diffractor.FRuleCallInfo)
            info = info.info
        end
        argtypes = CC.collect_argtypes(interp, stmt.args, nothing, irsv)[2:end]
        callee_result = dae_result_for_inst(interp, inst)
        callee_result === nothing && return RT(nothing, (false, false))
        if isa(callee_result, UncompilableIPOResult) || isa(callee_result.extended_rt, Const) || isa(callee_result.extended_rt, Type)
            return RT(nothing, (false, false))
        end
        mapping = CalleeMapping(CC.optimizer_lattice(interp), argtypes, callee_result)
        new_rt = apply_linear_incidence(callee_result.extended_rt, callee_result, interp.var_to_diff, interp.var_kind, interp.eq_kind, mapping)

        # If this callee has no internal equations and at most one non-linear return, then we are guaranteed
        # never to need to fission this, so we can just treat it as a primitive.
        if length(callee_result.total_incidence) == 0 && isa(callee_result.extended_rt, Incidence)

        end
        # Remember this mapping, both for performance of not having to recompute it
        # and because we may have assigned caller variables to internal variables
        # that we now need to remember.
        inst[:info] = MappingInfo(inst[:info], callee_result, mapping)
        if new_rt === nothing
            return RT(nothing, (false, false))
        end
        return RT(new_rt, (false, false))
    end
@label bail

    if isdefined(CC, :abstract_eval_invoke_inst)
        rt, effects = @invoke CC.abstract_eval_invoke_inst(interp::AbstractInterpreter, inst::Instruction, irsv::IRInterpretationState)
    else
        rt, effects = @invoke CC.concrete_eval_invoke(interp::AbstractInterpreter, stmt::Expr, mi::MethodInstance, irsv::IRInterpretationState)
    end

    if rt === nothing
        # E.g. recursion or bad effects
        return RT(nothing, effects)
    end

    if !isa(rt, Const) && !isa(rt, Incidence)
        argtypes = CC.collect_argtypes(interp, stmt.args, nothing, irsv)
        argtypes === nothing && return RT(Union{}, (false, true))
        if is_all_inc_or_const(argtypes) && !is_all_inc_or_const(Any[rt])
            fb_inci = _fallback_incidence(argtypes)
            if fb_inci !== nothing
                update_type(t::Type) = Incidence(t, fb_inci.row, fb_inci.eps)
                update_type(t::Incidence) = t
                update_type(t::Const) = t
                update_type(t::PartialStruct) = PartialStruct(t.typ, Any[update_type(f) for f in t.fields])
                return RT(update_type(rt), effects)
            end
        end
    end

    return RT(rt, effects)
end

@override function CC.abstract_eval_statement_expr(interp::DAEInterpreter, inst::Expr, vtypes::Nothing, irsv::IRInterpretationState)
    ret = @invoke CC.abstract_eval_statement_expr(interp::AbstractInterpreter, inst::Expr, vtypes::Nothing, irsv::IRInterpretationState)
    return Future{CC.RTEffects}(ret, interp, irsv) do ret, interp, irsv
        (; rt, exct, effects) = ret
        if (!interp.ipo_analysis_mode || interp.in_analysis) && !isa(rt, Const) && !isa(rt, Incidence) && !CC.isType(rt) && !is_all_inc_or_const(Any[rt])
            argtypes = CC.collect_argtypes(interp, inst.args, nothing, irsv)
            if argtypes === nothing
                return CC.RTEffects(rt, exct, effects)
            end
            if is_all_inc_or_const(argtypes)
                if inst.head in (:call, :invoke) && CC.hasintersect(widenconst(argtypes[inst.head === :call ? 1 : 2]), Union{typeof(variable), typeof(sim_time), typeof(state_ddt)})
                    # The `variable` and `state_ddt` intrinsics can source Incidence. For all other
                    # calls, if there's no incidence in the arguments, there cannot be any incidence
                    # in the result.
                    return CC.RTEffects(rt, exct, effects)
                end
                fb_inci = _fallback_incidence(argtypes)
                if fb_inci !== nothing
                    update_type(t::Type) = Incidence(t, fb_inci.row, fb_inci.eps)
                    update_type(t::Incidence) = t
                    update_type(t::Const) = t
                    update_type(t::CC.PartialTypeVar) = t
                    update_type(t::PartialStruct) = PartialStruct(t.typ, Any[Base.isvarargtype(f) ? f : update_type(f) for f in t.fields])
                    update_type(t::CC.Conditional) = CC.Conditional(t.slot, update_type(t.thentype), update_type(t.elsetype))
                    newrt = update_type(rt)
                    return CC.RTEffects(newrt, exct, effects)
                end
            end
        end
        return CC.RTEffects(rt, exct, effects)
    end
end

@override function CC.compute_forwarded_argtypes(interp::DAEInterpreter, arginfo::ArgInfo, sv::AbsIntState)
    if !interp.ipo_analysis_mode
        return @invoke CC.compute_forwarded_argtypes(interp::AbstractInterpreter, arginfo::ArgInfo, sv::AbsIntState)
    end
    #@assert !interp.in_analysis
    𝕃ᵢ = typeinf_lattice(interp)
    argtypes = arginfo.argtypes
    new_argtypes = Vector{Any}(undef,length(arginfo.argtypes))
    id = 1
    for i = 1:length(argtypes)
        argt = argtypes[i]
        if isa(argt, Const) && isa(argt.val, Scope)
            argt = Scope
        elseif isa(argt, PartialStruct) && argt.typ == Scope
            argt = Scope
        end
        if isa(argt, Incidence)
            argt = argt.typ
        end
        new_argtypes[i] = argt
    end

    return CC.WidenedArgtypes(new_argtypes)
end

# entry
# -----

function typeinf_dae(@nospecialize(tt), world::UInt=get_world_counter();
        ipo_analysis_mode::Bool = false)
    interp = DAEInterpreter(world; ipo_analysis_mode)
    match = Base._methods_by_ftype(tt, 1, world)
    isempty(match) && single_match_error(tt)
    match = only(match)
    mi = CC.specialize_method(match)
    ci = CC.typeinf_ext(interp, mi, Core.Compiler.SOURCE_MODE_ABI)
    return interp, ci
end

@noinline function single_match_error(@nospecialize tt)
    sig = sprint(Base.show_tuple_as_call, Symbol(""), tt)
    error(lazy"Could not find single target method for `$sig`")
end

# Cthulhu integration
# ===================

using Cthulhu

function Cthulhu.get_optimized_codeinst(interp::DAEInterpreter, curs::Cthulhu.CthulhuCursor)
    CC.getindex(CC.code_cache(interp), curs.mi)
end

function Cthulhu.lookup(interp::DAEInterpreter, curs::Cthulhu.CthulhuCursor, optimize::Bool)
    Cthulhu.lookup(interp, Cthulhu.get_mi(curs), optimize)
end

function Cthulhu.lookup(interp::DAEInterpreter, mi::MethodInstance, optimize::Bool; allow_no_src::Bool=false)
    if optimize
        return lookup_optimized(interp, mi, allow_no_src)
    else
        return lookup_unoptimized(interp, mi)
    end
end

function Cthulhu.lookup_semiconcrete(interp::DAEInterpreter, curs::Cthulhu.CthulhuCursor, override::Cthulhu.SemiConcreteCallInfo, optimize::Bool)
    src = CC.copy(override.ir)
    rt = Cthulhu.get_rt(override)
    exct = Any # TODO
    infos = src.stmts.info
    slottypes = src.argtypes
    effects = Cthulhu.get_effects(override)
    #(; codeinf) = Cthulhu.lookup(interp, Cthulhu.get_mi(override), optimize)
    return (; src, rt, exct, infos, slottypes, effects, codeinf=nothing)
end

# TODO: This seems somewhat redundant with Cthulhu.get_optimized_codeinst -
# maybe Cthulhu should have a default that just uses that?
function lookup_optimized(interp::DAEInterpreter, mi::MethodInstance, allow_no_src::Bool=false)
    codeinst = Cthulhu.get_optimized_codeinst(interp, Cthulhu.CthulhuCursor(mi))
    rt = Cthulhu.cached_return_type(codeinst)
    exct = Cthulhu.cached_exception_type(codeinst)
    opt = @atomic :monotonic codeinst.inferred
    if opt !== nothing
        opt = opt::DAECache
        src = CC.copy(opt.ir)
        codeinf = opt.inferred
        infos = src.stmts.info
        slottypes = src.argtypes
    elseif allow_no_src
        # This doesn't showed up as covered, but it is (see the CI test with `coverage=false`).
        # But with coverage on, the empty function body isn't empty due to :code_coverage_effect expressions.
        codeinf = src = nothing
        infos = []
        slottypes = Any[Base.unwrap_unionall(mi.specTypes).parameters...]
    else
        Core.eval(Main, quote
            interp = $interp
            mi = $mi
        end)
        error("couldn't find the source; inspect `Main.interp` and `Main.mi`")
    end
    effects = Cthulhu.get_effects(codeinst)
    return (; src, rt, exct, infos, slottypes, effects, codeinf)
end

Cthulhu.can_descend(interp::DAEInterpreter, @nospecialize(key), optimize::Bool) =
    optimize ? CC.haskey(CC.code_cache(interp), key) : haskey(interp.unopt, key)

# TODO: Why does Cthulhu have this separately from the lookup logic, which already
# returns effects
function Cthulhu.get_effects(interp::DAEInterpreter, mi::MethodInstance, opt::Bool)
    if opt
        # If we're asking for optimized effects, get the codeinst from our interpreter's
        # code cache, then decode the effects bits straight from the codeinst.
        codeinst = CC.get(CC.code_cache(interp), mi, nothing)
        if codeinst === nothing
            return CC.Effects()
        end
        return CC.decode_effects(codeinst.ipo_purity_bits)
    else
        # If we're not asking for optimized effects, do the same thing as for the `CthulhuInterpreter`.
        return haskey(interp.unopt, mi) ? Cthulhu.get_effects(interp.unopt[mi]) : CC.Effects()
    end
end

Cthulhu.get_remarks(interp::DAEInterpreter, key::Union{MethodInstance,InferenceResult}) =
    get(interp.remarks, key, nothing)


function lookup_unoptimized(interp::DAEInterpreter, mi::MethodInstance)
    codeinf = src = copy(interp.unopt[mi].src)
    (; rt, exct) = interp.unopt[mi]
    infos = interp.unopt[mi].stmt_info
    effects = Cthulhu.get_effects(interp.unopt[mi])
    slottypes = src.slottypes
    if isnothing(slottypes)
        slottypes = Any[ Any for i = 1:length(src.slotflags) ]
    end
    return (; src, rt, exct, infos, slottypes, effects, codeinf)
end

function Cthulhu.lookup_constproped(interp::DAEInterpreter, override::InferenceResult, optimize::Bool)
    if optimize
        return lookup_constproped_optimized(interp, override)
    else
        return lookup_constproped_unoptimized(interp, override)
    end
end

function lookup_constproped_optimized(interp::DAEInterpreter, override::InferenceResult)
    opt = override.src
    if isa(opt, DAECache)
        src = CC.copy(opt.ir)
        codeinf = opt.inferred
        infos = src.stmts.info
        slottypes = src.argtypes
        rt = override.result
        exct = override.exc_result
        effects = CC.Effects()
        # `(override::InferenceResult).src` might has been transformed to OptimizedSource already,
        # e.g. when we switch from constant-prop' unoptimized source
        return (; src, rt, exct, infos, slottypes, effects, codeinf)
    else
        # the source might be unavailable at this point,
        # when a result is fully constant-folded etc.
        return lookup(interp, override.linfo, optimize)
    end
end

function Cthulhu.lookup_constproped(interp::DAEInterpreter, curs::Cthulhu.CthulhuCursor, override::InferenceResult, optimize::Bool)
    Cthulhu.lookup_constproped(interp, override, optimize)
end

# TODO: This looks basically the same as `lookup_unoptimized` - can we simplify this
# interface?
function lookup_constproped_unoptimized(interp::DAEInterpreter, override::InferenceResult)
    unopt = get(interp.unopt, override, nothing)
    if unopt === nothing
        unopt = interp.unopt[override.linfo]
    end
    codeinf = src = copy(unopt.src)
    (; rt, exct) = unopt
    infos = unopt.stmt_info
    effects = Cthulhu.get_effects(unopt)
    slottypes = src.slottypes
    if isnothing(slottypes)
        slottypes = Any[ Any for i = 1:length(src.slotflags) ]
    end
    return (; src, rt, exct, infos, slottypes, effects, codeinf)
end


function CC.add_remark!(interp::DAEInterpreter, sv::InferenceState, msg)
    key = CC.is_constproped(sv) ? sv.result : sv.linfo
    push!(get!(Cthulhu.PC2Remarks, interp.remarks, key), sv.currpc=>msg)
end


#============================#
"run type inference and constant propagation on the ir"
function infer_ir!(ir, state, mi::MethodInstance)
    interp = getfield(get_sys(state), :interp)
    infer_ir!(ir, interp, mi)
end

function infer_ir!(ir, interp::AbstractInterpreter, mi::MethodInstance)
    for i = 1:length(ir.stmts)
        if ir[SSAValue(i)][:type] == Any && !isa(ir[SSAValue(i)][:inst], GotoNode) &&
                                            !isa(ir[SSAValue(i)][:inst], GotoIfNot)
            ir[SSAValue(i)][:flag] |= CC.IR_FLAG_REFINED
        end
    end

    (nargs, isva) = isa(mi.def, Method) ? (mi.def.nargs, mi.def.isva) : (0, false)
    method_info = CC.SpecInfo(nargs, isva, #=propagate_inbounds=#true, nothing)
    min_world = world = get_inference_world(interp)
    max_world = get_world_counter()
    irsv = IRInterpretationState(interp, method_info, ir, mi, ir.argtypes, world, min_world, max_world)
    (rt, nothrow) = CC.ir_abstract_constant_propagation(interp, irsv)
    return rt
end

"Given some IR generates a MethodInstance suitable for passing to infer_ir!"
get_toplevel_mi_from_ir(ir, sys) = get_toplevel_mi_from_ir(ir, getfield(sys, :mi).def.module)
function get_toplevel_mi_from_ir(ir, _module::Module)
    mi = ccall(:jl_new_method_instance_uninit, Ref{Core.MethodInstance}, ());
    # Use `widenconst` here to eliminate `Incidence` types if we happen to pass in a non-final `ir` value
    mi.specTypes = Tuple{map(CC.widenconst, ir.argtypes)...}
    mi.def = _module
    return mi
end
