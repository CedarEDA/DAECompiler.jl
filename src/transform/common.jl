@breadcrumb "ir_levels" function run_dae_passes_again(interp::AbstractInterpreter, ir::IRCode,
                                                      state::IRTransformationState)
    inline_state = CC.InliningState(interp)
    ir = ssa_inlining_pass!(ir, inline_state, #=propagate_inbounds=#true)
    record_ir!(state, "ssa_inlining_pass!", ir)

    if !isa(interp, DAEInterpreter)
        # This is needed because of the TODO noted at the callsite where we inline
        # SemiConcrete-eval results from the wrong interpreter.
        for i = 1:length(ir.stmts)
            ir.stmts[i][:type] = widenconst(ir.stmts[i][:type])
        end
    end

    ir = compact!(ir)
    record_ir!(state, "compact!.2", ir)

    ir = sroa_pass!(ir, inline_state)
    record_ir!(state, "sroa_pass!", ir)

    ir, #=made_changes=#_ = adce_pass!(ir, inline_state)
    record_ir!(state, "adce_pass!", ir)

    ir = compact_cfg(ir)
    record_ir!(state, "compact_cfg", ir)

    # ir = peephole_pass!(ir)
    # record_ir!(state, "peephole_pass!.1", ir)

    # ir = peephole_pass!(ir)
    # record_ir!(state, "peephole_pass!.2", ir)

    # Record it again for the "final output"
    record_ir!(state, "", ir)
    return ir
end

function compile_invokes!(ir, interp)
    for i = 1:length(ir.stmts)
        inst = ir.stmts[i]
        e = inst[:inst]
        if isexpr(e, :invoke)
            mi = e.args[1]::MethodInstance
            if !CC.haskey(CC.code_cache(interp), mi)
                CC.typeinf_ext_toplevel(interp, mi, CC.SOURCE_MODE_ABI)
            end
        end
    end
end

function widen_extra_info!(ir)
    for i = 1:length(ir.stmts)
        info = ir.stmts[i][:info]
        if isa(info, Diffractor.FRuleCallInfo)
            ir.stmts[i][:info] = info.info
        end
        inst = ir.stmts[i][:inst]
        if isa(inst, PiNode)
            ir.stmts[i][:inst] = PiNode(inst.val, widenconst(inst.typ))
        end
        ir.stmts[i][:type] = widenconst(ir.stmts[i][:type])
    end
end

function replace_call!(ir::Union{IRCode,IncrementalCompact}, idx::SSAValue, new_call::Expr)
    @assert !isa(ir[idx][:inst], PhiNode)
    ir[idx][:inst] = new_call
    ir[idx][:type] = Any
    ir[idx][:info] = CC.NoCallInfo()
    ir[idx][:flag] |= CC.IR_FLAG_REFINED
end

function compile_overload(ir, state, arg_types;
                          opt_params::OptimizationParams=OptimizationParams())
    # NB: we run inlining with the fallback interpreter. The versions we have cached
    # are not suitable for inlining, because the code we generated for these
    # themselves may have uninlined calls in them from FRuleCallInfo.
    # TODO: Potentially we should be owning more of the compilation pipeline here?

    fallback_interp = getfield(get_sys(state), :fallback_interp)

    subtype_ir = copy(ir)
    empty!(subtype_ir.argtypes)
    push!(subtype_ir.argtypes, Tuple{})  # function object
    append!(subtype_ir.argtypes, arg_types)
    subtype_mi = get_toplevel_mi_from_ir(subtype_ir, get_sys(state))
    infer_ir!(subtype_ir, fallback_interp, subtype_mi)
    record_ir!(state, "inferred", subtype_ir)

    NewInterp = typeof(fallback_interp)
    newinterp = NewInterp(fallback_interp; opt_params)
    subtype_ir = run_dae_passes_again(newinterp, subtype_ir, state)
    opt_params.compilesig_invokes || compile_invokes!(subtype_ir, fallback_interp)  # if they were not compiled we must do it

    record_ir!(state, "", subtype_ir)

    DebugConfig(state).verify_ir_levels && check_for_daecompiler_intrinstics(subtype_ir)
    subtype_f = Core.OpaqueClosure(subtype_ir)
    return subtype_f
end

"""
    replace_if_intrinsic!(compact, ssa, du, u, p, t, var_assignment)

In the DAE IR, there are a number of intrinsics such as `variable()`, `state_ddt()`, or `equation!()`.
When converting these to normal julia functions those intrinsics need to be mapped to concrete values;
generally one of the inputs to the function (e.g. the `u`, `p` and `t` arguments) which we typically address as `Argument`s, but literals or `SSAValues` can also be used.

This replaces:
 - `sim_time()` with `t`
 - `variable(...)`/`state_ddt` for selected states with `u[i]` or `du[i]` as appropriate
 - `variable(...)`/`state_ddt` for nonselected states with an unusued marker
 - `_1` with `p` (as in DAE IR the parameters are fields of a functor)
 - `equation!`, `observed!`, `singularity_root!`, `time_periodic_singularity!` are deleted.
 - `epsilon(...)` is replaced with `0.0`

If doing an ODE, then can put `nothing` for `du` argument as we know it will not be used

If `var_assignment` is `nothing`, all variables are assumed unassigned. In this
case `u` and `du` may be `nothing` as well.
"""
function replace_if_intrinsic!(compact, ssa, du, u, p, t, var_assignment)
    inst = compact[ssa]
    stmt = inst[:inst]
    # Transform references to `Argument(1)` into `p`
    if isexpr(stmt, :invoke) || isexpr(stmt, :call)
        for (arg_ii, arg) in enumerate(stmt.args)
            if arg == Argument(1)
                stmt.args[arg_ii] = p
            end
        end
    end

    # Transform calls from `variable()` or `state_ddt()` into `u[var_idx]`
    if is_known_invoke_or_call(stmt, variable, compact) || is_known_invoke_or_call(stmt, state_ddt, compact)
        var = idnum(inst[:type])
        if var_assignment === nothing
            var_idx = 0
        else
            var_idx, in_du = var_assignment[var]
            @assert !in_du || (du !== nothing)
        end

        if iszero(var_idx)
            # This is some `variable` or `state_ddt` that is not a selected state,
            # but for some reason, wasn't deleted in any prior pass.
            inst[:inst] = GlobalRef(DAECompiler.Intrinsics, :_VARIABLE_UNASSIGNED)
        else
            source = in_du ? du : u
            replace_call!(compact, ssa, Expr(:call, getindex, source, var_idx))
        end
    elseif is_known_invoke_or_call(stmt, sim_time, compact)
        inst[:inst] = t
    elseif is_known_invoke_or_call(stmt, equation, compact)
        inst[:inst] = Intrinsics.placeholder_equation
    elseif is_equation_call(stmt, compact) ||
        is_known_invoke_or_call(stmt, observed!, compact) ||
        is_known_invoke_or_call(stmt, singularity_root!, compact) ||
        is_known_invoke_or_call(stmt, time_periodic_singularity!, compact) ||
        is_solved_variable(stmt)
        # these have no meaning outside of DAE IR.
        # Its likely the transform has already converted these to something else
        # but if not then we are allowed to remove them
        inst[:inst] = nothing
    elseif is_known_invoke_or_call(stmt, epsilon, compact)
        # It is 0.0 unless we are running `epsjac` in which case it should have already been handled
        inst[:inst] = 0.0
    end
end

struct UnexpectedIntrinsicException <: Exception
    inst
end
function Base.showerror(io::IO, x::UnexpectedIntrinsicException)
    println(io, "It is expected that all DAECompiler intrinsics are removed before using the native compiler. But found $(x.inst).")
end

"""
Checks for any DAECompiler intrinstic (using the rule that any function defined in DAECompiler.Intrinstics is an intrinstic)
and if it finds one, throws an error.
"""
function check_for_daecompiler_intrinstics(ir::IRCode)
    for i in 1:length(ir.stmts)
        inst = ir[SSAValue(i)][:inst]
        isexpr(inst, :invoke) || continue
        mi = inst.args[1]
        if mi.def.module == DAECompiler.Intrinsics
            throw(UnexpectedIntrinsicException(inst))
        end
    end
end

"performs a copy of nonbits types, if possible"
function defensive_copy(x)
    if !isbits(x)
        try
            return copy(x)
        catch
        end
    end
    return x
end
defensive_copy(x::SciMLBase.AbstractODEIntegrator) = x

function store_args_for_replay!(ir, debug_config, name, extra_ssas = [])
    nargs=length(ir.argtypes)
    # If our debug config is asking us to log replay values, insert code to do so.
    if debug_config.replay_log !== nothing
        if !haskey(debug_config.replay_log, name)
            debug_config.replay_log[name] = Tuple[]
        end

        # First, copy all mutable args so that we can store them without them being modified by anyone else
        arg_ssas = []
        for arg_idx in 2:nargs
            push!(arg_ssas, insert_node!(
                ir,
                length(ir.stmts),
                NewInstruction(Expr(:call, defensive_copy, Argument(arg_idx)), Any),
            ))
        end

        # Add any extra values requested by the user
        for ssa in extra_ssas
            push!(arg_ssas, insert_node!(
                ir,
                length(ir.stmts),
                NewInstruction(Expr(:call, defensive_copy, ssa), Any),
            ))
        end

        # Next, bundle them into a tuple
        args_tuple_ssa = insert_node!(
            ir,
            length(ir.stmts),
            NewInstruction(Expr(:call, tuple, arg_ssas...), Any),
        )

        # Then, push that tuple onto the appropriate replay log
        insert_node!(
            ir,
            length(ir.stmts),
            NewInstruction(Expr(:call, Base.push!, debug_config.replay_log[name], args_tuple_ssa), Any),
        )
        ir = compact!(ir)
    end
    return ir
end
