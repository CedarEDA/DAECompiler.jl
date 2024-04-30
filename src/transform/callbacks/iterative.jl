
"""
    compile_iterative_callback_func(tsys, isdae)

Compiles a SciML IterativeCallback `time_choice` function as described in [0].
This function has a signature of `callback(integrator)`, and returns the time
of the next callback.  If no callbacks are scheduled, returns `nothing`.
This allows for dynamically-calculated discontinuities to surface to the solver
their timepoints.

[0]: https://docs.sciml.ai/DiffEqCallbacks/stable/timed_callbacks/#DiffEqCallbacks.IterativeCallback
"""
@breadcrumb "ir_levels" function compile_iterative_callback_func(tsys::TransformedIRODESystem, isdae::Bool, insert_pre_discontinuity_points::Bool = true)
    # Extract some parameters from our arguments
    debug_config = DebugConfig(tsys)

    var_assignment, _ = assign_vars_and_eqs(tsys, isdae)

    # Start from our compute_structure IR output.
    ir_callback = copy(tsys.state.ir)

    # Set the argtypes to just `integ`
    empty!(ir_callback.argtypes)
    push!(ir_callback.argtypes, Tuple{})
    push!(ir_callback.argtypes, SciMLBase.AbstractODEIntegrator) # integ
    integ = Argument(2)

    # We build our IR through the use of the `IncrementalCompact` API,
    # inserting nodes onto the end of the IR chunk via `insert_node_here!()`.
    compact = IncrementalCompact(ir_callback)

    # Pull our parameterization out of `integ`
    p = insert_node_here!(
        compact,
        NewInstruction(Expr(:call, Base.getproperty, integ, QuoteNode(:p)), parameter_type(get_sys(tsys)), Int32(1)),
    )

    # Pull our current time out of `integ`
    t = insert_node_here!(
        compact,
        NewInstruction(Expr(:call, Base.getproperty, integ, QuoteNode(:t)), Float64, Int32(1)),
    )

    # Initialize our next t to `Inf`
    next_t_ssa = insert_node_here!(compact, NewInstruction(Expr(:call, Ref, Inf), Ref{Float64}, Int32(1)))

    # Next, process the IR here
    for ((_, idx), stmt) in compact
        # We want `time_periodic_singularity!()` values
        if is_known_invoke(stmt, time_periodic_singularity!, compact)
            compact[SSAValue(idx)] = Expr(:call, next_periodic_minimum, next_t_ssa, t, stmt.args[3:5]..., insert_pre_discontinuity_points)
            compact[SSAValue(idx)][:type] = Any
        elseif isa(stmt, ReturnNode)
            # Change to fall-through
            compact[SSAValue(idx)] = nothing
        else
            replace_if_intrinsic!(compact, SSAValue(idx), nothing, nothing, p, t, nothing)
        end
    end

    # At the end, if `next_t_ssa == Inf`, return nothing, otherwise return it.
    next_t_ssa = insert_node_here!(compact, NewInstruction(Expr(:call, Base.getindex, next_t_ssa), Float64, Int32(1)), true)
    comp_ssa = insert_node_here!(compact, NewInstruction(Expr(:call, Base.:(==), next_t_ssa, Inf), Bool, Int32(1)), true)
    ret_val = insert_node_here!(compact, NewInstruction(Expr(:call, Core.ifelse, comp_ssa, nothing, next_t_ssa), Union{Nothing,Float64}, Int32(1)), true)
    insert_node_here!(compact, NewInstruction(ReturnNode(ret_val), Nothing, Int32(1)), true)

    # Finish, compact and optimize this callback function
    ir_callback = compact!(finish(compact), true)
    widen_extra_info!(ir_callback)
    ret_val_ssa = ir_callback.stmts[end][:inst].val
    ir_callback = store_args_for_replay!(ir_callback, debug_config, "iterative_callback", [ret_val_ssa])
    record_ir!(tsys.state, "compact!", ir_callback)

    DebugConfig(tsys).verify_ir_levels && check_for_daecompiler_intrinstics(ir_callback)

    F! = JITOpaqueClosure{:iterative_callback, Tuple{SciMLBase.AbstractODEIntegrator}}() do Integrator
        ir = copy(ir_callback)
        ir.argtypes[2] = Integrator
        mi = get_toplevel_mi_from_ir(ir, get_sys(tsys))
        fallback_interp = getfield(get_sys(tsys), :fallback_interp)
        NewInterp = typeof(fallback_interp)
        opt_params = OptimizationParams(; compilesig_invokes=false, preserve_local_sources=true)
        newinterp = NewInterp(fallback_interp; opt_params)
        infer_ir!(ir, newinterp, mi)
        ir = run_dae_passes_again(newinterp, ir, tsys.state)
        record_ir!(tsys.state, "", ir)
        return Core.OpaqueClosure(ir; do_compile=true)
    end
    return F!
end

"""
    next_periodic_minimum(next_tstop, t, offset, period, count)

Given the current best-guess `next_tstop`, the current time `t`, and a periodic
singularity definition via `(offset, period, count)`, calculate the next singularity
then store the minimum between that and `next_tstop` into `next_tstop`.  Doing this
for all periodic singularities results in the overall next tstop in `next_tstop`.
"""
@inline function next_periodic_minimum(next_tstop::Ref, t, offset, period, count, insert_pre_discontinuity_points::Bool)
    # Infinite period is an alternative way of saying `count=1`
    if period == Inf
        next_periodic_tstop = float(offset)

    # If count == 0, we do nothing
    elseif count == 0
        return

    # If count > 0, early-exit if we're beyond our reptition boundary
    elseif count > 0 && t > offset + count*period
        return

    # Otherwise, we know `period` is not `Inf` and we know we're not
    # past the end of the tstop train, so just figure out what the next
    # point is:
    else
        if t < offset
            next_periodic_tstop = float(offset)
        else
            next_period = ceil(Int, (t-offset)/period)

            # We get called with our precise `tstop` quite often, so if we're
            # right on the money, we need to go to the _next_ period
            if offset + next_period*period == t
                next_period = next_period+1
            end

            # Oops, we stepped out of our repetition boundary
            if count > 0 && next_period >= count
                return
            end
            next_periodic_tstop = float(offset + next_period*period)
        end
    end

    if next_periodic_tstop > t
        next_tstop[] = min(next_tstop[], next_periodic_tstop)
    end

    if insert_pre_discontinuity_points
        if prevfloat(next_periodic_tstop, 100) > t
            next_tstop[] = min(next_tstop[], prevfloat(next_periodic_tstop, 100))
        end
    end
    return
end
