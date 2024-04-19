

"""
    compile_vector_continuous_callback_func(tsys, isdae)

Compiles a SciML VectorContinuousCallback `condition` function as described in [0].
This function has a signature of `callback(out, u, t, integrator)`, and writes out
numeric values to `out` that signify a discontinuity or other interesting event when
one of the values in `out` reach zero.  This is used to dynamically calculate tstop
times in systems with state-dependent or otherwise dynamically-calculated
discontinuities.
"""
@breadcrumb "ir_levels" function compile_vector_continuous_callback_func(tsys::TransformedIRODESystem, isdae::Bool)
    # Extract some parameters from our arguments
    debug_config = DebugConfig(tsys)

    var_assignment, _ = assign_vars_and_eqs(tsys, isdae)

    # Start from our compute_structure IR output.
    ir_callback = copy(tsys.state.ir)

    # Set the argtypes to `out, u, t, integ`
    # we will refine these later in the JITOpaqueClosure
    empty!(ir_callback.argtypes)
    push!(ir_callback.argtypes, Tuple{})
    push!(ir_callback.argtypes, AbstractVector{<:Number}) # out
    push!(ir_callback.argtypes, AbstractVector{<:Number}) # u
    push!(ir_callback.argtypes, Number) # t
    push!(ir_callback.argtypes, SciMLBase.AbstractODEIntegrator) # integ
    (out, u, t, integ) = Argument.(2:5)

    # We build our IR through the use of the `IncrementalCompact` API,
    # inserting nodes onto the end of the IR chunk via `insert_node_here!()`.
    compact = IncrementalCompact(ir_callback)

    # If we're a DAE, we can get our current `du` value by calling `get_du(integ)`
    if isdae
        du = insert_node_here!(
            compact,
            NewInstruction(Expr(:call, SciMLBase.get_du, integ), Any, Int32(1)),
        )
    else
        du = nothing
    end

    # Pull our parameterization out of `integ`
    p = insert_node_here!(
        compact,
        NewInstruction(Expr(:call, Base.getproperty, integ, QuoteNode(:p)), parameter_type(get_sys(tsys)), Int32(1)),
    )

    # Next, process the IR here
    for ((_, idx), stmt) in compact
        if is_known_invoke(stmt, singularity_root!, compact)
            incidence = argextype(stmt.args[3], compact)
            compact[SSAValue(idx)] = Expr(:call, Base.setindex!, out, stmt.args[3], stmt.args[4])
            compact[SSAValue(idx)][:type] = Any
        else
            replace_if_intrinsic!(compact, SSAValue(idx), du, u, p, t, var_assignment)
        end
    end

    # Finish, compact and optimize this callback function
    ir_callback = compact!(compact!(finish(compact), true))
    widen_extra_info!(ir_callback)
    ir_callback = store_args_for_replay!(ir_callback, debug_config, "vector_continuous_callback")
    record_ir!(tsys.state, "compact!", ir_callback)

    DebugConfig(tsys).verify_ir_levels && check_for_daecompiler_intrinstics(ir_callback)
    
    goldclass_sig = Tuple{Vector{Float64}, Vector{Float64}, Float64, SciMLBase.AbstractODEIntegrator}
    F! = JITOpaqueClosure{:vector_continuous_callback, goldclass_sig}() do arg_types...
        ir = copy(ir_callback)
        ir.argtypes[2:end].=arg_types
        mi = get_toplevel_mi_from_ir(ir, get_sys(tsys))
        fallback_interp = getfield(get_sys(tsys), :fallback_interp)
        NewInterp = typeof(fallback_interp)
        opt_params = OptimizationParams(; compilesig_invokes=false,  preserve_local_sources=true)
        newinterp = NewInterp(fallback_interp; opt_params)
        infer_ir!(ir, newinterp, mi)
        ir = run_dae_passes_again(newinterp, ir, tsys.state)
        record_ir!(tsys.state, "", ir)
        return Core.OpaqueClosure(ir; do_compile=true)
    end
    return F!
end