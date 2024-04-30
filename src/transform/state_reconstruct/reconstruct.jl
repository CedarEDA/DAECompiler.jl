
"""
    compile_batched_reconstruct_func(tsys::TransformedIRODESystem, vars::AbstractVector{Int64}, isdae::Bool)

Compile a new reconstruction function for the given states, identified by their numeric
index in the list of all variables.  This function is capable of reconstructing both
variables and observed values.  The function has te form:

    u_out, obs_out = reconstruct(u_out, obs_out, [du_compressed], u_compressed, p, t)

Where `du_compressed` and `u_compressed` are generated via the function returned from
`compile_state_compression_func()`.  If this is an ODE problem, `du_compressed` is not
passed to `reconstruct()`.

The `vars` that are being reconstructed must appear in sorted order.
"""
@breadcrumb "ir_levels" function compile_batched_reconstruct_func(tsys::TransformedIRODESystem, vars::AbstractVector{Int64}, obs::AbstractVector{Int64}, isdae::Bool)
    debug_config = DebugConfig(tsys)
    (; var_assignment) = assign_vars_and_eqs(tsys, isdae)

    check_variable_specification_preconditions(tsys, vars, obs)

    # Next, generate the reconstruction function - this in general requires all the equations, so
    # we start with the full IR and go from there.
    ir_reconstruct = copy(tsys.state.ir)

    # Set the argtypes to (u_out, obs_out, [du], u, p, t)
    # We will overwrite this later, but we just put Abstract typed placeholders here for the compiler machinery
    empty!(ir_reconstruct.argtypes)
    push!(ir_reconstruct.argtypes, Tuple{})
    push!(ir_reconstruct.argtypes, AbstractVector{<:Number})
    push!(ir_reconstruct.argtypes, AbstractVector{<:Number})
    if isdae
        push!(ir_reconstruct.argtypes, AbstractVector{<:Number})
    end
    push!(ir_reconstruct.argtypes, AbstractVector{<:Number})
    push!(ir_reconstruct.argtypes, parameter_type(get_sys(tsys)))
    push!(ir_reconstruct.argtypes, AbstractVector{<:Number})

    if isdae
        (u_out, obs_out, du_compressed, u_compressed, p, t) = Argument.(2:7)
    else
        (u_out, obs_out, u_compressed, p, t) = Argument.(2:6)
    end

    # We build our IR through the use of the `IncrementalCompact` API,
    # inserting nodes onto the end of the IR chunk via `insert_node_here!()`.
    compact = IncrementalCompact(ir_reconstruct)
    slot_arg(in_du) = in_du ? ((@assert isdae); du_compressed) : u_compressed

    # Next, pull direct states out from input slot to output array.
    # These are easy, as there's no transformation needed, just a direct copy.
    for (v, (slot, in_du)) in enumerate(var_assignment)
        # Skip states that are eliminated by `compress()`
        slot == 0 && continue

        # Only do this for variables in `vars`
        state_idx = findfirst(==(v), vars)
        if state_idx !== nothing
            # Generate the moral equivalent of:
            # u_out[state_idx] = in_du ? du[slot] : u[slot]
            ref = insert_node_here!(
                compact,
                NewInstruction(Expr(:call, Base.getindex, slot_arg(in_du), slot), Any, Int32(1)),
            )
            insert_node_here!(
                compact,
                NewInstruction(Expr(:call, Base.setindex!, u_out, ref, state_idx), Any, Int32(1)),
            )
        end
    end

    # If we've found an `observed!()` or `solved_variable()` call,
    # we want to turn this into `setindex!`, but only if the state is one that we're
    # actually asking for.
    function set_state!(ssa_idx, out, out_set, val, v)
        state_idx = findfirst(==(v), out_set)
        if state_idx !== nothing
            replace_call!(compact, ssa_idx, Expr(:call, Base.setindex!, out, val, state_idx))
        else
            compact[ssa_idx] = nothing
        end
    end

    # Next up, observeds.  These are trickier, and we need to make use of the full IR code.
    # Our strategy is to make certain useful replacements to our IR code, namely:
    # - `variable(idx)` and `state_ddt(idx)` get translated to `u_compressed[idx]`.
    # - `equation(idx)` and `singularity_root!()` get nullified, we have no use for them here.
    # - `observed!(idx, val)` and `solved_variable(idx, val) get translated to the moral equivalent
    #    of `u_out[idx] = val`, since those are exactly what we're interested in here!
    #    However, we only insert these if `idx` is found within `vars`, otherwise we drop it.
    # - `sim_time()` gets replaced with our `t` parameter input.
    for ((_, idx), stmt) in compact
        # If we've found a `variable()` or `state_ddt()` call, pull it out of `u`/`du`:
        if is_known_invoke(stmt, variable, compact) ||
           is_known_invoke_or_call(stmt, state_ddt, compact)
            (slot, in_du) = var_assignment[idnum(compact[SSAValue(idx)][:type])]
            if slot == 0
                # Statements like `0.0 * a` drop incidence on `a`, but may
                # not be legal to drop in the compiler. We insert a GlobalRef
                # here that marks this and that can be varied to verify that
                # the final result indeed does not depend on this variable.
                # It's possible we should instead implement this as a postprocessing
                # step that ensures that `_VARIABLE_UNASSIGNED` is never used in an output.
                compact[SSAValue(idx)] = GlobalRef(DAECompiler.Intrinsics, :_VARIABLE_UNASSIGNED)
                compact[SSAValue(idx)][:type] = Float64
            else
                # If `slot` is not `0` this is a real state, and we can pull it out of `u`/`du`
                replace_call!(compact, SSAValue(idx), Expr(:call, Base.getindex, slot_arg(in_du), slot))
            end

        # If this is an `observed!()` or `solved_variable()` call, store it out
        elseif is_known_invoke(stmt, observed!, compact)
            @assert length(stmt.args) >= 5
            set_state!(SSAValue(idx), obs_out, obs, stmt.args[end-2], stmt.args[end])
        elseif is_solved_variable(stmt)
            set_state!(SSAValue(idx), u_out, vars, stmt.args[end], stmt.args[end-1])

        elseif is_known_invoke(stmt, equation, compact)
            compact[SSAValue(idx)] = Intrinsics.placeholder_equation

        # These kinds of statements get dropped
        elseif is_known_invoke(stmt, singularity_root!, compact) ||
               is_known_invoke(stmt, time_periodic_singularity!, compact) ||
               is_equation_call(stmt, compact) ||
               isa(stmt, ReturnNode)
            compact[SSAValue(idx)] = nothing
            compact[SSAValue(idx)][:type] = Any

        elseif is_known_invoke(stmt, epsilon, compact)
            compact[SSAValue(idx)] = 0.0
            compact[SSAValue(idx)][:type] = Float64

        # `sim_time()` gets turned into just `t`.
        elseif is_known_invoke(stmt, sim_time, compact)
            compact[SSAValue(idx)] = t
            compact[SSAValue(idx)][:type] = Any
        else
            # For all other values, just make sure to replace a reference to `Argument(1)` with `p`,
            # since it is no longer true that `_1` is `p` in this function.
            replace_argument!(compact, idx, Argument(1), p)
        end
    end

    # Construct a tuple for our return value:
    tup = insert_node_here!(compact, NewInstruction(Expr(:call, Core.tuple, u_out, obs_out), Any, Int32(1)), true)
    insert_node_here!(compact, NewInstruction(ReturnNode(tup), Nothing, Int32(1)), true)

    # Finish, compact and optimize this reconstruction function
    ir_reconstruct = compact!(finish(compact), true)
    ir_reconstruct = store_args_for_replay!(ir_reconstruct, debug_config, "reconstruct")
    widen_extra_info!(ir_reconstruct)

    p_type = parameter_type(get_sys(tsys))
    goldclass_sig = if isdae
        Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, p_type, Float64}
    else
        Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, p_type, Float64}
    end
    F! = JITOpaqueClosure{:reconstruct, goldclass_sig}() do arg_types...
        ir = copy(ir_reconstruct)
        ir.argtypes[2:end] .= arg_types

        fallback_interp = getfield(get_sys(tsys), :fallback_interp)
        NewInterp = typeof(fallback_interp)
        opt_params = OptimizationParams(; compilesig_invokes=false, preserve_local_sources=true)
        newinterp = NewInterp(fallback_interp; opt_params)

        mi = get_toplevel_mi_from_ir(ir, get_sys(tsys))
        infer_ir!(ir, newinterp, mi)
        DebugConfig(tsys).verify_ir_levels && check_for_daecompiler_intrinstics(ir)
        fallback_interp = getfield(get_sys(tsys), :fallback_interp)
        vars_str = join(vars, ",")
        obs_str = join(obs, ",")
        breadcrumb_name = "vars=$(vars_str),obs=$(obs_str),state_type=$(eltype(arg_types[1]))"
        with_breadcrumb("ir_levels", breadcrumb_name) do
            ir = run_dae_passes_again(newinterp, ir, tsys.state)
            record_ir!(debug_config, "", ir)
        end
        return Core.OpaqueClosure(ir; do_compile=true)
    end
    return F!
end
