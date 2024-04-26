"""
    compile_batched_reconstruct_derivatives(tsys::TransformedIRODESystem, vars::AbstractVector{Int64}, isdae::Bool)

Similar to `compile_batched_reconstruct_func()`, but for reconstructing derivatives
instead of states.  Returns a function `reconstruct_der!` of the form:

    reconstruct_der!(dvar_du, dvar_dp, dvar_deps, dobs_du, dobs_dp, dobs_deps, [du_compressed], u_compressed, p, t)

Where `du_compressed` and `u_compressed` are generated via the function returned from
`compile_state_compression_func()`.  If this is an ODE problem, `du_compressed` is not
passed to `reconstruct()`.

The `vars` that are being reconstructed must appear in sorted order.
"""
@breadcrumb "ir_levels" function compile_batched_reconstruct_derivatives(
    tsys::TransformedIRODESystem, vars::AbstractVector{Int64}, obs::AbstractVector{Int64}, with_eps::Bool, isdae::Bool;
)
    debug_config=DebugConfig(tsys)
    check_variable_specification_preconditions(tsys, vars, obs)

    # Extract some parameters from our arguments
    (; var_assignment, neqs) = assign_vars_and_eqs(tsys, isdae)

    # Now we must make the code we are compiling
    ir = copy(tsys.state.ir)  # TODO: maybe we use the IR that we get from prepare_ir_for_differentiation
    if !with_eps
        ir = prepare_ir_for_differentiation(ir, tsys,
            keep_epsilon=false, keep_observerd=true, keep_nonselected=true)
    end

    # Set the argtypes to (dvar_du, dvar_dp, [dvar_deps,] dobs_du, dobs_dp, [dobs_deps,] du, u, p, t) we will refine this later
    empty!(ir.argtypes)
    push!(ir.argtypes, Tuple{})
    for i = 1:(with_eps ? 6 : 4)
        push!(ir.argtypes, AbstractMatrix{<:Number})
    end
    if isdae
        push!(ir.argtypes, AbstractVector{<:Number})
    end
    push!(ir.argtypes, AbstractVector{<:Number})
    p_type = parameter_type(get_sys(tsys))
    push!(ir.argtypes, p_type)
    push!(ir.argtypes, Number)

    if with_eps
        maxoutarg = 7
        dvar_du, dvar_dp, dvar_deps, dobs_du, dobs_dp, dobs_deps = Argument.(2:maxoutarg)
    else
        maxoutarg = 5
        dvar_du, dvar_dp, dobs_du, dobs_dp = Argument.(2:maxoutarg)
    end

    if isdae
        (du_compressed, u_compressed, p, t) = Argument.((maxoutarg+1):(maxoutarg + 5))
    else
        (u_compressed, p, t) = Argument.((maxoutarg+1):(maxoutarg + 4))
    end

    diff_ssas = filter_reconstruction_output_ssas(ir)

    # Perform AD transform
    neps = with_eps ? getfield(tsys.state.sys, :result).neps : 0
    nparams = determine_num_tangents(p_type)
    param_bob = insert_param_bob(ir, p; left_padding=neqs, right_padding=neps)
    Diffractor.forward_diff_no_inf!(
        ir, diff_ssas .=> 1;
        visit_custom! = get_reconstruct_der_visit_custom!(var_assignment),
        transform! = define_transform_for_reconstruct_der(var_assignment, vars, obs, param_bob, neqs, nparams, neps, with_eps, isdae),
        eras_mode=true,
    )

    insert_selected_state_ders!(ir, var_assignment, vars, dvar_du, dvar_dp, neqs, nparams)

    # hard code SSAValue for returns so valid after compact! (This is a bad hack since it is fragile, Keno says there is a better way)
    ir = conclude_reconstruct_like!(
         ir, tuple((Argument.(2:maxoutarg))...),
         (isdae ? du_compressed : nothing), u_compressed, p, t, var_assignment;
    )

    ir = compact!(ir)
    ir = store_args_for_replay!(ir, debug_config, "reconstruct_der")

    goldclass_sig = Tuple{map(2:length(ir.argtypes)) do i
        T = ir.argtypes[i]
        T == AbstractMatrix{<:Number} ? Matrix{Float64} :
        T == AbstractVector{<:Number} ? Vector{Float64} :
        T
    end...}
    F! = JITOpaqueClosure{:reconstruct_derivative, goldclass_sig}() do arg_types...
        ir = copy(ir)
        ir.argtypes[2:end] .= arg_types

        mi = get_toplevel_mi_from_ir(ir, get_sys(tsys))
        fallback_interp = getfield(get_sys(tsys), :fallback_interp)
        NewInterp = typeof(fallback_interp)
        opt_params = OptimizationParams(; compilesig_invokes=false,  preserve_local_sources=true)
        newinterp = NewInterp(fallback_interp; opt_params)
        infer_ir!(ir, newinterp, mi)
        widen_extra_info!(ir)
        DebugConfig(tsys).verify_ir_levels && check_for_daecompiler_intrinstics(ir)

        vars_str = join(string.(vars), ",")
        obs_str = join(string.(obs), ",")
        breadcrumb_name = "reconstruct_der.vars=$(vars_str),obs=$(obs_str),state_type=$(eltype(arg_types[4]))"
        with_breadcrumb("ir_levels", breadcrumb_name) do
            ir = run_dae_passes_again(newinterp, ir, tsys.state)
            record_ir!(debug_config, "", ir)
        end
        return Core.OpaqueClosure(ir; do_compile=true)
    end
    return F!
end



"Insert a 1 into the derivative, for derivative this as a selected state wrt this as a variable"
function insert_selected_state_ders!(ir, var_assignment, vars, dvar_du, dvar_dp, neqs, nparams)
    for (out_idx, var) in enumerate(vars)
        var_ii, in_du = var_assignment[var]
        # XXX: This over-initializes the matrix for DAEProblems
        #      (meaning it initializes indices that we end up setting later)
        iszero(var_ii) && continue  # not selected
        for col in 1:neqs
            partial = Float64(col == var_ii)
            insert_node!(ir, 1, NewInstruction(Expr(:call, Base.setindex!, dvar_du, partial, out_idx, col)), true)
        end
        for col in 1:nparams
            insert_node!(ir, 1, NewInstruction(Expr(:call, Base.setindex!, dvar_dp, 0.0, out_idx, col)), true)
        end
    end
    return ir
end

function define_transform_for_reconstruct_der(var_assignment, vars, obs, param_bob, neqs::Int, nparams::Int, neps::Int, with_eps::Bool, isdae::Bool)
    # If its a `variable`/`state_ddt` for a selected state attach a BoB to it.
        # and do make it read from the argument to get it's primal value
        # And insert a 1 into it's derivative output
    # If its a `variable`/`state_ddt` for a nonselected state mark it as UNUSED and don't AD wrt it
    # If it is an `observed!` or a `is_solved_variable` extract out the derivatives
        # `is_solved_variable` gets us all the nonselected variables's derivatives
        # I have been promised that `observed!` never depends on nonselected variables so its all good.
    function transform!(ir, ssa, order, maparg)
        @assert order==1  # reconstruct_der is first order only.
        if isa(ssa, Argument)
            if ssa == Argument(1)
                return param_bob
            else
                return insert_node!(ir, SSAValue(1), NewInstruction(Expr(:call, Diffractor.zero_bundle{1}(), ssa)))
            end
        end

        # Name the arguments for ease of reference
        if with_eps
            maxoutarg = 7
            dvar_du, dvar_dp, dvar_deps, dobs_du, dobs_dp, dobs_deps = Argument.(2:maxoutarg)
        else
            maxoutarg = 5
            dvar_du, dvar_dp, dobs_du, dobs_dp = Argument.(2:maxoutarg)
        end

        if isdae
            (du_compressed, u_compressed, p, t) = Argument.((maxoutarg+1):(maxoutarg + 5))
        else
            (u_compressed, p, t) = Argument.((maxoutarg+1):(maxoutarg + 4))
        end
        slot_arg(in_du) = in_du ? ((@assert isdae); du_compressed) : u_compressed

        inst = ir[ssa]
        stmt = inst[:inst]
        while isa(stmt, SSAValue)
            # It's possible an earlier call to transform! from e.g. Pantelides moved this call, so follow references.
            stmt = ir[stmt][:inst]
        end
        if is_known_invoke_or_call(stmt, variable, ir) || is_known_invoke_or_call(stmt, state_ddt, ir)
            var = idnum(inst[:type])
            var_ii, in_du = var_assignment[var]
            @assert isdae || !in_du
            if iszero(var_ii)
                # This is some `variable` or `state_ddt` that not a selected state, but for some reason, wasn't deleted in any prior pass.
                dummy_primal = insert_node!(ir, ssa, NewInstruction(inst; stmt=GlobalRef(DAECompiler.Intrinsics, :_VARIABLE_UNASSIGNED)))
                replace_call!(ir, ssa, Expr(:call, Diffractor.zero_bundle{1}(), dummy_primal))
                return nothing
            end

            u_ii = insert_node!(ir, ssa, NewInstruction(Expr(:call, getindex, slot_arg(in_du), var_ii)))
            input_basis_row = ntuple(neqs + nparams + neps) do active_state_ii
                Float64(var_ii == active_state_ii)
            end
            replace_call!(ir, ssa, Expr(:call, BatchOfBundles{neqs + nparams + neps}, u_ii, input_basis_row...))
            return nothing
        elseif is_known_invoke(stmt, epsilon, ir)
            @assert with_eps
            eps_ii = epsnum(inst[:type])
            input_basis_row = ntuple(neqs + nparams + neps) do active_state_ii
                Float64(var_ii == (active_state_ii - neqs + nparams))
            end
            replace_call!(ir, ssa, Expr(:call, BatchOfBundles{neqs + nparams + neps}, u_ii, input_basis_row...))
            return nothing
        elseif is_solved_variable(stmt) || is_known_invoke(stmt, observed!, ir)
            if is_solved_variable(stmt)
                d_du = dvar_du
                d_dp = dvar_dp
                if with_eps
                    d_deps = dvar_deps
                end
                v = stmt.args[end-1]
                bundles = stmt.args[end]
                out_idx = findfirst(==(v), vars)
            else  # observed!
                d_du = dobs_du
                d_dp = dobs_dp
                if with_eps
                    d_deps = dobs_deps
                end
                v = stmt.args[end]
                bundles = stmt.args[end-2]
                out_idx = findfirst(==(v), obs)
            end
            if out_idx !== nothing
                for col in 1:neqs
                    partial = insert_node!(ir, ssa, NewInstruction(Expr(:call, extract_partial, bundles, col)), true)
                    insert_node!(ir, ssa, NewInstruction(Expr(:call, Base.setindex!, d_du, partial, out_idx, col)), true)
                end
                for col in 1:nparams
                    partial = insert_node!(ir, ssa, NewInstruction(Expr(:call, extract_partial, bundles, neqs + col)), true)
                    insert_node!(ir, ssa, NewInstruction(Expr(:call, Base.setindex!, d_dp, partial, out_idx, col)), true)
                end
                if with_eps
                    for col in 1:neps
                        partial = insert_node!(ir, ssa, NewInstruction(Expr(:call, extract_partial, bundles, neqs + nparams + col)), true)
                        insert_node!(ir, ssa, NewInstruction(Expr(:call, Base.setindex!, d_deps, partial, out_idx, col)), true)
                    end
                end
            end
            ir[ssa] = nothing
            return nothing
        else @assert false end
    end
end


"Determines if there is something special to do for purposes of AD when trying to find jacobian"
function get_reconstruct_der_visit_custom!(var_assignment)
    selected_states=[var_num for (var_num, (slot, _)) in enumerate(var_assignment) if !iszero(slot)]
    function visit_custom!(ir::IRCode, ssa::Union{SSAValue,Argument}, order, recurse)
        if isa(ssa, Argument)
            return ssa == Argument(1)
        end

        stmt = ir[ssa][:inst]
        if is_known_invoke_or_call(stmt, variable, ir) || is_known_invoke_or_call(stmt, state_ddt, ir)
            return true
        elseif is_known_invoke_or_call(stmt, solved_variable, ir)
            recurse(stmt.args[end])
            return true
        elseif is_known_invoke_or_call(stmt, observed!, ir)
            recurse(stmt.args[end-2])
            return true
        elseif is_equation_call(stmt, ir)
            return false
        end

        return false
    end
end
