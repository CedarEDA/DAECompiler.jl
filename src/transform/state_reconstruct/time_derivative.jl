"""
    construct_reconstruction_time_derivative

This constructs a function, which for the listed `vars` and `obs` returns their deriviative with respect to time.
"""
@breadcrumb "ir_levels" function construct_reconstruction_time_derivative(
    tsys::TransformedIRODESystem, vars::AbstractVector{Int64}, obs::AbstractVector{Int64}, isdae::Bool;
)
    isdae && error("DAE not yet supported")
    debug_config=DebugConfig(tsys)
    (; var_assignment, neqs) = assign_vars_and_eqs(tsys, isdae)
    check_variable_specification_preconditions(tsys, vars, obs)

    ir = copy(tsys.state.ir)  # TODO: maybe we use the IR that we get from prepare_ir_for_differentiation

    # Set the argtypes to (du, u, p, t) we will refine this later
    empty!(ir.argtypes)
    push!(ir.argtypes, Tuple{})
    push!(ir.argtypes, AbstractVector{<:Number}) # dvar_out
    push!(ir.argtypes, AbstractVector{<:Number}) # dobs_out
    push!(ir.argtypes, AbstractVector{<:Number}) # du -- note we even have a du for ODE (for DAE we would also ned a `ddu` second deriviative, I think)
    push!(ir.argtypes, AbstractVector{<:Number}) # u
    p_type = parameter_type(get_sys(tsys))
    push!(ir.argtypes, p_type)                   # p
    push!(ir.argtypes, Number)                   # t

    (dvar_out, dobs_out, du, u, p, t) = Argument.(2:7)


    diff_ssas = filter_reconstruction_output_ssas(ir)
    # Perform AD transform
    Diffractor.forward_diff_no_inf!(
        ir, diff_ssas .=> 1;
        visit_custom! = reconstruction_time_derivative_visit_custom!,
        transform! = define_transform_for_pushingforward_reconstruct_time_derivatives(
            var_assignment, vars, obs, isdae
        ),
        eras_mode=true,
    )
    insert_selected_state_time_derivatives!(ir, var_assignment, vars, dvar_out, du)

    ir = conclude_reconstruct_like!(ir, (dvar_out, dobs_out), nothing, u, p, t, var_assignment)

    store_args_for_replay!(ir, debug_config, "reconstruct_time_der")
    ir = compact!(ir)
    DebugConfig(tsys).verify_ir_levels && check_for_daecompiler_intrinstics(ir)

    goldclass_sig = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, p_type, Float64}
    return JITOpaqueClosure{:reconstruct_time_derivative, goldclass_sig}() do arg_types...
        ir = copy(ir)
        ir.argtypes[2:end] .= arg_types

        fallback_interp = getfield(get_sys(tsys), :fallback_interp)
        NewInterp = typeof(fallback_interp)
        opt_params = OptimizationParams(; compilesig_invokes=false, preserve_local_sources=true)
        newinterp = NewInterp(fallback_interp; opt_params)

        # Just do a little bit of optimization so that it's properly inferred, etc...
        mi = get_toplevel_mi_from_ir(ir, get_sys(tsys))
        infer_ir!(ir, newinterp, mi)

        vars_str = join(vars, ",")
        obs_str = join(obs, ",")
        breadcrumb_name = "vars=$(vars_str),obs=$(obs_str),state_type=$(eltype(arg_types[4]))"
        with_breadcrumb("ir_levels", breadcrumb_name) do
            record_ir!(debug_config, "", ir)
        end
        return Core.OpaqueClosure(ir; do_compile=true)
    end
end


"Insert a 1 into the derivative, for derivative this as a selected state wrt this as a variable"
function insert_selected_state_time_derivatives!(ir, var_assignment, vars, dvar_out, du)
    for (out_idx, var) in enumerate(vars)
        var_ii, in_du = var_assignment[var]
        iszero(var_ii) && continue  # not selected
        partial_ssa = insert_node!(ir, SSAValue(1), NewInstruction(Expr(:call, Base.getindex, du, var_ii)), false)
        insert_node!(ir, partial_ssa, NewInstruction(Expr(:call, Base.setindex!, dvar_out, partial_ssa, out_idx)), true)
    end
    return ir
end


"Determines if there is something special to do for purposes of AD when trying to find tgrad"
function reconstruction_time_derivative_visit_custom!(ir::IRCode, ssa::Union{SSAValue,Argument}, order, recurse)
    isa(ssa, Argument) && return false

    stmt = ir[ssa][:inst]
    if is_known_invoke_or_call(stmt, variable, ir) || is_known_invoke_or_call(stmt, state_ddt, ir)
        return true
    elseif is_known_invoke_or_call(stmt, solved_variable, ir)
        recurse(stmt.args[end])
        return true
    elseif is_known_invoke_or_call(stmt, observed!, ir)
        recurse(stmt.args[end-2])
        return true
    elseif is_known_invoke(stmt, sim_time, ir)
        return true
    elseif is_equation_call(stmt, ir)
        return false
    end

    if isa(stmt, PhiNode)
        # Don't run our custom transform for PhiNodes - we don't have a place
        # to put the call and the regular recursion will handle it fine.
        return false
    end

    # We have a custom transform for every statement with no dependence
    typ = ir[ssa][:type]
    has_simple_incidence_info(typ) || return false
    return !has_dependence(typ)
end


function define_transform_for_pushingforward_reconstruct_time_derivatives(var_assignment, vars, obs, isdae)
    function transform!(ir, ssa, order, maparg)
        @assert order==1
        if isa(ssa, Argument)
            return insert_node!(ir, SSAValue(1), NewInstruction(Expr(:call, Diffractor.zero_bundle{1}(), ssa)))
        end

        (dvar_out, dobs_out, du, u, p, t) = Argument.(2:7)

        inst = ir[ssa]
        stmt = inst[:inst]
        while isa(stmt, SSAValue)
            # It's possible an earlier call to transform! from e.g. Pantelides moved this call, so follow references.
            stmt = ir[stmt][:inst]
        end

        if is_known_invoke_or_call(stmt, variable, ir) || is_known_invoke_or_call(stmt, state_ddt, ir)
            # if selected insert TaylorBundle{1}(u[i], (du[i],))
            # else set with unused marker
            var = idnum(inst[:type])
            var_ii, in_du = var_assignment[var]
            @assert isdae || !in_du
            if iszero(var_ii)
                # This is some `variable` or `state_ddt` that not a selected state, but for some reason, wasn't deleted in any prior pass.
                dummy_primal = insert_node!(ir, ssa, NewInstruction(inst; stmt=GlobalRef(DAECompiler.Intrinsics, :_VARIABLE_UNASSIGNED)))
                replace_call!(ir, ssa, Expr(:call, Diffractor.zero_bundle{1}(), dummy_primal))
                return nothing
            end

            u_ii = insert_node!(ir, ssa, NewInstruction(Expr(:call, getindex, u, var_ii)))
            du_ii = insert_node!(ir, ssa, NewInstruction(Expr(:call, getindex, du, var_ii)))
            _taylor1(primal, partial) = Diffractor.TaylorBundle{1}(primal, (partial,))
            replace_call!(ir, ssa, Expr(:call, _taylor1, u_ii, du_ii))
            return nothing
        elseif is_solved_variable(stmt) || is_known_invoke(stmt, observed!, ir)
            if is_solved_variable(stmt)
                out = dvar_out
                v = stmt.args[end-1]
                bundle = stmt.args[end]
                out_idx = findfirst(==(v), vars)
            else  # observed!
                out = dobs_out
                v = stmt.args[end]
                bundle = stmt.args[end-2]
                out_idx = findfirst(==(v), obs)
            end
            if out_idx !== nothing
                partial = insert_node!(ir, ssa, NewInstruction(Expr(:call, _first_partial, bundle)), true)
                insert_node!(ir, ssa, NewInstruction(Expr(:call, Base.setindex!, out, partial, out_idx)), true)
            end
            ir[ssa] = nothing
            return nothing
        elseif is_known_invoke(stmt, sim_time, ir)
            replace_call!(ir, ssa, Expr(:call, Diffractor.TaylorBundle{1}, t, (1.0,)))
            return nothing
        else
            # must be something the no dependency
            exclude_from_ad!(ir, ssa, maparg)
            return nothing
        end
    end
end
