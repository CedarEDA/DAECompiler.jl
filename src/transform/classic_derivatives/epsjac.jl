"""
    get_epsjac(obj|tsys)

For a SciML object such as an ODEProblem, ODESolution etc, or a `TransformedIRODESystem` this returns the 
`espjac!(J, u, p, t)`  which when called writes directional derivative of the system of equations with
respects to the epislons in the system into J.
"""
get_epsjac(obj) = get_epsjac(get_transformed_sys(obj))

function get_epsjac(tsys::TransformedIRODESystem)
    if isdefined(tsys.epsjac_cache, 1)
        tsys.epsjac_cache[]
    else
        tsys.epsjac_cache[] = construct_epsjac(tsys)
    end
end


"""
construct_epsjac

Constructions a function `epsjac(J,u,p,t)!` which when called writes directional derivative of the system of equations with
respects to the epislons in the system.
"""
function construct_epsjac end
function construct_epsjac(transformed_sys::TransformedIRODESystem)
    input_ir = prepare_ir_for_differentiation(transformed_sys.state.ir, transformed_sys)
    (;var_assignment, eq_assignment) = assign_vars_and_eqs(transformed_sys, false)
    return construct_epsjac(input_ir, transformed_sys, var_assignment, eq_assignment);
end

@breadcrumb "ir_levels" function construct_epsjac(
    input_ir::IRCode, transformed_sys::TransformedIRODESystem,
    var_assignment::Vector{Pair{Int, Bool}}, eq_assignment;
)
    (; var_eq_matching) = transformed_sys
    (; var_to_diff) = transformed_sys.state.structure
    debug_config = DebugConfig(transformed_sys)

    ir = copy(input_ir)

    diff_ssas = filter_output_ssas(ir)

    # Perform AD transform
    Diffractor.forward_diff_no_inf!(
        ir, diff_ssas .=> 1;
        visit_custom! = epsjac_visit_custom!,
        transform! = define_transform_for_pushingforward_epsilon(
            eq_assignment, var_to_diff, var_eq_matching, var_assignment, transformed_sys.state.neps
        ),
        eras_mode=true,
    )
    ir = store_args_for_replay!(ir, debug_config, "epsjac")

    compact = IncrementalCompact(ir)
    # zero J
    J = Argument(2)
    z = insert_node_here!(compact, NewInstruction(Expr(:call, zero!, J), Any, Int32(1)))
    compact[z][:flag] |= CC.IR_FLAG_REFINED

    return construct_derivative_function(:epsjac, compact, transformed_sys, var_assignment, debug_config, false)
end


"Determines if there is something special to do for purposes of AD when trying to find paramjac"
function epsjac_visit_custom!(ir::IRCode, ssa::Union{SSAValue,Argument}, order, recurse)
    if isa(ssa, Argument)
        return ssa==Argument(1)
    end

    stmt = ir[ssa][:inst]
    if is_known_invoke(stmt, epsilon, ir)
        return true
    elseif is_equation_call(stmt, ir)
        recurse(_eq_val_arg(stmt))
        return true
    elseif is_solved_variable(stmt)
        recurse(stmt.args[end])
        return true
    end

    typ = ir[ssa][:type]
    has_simple_incidence_info(typ) || return false
    return !has_epsilon_dependence(typ)  # if it doesn't have epsilon dependence we have special handling for it.
end

function define_transform_for_pushingforward_epsilon(eq_assignment, var_to_diff, var_eq_matching, var_assignment, neps)
    function transform!(ir, ssa, order, maparg)
        @assert order==1  # paramjac is first order only.
        if isa(ssa, Argument)
            return insert_node!(ir, SSAValue(1), NewInstruction(Expr(:call, Diffractor.zero_bundle{1}(), ssa)))
        end

        # Name the arguments for ease of reference
        J,u,p,t = Argument.(2:5)

        inst = ir[ssa]
        stmt = inst[:inst]
        while isa(stmt, SSAValue)
            # It's possible an earlier call to transform! from e.g. Pantelides moved this call, so follow references.
            stmt = ir[stmt][:inst]
        end

        if is_equation_call(stmt, ir) || is_solved_variable(stmt)
            row, bundles = determine_jacobian_row_and_bundle(ir, stmt, eq_assignment, var_to_diff, var_eq_matching, var_assignment)
            isnothing(row) && return dnullout_inst!(inst)
            row::Integer
            @assert row > 0 "out of state: $ssa $stmt"
            
            # here we write into J the result of differenting this equation/solved_variable wrt parameters
            for col in 1:neps
                partial = insert_node!(ir, ssa, NewInstruction(Expr(:call, extract_partial, bundles, col)), true)
                addi = insert_node!(ir, ssa, NewInstruction(Expr(:call, addindex!, J, partial, row, col)), true)
                ir[addi][:flag] |= CC.IR_FLAG_INBOUNDS
            end
            return dnullout_inst!(inst)
        elseif is_known_invoke(stmt, epsilon, ir)
            eps_ii = epsnum(inst[:type])
            input_basis_row = ntuple(neps) do active_eps_ii
                if eps_ii == active_eps_ii
                    1.0
                else
                    # for this batch step some inactive element of the state, so zero derivative
                    0.0
                end
            end
            replace_call!(ir, ssa, Expr(:call, BatchOfBundles{neps}, 0.0, input_basis_row...))
            return nothing
        else
            # this is something that doesn't depend on epsilons, We don't need to AD it at all
            # A special case of this is variable, state_ddt, and sim_time
            @assert !has_epsilon_dependence(inst[:type])
            exclude_from_ad!(ir, ssa, maparg)
            return nothing
        end
    end
end
