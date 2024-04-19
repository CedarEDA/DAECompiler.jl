"""
construct_paramjac

Constructions a function `paramjac(J,u,p,t)` which when called writes directional derivative of the system of equations with respect to the parameters `p`.
"""
function construct_paramjac end

# This one is only for testing purposes
function construct_paramjac(sys::IRODESystem)
    transformed_sys = TransformedIRODESystem(sys)
    (;state, var_eq_matching) = transformed_sys
    input_ir = prepare_ir_for_differentiation(transformed_sys.state.ir, transformed_sys)
    (;var_assignment, eq_assignment) = assign_vars_and_eqs(transformed_sys, false)
    return construct_paramjac(input_ir, transformed_sys, var_assignment, eq_assignment);
end

@breadcrumb "ir_levels" function construct_paramjac(
    input_ir::IRCode, transformed_sys::TransformedIRODESystem,
    var_assignment::Vector{Pair{Int, Bool}}, eq_assignment;
)
    (; var_eq_matching) = transformed_sys
    (; var_to_diff) = transformed_sys.state.structure
    debug_config = DebugConfig(transformed_sys)

    # Note: this is only for ODEFunctions as DAEFunction doesn't accept a paramjac.
    ir = copy(input_ir)

    diff_ssas = filter_output_ssas(ir)

    p = Argument(4)
    num_params = determine_num_tangents(parameter_type(get_sys(transformed_sys)))
    param_bob_ssa = insert_param_bob(ir, p)
    # Perform AD transform
    Diffractor.forward_diff_no_inf!(
        ir, diff_ssas .=> 1;
        visit_custom! = paramjac_visit_custom!,
        transform! = define_transform_for_pushingforward_all_params(
            eq_assignment, var_to_diff, var_eq_matching, var_assignment, num_params, param_bob_ssa
        ),
        eras_mode=true,
    )
    ir = store_args_for_replay!(ir, debug_config, "paramjac")

    compact = IncrementalCompact(ir)
    # zero J
    J = Argument(2)
    z = insert_node_here!(compact, NewInstruction(Expr(:call, zero!, J), Any, Int32(1)))
    compact[z][:flag] |= CC.IR_FLAG_REFINED

    return construct_derivative_function(:param_jac, compact, transformed_sys, var_assignment, debug_config, false)
end


"Determines if there is something special to do for purposes of AD when trying to find paramjac"
function paramjac_visit_custom!(ir::IRCode, ssa::Union{SSAValue,Argument}, order, recurse)
    if isa(ssa, Argument)
        return ssa==Argument(1)
    end

    stmt = ir[ssa][:inst]
    if is_known_invoke_or_call(stmt, variable, ir)
        return true
    elseif is_known_invoke_or_call(stmt, state_ddt, ir)
        return true
    elseif is_known_invoke(stmt, sim_time, ir)
        return true
    elseif is_equation_call(stmt, ir)
        recurse(_eq_val_arg(stmt))
        return true
    elseif is_solved_variable(stmt)
        recurse(stmt.args[end])
        return true
    end
    return false
end

zero_tangents(primal, quantity) = ntuple((_)->DAECompiler.deep_zero(typeof(primal)), quantity)
padded_basis_bob(primal, left_padding, right_padding) = BatchOfBundles(primal, (
    zero_tangents(primal, left_padding)...,
    basis_tangents(primal)...,
    zero_tangents(primal, right_padding)...,
))
function insert_param_bob(ir, p; left_padding=0, right_padding=0)
    if left_padding == 0 && right_padding == 0
        new_inst = NewInstruction(Expr(:call, basis_bob, p))
    else
        # XXX: Using this function call for all of our jacobians seems to trigger a segfault
        #      in the `sensitivity/amplifier_ibias` example in CedarEDA
        new_inst = NewInstruction(Expr(:call, padded_basis_bob, p, left_padding, right_padding))
    end
    return insert_node!(ir, SSAValue(1), new_inst, #= attach_after =# false)
end


function define_transform_for_pushingforward_all_params(eq_assignment, var_to_diff, var_eq_matching, var_assignment, num_params, param_bob_ssa)
    function transform!(ir, ssa, order, maparg)
        @assert order==1  # paramjac is first order only.
        if isa(ssa, Argument)
            if ssa == Argument(1)
                return param_bob_ssa
            else
                return insert_node!(ir, SSAValue(1), NewInstruction(Expr(:call, Diffractor.zero_bundle{1}(), ssa)))
            end
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
            for col in 1:num_params
                partial = insert_node!(ir, ssa, NewInstruction(Expr(:call, extract_partial, bundles, col)), true)
                addi = insert_node!(ir, ssa, NewInstruction(Expr(:call, addindex!, J, partial, row, col)), true)
                ir[addi][:flag] |= CC.IR_FLAG_INBOUNDS
            end
            return dnullout_inst!(inst)
        else
            if is_known_invoke(stmt, sim_time, ir)
                # use time argument in-place of sim_time
                replace_call!(ir, ssa, Expr(:call, Diffractor.zero_bundle{1}(), t))
            else
                @assert is_known_invoke_or_call(stmt, variable, ir) || is_known_invoke_or_call(stmt, state_ddt, ir)
                # We want to do nothing to these, not even AD them.
                # later `conclude_derivative_construction` will clean them up, replacing with references to `u` etc.            
                primal = insert_node!(ir, ssa, NewInstruction(inst))
                replace_call!(ir, ssa, Expr(:call, Diffractor.zero_bundle{1}(), primal))
                return nothing
            end
        end
    end
end
