"""
construct_tgrad

Constructions a function `tgrad(dT,u,p,t)` which when called writes directional derivative of the system of equations with respect to the t direction.
"""
function construct_tgrad end

# This one is only for testing purposes
function construct_tgrad(sys::IRODESystem)
    transformed_sys = TransformedIRODESystem(sys)
    input_ir = prepare_ir_for_differentiation(transformed_sys.state.ir, transformed_sys)
    (;var_assignment, eq_assignment) = assign_vars_and_eqs(transformed_sys, false)
    return construct_tgrad(input_ir, transformed_sys, var_assignment, eq_assignment);
end

@breadcrumb "ir_levels" function construct_tgrad(input_ir::IRCode, transformed_sys::TransformedIRODESystem,
    var_assignment::Vector{Pair{Int, Bool}}, eq_assignment
)
    debug_config = DebugConfig(transformed_sys)
    (; var_eq_matching) = transformed_sys
    (; var_to_diff) = transformed_sys.state.structure


    # Note: this is only for ODEFunctions as DAEFunction doesn't accept a tgrad.
    ir = copy(input_ir)
    empty!(ir.argtypes)
    push!(ir.argtypes, Tuple{})  #OpaqueClosure
    push!(ir.argtypes, AbstractVector{<:Real})  # out
    push!(ir.argtypes, AbstractVector{<:Real})  # u,
    push!(ir.argtypes, parameter_type(get_sys(transformed_sys)))  # p
    push!(ir.argtypes, Real)  #  t

    interp = getfield(get_sys(transformed_sys), :interp)

    diff_ssas = filter_output_ssas(ir)

    # Perform AD transform
    Diffractor.forward_diff_no_inf!(
        ir, diff_ssas .=> 1;
        visit_custom! = tgrad_visit_custom!,
        transform! = define_transform_for_pushingforward_time(var_assignment, eq_assignment, var_eq_matching, var_to_diff),
        eras_mode=true,
    )

    ir = store_args_for_replay!(ir, debug_config, "tgrad")
    
    compact = IncrementalCompact(ir)
    # zero dudt
    dudt = Argument(2)
    z = insert_node_here!(compact, NewInstruction(Expr(:call, zero!, dudt), Any, Int32(1)))
    compact[z][:flag] |= CC.IR_FLAG_REFINED
    return construct_derivative_function(:tgrad, compact, transformed_sys, var_assignment, debug_config, false)
end

"Determines if there is something special to do for purposes of AD when trying to find tgrad"
function tgrad_visit_custom!(ir::IRCode, ssa::Union{SSAValue,Argument}, order, recurse)
    if isa(ssa, Argument)
        return false
    end

    stmt = ir[ssa][:inst]

    if is_known_invoke(stmt, sim_time, ir)
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

    if isa(stmt, PhiNode)
        # Don't run our custom transform for PhiNodes - we don't have a place
        # to put the call and the regular recursion will handle it fine.
        return false
    end

    # We have a custom transform for every statement with no time dependence
    return !has_time_dependence(typ)
end


function define_transform_for_pushingforward_time(var_assignment, eq_assignment, var_eq_matching, var_to_diff)
    function transform!(ir, ssa, order, maparg)
        @assert order==1  # tgrad is first order only.
        if isa(ssa, Argument)
            return insert_node!(ir, SSAValue(1), NewInstruction(Expr(:call, Diffractor.zero_bundle{1}(), ssa)))
        end

        # Name the arguments for ease of reference
        dT,u,p,t = Argument.(2:5)

        inst = ir[ssa]
        stmt = inst[:inst]
        while isa(stmt, SSAValue)
            # It's possible an earlier call to transform! from e.g. Pantelides moved this call, so follow references.
            stmt = ir[stmt][:inst]
        end

        if is_known_invoke(stmt, sim_time, ir)
            replace_call!(ir, ssa, Expr(:call, Diffractor.TaylorBundle{1}, t, (1.0,)))
            return nothing
        elseif is_equation_call(stmt, ir) || is_solved_variable(stmt)
            row, bundle = determine_jacobian_row_and_bundle(ir, stmt, eq_assignment, var_to_diff, var_eq_matching, var_assignment)
            isnothing(row) && return dnullout_inst!(inst)
            row::Integer
            @assert row > 0 "out of state: $ssa $stmt"
            if !isa(bundle, SSAValue)
                @assert isa(bundle, Real)  # It is just a literal, thus no derivative was computed
                bundle = Diffractor.zero_bundle{1}()(bundle)  # thus we know the derivative wrt time is zero
            end
            # here we write into dT the result of differenting this equation/solved_variable wrt sim_time
            partial = insert_node!(ir, ssa, NewInstruction(Expr(:call, _first_partial, bundle)), true)
            addi = insert_node!(ir, ssa, NewInstruction(Expr(:call, addindex!, dT, partial, row)), true)
            ir[addi][:flag] |= CC.IR_FLAG_INBOUNDS
            return dnullout_inst!(inst)
        else
            typ = inst[:type]
            @assert !has_time_dependence(typ)
            exclude_from_ad!(ir, ssa, maparg)
            return nothing
        end
    end
end
