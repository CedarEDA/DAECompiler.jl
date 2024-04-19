"""
construct_jacobian

Constructions a function `jac(J,u,p,t)` which when called writes the jacobian into `J`.
"""
function construct_jacobian end

# This one is only for testing purposes
function construct_jacobian(sys::IRODESystem; isdae=false, kwargs...)
    transformed_sys = TransformedIRODESystem(sys)
    input_ir = prepare_ir_for_differentiation(transformed_sys.state.ir, transformed_sys)
    (; neqs, var_assignment, eq_assignment, dummy_map) = assign_vars_and_eqs(transformed_sys, isdae)
    return construct_jacobian(
        input_ir, transformed_sys, var_assignment, eq_assignment, dummy_map, neqs; isdae, kwargs...
    );
end

@breadcrumb "ir_levels" function construct_jacobian(input_ir::IRCode, transformed_sys::TransformedIRODESystem,
    var_assignment::Vector{Pair{Int, Bool}}, eq_assignment, dummy_map, neqs::Int;
    isdae=false)
    debug_config = DebugConfig(transformed_sys)
    (; var_eq_matching) = transformed_sys
    (; var_to_diff) = transformed_sys.state.structure

    ir = copy(input_ir)
    # Change argtypes (Note, we will replace these again during specialization, but we need them anyway for some of the IR manipulation tools to work right)
    empty!(ir.argtypes)   # it is the arguments of the function that defines the DAE
    push!(ir.argtypes, Tuple{})  #OpaqueClosure
    push!(ir.argtypes, AbstractMatrix{<:Real})  # J
    isdae && push!(ir.argtypes, AbstractVector{<:Real})  # du,
    push!(ir.argtypes, AbstractVector{<:Real})  # u,
    push!(ir.argtypes, parameter_type(get_sys(transformed_sys)))  # p
    isdae && push!(ir.argtypes, Real)  #  γ
    push!(ir.argtypes, Real)  #  t

    diff_ssas = filter_output_ssas(ir)

    # Perform AD transform
    @may_timeit debug_config "forward_diff_no_inf" begin
        Diffractor.forward_diff_no_inf!(
            ir, diff_ssas .=> 1;
            visit_custom! = get_jac_visit_custom!(var_assignment),
            transform! = define_transform_for_pushingforward_all_states(neqs, var_assignment, eq_assignment, var_eq_matching, var_to_diff, isdae),
            eras_mode=true,
        )
    end
    record_ir!(debug_config, "post_diffractor", ir)
    ir = store_args_for_replay!(ir, debug_config, "jacobian")

    # post-processing, including compacting
    compact = IncrementalCompact(ir)

    J = Argument(2)
    # zero the jacobian
    z = insert_node_here!(compact, NewInstruction(Expr(:call, zero!, J), Any, Int32(1)))
    compact[z][:flag] |= CC.IR_FLAG_REFINED
    # Process various kinds of aliases
    # dummy_map
    if !isdae
        for (diff, var) in dummy_map
            insert_node_here!(compact, NewInstruction(Expr(:call, setindex!, J, 1.0, diff, var), Any, ir.stmts[1][:line]))
        end
    end

    # implicit equation:
    # Only occurs if my active state is a derivative of some other selected state and both are selected
    # the active state is of-course selected as otherwise it wouldn't be part of the state vector
    # both ODE and DAE have a u state
    for (vint, var) in enumerate(var_to_diff)
        isnothing(var) && continue


        if var_eq_matching[vint] == var_eq_matching[var] == SelectedState()
            var_slot, var_indu = var_assignment[var]
            @assert var_slot > 0
            vint_slot, vint_in_du = var_assignment[vint]
            @assert vint_slot > 0
            insert_node_here!(compact, NewInstruction(Expr(:call, setindex!, J, 1.0, vint_slot, var_slot), Any, ir.stmts[1][:line]))

            if isdae  # only DAE has a du state
                # By convention, the equation slot for the implicit equation matches the
                # variable slot for the integrated variable.
                # so:
                row = col = vint_slot
                γ = Argument(6)
                partial = insert_node_here!(compact, NewInstruction(Expr(:call, -, γ), Any, ir.stmts[1][:line]))
                insert_node_here!(compact, NewInstruction(Expr(:call, setindex!, J, partial, row, col), Any, ir.stmts[1][:line]))
            end
        end
    end

    return construct_derivative_function(:jac, compact, transformed_sys, var_assignment, debug_config, isdae; has_γ=true)
end

function define_transform_for_pushingforward_all_states(neqs, var_assignment, eq_assignment, var_eq_matching, var_to_diff, isdae::Bool)
    # Custom tranform needs to be different:
    # Custom transform for `variable` needs to construct a basis element for that element
    # Custom tranform for `state_ddt` similarly
    # Custom for equation! and solved_variable needs to write the row into the jacobian
    function transform!(ir, ssa, order, maparg)
        @assert order==1  # jacobian is first order only.
        if isa(ssa, Argument)
            return insert_node!(ir, SSAValue(1), NewInstruction(Expr(:call, Diffractor.zero_bundle{order}(), ssa)))
        end

        # Name the arguments for ease of reference
        if isdae
            J,du,u,p,γ,t = Argument.(2:7)
        else
            J,u,p,t = Argument.(2:5)
        end


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
                @assert var_eq_matching[var] !== SelectedState()
                dummy_primal = insert_node!(ir, ssa, NewInstruction(inst; stmt=GlobalRef(DAECompiler.Intrinsics, :_VARIABLE_UNASSIGNED)))
                replace_call!(ir, ssa, Expr(:call, Diffractor.zero_bundle{1}(), dummy_primal))
                return nothing
            end

            u_ii = insert_node!(ir, ssa, NewInstruction(Expr(:call, getindex, in_du ? du : u, var_ii)))
            input_basis_row = ntuple(neqs) do active_state_ii
                if var_ii == active_state_ii
                    # This is our active element, normally we insert a 1 here
                    # but for a DAE, if it is an element of `du`, then we insert a γ here as we are
                    # computing `γ*dG/d(du)`
                    in_du ? γ : 1.0
                else
                    # for this batch step some inactive element of the state, so zero derivative
                    0.0
                end
            end
            replace_call!(ir, ssa, Expr(:call, BatchOfBundles{neqs}, u_ii, input_basis_row...))
            return nothing
        elseif is_solved_variable(stmt) || is_equation_call(stmt, ir)
            row, bundles = determine_jacobian_row_and_bundle(ir, stmt, eq_assignment, var_to_diff, var_eq_matching, var_assignment)
            isnothing(row) && return dnullout_inst!(inst)
            row::Integer
            @assert row > 0 "out of state: $ssa $stmt"

            # here we write into J the result of differenting this equation/solved_variable
            for ii in 1:neqs
                partial = insert_node!(ir, ssa, NewInstruction(Expr(:call, extract_partial, bundles, ii)), true)
                if isdae && row == ii && is_solved_variable(stmt)
                    # Implicit equations are of the form `f(u) - γ*du`, so we need an implicit -γ here.
                    partial = insert_node!(ir, ssa, NewInstruction(Expr(:call, -, partial, γ)), true)
                end
                addi = insert_node!(ir, ssa, NewInstruction(Expr(:call, addindex!, J, partial, row, ii)), true)
                ir[addi][:flag] |= CC.IR_FLAG_INBOUNDS
            end
            return dnullout_inst!(inst)
        else
            # must be something the no dependency on selected states
            exclude_from_ad!(ir, ssa, maparg)
            return nothing
        end
    end
end


"Determines if there is something special to do for purposes of AD when trying to find jacobian"
function get_jac_visit_custom!(var_assignment)
    selected_states=[var_num for (var_num, (slot, _)) in enumerate(var_assignment) if !iszero(slot)]
    function jac_visit_custom!(ir::IRCode, ssa::Union{SSAValue,Argument}, order, recurse)
        if isa(ssa, Argument)
            return false
        end

        stmt = ir[ssa][:inst]
        if is_known_invoke_or_call(stmt, variable, ir)
            return true
        elseif is_known_invoke_or_call(stmt, state_ddt, ir)
            return true
        elseif is_equation_call(stmt, ir)
            recurse(_eq_val_arg(stmt))
            return true
        elseif is_solved_variable(stmt)
            recurse(stmt.args[end])
            return true
        end

        if isa(stmt, PhiNode)
            # Don't run our custom transform for PhiNodes - we don't have a place
            # to put the call and the regular recursion will handle it fine.
            return false
        end

        typ = ir[ssa][:type]
        has_simple_incidence_info(typ) || return false

        if isa(typ, Incidence)
            if has_only_time_dependence(typ) || !has_dependence(typ)
                return true
            end
            @assert has_state_dependence(typ, selected_states)
            return false
        else
            # we have custom handling for things without any dependency on our selected states
            return true
        end
    end
end
