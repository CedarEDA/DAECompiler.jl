"""
Call this within `transform` to exclude the statement at `ir[ssa]` from the AD process.
You must know that this function does not depend on any "input" being ADed wrt.
"""
function exclude_from_ad!(ir, ssa, maparg)
    inst = ir[ssa]
    stmt = inst[:inst]
    urs = userefs(stmt)
    for ur in urs
        ur[] = maparg(ur[], ssa, 0)
    end
    inst[:inst] = urs[]
    primal = insert_node!(ir, ssa, NewInstruction(inst))
    replace_call!(ir, ssa, Expr(:call, Diffractor.zero_bundle{1}(), primal))
end

@breadcrumb "ir_levels" function prepare_ir_for_differentiation(ir, transformed_sys; keep_epsilon=true, keep_observerd=false, keep_nonselected=false)
    interp = getfield(get_sys(transformed_sys), :interp)
    (;var_eq_matching,) = transformed_sys
    (;var_to_diff,) = transformed_sys.state.structure
    ir = copy(ir)

    # Remove intrinsics that are not used in the jacobian lowering
    for ii in 1:length(ir.stmts)
        inst = ir[SSAValue(ii)]
        stmt = inst[:inst]
        if is_known_invoke(stmt, singularity_root!, ir) ||
           is_known_invoke(stmt, time_periodic_singularity!, ir)
            inst[:inst] = nothing

        elseif is_known_invoke(stmt, observed!, ir) && !keep_observerd
            inst[:inst] = nothing

        elseif is_known_invoke(stmt, epsilon, ir) && !keep_epsilon
            inst[:inst] = 0.0

        # Filter out `solved_variable!()` calls that are for non-selected states, which
        # we don't use here; we're only interested in selected states.
        elseif is_solved_variable(stmt) && !keep_nonselected
            var = stmt.args[end-1]
            vint = invview(var_to_diff)[var]
            if vint === nothing || var_eq_matching[vint] !== SelectedState()
                # Solved algebric variable, not used in this lowering
                inst[:inst] = nothing
            end
        end
    end
    ir = run_dae_passes_again(interp, copy(ir), transformed_sys.state)
    return ir
end


"Set inst to be a differentiable nothing of given order"
function dnullout_inst!(inst, order=1)
    inst[:inst] = quoted(Diffractor.DNEBundle{order}(nothing))
    inst[:type] = typeof(Diffractor.DNEBundle{order}(nothing))
    inst[:flag] = CC.IR_FLAG_EFFECT_FREE | CC.IR_FLAG_NOTHROW | CC.IR_FLAG_CONSISTENT
    return nothing
end

"""
    _first_partial(x)

Extracts the first partial from any bundle, treats nonbundles as if they were `ZeroBundles`.
This is useful over `Diffractor.first_partial` if the AD transform might have been optimized to leave part of the code alone due to it having known zero derivative.
"""
_first_partial(x::Diffractor.ATB) = Diffractor.first_partial(x)
_first_partial(x) = zero(x)
