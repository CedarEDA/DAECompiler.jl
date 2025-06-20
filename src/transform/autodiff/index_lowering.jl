function tearing_visit_custom!(ir::IRCode, ssa::Union{SSAValue,Argument}, order, recurse)
    if isa(ssa, Argument)
        return false
    end

    stmt = ir[ssa][:inst]
    if is_known_invoke_or_call(stmt, variable, ir)
        return true
    elseif is_known_invoke(stmt, equation, ir)
        return true
    elseif is_known_invoke(stmt, sim_time, ir)
        return true
    elseif is_equation_call(stmt, ir)
        recurse(_eq_function_arg(stmt))
        recurse(_eq_val_arg(stmt))
        return true
    elseif is_known_invoke(stmt, ddt, ir)
        recurse(stmt.args[end], order+1)
        return true
    end

    if isa(stmt, PhiNode)
        # Don't run our custom transform for PhiNodes - we don't have a place
        # to put the call and the regular recursion will handle it fine.
        return false
    end

    typ = ir[ssa][:type]
    has_simple_incidence_info(typ) || return false

    # we have custom handling for things without any dependency on time nor state
    return !has_dependence(typ)
end

function is_diffed_equation_call_invoke_or_call(@nospecialize(stmt), ir::IRCode)
    (isexpr(stmt, :invoke) || isexpr(stmt, :call)) || return false
    callee = _eq_function_arg(stmt)
    isa(callee, SSAValue) || return false
    bundlecall = ir[callee][:inst]
    isexpr(bundlecall, :call) || return false
    bt = bundlecall.args[1]
    isa(bt, Type) || return false
    bt <: Diffractor.TaylorBundle || return false
    ft = argextype(bundlecall.args[2], ir)
    return widenconst(ft) === equation
end

function index_lowering_ad!(state::TransformationState, key::TornCacheKey, settings::Settings)
    (; result, structure) = state
    (; var_to_diff, eq_to_diff, graph, solvable_graph) = structure

    ir = state.result.ir

    # Figure out which equations we need to differentiate
    # TODO: Should have some nicer interface in MTK
    diff_eqs = Pair{Int, Int}[]
    for i = 1:length(eq_to_diff)
        # If this is a linear equation, we cannot differentiate it, because
        # alias elimination changed the equation on us, but didn't update the
        # IR. We codegen it directly below.
        if invview(eq_to_diff)[i] === nothing && eq_to_diff[i] !== nothing && !isempty(𝑠neighbors(graph, eq_to_diff[i]))
            level = 1
            diff = eq_to_diff[i]
            is_fully_state_linear(result.total_incidence[i], key.param_vars) && continue
            while (diff = eq_to_diff[diff]) !== nothing
                level += 1
            end
            push!(diff_eqs, i => level)
        end
    end

    # Mark all non-trivial `ddt()` statements as ones that we should differentiate
    diff_ssas = Pair{SSAValue,Int}[]
    for i = 1:length(ir.stmts)
        if is_known_invoke(ir.stmts[i][:stmt], ddt, ir) && !is_const_plus_var_known_linear(argextype(ir.stmts[i][:stmt].args[end], ir))
            push!(diff_ssas, SSAValue(i) => 0)
        end
    end

    if isempty(diff_ssas) && isempty(diff_eqs)
        return copy(ir)
    end

    compact = IncrementalCompact(copy(ir))
    # TODO: This could all be combined with the below into a single pass
    (eqs, vars) = find_eqs_vars(state.structure.graph, compact)
    ir = Compiler.finish(compact)

    for (eq, level) in diff_eqs
        for ssa in eqs[eq][2]
            push!(diff_ssas, ssa => level)
        end
    end

    append!(eqs, (SSAValue(0)=>SSAValue[] for _ in 1:(length(eq_to_diff)-length(eqs))))
    append!(vars, fill(SSAValue(0), length(var_to_diff)-length(vars)))
    domtree = Compiler.construct_domtree(ir.cfg.blocks)

    function diff_one!(ir, ssa, dvar)
        if dvar === nothing
            # dvar can be `nothing` if we are differentiating a variable that doesn't actually appear
            # in the matched system structure's incidence analysis for the equation currently being differentiated.
            # This can occur because Diffractor's types bundle both the primal and the tangent derivatives
            # in a single type, causing differentiation of all listed variables to hit this function.
            # We emit here a `_DIFF_UNUSED` value that we expect to never be used and DCE'd later on in the pipeline.
            return insert_node!(ir, ssa, NewInstruction(GlobalRef(DAECompiler.Intrinsics, :_DIFF_UNUSED), Incidence(Float64), Int32(1)))
        end
        if vars[dvar] == SSAValue(0)
            vars[dvar] = insert_node!(ir, ssa, NewInstruction(Expr(:invoke, nothing, variable), Incidence(dvar)))
        elseif !dominates_ssa(ir, domtree, vars[dvar], ssa; dominates_after=true)
            varssa = vars[dvar]
            inst = ir[varssa]
            vars[dvar] = insert_node!(ir, ssa, NewInstruction(inst))
            ir[varssa][:inst] = vars[dvar]
        end
        return vars[dvar]
    end

    function diff_variable!(ir, settings, ssa, stmt, order)
        inst = ir[ssa]
        var = idnum(ir[ssa][:type])
        primal = insert_node!(ir, ssa, NewInstruction(inst))
        vars[var] = primal
        diffs = SSAValue[]
        for i = 1:order
            var !== nothing && (var = var_to_diff[var])
            push!(diffs, diff_one!(ir, ssa, var))
        end
        duals = insert_node!(ir, ssa, NewInstruction(
            Expr(:call, tuple, diffs...), Any
        ))
        replace_call!(ir, ssa, Expr(:call, Diffractor.TaylorBundle{order}, primal, duals), settings, @__SOURCE__)
    end

    function transform!(ir, ssa, order, maparg)
        if isa(ssa, Argument)
            # at start of function define a SSA holding the initially accumulated derivative of each argument, (i.e. 0)
            return insert_node!(ir, SSAValue(1), NewInstruction(Expr(:call, Diffractor.zero_bundle{order}(), ssa), Any))
        end
        inst = ir[ssa]
        stmt = inst[:inst]
        while isa(stmt, SSAValue)
            # It's possible an earlier call to transform! moved this call, so follow references.
            stmt = ir[stmt][:inst]
        end
        if is_known_invoke(stmt, variable, ir)
            diff_variable!(ir, ssa, stmt, order)
            return nothing
        elseif is_known_invoke(stmt, equation, ir)
            eq = inst[:type].id
            primal = insert_node!(ir, ssa, NewInstruction(inst))
            eqs[eq] = primal=>eqs[eq][2]
            duals = SSAValue[]
            for _ = 1:order
                deq = eq_to_diff[eq]
                # If `deq` is nothing, that means we're asking for a derivative of an equation
                # that does not exist.  This is possible if we, for instance, have a tuple of
                # equation-related values that does not get SROA'ed, and is then differentiated
                # by Diffractor due to _one_ of the equations being differentiated.  But that
                # results in this loop asking for derivatives of the _other_ equations that
                # don't exist.  To handle this, we insert a bogus equation node, similar in
                # spirit to the `_DIFF_UNUSED` value.
                if deq === nothing
                    diff = insert_node!(ir, ssa, NewInstruction(GlobalRef(DAECompiler.Intrinsics, :_EQ_UNUSED), equation))
                else
                    diff = insert_node!(ir, ssa, NewInstruction(inst))
                    diffinst = ir[diff]
                    diffinst[:type] = Eq(deq)
                    eqs[deq] = diff=>eqs[deq][2]
                    eq = deq
                end
                push!(duals, diff)
            end
            dtup = insert_node!(ir, ssa, NewInstruction(
                Expr(:call, tuple, duals...), Any
            ))
            # N.B.: No replace_call!, because we rely on the type of this call.
            inst[:inst] = Expr(:call, Diffractor.TaylorBundle{order}, primal, dtup)
            inst[:info] = Compiler.NoCallInfo()
            return nothing
        elseif is_known_invoke(stmt, sim_time, ir)
            time = insert_node!(ir, ssa, NewInstruction(inst))
            replace_call!(ir, ssa, Expr(:call, Diffractor.∂xⁿ{order}(), time), settings, @__SOURCE__)
            return nothing
        elseif is_diffed_equation_call_invoke_or_call(stmt, ir)
            eq = idnum(argextype(_eq_function_arg(stmt), ir))
            bundle = _eq_val_arg(stmt)
            # Rewrite the equation (we could extract it from the bundle, but we already know where it is)
            # N.B.: We don't need replace_call! here, because we're not changing the call target,
            # we're just rearranging the SSA.
            inst[:inst] = Expr(
                :call,
                eqs[eq][1],
                insert_node!(ir, ssa, NewInstruction(Expr(:call, getfield, bundle, 1), Any)),  # primal
            )
            # Pull out the equation from the primal, so we can null it out below
            new_primal = insert_node!(ir, ssa, NewInstruction(inst))
            replace!(eqs[eq][2], ssa=>new_primal)
            for i = 1:order
                val = insert_node!(ir, ssa, NewInstruction(Expr(:call, getindex, bundle, Diffractor.TaylorTangentIndex(i)), Any))
                push!(
                    eqs[eq_to_diff[eq]][2],
                    insert_node!(ir, ssa, NewInstruction(Expr(:call, eqs[eq_to_diff[eq]][1], val), Any))
                )
                eq = eq_to_diff[eq]
            end
            # equation! also returns nothing, but it's possible for the value
            # to be used (e.g. by a return, so conform to the interface)
            dnullout_inst!(inst, order)
        elseif is_known_invoke(stmt, ddt, ir)
            arg = maparg(stmt.args[end], ssa, order+1)
            if order == 0
                replace_call!(ir, ssa, Expr(:call, Diffractor.partial, arg, 1), settings, @__SOURCE__())
            else
                replace_call!(ir, ssa, Expr(:call, diff_bundle, arg), settings, @__SOURCE__())
            end
            return nothing
        else
            # must be something with no dependency
            @assert !has_dependence(inst[:type])
            urs = userefs(stmt)
            for ur in urs
                ur[] = maparg(ur[], ssa, 0)
            end
            inst[:inst] = urs[]
            primal = insert_node!(ir, ssa, NewInstruction(inst))
            replace_call!(ir, ssa, Expr(:call, Diffractor.zero_bundle{order}(), primal), settings, @__SOURCE__)
            return nothing
        end
    end
    Diffractor.forward_diff_no_inf!(ir, diff_ssas; visit_custom! = tearing_visit_custom!, transform!, eras_mode=true)

    # Rename state
    compact = IncrementalCompact(ir)
    (eqs, vars) = find_eqs_vars(state.structure.graph, compact)
    # Some variables may look dead, but are used in linear equations
    # don't dce them just yet - we'll dce them below
    Compiler.non_dce_finish!(compact)
    ir = Compiler.complete(compact)

    # Derivatives can appear out of "thin air" due to implicit dependencies
    # (i.e. an equation that depends on 1 also depends on ddt(1)), or due to
    # imprecision introduced by the AD transform (causing a primal to
    # spuriously be carried along in the Incidence with its derivative).
    #
    # Allow this by verifying there is an element in `g` whose k-derivative
    # is `var` (k ∈ ℤ).
    function in_any_derivative(var, g)
        while var_to_diff[var] !== nothing
            var = var_to_diff[var] # Normalize to highest-derivative
        end
        while true
            var in g && return true
            invview(var_to_diff)[var] === nothing && return false
            var = invview(var_to_diff)[var]
        end
    end

    # Update solvable graph
#=
    for (eq, (_, eqssas)) in enumerate(eqs)
        is_fully_state_linear(state.total_incidence[eq], key.param_vars) && continue
        old_graph = empty_eq_list!(graph, eq)
        old_solvable_graph = empty_eq_list!(solvable_graph, eq)
        for eqssa in eqssas
            if ir[eqssa][:inst] === nothing
                # Could have been in a dead branch and deleted - allow that for now.
                continue
            end
            eqssaval = _eq_val_arg(ir[eqssa][:inst])
            inc = ir[eqssaval][:type]
            if !isa(inc, Incidence)
                throw(UnsupportedIRException("Expected incidence analysis to produce result for $eqssaval, got $inc", ir))
            end
            for (v, coeff) in zip(rowvals(inc.row), nonzeros(inc.row))
                v == 1 && continue
                @assert in_any_derivative(v-1, old_graph)
                @assert !has_edge(graph, BipartiteEdge(eq, v-1))
                add_edge!(graph, eq, v-1)
                if coeff !== nonlinear
                    add_edge!(solvable_graph, eq, v-1)
                else
                    # TODO: solvable should generally not become unsolvable but in some cases
                    # our AD transform widens Incidence propagation in a way that artificially
                    # makes tearing's life harder (see downstream BSIM-CMG test)
                    # @assert !(v-1 in old_solvable_graph)
                    if v-1 in old_solvable_graph
                        @debug "Variable $(v-1) in Eq. $(eq) went from solvable -> unsolvable after AD transform"
                    end
                end
            end
        end
    end
=#

    return ir
end

function empty_eq_list!(graph::BipartiteGraph, eq)
    vs = copy(𝑠neighbors(graph, eq))
    foreach(vs) do v
        rem_edge!(graph, eq, v)
    end
    return vs
end
