using ModelingToolkit.StructuralTransformations: DiCMOBiGraph, topological_sort_by_dfs
using Core.Compiler: NewSSAValue, OldSSAValue, Instruction, insert_node_here!, construct_domtree

function type_contains_taint(@nospecialize(incT), var, available)
    if isa(incT, Incidence)
        tainted_by_solved = tainted_by_unavailable = false
        for (k, v) in zip(rowvals(incT.row), nonzeros(incT.row))
            k -= 1
            k == 0 && continue
            if !iszero(v)
                if k == var
                    tainted_by_solved = true
                elseif !(k in available)
                    tainted_by_unavailable = true
                end
            end
        end
        return (tainted_by_solved, tainted_by_unavailable)
    end
    if isa(incT, PartialStruct)
        return mapreduce(x->type_contains_taint(x, var, available), ((a,b),(c,d))->(a|c,b|d), incT.fields)
    end
    return (false, false)
end

coeff_from_incidence(var, row) = row[var+1]  # indicies in row are offset by 1 from var numbers as time is row[1]

function lanewise(fs...)
    function (as, bs)
        map(fs, as, bs) do f, a, b
            f(a, b)
        end
    end
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

function diff_bundle(bundle::Diffractor.UniformBundle{N, B, U}) where {N, B, U}
    return Diffractor.UniformBundle{N-1}(bundle.tangent.val, bundle.tangent)
end

function diff_bundle(bundle::Diffractor.TaylorBundle{N}) where {N}
    return Diffractor.TaylorBundle{N-1}(bundle.tangent.coeffs[1], bundle.tangent.coeffs[2:end])
end

@breadcrumb "ir_levels" function tearing_schedule!(state::IRTransformationState, linear_eqs, ils, var_eq_matching)
    ir = state.ir
    debug_config = DebugConfig(state)
    (; var_to_diff, graph) = state.structure
    record_ir!(state, "pre_tearing", ir)

    compact = IncrementalCompact(ir)
    (eqs, vars) = find_eqs_vars(state, compact)
    ir = Core.Compiler.finish(compact)

    @may_timeit debug_config "tearing" begin
        domtree = construct_domtree(ir.cfg.blocks)
        compact = IncrementalCompact(ir)
        record_ir!(state, "tearing.compact", ir)

        toporder = topological_sort_by_dfs(DiCMOBiGraph{false}(graph, var_eq_matching))

        rename = Dict{SSAValue, Union{SSAValue, Float64}}()

        function schedule_missing_derivative!(v::Int, line=Int32(1))
            @assert vars[v] == SSAValue(0)
            @assert var_eq_matching[v] in (StructuralTransformations.SelectedState(), unassigned)
            new_node = insert_node_here!(compact, NewInstruction(Expr(:call, state_ddt, v), Incidence(v), line))
            vars[v] = NewSSAValue(new_node.id)
            return new_node
        end

        # If we try to get the variable IR instruction for a `NewSSAValue`,
        # that signifies that we're looking up something that was recently added,
        # and therefore we should look in `compact`, not `ir`.
        get_var_ir(s::SSAValue) = ir[s]
        get_var_ir(s::NewSSAValue) = compact[SSAValue(s.id)]


        # Given an input incidence type `incT`, iterate over its component
        # variables, map those through `vars`, then return a new incidence
        # with the new, compressed incidence.
        function remap_incidence(incT::Incidence, var)
            const_val = incT.typ
            new_row = _zero_row()
            # Map old incidence variable numbers to only the selected states that have been chosen by tearing.
            for (v_offset, coeff) in zip(rowvals(incT.row), nonzeros(incT.row))
                v = v_offset - 1

                # Time dependence persists as itself
                if v == 0
                    new_row[v_offset] += coeff
                    continue
                end

                # Skip ourselves
                if v == var
                    continue
                end

                canonical_stmt = if isa(vars[v], SSAValue)
                    # Special-case values that are an old SSAValue
                    # This only occurs in our final compaction, (hence `var == 0`)
                    # and should only happen for selected states/algebraic variables
                    if !haskey(rename, vars[v])
                        if var != 0
                            # This statement was tainted by an unavailable variable
                            return widenconst(incT)
                        end
                        @assert var_eq_matching[v] == StructuralTransformations.SelectedState() ||
                                var_eq_matching[v] === unassigned
                        new_row[v_offset] += coeff
                        continue
                    end
                    rename[vars[v]]
                else
                    SSAValue(vars[v].id)
                end
                canonical_type = argextype(canonical_stmt, compact)
                if isa(canonical_type, Incidence)
                    new_row .+= canonical_type.row .* coeff
                else
                    if isa(const_val, Const) && isa(canonical_type, Const)
                        new_const_val = const_val.val + canonical_type.val * coeff
                        if isa(new_const_val, Float64)
                            const_val = Const(new_const_val)
                        else
                            const_val = widenconst(const_val)
                        end
                    else
                        # The replacement has some unknown type - we need to widen
                        # all the way here.
                        return widenconst(const_val)
                    end
                end
            end
            if isa(const_val, Const) && !any(!iszero, rowvals(new_row)) && isempty(incT.eps)
                return const_val
            end
            return Incidence(const_val, new_row, incT.eps)
        end
        remap_incidence(t::PartialStruct, var) = PartialStruct(t.typ, Any[remap_incidence(f, var) for f in t.fields])
        remap_incidence(t::Union{Type, Const, Eq}, var) = t

        function remap_ir!(ir, var)
            for i = 1:length(ir.stmts)
                typ = ir[SSAValue(i)][:type]
                if isa(typ, Incidence) || isa(typ, PartialStruct)
                    ir[SSAValue(i)][:type] = remap_incidence(typ, var)
                end
                ir[SSAValue(i)][:info] = remap_info(ir2->remap_ir!(ir2, var), ir[SSAValue(i)][:info])
            end
        end

        # For everything else, we don't want to schedule the alias until all dependencies
        # have been scheduled.
        function schedule_alias_or_var(v, inst::Instruction; unassigned_is_nothing::Bool = false)
            # This branch covers when we're processing something that we previously wrote out to
            # the compactor; e.g. it avoids double-processing.
            if isa(vars[v], NewSSAValue)
                return vars[v]
            # This branch covers variables that are unused in any equations
            # It should have been deleted, but in some cases that does not happen, so we just insert
            # a dummy value and it should get dropped by future compiler passes (e.g. future inlining).
            elseif isempty(𝑑neighbors(graph, v)) && var_eq_matching[v] isa ModelingToolkit.BipartiteGraphs.Unassigned
                if unassigned_is_nothing
                    return nothing
                else
                    # This variable could have been eliminated entirely in earlier simplification. Just set this to a dummy value.
                    return NewSSAValue(insert_node_here!(compact, NewInstruction(inst; stmt=GlobalRef(DAECompiler.Intrinsics, :_VARIABLE_UNASSIGNED), type=Incidence(Float64))).id)
                end

            # This else branch covers everything else that we might want to reschedule, such as:
            # - `state_ddt()`
            # - `variable()` statements with no aliases
            else
                ii = inst[:inst]
                # Construct new SSA Value to insert into `compact` that invokes `Intrinsics.state_ddt()` or `Intrinsics.variable()`.
                # Note that this is not a legal `invoke()` invocation, it just matches the pattern of
                # (`:invoke`, `MethodInstance`,`variable/state_ddt`) that we will look for later on down the line
                # (see, for example, `dae_finish!()` checking `is_known_invoke(stmt, variable, ir)`)
                # We manually insert `Intrinsics.variable` here, instead of `ii.args[2]`, since we don't
                # want to have to deal with re-naming the reference to the SSAValue that `ii.args[2]`
                # points to.  We already know it refers to a `GlobalRef` of a variable, so we just
                # insert the `Intrinsics.variable` directly.
                @assert var_eq_matching[idnum(inst[:type])] == SelectedState() ||
                        var_eq_matching[idnum(inst[:type])] === unassigned
                if isexpr(ii, :invoke)
                    r = NewSSAValue(insert_node_here!(compact,
                        NewInstruction(inst; stmt=Expr(:invoke, #=mi=#ii.args[1], Intrinsics.variable, ii.args[3:end]...))).id)
                else
                    # Note: I do not understand why if it is a call rather than an invoke we can't just claim it is always a variable
                    # and never a state_ddt. And have to actually tell the truth (which for case in tests is always a state_ddt)
                    # but we do or one of the jacobian tests failed -- by generating too many entries.

                    @assert isexpr(ii, :call)
                    @assert (ii.args[1] == Intrinsics.variable) || (ii.args[1] == Intrinsics.state_ddt)  #not a SSAValue. A function object literal, so it is recognisable
                    r = NewSSAValue(insert_node_here!(compact,
                        NewInstruction(inst; stmt=copy(ii))).id)
                end
                rename[SSAValue(inst.idx)] = SSAValue(r.id)

                # Remove original variable definition
                if isa(vars[v], SSAValue) && vars[v] != SSAValue(0)
                    ir[vars[v]] = r
                end
                vars[v] = r
                return r
            end
        end

        ssa_to_var = Dict(ssa=>var for(var, ssa) in pairs(vars))

        available = BitSet()

        for (var, eq) in pairs(var_eq_matching)
            # Everything that's unassigned or a selected state is considered available
            if !isa(eq, Int)
                push!(available, var)
            end
        end

        for eq in reverse(toporder)
            vs = 𝑠neighbors(graph, eq)
            var = invview(var_eq_matching)[eq]
            if isempty(vs)
                @assert var === unassigned
                # remove empty equations
                eqs[eq][1] == SSAValue(0) && continue
                for ssa in eqs[eq][2]
                    ir[ssa] = nothing
                end
                continue
            end
            if haskey(linear_eqs, eq)
                (ei, diff_level) = linear_eqs[eq]
                eqintrossa = eqs[eq][1]
                idx = eqintrossa.id
                # If we have line info for this equation, use it, otherwise default to `:1`.
                line_entry = idx != 0 ? ir[SSAValue(idx)][:line] : Int32(1)
                function ir_add!(_a, _b)
                    a, b = _a, _b
                    b === nothing && return _a
                    a === nothing && return _b
                    ni = NewInstruction(Expr(:call, +, a, b), Any, line_entry)
                    z = insert_node_here!(compact, ni)
                    compact[z][:flag] |= CC.IR_FLAG_REFINED
                    z
                end
                function var_at_level(var)
                    let vdiff_level = diff_level
                        while vdiff_level > 0
                            var = var_to_diff[var]
                            vdiff_level -= 1
                        end
                    end
                    return var
                end
                coeffs = [var_at_level(v)=>c for (v, c) in pairs(nonzerosmap(@view ils[ei, :])) if !iszero(c)]
                avarcoeff = nothing
                # if length(coeffs) == 1, we implicitly have `coeff * var = 0`, which
                # forces the result to zero (not by the coefficient being zero, but the
                # variable being forced to zero).  We handle this by just setting the
                # new SSA value to `0.0`, as we have implicitly solved this variable to 0.
                if length(coeffs) == 1
                    thisvar, coeff = coeffs[1]
                    @assert thisvar == var
                    avarcoeff = 1.0
                    new_ssa = nothing
                else
                    # We have avarcoeff * avar + ∑ coeff * var = 0.
                    # Rewrite to avar = (∑ (-coeff * var)) / avarcoeff
                    new_ssa = mapfoldl(ir_add!, coeffs) do (thisvar, coeff)
                        if thisvar === var
                            avarcoeff = coeff
                            return nothing
                        end
                        thisvaridx = vars[thisvar]
                        if isa(thisvaridx, NewSSAValue)
                            thisvar = SSAValue(thisvaridx.id)
                        elseif thisvaridx == SSAValue(0)
                            thisvar = schedule_missing_derivative!(thisvar, line_entry)
                        else
                            if !haskey(rename, thisvaridx)
                                inst = ir[thisvaridx]
                                (inst[:inst] === nothing) && return nothing
                                @assert is_known_invoke(inst[:inst], variable, ir) || is_known_invoke_or_call(inst[:inst], state_ddt, ir)
                                @assert idnum(inst[:type]) == thisvar
                                r = schedule_alias_or_var(thisvar, inst)
                                rename[thisvaridx] = thisvar = isa(r, NewSSAValue) ? SSAValue(r.id) : r
                                ir[thisvaridx][:inst] = r
                            else
                                thisvar = rename[thisvaridx]
                            end
                        end
                        if isa(thisvar, Float64)
                            thisvar_type = Const(thisvar)
                        else
                            thisvar_type = compact[thisvar][:type]
                        end
                        ni = NewInstruction(Expr(:call, *, -Float64(coeff), thisvar),
                            Any, line_entry)
                        z = insert_node_here!(compact, ni)
                        compact[z][:flag] |= CC.IR_FLAG_REFINED
                        return z
                    end
                end
                for eqvalssa in eqs[eq][2]
                    ir[eqvalssa] = nothing
                end

                # If this variable was assigned, this means that `var` is the variable we are solving for
                # in this equation, e.g. we had `a + b + c = 0` and we chose `a = -(b + c)` in tearing.
                # Therefore, we can rewrite this equation to just directly compute the value of `a` here.
                # We do this with the `ir_add()` invocations above to do the arithmetic, then emit a
                # `solved_variable()` call here.
                if var !== unassigned
                    @assert avarcoeff !== nothing
                    if avarcoeff != 1.0
                        new_ssa = insert_node_here!(compact,
                            NewInstruction(Expr(:call, /, new_ssa, Float64(avarcoeff)),
                                Any, line_entry))
                        compact[new_ssa][:flag] |= CC.IR_FLAG_REFINED
                    end

                    if vars[var] == SSAValue(0)
                        # If `new_ssa !== nothing`, we have a previous entry to assign into `vars`
                        # otherwise, we insert a raw `0.0` node and use that SSA value.
                        if new_ssa !== nothing
                            vars[var] = NewSSAValue(new_ssa.id)
                        else
                            vars[var] = NewSSAValue(insert_node_here!(compact, NewInstruction(0.0, Const(0.0), line_entry)).id)
                        end
                    elseif !isa(vars[var], NewSSAValue)
                        rename[vars[var]] = new_ssa === nothing ? 0.0 : new_ssa

                        if new_ssa === nothing
                            ir[vars[var]][:inst] = 0.0
                        else
                            ir[vars[var]] = NewSSAValue(new_ssa.id)
                        end
                    end
                    newvarssa = new_ssa === nothing ? 0.0 : SSAValue(new_ssa.id)
                    insert_node_here!(compact, NewInstruction(Expr(:call, solved_variable, var, newvarssa), Nothing, line_entry))
                else
                    # If the variable was not assigned, we have some linear equation that could not be solved for directly.
                    # Examples of this are algebraic loops, and pantelides-generated equations.
                    if eqintrossa.id == 0
                        # If our equation SSA value is `%0`, we must generate the `equation()` call here.
                        eq_ssa = insert_node_here!(compact, NewInstruction(Expr(:call, equation), Eq(eq), line_entry))
                        # This is one of two (other directly below) locations we insert a invoke with no method instances
                        # This can break compiler passes, if we don't removed it before running them
                        insert_node_here!(compact, NewInstruction(Expr(:call, eq_ssa, new_ssa), Nothing, line_entry))
                    else
                        # This is one of two (other directly above) locations we insert a invoke with no method instances
                        # This can break compiler passes, if we don't removed it before running them
                        insert_node!(compact, OldSSAValue(eqintrossa.id),
                            NewInstruction(Expr(:call, OldSSAValue(eqintrossa.id), new_ssa), Nothing, line_entry),
                            #= attach_after =# true)
                    end
                end
            else
                isa(var, Int) || continue

                (eqintrossa, eqcalls) = eqs[eq]
                if eqintrossa == SSAValue(0)
                    @error "Missing code for equation $eq. AD Problem?"
                    continue
                end
                eqline = ir[eqintrossa][:line]

                function ir_add_compact!(_a, _b)
                    _b === nothing && return _a
                    _a === nothing && return _b
                    a = _a
                    b = _b
                    if (a === NewSSAValue(0) || b === NewSSAValue(0))
                        return NewSSAValue(0)
                    end
                    ni = NewInstruction(Expr(:call, +, a, b), Any, eqline)
                    z = insert_node_here!(compact, ni)
                    compact[z][:flag] |= CC.IR_FLAG_REFINED
                    return z
                end

                any_renamed = false
                found_solvable = false
                (varcoeff, varssa) = mapfoldl(lanewise(+, ir_add_compact!), eqcalls) do eqssa
                    eqssaval = _eq_val_arg(ir[eqssa][:inst])
                    inc = ir[eqssaval][:type]
                    if !isa(inc, Incidence)
                        var_eq_matching[var] = unassigned
                        @warn "Previous analysis expected proper incidence for equation $eq's (matched to $var) argument ($eqssaval), but got $inc. Investigate ir_levels[:pre_tearing]. Processing will continue, but model may be incorrect."
                        return ((0., Float64), (NewSSAValue(0), Float64))
                    end

                    # First, collect all statements that contribute to this equation
                    stmts = Set{SSAValue}()
                    worklist = SSAValue[eqssa]
                    idx = vars[var].id

                    function ssa_is_solvable(val)
                        val.id == idx && return true
                        haskey(ssa_to_var, val) || return false
                        xvar = ssa_to_var[val]
                        @assert var != xvar
                        return false
                    end

                    function maybe_add_to_worklist!(v)
                        if !(v in stmts)
                            push!(stmts, v)
                            push!(worklist, v)
                        end
                    end
                    while !isempty(worklist)
                        w = pop!(worklist)
                        stmt = ir[w][:inst]
                        if isa(stmt, PhiNode)
                            if length(stmt.edges) == 1
                            else
                                bbs = dominator_bb_set(domtree, ir, block_for_inst(ir, w.id))
                                for bb in bbs
                                    bb_term = ir[SSAValue(ir.cfg.blocks[bb].stmts[end])][:inst].cond
                                    @assert isa(bb_term, SSAValue)
                                    maybe_add_to_worklist!(bb_term)
                                end
                            end
                        elseif isa(stmt, SSAValue)
                            maybe_add_to_worklist!(stmt)
                            continue
                        end
                        for ur in userefs(stmt)
                            v = ur[]
                            if isa(v, Core.SSAValue)
                                vv = ir[v][:inst]
                                maybe_add_to_worklist!(v)
                            end
                        end
                    end
                    function by(x)
                        x.id <= length(ir.stmts) ? (x.id, length(ir.stmts) + length(ir.new_nodes.info)) :
                            (ir.new_nodes.info[x.id - length(ir.stmts)].pos, x.id)
                    end
                    ssas = sort(collect(stmts); by)
                    rename_local = Dict{SSAValue, SSAValue}()
                    local tainted_by_solved
                    local ssa

                    function rename_ssa(val::SSAValue)
                        if haskey(rename_local, val)
                            return rename_local[val]
                        elseif haskey(rename, val)
                            return rename[val]
                        else
                            record_ir!(state, "error", ir)
                            throw(UnsupportedIRException("rename_ssa received unexpected value ($(val))", ir))
                        end
                    end

                    function update_for_rename(stmt)
                        if isa(stmt, SSAValue)
                            return rename_ssa(stmt)
                        else
                            urs = userefs(stmt)
                            for ur in urs
                                val = ur[]
                                if isa(val, Core.SSAValue)
                                    ur[] = rename_ssa(val)
                                end
                            end
                            return urs[]
                        end
                    end

                    for outer ssa in ssas
                        inst = ir[ssa]
                        stmt = inst[:inst]
                        inst[:info] = remap_info(ir2->remap_ir!(ir2, var), inst[:info])
                        # Determine if this statement is tainted by the variable we're solving for.
                        # If so, we need to copy it, because we'll be re-arranging the code to zero
                        # out that variable. If not, we simply need to move the statement to maintain
                        # SSA.

                        # TODO: This isn't really a good way to solve this problem
                        incT = inst[:type]
                        (tainted_by_solved, tainted_by_unavailable) = type_contains_taint(incT, var, available)
                        if !tainted_by_solved && haskey(rename, ssa)
                            continue
                        end
                        if isa(stmt, Union{Expr, PhiNode})
                            stmt = copy(stmt)
                        end
                        if isa(stmt, GotoNode) || isa(stmt, GotoIfNot)
                            error()
                        end
                        preIncT = incT
                        if ssa_is_solvable(ssa)
                            found_solvable = true
                            stmt = 0.0
                            incT = Const(0.0)
                        elseif is_known_invoke(stmt, variable, ir) || is_known_invoke_or_call(stmt, state_ddt, ir)
                            if tainted_by_unavailable
                                stmt = NewSSAValue(insert_node_here!(compact, NewInstruction(inst; stmt=GlobalRef(DAECompiler.Intrinsics, :_VARIABLE_UNAVAILABLE), type=Incidence(Float64))).id)
                            elseif !tainted_by_solved
                                var_num = idnum(inst[:type])
                                stmt = schedule_alias_or_var(var_num, inst)
                            else
                                stmt = update_for_rename(stmt)
                            end
                            incT = argextype(SSAValue(stmt.id), compact)
                        else
                            stmt = update_for_rename(stmt)
                            if isa(incT, Incidence) || isa(incT, PartialStruct)
                                incT = remap_incidence(incT, var)
                            end
                        end
                        isa(stmt, NewSSAValue) && (stmt = SSAValue(stmt.id))
                        if isa(stmt, PhiNode)
                            if length(stmt.edges) == 1
                                stmt = stmt.values[1]
                            else
                                thisbb = block_for_inst(ir, ssa.id)
                                bbs = dominator_bb_set(domtree, ir, thisbb)
                                # Will be filled in. N.B., we don't use our global trick,
                                # because these NaNs should be selected away by ifelse, or
                                # otherwise never contribute to the result. Otherwise, there's
                                # something wrong with the original IR.
                                bb_ifelse = Any[Expr(:call, Core.ifelse, update_for_rename(ir[SSAValue(ir.cfg.blocks[bb].stmts[end])][:inst].cond), NaN, NaN) for bb in bbs]
                                bb_placeholders = Any[SSAValue(compact.result_idx + i) for i = 0:(length(bbs)-1)]
                                function fill_in_edge(edge, value, last_edge)
                                    while !(edge in bbs)
                                        last_edge = edge
                                        edge = domtree.idoms_bb[edge]
                                    end
                                    edge_term = ir[SSAValue(ir.cfg.blocks[edge].stmts[end])][:inst]
                                    reversed = last_edge == edge_term.dest
                                    bb_ifelse[findfirst(isequal(edge), bbs)].args[reversed ? 4 : 3] = value
                                end
                                for (i, edge) in enumerate(stmt.edges)
                                    if isassigned(stmt.values, i)
                                        fill_in_edge(edge, stmt.values[i], thisbb)
                                    end
                                end
                                for i = 1:length(bbs)-1
                                    fill_in_edge(domtree.idoms_bb[bbs[i]], bb_placeholders[i], bbs[i])
                                end
                                bb_ssas = Any[insert_node_here!(compact, NewInstruction(inst; stmt, type=incT)) for stmt in bb_ifelse]
                                @assert bb_ssas == bb_placeholders
                                stmt = bb_ssas[end]
                            end
                        end
                        r = insert_node_here!(compact, NewInstruction(inst; stmt, type=incT))
                        if tainted_by_solved || tainted_by_unavailable
                            # Make a copy of this in the appropriate place.
                            rename_local[ssa] = r
                        else
                            # Just move the instruction up.
                            # TODO: We should have done some legality analysis at some point that says this is ok.
                            rename[ssa] = r
                        end
                    end
                    eqssaval = _eq_val_arg(ir[eqssa][:inst])
                    if haskey(rename_local, eqssaval)
                        r = rename_local[eqssaval]
                        any_renamed = true
                    else
                        r = rename[eqssaval]
                    end
                    r_type = argextype(SSAValue(r.id), compact)
                    thisvarcoeff = coeff_from_incidence(var, inc.row)
                    if thisvarcoeff === nonlinear
                        var_eq_matching[var] = unassigned
                        @warn "Previous analysis expected equation $eq to be solvable for var $var, but incidence was too imprecise at $eqssaval. This should be fixed. Skipping equation."
                    end
                    return thisvarcoeff, r
                end
                varssa == NewSSAValue(0) && continue

                if !found_solvable
                    throw(UnsupportedIRException(
                        "Equation $eq was supposed to be solved for variable $var, but the variable def was not found.",
                        ir,
                    ))
                end
                if !any_renamed
                    throw(UnsupportedIRException(
                        "Equation $eq was supposed to be solved for variable $var, but the equation argument was not tainted by it.",
                        ir,
                    ))
                end
                if varcoeff === nonlinear
                    var_eq_matching[var] = unassigned
                    continue
                elseif iszero(varcoeff)
                    var_eq_matching[var] = unassigned
                    @warn "Previous analysis expected equation $eq to be solvable for var $var, but the variable is not incident. This should be fixed. Skipping equation."
                    continue
                end

                # All error checks done - delete the equation side effects now
                for eqssa in eqcalls
                    ir[eqssa] = nothing
                end

                if varcoeff != -1
                    varssa = insert_node_here!(compact, NewInstruction(Expr(:call, /, varssa, Float64(-varcoeff)), Any, ir[eqintrossa][:line]))
                    compact[varssa][:flag] |= CC.IR_FLAG_REFINED
                end

                # Update `ir` with new value; `idx == 0` represents equations added by Pantelides
                if idx != 0
                    rename[SSAValue(idx)] = varssa
                    ir[SSAValue(idx)] = NewSSAValue(varssa.id)
                else
                    vars[var] = NewSSAValue(varssa.id)
                end
                insert_node_here!(compact, NewInstruction(Expr(:call, solved_variable, var, varssa), Nothing, ir[eqintrossa][:line]))
            end
            isa(var, Int) && push!(available, var)
        end
        for (ssa, newssa) in rename
            ir[ssa][:inst] = isa(newssa, Float64) ? newssa : NewSSAValue(newssa.id)
        end

        # Finish compaction (iterating over `compact` performs the compacting operation)
        for ((_, idx), stmt) in compact
            # Ensure that no variable statements remain that are aliases, they should all
            # have been moved/transformed by `schedule_alias_or_var()` above.
            if is_known_invoke(stmt, variable, compact)
                varnum = idnum(compact[SSAValue(idx)][:type])
                #TODO: above code seems to do nothing. Did we forget to put validation logic here?
            end

            ssa_idx = SSAValue(idx)
            inst = compact[ssa_idx]
            if isa(inst[:type], Union{Incidence, PartialStruct})
                inst[:type] = remap_incidence(inst[:type], 0)
            end
            inst[:info] = remap_info(ir2->remap_ir!(ir2, 0), inst[:info])
        end

        ir = Core.Compiler.finish(compact)
    end

    record_ir!(state, "", ir)
    rt = infer_ir!(ir, state, getfield(get_sys(state), :mi))
    if rt === Union{}
        display(ir)
        error("After tearing schedule, function unconditionally errors")
    end

    return state.ir = ir
end
