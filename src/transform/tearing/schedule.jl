
using StateSelection.BipartiteGraphs: nsrcs, ndsts, DiCMOBiGraph, invview
using Compiler: userefs, insert_node!, insert_node_here!, NewInstruction, NewSSAValue

struct SolvedVariable
    ssa::Union{SSAValue, Float64}
end

"""
Scan `compact` for all variables and equations.
"""
function find_eqs_vars(graph, compact::IncrementalCompact)
    eqs = Pair{SSAValue, Vector{SSAValue}}[ SSAValue(0) => SSAValue[] for _ in 1:nsrcs(graph)]
    vars = Union{SSAValue, NewSSAValue, SolvedVariable}[SSAValue(0) for _ in 1:ndsts(graph)]
    for ((_, i), stmt) in compact
        if is_known_invoke_or_call(stmt, equation, compact)
            eq = compact[SSAValue(i)][:type].id
            eqs[eq] = SSAValue(i) => eqs[eq][2]
        elseif is_known_invoke_or_call(stmt, variable, compact)
            var = only(rowvals(compact[SSAValue(i)][:type].row)) - 1
            vars[var] = SSAValue(i)
        elseif is_equation_call(stmt, compact)
            push!(eqs[idnum(argextype(_eq_function_arg(stmt), compact))][2], SSAValue(i))
        elseif is_known_invoke_or_call(stmt, solved_variable, compact)
            vars[stmt.args[end-1]] = SolvedVariable(stmt.args[end])
        end
    end
    (eqs, vars)
end

function find_eqs_vars(state)
    compact = IncrementalCompact(copy(state.ir))
    find_eqs_vars(state.structure.graph, compact)
end

function ir_add!(compact, line, _a, _b)
    a, b = _a, _b
    b === nothing && return _a
    a === nothing && return _b
    ni = NewInstruction(Expr(:call, +, a, b), Any, line)
    z = insert_node_here!(compact, ni)
    compact[z][:flag] |= Compiler.IR_FLAG_REFINED
    z
end

function ir_mul_const!(compact, line, coeff::Float64, _a)
    if isone(coeff)
        return _a
    end
    ni = NewInstruction(Expr(:call, *, coeff, _a), Any, line)
    z = insert_node_here!(compact, ni)
    compact[z][:flag] |= Compiler.IR_FLAG_REFINED
    return z
end

Base.IteratorSize(::Type{Compiler.UseRefIterator}) = Base.SizeUnknown()

function schedule_incidence!(compact, var_eq_matching, curval, ::Type, var, line; vars=nothing, schedule_missing_var! = nothing)
    # This just needs the linear part, which is `0` in `Type`
    return (curval, nothing)
end

function schedule_incidence!(compact, var_eq_matching, curval, incT::Const, var, line; vars=nothing, schedule_missing_var! = nothing)
    if curval !== nothing
        return (ir_add!(compact, line, curval, incT.val), nothing)
    end
    return (incT.val, nothing)
end

function schedule_incidence!(compact, var_eq_matching, curval, incT::Incidence, var, line; vars=nothing, schedule_missing_var! = nothing)
    thiscoeff = nothing

    # We do need to materialize the linear parts of the incidence here
    for (lin_var_offset, coeff) in zip(rowvals(incT.row), nonzeros(incT.row))
        lin_var = lin_var_offset - 1

        if lin_var == var
            # This is the value we're solving for
            thiscoeff = -coeff
            continue
        end

        # XXX: what to do with the linear parts that have an unknown coefficient?
        isa(coeff, Float64) || continue

        if lin_var == 0
            lin_var_ssa = insert_node_here!(compact,
                    NewInstruction(
                        Expr(:invoke, nothing, Intrinsics.sim_time),
                        Incidence(0),
                        line))
        else
            if !isassigned(vars, lin_var)
                schedule_missing_var!(lin_var)
            end
            lin_var_ssa = vars[lin_var]
            if lin_var_ssa === 0.0
                continue
            end
        end

        acc = ir_mul_const!(compact, line, coeff, lin_var_ssa)
        curval = curval === nothing ? acc : ir_add!(compact, line, curval, acc)
    end
    (curval, _) = schedule_incidence!(compact, var_eq_matching, curval, incT.typ, var, line; vars, schedule_missing_var!)
    return (curval, thiscoeff)
end

struct RenameOverlayVector <: AbstractVector{Any}
    base::Vector{Any}
    overlay::Vector{Any}
end

function Base.getindex(rno::RenameOverlayVector, i::Int)
    val = rno.overlay[i]
    val !== nothing && return val
    return rno.base[i]
end
Base.setindex!(rno::RenameOverlayVector, @nospecialize(val), i::Int) =
    Base.setindex!(rno.overlay, val, i)
Base.size(rno::RenameOverlayVector) = Base.size(rno.base)

is_state_part_linear(incT::Const, param_vars) = true
is_state_part_linear(incT::Incidence, param_vars) = !(any(==(nonlinear), incT.row) || any(x->((x-1) in param_vars), rowvals(incT.row)))
is_const_plus_state_linear(incT, param_vars) = is_state_part_linear(incT, param_vars) && isa(incT.typ, Const)

is_fully_state_linear(incT, param_vars) = is_const_plus_state_linear(incT, param_vars) && is_fully_state_linear(incT.typ, param_vars)
is_fully_state_linear(incT::Const, param_vars) = iszero(incT.val)

function schedule_nonlinear!(compact, param_vars, var_eq_matching, ir, val::Union{SSAValue, Argument}, ssa_rename::AbstractVector{Any}; vars, schedule_missing_var! = nothing)
    isa(val, Argument) && return nothing

    if isassigned(ssa_rename, val.id)
        return ssa_rename[val.id]
    end

    inst = ir[val]

    if all(x->((x-1) in param_vars), rowvals(inst[:type].row))
        @assert (inst[:stmt]::Expr).args[1] === getfield
        this = insert_node_here!(compact, NewInstruction(inst))
        # No state dependence, we're done
        ssa_rename[val.id] = this
        return this
    end


    stmt = inst[:stmt]
    info = inst[:info]
    incT = inst[:type]::Incidence
    if isa(info, Diffractor.FRuleCallInfo)
        info = info.info
    end
    call_is_linear = false
    if isa(info, MappingInfo)
        result = info.result
        extended_rt = result.extended_rt
    else
        @assert isexpr(stmt, :call)
        f = argextype(stmt.args[1], ir)
        @assert isa(f, Const)
        f = f.val
        @assert f in (Core.Intrinsics.sub_float, Core.Intrinsics.add_float,
                      Core.Intrinsics.mul_float, Core.Intrinsics.copysign_float,
                      Core.ifelse, Core.Intrinsics.or_int, Core.Intrinsics.and_int,
                      Core.Intrinsics.fma_float, Core.Intrinsics.muladd_float)
        # TODO: or_int is linear in Bool
        # TODO: {fma, muladd}_float is linear in one of its arguments
        call_is_linear = f in (Core.Intrinsics.sub_float, Core.Intrinsics.add_float)
    end

    args = map(enumerate(Iterators.drop(userefs(stmt), 1))) do (i, ur)
        arg = ur[]
        typ = argextype(arg, ir)
        if isa(typ, Const)
            return nothing
        end

        # TODO: SICM
        if !is_const_plus_state_linear(typ::Incidence, param_vars)
            this_nonlinear = schedule_nonlinear!(compact, param_vars, var_eq_matching, ir, arg, ssa_rename; vars, schedule_missing_var!)
        else
            if @isdefined(result)
                template_argtype = result.argtypes[i]
                template_v = only(rowvals(template_argtype.row))-1
                if (extended_rt::Incidence).row[template_v+1] !== nonlinear
                    return nothing
                end
            end
            this_nonlinear = nothing
        end

        if call_is_linear
            return this_nonlinear
        end

        return schedule_incidence!(compact, var_eq_matching, this_nonlinear, typ, -1, inst[:line]; vars, schedule_missing_var!)[1]
    end

    if is_const_plus_state_linear(incT, param_vars)
        # TODO: This needs to do a proper template match
        ret = schedule_incidence!(compact, var_eq_matching, nothing, info.result.extended_rt, -1, inst[:line]; vars=
            [arg === nothing ? 0.0 : arg for arg in args[2:end]])[1]
    else
        # TODO: This needs to be the
        new_stmt = copy(stmt)
        for (i, arg) in enumerate(args)
            i += 1
            if arg == nothing
                if isa(new_stmt.args[i], NewSSAValue)
                    new_stmt.args[i] = SSAValue(new_stmt.args[i].id)
                elseif !isa(argextype(new_stmt.args[i], ir), Const)
                    new_stmt.args[i] = 0.0
                end
                continue
            end
            new_stmt.args[i] = arg
        end

        ret = insert_node_here!(compact, NewInstruction(inst; stmt=new_stmt))
    end

    ssa_rename[val.id] = ret
    return ret
end

struct SICMSSAValue
    id::Int
end

function is_fully_ready(info::MappingInfo, available::BitSet)
    callee_vars = BitSet()

    # TODO: this needs to count returned vars as external
    # TODO: We only care about variables used non-linearly
    for i = 1:info.result.nexternalargvars
        caller_map = info.mapping.var_coeffs[i]
        isa(caller_map, Const) && continue
        union!(callee_vars, rowvals(caller_map.row) .- 1)
    end

    all(cv->(cv in available), callee_vars)
end

function compute_eq_schedule(key::TornCacheKey, total_incidence, result, mss::StateSelection.MatchedSystemStructure)
    (; diff_states, alg_states, var_schedule) = key
    (; structure, var_eq_matching) = mss

    eq_orders  = Vector{Union{Int, SSAValue}}[]
    callee_schedules = Dict{SSAValue, Vector{Pair{BitSet, BitSet}}}()

    available = BitSet()
    # Add all selected states to available
    if diff_states !== nothing
        union!(available, diff_states) # Computed by integrator
    end
    union!(available, alg_states)  # Computed by NL solver
    union!(available, key.param_vars)  # Computed by SCIM hoist

    frontier = BitSet()

    isempty(var_schedule) && (var_schedule = Pair{BitSet, BitSet}[available=>BitSet()])
    vargraph = DiCMOBiGraph{true}(structure.graph, var_eq_matching)

    function may_enqueue_frontier_var!(var)
        isa(var_eq_matching[var], Int) || return
        (var in available) && return # Already processed
        (var in frontier) && return # Already queued
        # Check if this neighbor is ready
        if all(inn->(inn in available), inneighbors(vargraph, var))
            push!(frontier, var)
        end
    end

    for sched in var_schedule
        eq_order = Union{Int, SSAValue}[]
        push!(eq_orders, eq_order)

        (in_vars, _) = sched
        union!(available, in_vars)

        for neighbor in 1:ndsts(structure.graph)
            may_enqueue_frontier_var!(neighbor)
        end

        new_available = BitSet()

        function schedule_callee_now!(ssa::SSAValue)
            schedule = get!(callee_schedules, ssa, Vector{Pair{BitSet, BitSet}}())
            callee_info = result.ir[ssa][:info]::MappingInfo

            # Determine which of this callee's external variables we have
            this_callee_in_vars = BitSet()
            for i = 1:callee_info.result.nexternalargvars
                caller_map = callee_info.mapping.var_coeffs[i]
                isa(caller_map, Const) && continue
                caller_vars_for_this_callee_var = rowvals(caller_map.row) .- 1
                if all(x->(x == 0 || x in available || x in key.param_vars), caller_vars_for_this_callee_var) &&
                # If these are all param_vars, the var itself becomes a param var,
                # not an in var.
                !all(in(key.param_vars), caller_vars_for_this_callee_var)
                    push!(this_callee_in_vars, i)
                end
            end

            previously_scheduled_or_ignored = isempty(schedule) ? BitSet() : mapreduce(x->x[2], union, schedule)

            # Schedule all equations that do not depend non-linearily on a variable we do not have
            this_callee_eqs = BitSet()
            found_any = true
            while found_any
                found_any = false
                for i = 1:length(callee_info.result.total_incidence)
                    i in previously_scheduled_or_ignored && continue # We scheduled this previously
                    i in this_callee_eqs && continue # We already scheduled this
                    callee_incidence = callee_info.result.total_incidence[i]
                    incidence = apply_linear_incidence(nothing, callee_incidence, nothing, callee_info.mapping)
                    if is_const_plus_state_linear(incidence, key.param_vars)
                        # No non-linear components - skip it
                        push!(previously_scheduled_or_ignored, i)
                        continue
                    end
                    for (caller_var_offset, coeff) in zip(rowvals(incidence.row), nonzeros(incidence.row))
                        caller_var = caller_var_offset - 1
                        if caller_var == 0 || coeff !== nonlinear
                            continue
                        end
                        if !(caller_var in available) && !(caller_var in new_available)
                            # Can't schedule this yet - we don't have a variable that is used non-linearly
                            @goto outer_continue
                        end
                    end
                    # We have all variables for this incidence, schedule it
                    mapped_eq = callee_info.mapping.eqs[i]
                    assigned_var = invview(var_eq_matching)[mapped_eq]
                    if assigned_var === unassigned
                        push!(previously_scheduled_or_ignored, i)
                        continue
                    elseif isa(assigned_var, Int) && varclassification(result, structure, assigned_var) == CalleeInternal &&
                        eqclassification(result, structure, i) == CalleeInternal
                        # If this is callee internal or unassigned, let the callee handle it
                        push!(new_available, assigned_var)
                        push!(previously_scheduled_or_ignored, i)
                        found_any = true
                        continue
                    end
                    push!(this_callee_eqs, i)
                    @label outer_continue
                end
            end

            @assert isempty(eq_order) || eq_order[end] != ssa
            push!(eq_order, ssa)
            push!(schedule, this_callee_in_vars => this_callee_eqs)
        end

        function schedule_frontier_var!(var; force=false)
            # Find all frontier variables that in a callee that is fully available
            eq = var_eq_matching[var]::Int
            if is_const_plus_state_linear(total_incidence[eq], key.param_vars)
                # This is a linear equation, we can schedule it now
                push!(eq_order, eq)
                push!(new_available, var)
                return
            end

            # Otherwise, we need to compute some non-linear component
            eb = baseeq(result, structure, eq)
            mapping = result.eq_callee_mapping[eb]
            if mapping === nothing
                # Eq is toplevel, can be scheduled now
                push!(eq_order, eq)
                push!(new_available, var)
                return
            end
            @assert eq == eb # TODO

            for (ssa, callee_eq) in mapping
                schedule = get!(callee_schedules, ssa, Vector{Pair{BitSet, BitSet}}())
                callee_info = result.ir[ssa][:info]::MappingInfo

                callee_incidence_part = apply_linear_incidence(nothing, callee_info.result.total_incidence[callee_eq], nothing, callee_info.mapping)
                if is_const_plus_state_linear(callee_incidence_part, key.param_vars)
                    # This portion of the calle is linear, we can schedule it
                    continue
                end

                # Check if this equation was already scheduled
                any(((_,eqs),)->(callee_eq in eqs), schedule) && continue

                # Check if this callee is ready to be fully scheduled
                if !force && !is_fully_ready(callee_info, available)
                    return
                end

                # Schedule (a portion of) this callee now
                schedule_callee_now!(ssa)

                @assert (callee_eq in schedule[end][2] || var in new_available)
            end
            if !(var in new_available)
                push!(new_available, var)
                push!(eq_order, eq)
            end
        end

        while !isempty(frontier)
            for var in frontier
                var in new_available && continue
                schedule_frontier_var!(var)
            end

            if !isempty(new_available)
                union!(available, new_available)
                for var in new_available
                    for neighbor in outneighbors(vargraph, var)
                        may_enqueue_frontier_var!(neighbor)
                    end
                end
                setdiff!(frontier, new_available)
                empty!(new_available)
                continue
            end

            # TODO: We should have a heuristic for which callee to partition here rather
            # than just picking the first one

            schedule_frontier_var!(first(frontier); force=true)
            @assert !isempty(new_available)
            union!(available, new_available)
            for var in new_available
                for neighbor in outneighbors(vargraph, var)
                    may_enqueue_frontier_var!(neighbor)
                end
            end
            setdiff!(frontier, new_available)
            empty!(new_available)
        end
    end

    return (eq_orders, callee_schedules)
end

function invert_eq_callee_mapping(eq_callee_mapping)
    callee_eq_mapping = Dict{SSAValue, Vector{Int}}()

    for (ieq, mapping) in pairs(eq_callee_mapping)
        mapping === nothing && continue
        for val in mapping
            val === nothing && continue
            (eq, callee_eq) = val
            vec = get!(callee_eq_mapping, eq, Vector{Int}())
            n = length(vec)
            if n < callee_eq
                resize!(vec, callee_eq)
                vec[(n+1):callee_eq] .= 0
            end
            vec[callee_eq] = ieq
        end
    end

    return callee_eq_mapping
end

function classify_var(var_to_diff, key::TornCacheKey, var)
    if var in key.alg_states
        vint = invview(var_to_diff)[var]
        if vint in key.diff_states
            kind = AlgebraicDerivative
        else
            kind = Algebraic
        end
    elseif key.diff_states !== nothing && (var in key.diff_states)
        vdiff = var_to_diff[var]
        if vdiff in key.alg_states
            kind = UnassignedDiff
        else
            kind = AssignedDiff
        end
    else
        return nothing
    end
    return kind
end

function assign_slots(state::TransformationState, key::TornCacheKey, var_eq_matching::Union{StateSelection.Matching, Nothing})
    (; result, structure) = state
    slot_assignments = zeros(Int, Int(LastEquationStateKind))
    var_assignment = Vector{Union{Nothing, Pair{StateKind, Int}}}(nothing, length(structure.var_to_diff))
    eq_assignment = var_eq_matching === nothing ? nothing : Vector{Union{Nothing, Pair{EquationStateKind, Int}}}(nothing, length(state.total_incidence))

    function assign_slot!(kind, varnum)
        @assert kind != AlgebraicDerivative # Always the same assignment as the UnassignedDiff of the vint
        @assert kind != StateDiff # Always the same as the assignment as the AssignedDiff of the var
        iseq = kind in (StateDiff, Explicit)
        arr = (iseq ? eq_assignment : var_assignment)
        if arr[varnum] === nothing
            slot_assignments[kind] += 1
            arr[varnum] = kind => slot_assignments[kind]
        end
        (skind, slot) = arr[varnum]
        @assert skind == kind
        return slot
    end

    for i = 1:length(var_assignment)
        varclassification(result, structure, i) == External && continue
        kind = classify_var(state.structure.var_to_diff, key, i)
        kind === nothing && continue
        cache_kind = kind
        varnum = i
        if kind == AlgebraicDerivative
            cache_kind = UnassignedDiff
            varnum = invview(state.structure.var_to_diff)[varnum]::Int
        end
        slot = assign_slot!(cache_kind, varnum)
        if kind == AlgebraicDerivative
            var_assignment[i] = kind => slot
        elseif kind == AssignedDiff && var_eq_matching !== nothing
            eq = var_eq_matching[state.structure.var_to_diff[i]]
            if isa(eq, Int)
                eq_assignment[eq] = StateDiff => slot
            end
        end
    end

    if var_eq_matching !== nothing
        for eq = 1:length(state.total_incidence)
            eqclassification(result, structure, eq) == External && continue
            (invview(var_eq_matching)[eq] === unassigned) || continue
            assign_slot!(Explicit, eq)
        end
    end

    (slot_assignments, var_assignment, eq_assignment)
end

"""
    struct SICMSpec

Cache partition for the state-invariant prologue
"""
struct SICMSpec
    key::TornCacheKey
end

struct TornIRSpec
    key::TornCacheKey
end

function matching_for_key(result::DAEIPOResult, key::TornCacheKey, structure = make_structure_from_ipo(result))
    (; diff_states, alg_states, explicit_eqs, var_schedule) = key

    allow_init_eqs = key.diff_states === nothing

    may_use_var(var) = varclassification(result, structure, var) != External && (diff_states === nothing || !(var in diff_states)) && !(var in alg_states) && varkind(result, structure, var) == Intrinsics.Continuous
    may_use_eq(eq) = !(eq in explicit_eqs) && eqclassification(result, structure, eq) != External && eqkind(result, structure, eq) in (allow_init_eqs ? (Intrinsics.Initial, Intrinsics.Always) : (Intrinsics.Always,))

    # Max match is the (unique) tearing result given the choice of states
    var_eq_matching = StateSelection.complete(StateSelection.maximal_matching(structure.solvable_graph, IPOMatches;
        dstfilter = may_use_var, srcfilter = may_use_eq), nsrcs(structure.solvable_graph))

    if diff_states !== nothing
        for var in diff_states
            var_eq_matching[var] = SelectedState()
        end
    end

    for var in key.param_vars
        var_eq_matching[var] = StateInvariant()
    end

    for (ordinal, (in_vars, out_eqs)) in enumerate(key.var_schedule)
        for in_var in in_vars
            isa(var_eq_matching[in_var], InOut) && continue
            @assert var_eq_matching[in_var] === unassigned
            var_eq_matching[in_var] = InOut(ordinal)
        end

        for out_eq in out_eqs
            invview(var_eq_matching)[out_eq] = InOut(ordinal)
        end
    end

    return var_eq_matching
end

function tearing_schedule!(result::DAEIPOResult, ci::CodeInstance, key::TornCacheKey, world::UInt)
    structure = make_structure_from_ipo(result)
    tstate = TransformationState(result, structure, copy(result.total_incidence))
    return tearing_schedule!(tstate, ci, key, world)
end

function tearing_schedule!(state::TransformationState, ci::CodeInstance, key::TornCacheKey, world::UInt)
    result_ci = find_matching_ci(ci->isa(ci.owner, SICMSpec) && ci.owner.key == key, ci.def, world)
    if result_ci !== nothing
        return result_ci
    end

    (; diff_states, alg_states, var_schedule) = key
    (; result, structure, total_incidence ) = state

    var_eq_matching = matching_for_key(result, key, structure)

    mss = StateSelection.MatchedSystemStructure(structure, var_eq_matching)
    (eq_orders, callee_schedules) = compute_eq_schedule(key, total_incidence, result, mss)

    ir = index_lowering_ad!(state, key)

    # First, schedule any statements that do not have state dependence
    sicm_rename = Vector{Any}(undef, length(ir.stmts))
    nonlin_sicm_rename = Vector{Any}(undef, length(ir.stmts))

    compact = IncrementalCompact(copy(ir))
    line = ir[SSAValue(1)][:line]
    #insert_node_here!(compact, NewInstruction(Expr(:call, println, "Trace: SICM"), Cvoid, line))

    isready(ssa::SSAValue) = isassigned(sicm_rename, ssa.id)
    function isready(inst::Compiler.Instruction)
        for arg in userefs(inst[:stmt])
            isa(arg[], SSAValue) || continue
            isready(arg[]) || return false
        end
        return true
    end

    non_sicm_use_counter = zeros(Int64, length(ir.stmts))

    function maybe_schedule_sicm_inst!(inst)
        isa(inst[:type], Eq) && return false
        if isready(inst) && (inst[:flag] & Compiler.IR_FLAG_UNUSED) == 0 && !has_dependence_other_than(inst[:type], key.param_vars)
            stmt = inst[:stmt]
            if isexpr(stmt, :invoke)
                info = inst[:info]
                if isa(info, MappingInfo) && length(info.result.total_incidence) != 0
                    return false
                end
            end
            isa(stmt, Expr) && (stmt = copy(stmt))
            isa(stmt, ReturnNode) && return false
            urs = userefs(stmt)
            for ur in urs
                isa(ur[], SSAValue) || continue
                ur[] = sicm_rename[ur[].id]
            end

            sicm_rename[inst.idx] = insert_node_here!(compact, NewInstruction(inst; stmt=urs[]))
            return true
        end
        return false
    end

    slot_assignments = zeros(Int, Int(LastEquationStateKind))

    callee_eq_mapping = invert_eq_callee_mapping(result.eq_callee_mapping)

    diff_states_in_callee = BitSet()
    processed_variables = Set{Int}()

    # Generate SICM partition
    for i = 1:length(ir.stmts)
        inst = ir[SSAValue(i)]
        # TODO: This ignores all control flow
        if !maybe_schedule_sicm_inst!(inst)
            stmt = inst[:stmt]
            # Return value with state-independent part or contained equations with state-invariant non-linear component
            if is_equation_call(stmt, ir)
                #=
                eqnum = idnum(argextype(_eq_function_arg(stmt), compact))
                val = _eq_val_arg(stmt)

                display(ir)
                nonlin = schedule_nonlinear!(compact, key.param_vars, nothing, ir, val, RenameOverlayVector(sicm_rename, nonlin_sicm_rename); vars=param_arg_rename)
                val = schedule_incidence!(compact, nothing, nonlin, argextype(val, ir), -1, inst[:line]; vars=param_arg_rename)[1]
                =#
                #ir_add!(compact, inst[:line], eq_resid.args[eqnum+1], val)
                inst[:type] = Any # Make sure this shows up in rename
            elseif is_known_invoke(stmt, equation, ir)
                # Retain this in both the SICM (to be able to make additional vars parameters) and
                # the RHS
                sicm_rename[inst.idx] = insert_node_here!(compact, NewInstruction(inst))
            elseif is_known_invoke(stmt, variable, ir)
                # Receive the variable number based on the incidence -- we know it will be there
                varnum = idnum(ir.stmts.type[i])

                # Ensure that we only process each variable once
                if varnum ∈ processed_variables
                    record_ir!(state, "error", ir)
                    throw(UnsupportedIRException("Duplicate variable ($(varnum))", ir))
                end
                push!(processed_variables, varnum)
            elseif isexpr(stmt, :invoke)
                # Project the state mapping
                callee_diff_vars = BitSet()
                callee_alg_vars = BitSet()
                callee_param_vars = BitSet()
                callee_explicit_eqs = BitSet()
                externally_solved_vars = BitSet()

                info = inst[:info]
                isa(info, MappingInfo) || continue

                callee_result = (info::MappingInfo).result

                type = inst[:type]
                if isa(type, Incidence)
                    if length(callee_result.total_incidence) == 0
                        continue
                    end
                end

                for callee_var = 1:length(callee_result.var_to_diff)
                    caller_map = info.mapping.var_coeffs[callee_var]
                    if isa(caller_map, Const)
                        push!(callee_param_vars, callee_var)
                    else
                        if varclassification(callee_result, callee_var) == External
                            if all(x->(x in key.param_vars), rowvals(caller_map.row) .- 1)
                                push!(callee_param_vars, callee_var)
                            else
                                push!(externally_solved_vars, callee_var)
                            end
                            continue
                        end
                        caller_var = only(rowvals(caller_map.row))-1
                        if caller_var in diff_states
                            push!(callee_diff_vars, callee_var)
                        elseif caller_var in alg_states
                            push!(callee_alg_vars, callee_var)
                        elseif caller_var in key.param_vars
                            push!(callee_param_vars, callee_var)
                        end
                        eq = var_eq_matching[caller_var]
                        if isa(eq, Int) && result.eqclassification[eq] == Owned
                            push!(externally_solved_vars, callee_var)
                        end
                    end
                end

                for (callee_eq, caller_eq) in enumerate(info.mapping.eqs)
                    if caller_eq in key.explicit_eqs
                        push!(callee_explicit_eqs, callee_eq)
                    end
                end

                callee_final_available = externally_solved_vars
                if !haskey(callee_schedules, SSAValue(i))
                    # This call may not be needed
                    if isempty(callee_diff_vars) && isempty(callee_alg_vars) && length(callee_result.total_incidence) == 0
                        continue
                    end
                    callee_schedules[SSAValue(i)] = Pair{BitSet,BitSet}[callee_final_available => BitSet()]
                    push!(eq_orders[end], SSAValue(i))
                else
                    final_sched_avail = callee_schedules[SSAValue(i)][end][1]
                    if !all(x->(x in final_sched_avail || x in callee_param_vars), callee_final_available)
                        push!(callee_schedules[SSAValue(i)], callee_final_available => BitSet())
                        push!(eq_orders[end], SSAValue(i))
                    end
                end

                callee_var_schedule = callee_schedules[SSAValue(i)]
                callee_key = TornCacheKey(callee_diff_vars, callee_alg_vars, callee_param_vars, callee_explicit_eqs, callee_var_schedule)

                callee_codeinst = stmt.args[1]
                if isa(callee_codeinst, MethodInstance)
                    callee_codeinst = Compiler.get(Compiler.code_cache(interp), callee_codeinst, nothing)
                end
                callee_result = structural_analysis!(callee_codeinst, world)
                callee_sicm_ci = tearing_schedule!(callee_result, callee_codeinst, callee_key, world)

                inst[:type] = Any
                inst[:flag] = UInt32(0)
                new_stmt = copy(stmt)

                stmt.args[1] = (stmt.args[1], callee_key)
                resize!(stmt.args, 1)

                if !isdefined(callee_sicm_ci, :rettype_const)
                    new_stmt.args[1] = callee_sicm_ci

                    urs = userefs(new_stmt)
                    for ur in urs
                        isa(ur[], SSAValue) || continue
                        if !isassigned(sicm_rename, ur[].id)
                            ur[] = 0.
                            continue
                        end
                        ur[] = sicm_rename[ur[].id]
                    end

                    sstate = insert_node_here!(compact, NewInstruction(inst; stmt=new_stmt, type=Tuple, flag=UInt32(0)))
                    push!(stmt.args, SICMSSAValue(sstate.id))
                else
                    push!(stmt.args, callee_sicm_ci.rettype_const)
                end
            elseif stmt === nothing || isa(stmt, ReturnNode)
                continue
            elseif isexpr(stmt, :call) || isexpr(stmt, :new) || isa(stmt, GotoNode)
                # TODO: Pull this up, if arguments are state-independent
                continue
            else
                @show stmt
                error()
            end
        else
            ir[SSAValue(i)][:stmt] = SICMSSAValue(sicm_rename[i].id)
        end
    end

    # Now make sure to schedule all unassigned equations that were not needed
    # to satisfy any variable/out requests.
    # TODO: This schedules them in the very last partition - would it be better
    # to schedule them as soon as they're available?
    scheduled_eqs = BitSet()
    for eq_order in eq_orders
        for eq in eq_order
            isa(eq, SSAValue) && continue
            push!(scheduled_eqs, eq::Int)
        end
    end

    for (_, out_eqs) in key.var_schedule
        union!(scheduled_eqs, out_eqs)
    end

    for eq in 1:nsrcs(structure.graph)
        if !(eq in scheduled_eqs) &&
                result.eq_callee_mapping[baseeq(result, structure, eq)] === nothing &&
                eqclassification(result, structure, eq) != External &&
                eqkind(result, structure, eq) == Intrinsics.Always &&
                !is_const_plus_state_linear(total_incidence[eq], key.param_vars)
            push!(eq_orders[end], eq)
        end
    end

    resid = Expr(:call, tuple)
    sicm_resid_rename = Vector{Any}(undef, compact.result_idx)

    compact1 = IncrementalCompact(ir)
    foreach(_->nothing, compact1)
    rename_hack = copy(compact1.ssa_rename)
    ir = Compiler.finish(compact1)

    # Rewrite SICM to state references
    line = ir[SSAValue(1)][:line]
    resid_recv = Argument(1)
    for i = 1:length(ir.stmts)
        inst = ir[SSAValue(i)]
        stmt = inst[:stmt]
        @assert !isa(stmt, SICMSSAValue)
        urs = userefs(stmt)
        for ur in urs
            isa(ur[], SICMSSAValue) || continue
            if !isassigned(sicm_resid_rename, ur[].id)
                push!(resid.args, SSAValue(ur[].id))
                sicm_resid_rename[ur[].id] = insert_node!(ir, SSAValue(1),
                    NewInstruction(Expr(:call, getfield, resid_recv, length(resid.args)-1), Incidence(Any), line))
            end
            ur[] = sicm_resid_rename[ur[].id]
        end
        inst[:stmt] = urs[]
    end

    # Normalize by explicitly inserting implicit equations for return
    if result.nimplicitoutpairs > 0
        lastssa = SSAValue(length(ir.stmts))
        ret = ir[lastssa]
        @assert isa(ret[:stmt], ReturnNode)
        # TODO: Handle structures
        #=
        @assert widenconst(argextype(ret[:stmt].val, ir)) === Float64
        implicitvar = length(result.var_to_diff)
        @assert var_eq_matching[implicitvar] == length(result.total_incidence)
        ssavar = insert_node!(ir, lastssa, NewInstruction(ret; stmt=Expr(:call, solved_variable, implicitvar, ret[:stmt].val), type=Nothing))
        =#
        ret[:stmt] = ReturnNode(nothing)
    end

    # Now schedule the state-invariant non-linear part of all contained equations
    compact1 = IncrementalCompact(ir)
    (eqs, vars) = find_eqs_vars(structure.graph, compact1)
    eqs = convert(Vector{Pair{SSAValue, Vector{Union{SSAValue, NewSSAValue}}}}, eqs)
    rename_hack2 = copy(compact1.ssa_rename)

    ir = Compiler.finish(compact1)

    # Done with state-independent
    if compact.result_idx == 1
        ir_sicm = nothing
    else
        if length(resid.args) != 1
            resid_ssa = insert_node_here!(compact, NewInstruction(resid, Tuple, line))
        else
            resid_ssa = ()
        end

#=
        if length(eq_resid.args) != 1
            eq_resid_ssa = insert_node_here!(compact, NewInstruction(eq_resid, Tuple, line))
        else
            eq_resid_ssa = ()
        end
        ret_ssa = insert_node_here!(compact, NewInstruction(Expr(:call, tuple, resid_ssa, eq_resid_ssa), Tuple{Tuple, Tuple}, line))
=#

        insert_node_here!(compact, NewInstruction(ReturnNode(resid_ssa), Union{}, line))
        ir_sicm = Compiler.finish(compact)
    end

    var_sols = Vector{Any}(undef, length(structure.var_to_diff))

    for var in key.param_vars
        var_sols[var] = 0.0
    end

    ssa_rename = Vector{Any}(undef, length(result.ir.stmts))

    function schedule_missing_var!(lin_var)
        if lin_var in key.param_vars
            error()
        else
            @assert lin_var > result.nexternalargvars
            var_sols[lin_var] = insert_node_here!(compact1,
                NewInstruction(
                    Expr(:invoke, nothing, Intrinsics.variable),
                    Incidence(lin_var),
                    line))
        end
    end

    function insert_solved_var_here!(compact1, var, curval, line)
        if result.varclassification[basevar(result, structure, var)] != Owned
            return
        end
        insert_node_here!(compact1, NewInstruction(Expr(:call, solved_variable, var, curval), Nothing, line))
    end

    isempty(var_schedule) && (var_schedule = Pair{BitSet, BitSet}[BitSet()=>BitSet()])

    callee_ordinals = Dict{SSAValue, Int}()

    irs = IRCode[]
    for (ordinal, (eq_order, sched)) in enumerate(zip(eq_orders, var_schedule))
        # Schedule internal var-eq pairs
        nir = copy(ir)
        empty!(nir.argtypes)
        push!(nir.argtypes, Tuple)
        push!(nir.argtypes, Tuple)
        compact1 = IncrementalCompact(nir)
        line = nir[SSAValue(1)][:line]

        #insert_node_here!(compact1, NewInstruction(Expr(:call, println, "Trace: Ordinal $ordinal"), Cvoid, line))

        (in_vars, out_eqs) = sched
        for (idx, var) in enumerate(in_vars)
            var_sols[var] = insert_node_here!(compact1,
                NewInstruction(Expr(:call, getfield, Argument(2), idx), Any, line))
            insert_solved_var_here!(compact1, var, var_sols[var], line)
        end

        for eq in eq_order
            if isa(eq, SSAValue)
                eqinst = ir[rename_hack2[rename_hack[eq.id].id]]
                @assert isexpr(eqinst[:stmt], :invoke)

                info = eqinst[:info]::MappingInfo
                callee_result = info.result

                callee_ordinal = get!(callee_ordinals, eq, 1)
                callee_sched = callee_schedules[eq][callee_ordinal]
                (callee_in_vars, callee_out_eqs) = callee_sched

                in_vars = Expr(:call, tuple)
                for var in callee_in_vars
                    (argval, _) = schedule_incidence!(compact1,
                        var_eq_matching, nothing, info.mapping.var_coeffs[var], -1, line; vars=var_sols, schedule_missing_var!)

                    push!(in_vars.args, argval)
                end

                in_vars_ssa = insert_node_here!(compact1, NewInstruction(eqinst; stmt=in_vars, type=Tuple))

                new_stmt = copy(eqinst[:stmt])
                spec = new_stmt.args[1]
                @assert isa(spec, NTuple{2, Any})
                new_stmt.args[1] = (spec..., callee_ordinal)
                push!(new_stmt.args, in_vars_ssa)

                if isa(new_stmt.args[2], SSAValue)
                    new_stmt.args[2] = insert_node_here!(compact1, NewInstruction(eqinst; stmt=ir[new_stmt.args[2]][:stmt], type=Tuple))
                end

                urs = userefs(new_stmt)
                for ur in urs
                    isa(ur[], NewSSAValue) || continue
                    ur[] = SSAValue(ur[].id)
                end

                callee_ordinals[eq] = callee_ordinal+1

                this_call = insert_node_here!(compact1, NewInstruction(eqinst; stmt=urs[]))

                for (idx, this_callee_eq) in enumerate(callee_out_eqs)
                    this_eq = callee_eq_mapping[eq][this_callee_eq]
                    incT = state.total_incidence[this_eq]
                    var = invview(var_eq_matching)[this_eq]
                    curval = insert_node_here!(compact1, NewInstruction(eqinst; stmt=Expr(:call, getfield, this_call, idx), type=Any))
                    push!(eqs[this_eq][2], NewSSAValue(curval.id))
                end
            else
                var = invview(var_eq_matching)[eq]

                incT = state.total_incidence[eq]
                anynonlinear = !is_const_plus_state_linear(incT, key.param_vars)
                nonlinearssa = nothing
                if anynonlinear
                    if isa(var, Int) && isa(vars[var], SolvedVariable)
                        nonlinearssa = schedule_nonlinear!(compact1, key.param_vars, var_eq_matching, ir, vars[var].ssa, ssa_rename; vars=var_sols, schedule_missing_var!)
                    else
                        for eqcallssa in eqs[eq][2]
                            if !isa(eqcallssa, NewSSAValue)
                                inst = ir[eqcallssa]
                                this_nonlinearssa = schedule_nonlinear!(compact1, key.param_vars, var_eq_matching, ir, inst[:stmt].args[end], ssa_rename; vars=var_sols, schedule_missing_var!)
                                line = ir[eqcallssa][:line]
                            else
                                # From getfield from a callee
                                this_nonlinearssa = SSAValue(eqcallssa.id)
                                line = compact1[eqcallssa][:line]
                            end
                            nonlinearssa = nonlinearssa === nothing ? this_nonlinearssa : ir_add!(compact1, line, this_nonlinearssa, nonlinearssa)
                        end
                    end
                    @assert nonlinearssa !== nothing
                end

                eqssa = eqs[eq][1]
                line = ir[eqssa == SSAValue(0) ? SSAValue(1) : eqssa][:line]

                if isa(var, Int)
                    curval = nonlinearssa
                    (curval, thiscoeff) = schedule_incidence!(compact1, var_eq_matching, curval, incT, var, line; vars=var_sols, schedule_missing_var!)
                    @assert thiscoeff != nonlinear
                    curval = ir_mul_const!(compact1, line, 1/thiscoeff, curval)
                    var_sols[var] = curval
                    insert_solved_var_here!(compact1, var, curval, line)
                else
                    curval = nonlinearssa
                    (curval, thiscoeff) = schedule_incidence!(compact1, var_eq_matching, curval, incT, -1, line; vars=var_sols, schedule_missing_var!)
                    insert_node_here!(compact1, NewInstruction(Expr(:call, InternalIntrinsics.contribution!, eq, Explicit, curval), Nothing, line))
                end
            end
        end

        # Schedule non-linear part of equations that are returned
        eq_resids = Expr(:call, tuple)
        for eq in out_eqs
            var = invview(var_eq_matching)[eq]
            if isa(var, Int) && isa(vars[var], SolvedVariable)
                nonlinearssa = schedule_nonlinear!(compact1, key.param_vars, var_eq_matching, ir, vars[var].ssa, ssa_rename; vars=var_sols, schedule_missing_var!)
            else
                if isempty(eqs[eq][2])
                    nonlinearssa = nothing
                else
                    eqcallssa = only(eqs[eq][2])
                    if isa(eqcallssa, NewSSAValue)
                        nonlinearssa = SSAValue(eqcallssa.id)
                    else
                        nonlinearssa = schedule_nonlinear!(compact1, key.param_vars, var_eq_matching, ir, ir[eqcallssa][:stmt].args[3], ssa_rename; vars=var_sols, schedule_missing_var!)
                    end
                end
            end
            push!(eq_resids.args, nonlinearssa === nothing ? 0.0 : nonlinearssa)
        end

        eq_resid_ssa = isempty(out_eqs) ? () :
            insert_node_here!(compact1, NewInstruction(eq_resids, Tuple,
                ir[SSAValue(length(ir.stmts))][:line]))

        insert_node_here!(compact1, NewInstruction(ReturnNode(eq_resid_ssa), Union{},
            ir[SSAValue(length(ir.stmts))][:line]))

        this_ir = Compiler.finish(compact1)
        push!(irs, this_ir)
    end

    if ir_sicm === nothing
        src = nothing
        sig = Tuple
        debuginfo = Core.DebugInfo(:sicm)
    else
        widen_extra_info!(ir_sicm)
        src = ir_to_src(ir_sicm)
        sig = Tuple{map(Compiler.widenconst, ir_sicm.argtypes)...}
        debuginfo = src.debuginfo
    end

    sicm_ci = cache_dae_ci!(ci, src, debuginfo, sig, SICMSpec(key))
    ccall(:jl_add_codeinst_to_jit, Cvoid, (Any, Any), sicm_ci, src)

    torn_ci = cache_dae_ci!(ci, TornIR(ir_sicm, irs), nothing, sig, TornIRSpec(key))

    return sicm_ci
end
