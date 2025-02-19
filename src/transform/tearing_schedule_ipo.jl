
function ir_add!(compact, line, _a, _b)
    a, b = _a, _b
    b === nothing && return _a
    a === nothing && return _b
    ni = NewInstruction(Expr(:call, +, a, b), Any, line)
    z = insert_node_here!(compact, ni)
    compact[z][:flag] |= CC.IR_FLAG_REFINED
    z
end

function ir_mul_const!(compact, line, coeff::Float64, _a)
    if isone(coeff)
        return _a
    end
    ni = NewInstruction(Expr(:call, *, coeff, _a), Any, line)
    z = insert_node_here!(compact, ni)
    compact[z][:flag] |= CC.IR_FLAG_REFINED
    return z
end

Base.IteratorSize(::Type{CC.UseRefIterator}) = Base.SizeUnknown()

struct StateInvariant; end
StateSelection.BipartiteGraphs.overview_label(::Type{StateInvariant}) = ('P', "State Invariant / Parameter", :red)

struct InOut
    ordinal::Int
end
StateSelection.BipartiteGraphs.overview_label(::Type{InOut}) = ('#', "IPO in var / out eq", :green)
StateSelection.BipartiteGraphs.overview_label(io::InOut) = (string(io.ordinal), "IPO in var / out eq", :green)

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

        coeff == nonlinear && continue

        if !isassigned(vars, lin_var)
            schedule_missing_var!(lin_var)
        end
        lin_var_ssa = vars[lin_var]
        if lin_var_ssa === 0.0
            continue
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

is_fully_state_linear(incT, param_vars) = !(any(==(nonlinear), incT.row) || any(x->((x-1) in param_vars), rowvals(incT.row)) || incT.typ != Const(0.0))
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


    info = inst[:info]
    if isa(info, Diffractor.FRuleCallInfo)
        info = info.info
    end
    @show inst[:stmt]
    info::MappingInfo
    result = info.result
    extended_rt = result.extended_rt
    incT = inst[:type]::Incidence

    stmt = inst[:stmt]

    @show result.argtypes
    args = map(zip(Iterators.drop(userefs(stmt), 1), result.argtypes)) do (ur, template_argtype)
        arg = ur[]
        typ = argextype(arg, ir)
        if isa(typ, Const)
            return nothing
        end

        # TODO: SICM

        @show is_fully_state_linear(typ::Incidence, param_vars)
        if !is_fully_state_linear(typ::Incidence, param_vars)
            this_nonlinear = schedule_nonlinear!(compact, param_vars, var_eq_matching, ir, arg, ssa_rename; vars, schedule_missing_var!)
        else
            template_v = only(rowvals(template_argtype.row))-1
            if (extended_rt::Incidence).row[template_v+1] !== nonlinear
                return nothing
            end
            this_nonlinear = nothing
        end
        @show (arg, typ, this_nonlinear)

        return schedule_incidence!(compact, var_eq_matching, this_nonlinear, typ, -1, inst[:line]; vars, schedule_missing_var!)[1]
    end
    @show args

    if is_fully_state_linear(incT, param_vars)
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

function compute_eq_schedule(key::TornCacheKey, result, mss::MatchedSystemStructure)
    (; diff_states, alg_states, var_schedule) = key
    (; structure, var_eq_matching) = mss

    eq_orders  = Vector{Union{Int, SSAValue}}[]
    callee_schedules = Dict{SSAValue, Vector{Pair{BitSet, BitSet}}}()

    available = BitSet()
    # Add all selected states to available
    union!(available, diff_states) # Computed by integrator
    union!(available, alg_states)  # Computed by NL solver
    union!(available, key.param_vars)  # Computed by SCIM hoist

    frontier = BitSet()

    isempty(var_schedule) && (var_schedule = Pair{BitSet, BitSet}[available=>BitSet()])

    for sched in var_schedule
        eq_order = Union{Int, SSAValue}[]
        push!(eq_orders, eq_order)
        vargraph = DiCMOBiGraph{true}(structure.graph, var_eq_matching)

        (in_vars, _) = sched
        union!(available, in_vars)

        for neighbor in 1:ndsts(structure.graph)
            isa(var_eq_matching[neighbor], Int) || continue
            (neighbor in available) && continue # Already processed
            (neighbor in frontier) && continue # Already queued
            # Check if this neighbor is ready
            if all(inn->(inn in available), inneighbors(vargraph, neighbor))
                push!(frontier, neighbor)
            end
        end

        new_available = BitSet()
        function schedule_frontier_var!(var; force=false)
            # Find all frontier variables that in a callee that is fully available
            eq = var_eq_matching[var]::Int
            mapping = result.eq_callee_mapping[eq]
            if mapping === nothing
                # Eq is toplevel, can be scheduled now
                push!(eq_order, eq)
                push!(new_available, var)
                return
            end

            # Check if this callee has all its external vars available
            all_callee_vars_available = all(mapping) do (ssa, callee_eq)
                info = result.ir[ssa][:info]::MappingInfo

                callee_vars = BitSet()

                # TODO: this needs to count returned vars as external
                # TODO: We only care about variables used non-linearly
                for i = 1:info.result.nexternalvars
                    caller_map = info.mapping.var_coeffs[i]
                    isa(caller_map, Const) && continue
                    union!(callee_vars, rowvals(caller_map.row) .- 1)
                end

                all(cv->(cv in available), callee_vars)
            end

            if force || all_callee_vars_available
                # If so, schedule it now
                push!(new_available, var)
                for (ssa, callee_eq) in mapping
                    push!(eq_order, ssa)
                    info = result.ir[ssa][:info]::MappingInfo

                    schedule = get!(callee_schedules, ssa, Vector{Pair{BitSet, BitSet}}())

                    this_callee_in_vars = BitSet()
                    this_callee_eqs = BitSet()

                    if result.eq_kind[eq] == Owned || result.var_kind[var] == Owned
                        push!(this_callee_eqs, callee_eq)
                    end

                    for i = 1:info.result.nexternalvars
                        caller_map = info.mapping.var_coeffs[i]
                        isa(caller_map, Const) && continue
                        caller_vars_for_this_callee_var = rowvals(caller_map.row) .- 1
                        if all(in(available), caller_vars_for_this_callee_var) &&
                        # If these are all param_vars, the var itself becomes a param var,
                        # not an in var.
                        !all(in(key.param_vars), caller_vars_for_this_callee_var)
                            push!(this_callee_in_vars, i)
                        end
                    end

                    push!(schedule, this_callee_in_vars=>this_callee_eqs)

                    # Newly available are all variables reachable from here that do not leave the callee
                    worklist = Int[var]
                    while !isempty(worklist)
                        work = pop!(worklist)
                        for neighbor in outneighbors(vargraph, work)
                            neighbor_eq_mapping = result.eq_callee_mapping[var_eq_matching[neighbor]]
                            if neighbor_eq_mapping[1] == ssa
                                if !(neighbor in new_available)
                                    error() # TODO: I think this needs to check inneighbors
                                    push!(this_callee_eqs, neighbor_eq_mapping[2])
                                    push!(worklist, neighbor)
                                end
                            else
                                push!(frontier, neighbor)
                            end
                        end
                    end
                end

                if result.eq_kind[eq] == Owned || result.var_kind[var] == Owned
                    push!(eq_order, eq)
                end
            end
        end

        while !isempty(frontier)
            for var in frontier
                schedule_frontier_var!(var)
            end

            if !isempty(new_available)
                union!(available, new_available)
                setdiff!(frontier, new_available)
                empty!(new_available)
                continue
            end

            if length(frontier) > 1
                # We need to make a codegen choice about which callee to partition
                error()
            end

            schedule_frontier_var!(only(frontier); force=true)
            @show new_available
            @assert !isempty(new_available)
            union!(available, new_available)
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
    elseif var in key.diff_states
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

function tearing_schedule!(interp, ci::CodeInstance, key::TornCacheKey)
    @show key
    result = CC.traverse_analysis_results(ci) do @nospecialize result
        return result isa Union{DAEIPOResult, UncompilableIPOResult} ? result : nothing
    end

    if haskey(result.sicm_cache, key)
        return result.sicm_cache[key]
    end

    (; diff_states, alg_states, var_schedule) = key
    structure = make_structure_from_ipo(result)
    complete!(structure)

    # Perform state selection for this sub-problem
    varwhitelist = StateSelection.computed_highest_diff_variables(structure)

    # Max match is the (unique) tearing result given the choice of states
    var_eq_matching = complete(maximal_matching(structure.graph, Union{Unassigned, SelectedState, StateInvariant, InOut};
        dstfilter = var->(var > result.nexternalvars && varwhitelist[var] && !(var in diff_states) && !(var in alg_states)),
        srcfilter = eq -> result.eq_kind[eq] != External), nsrcs(structure.graph))

    for var in diff_states
        var_eq_matching[var] = SelectedState()
    end

    for var in key.param_vars
        var_eq_matching[var] = StateInvariant()
    end

    for (ordinal, (in_vars, out_eqs)) in enumerate(key.var_schedule)
        @show (ordinal, (in_vars, out_eqs))
        for in_var in in_vars
            isa(var_eq_matching[in_var], InOut) && continue
            var_eq_matching[in_var] = InOut(ordinal)
        end

        for out_eq in out_eqs
            invview(var_eq_matching)[out_eq] = InOut(ordinal)
        end
    end

    mss = MatchedSystemStructure(structure, var_eq_matching)
    (eq_orders, callee_schedules) = compute_eq_schedule(key, result, mss)
    display(mss)

    # TODO: This should be the post-AD IR
    ir = copy(result.ir)

    display(ir)

    # First, schedule any statements that do not have state dependence
    sicm_rename = Vector{Any}(undef, length(ir.stmts))
    nonlin_sicm_rename = Vector{Any}(undef, length(ir.stmts))

    compact = IncrementalCompact(copy(ir))
    line = ir[SSAValue(1)][:line]
    #insert_node_here!(compact, NewInstruction(Expr(:call, println, "Trace: SICM"), Cvoid, line))

    isready(ssa::SSAValue) = isassigned(sicm_rename, ssa.id)
    function isready(inst::Instruction)
        for arg in userefs(inst[:stmt])
            isa(arg[], SSAValue) || continue
            isready(arg[]) || return false
        end
        return true
    end

    non_sicm_use_counter = zeros(Int64, length(ir.stmts))

    function maybe_schedule_sicm_inst!(inst)
        isa(inst[:type], Eq) && return false
        if isready(inst) && (inst[:flag] & CC.IR_FLAG_UNUSED) == 0 && !has_dependence_other_than(inst[:type], key.param_vars)
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

    slot_assignments = zeros(Int, Int(LastEquationKind))

    callee_eq_mapping = invert_eq_callee_mapping(result.eq_callee_mapping)

    var_assignment = Vector{Union{Nothing, Pair{StateKind, Int}}}(nothing, length(result.var_to_diff))
    eq_assignment = Vector{Union{Nothing, Pair{EquationKind, Int}}}(nothing, length(result.total_incidence))
    diff_states_in_callee = BitSet()
    processed_variables = Set{Int}()

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

    function get_var_slot_assignment(varnum::Int)
        kind = classify_var(result.var_to_diff, key, varnum)
        kind === nothing && return nothing

        cache_kind = kind
        if kind == AlgebraicDerivative
            cache_kind = UnassignedDiff
            varnum = invview(result.var_to_diff)[varnum]::Int
        end

        slot = assign_slot!(cache_kind, varnum)
        return (slot, kind)
    end

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
            elseif is_known_invoke(stmt, variable, ir) || is_known_invoke_or_call(stmt, state_ddt, ir)
                # Receive the variable number based on the incidence -- we know it will be there
                varnum = idnum(ir.stmts.type[i])

                # Ensure that we only process each variable once
                if varnum ∈ processed_variables
                    record_ir!(state, "error", ir)
                    throw(UnsupportedIRException("Duplicate variable ($(varnum))", ir))
                end
                push!(processed_variables, varnum)

                assgn = get_var_slot_assignment(varnum)

                if assgn !== nothing
                    resize!(stmt.args, 4)
                    stmt.args[1] = nothing
                    stmt.args[2] = Intrinsics.state
                    stmt.args[3] = assgn[1]
                    stmt.args[4] = assgn[2]
                end
            elseif isexpr(stmt, :invoke)
                # Project the state mapping
                callee_diff_states = BitSet()
                callee_alg_states = BitSet()
                callee_param_vars = BitSet()
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
                        caller_var = only(rowvals(caller_map.row))-1
                        if caller_var in diff_states && callee_var > callee_result.nexternalvars
                            push!(callee_diff_states, callee_var)
                        elseif caller_var in alg_states && callee_var > callee_result.nexternalvars
                            push!(callee_alg_states, callee_var)
                        elseif caller_var in key.param_vars
                            push!(callee_param_vars, callee_var)
                        end
                        eq = var_eq_matching[caller_var]
                        if isa(eq, Int) && result.eq_kind[eq] == Owned
                            push!(externally_solved_vars, callee_var)
                        end
                    end
                end

                callee_final_available = union!(BitSet(1:callee_result.nexternalvars), externally_solved_vars)
                if !haskey(callee_schedules, SSAValue(i))
                    # This call may not be needed
                    @show callee_final_available
                    if isempty(callee_diff_states) && isempty(callee_alg_states) && length(callee_result.total_incidence) == 0
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
                callee_key = TornCacheKey(callee_diff_states, callee_alg_states, callee_param_vars, callee_var_schedule)

                # Allocate state/equation indices
                slot_starts = copy(slot_assignments) .+ 1

                # States
                for callee_var = 1:length(info.result.var_to_diff)
                    caller_map = info.mapping.var_coeffs[callee_var]
                    isa(caller_map, Const) && continue
                    caller_var = only(rowvals(caller_map.row))-1

                    kind = classify_var(info.result.var_to_diff, callee_key, callee_var)
                    kind === nothing && continue
                    if kind == AlgebraicDerivative
                        assign_slot!(UnassignedDiff, invview(result.var_to_diff)[caller_var])
                    else
                        assign_slot!(kind, caller_var)
                    end
                    callee_var in callee_key.diff_states && push!(diff_states_in_callee, caller_var)
                end

                # Explicit equations
                if haskey(callee_eq_mapping, SSAValue(i))
                    for eq in callee_eq_mapping[SSAValue(i)]
                        if invview(var_eq_matching)[eq] === unassigned
                            assign_slot!(Explicit, eq)
                        end
                    end
                end

                callee_codeinst = stmt.args[1]
                if isa(callee_codeinst, MethodInstance)
                    callee_codeinst = CC.get(CC.code_cache(interp), callee_codeinst, nothing)
                end
                callee_sicm_ci = tearing_schedule!(interp, callee_codeinst, callee_key)

                inst[:type] = Any
                inst[:flag] = UInt32(0)
                new_stmt = copy(stmt)

                stmt.args[1] = (stmt.args[1], callee_key,
                    (slot_starts[i]:slot_assignments[i] for i in
                        (AssignedDiff, UnassignedDiff, Algebraic, Explicit))...)
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

                    state = insert_node_here!(compact, NewInstruction(inst; stmt=new_stmt, type=Tuple, flag=UInt32(0)))
                    push!(stmt.args, SICMSSAValue(state.id))
                else
                    push!(stmt.args, callee_sicm_ci.rettype_const)
                end
            elseif stmt === nothing || isa(stmt, ReturnNode)
                continue
            elseif isexpr(stmt, :call)
                urs = userefs(stmt)
                for ur in urs
                    isa(ur[], SSAValue) || continue
                    if !isassigned(sicm_rename, ur[].id)
                        ur[] = 0.
                        continue
                    end
                    ur[] = sicm_rename[ur[].id]
                end
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
        if !(eq in scheduled_eqs) && result.eq_callee_mapping[eq] === nothing && result.eq_kind[eq] != External
            push!(eq_orders[end], eq)
        end
    end

    resid = Expr(:call, tuple)
    sicm_resid_rename = Vector{Any}(undef, compact.result_idx)

    compact1 = IncrementalCompact(ir)
    foreach(_->nothing, compact1)
    rename_hack = copy(compact1.ssa_rename)
    ir = finish(compact1)

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

    ir = finish(compact1)

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
        ir_sicm = finish(compact)
    end

    var_sols = Vector{Any}(undef, length(result.var_to_diff))

    for var in key.param_vars
        var_sols[var] = 0.0
    end

    ssa_rename = Vector{Any}(undef, length(result.ir.stmts))

    function schedule_missing_var!(lin_var)
        if lin_var in key.param_vars
            error()
        else
            @assert lin_var > result.nexternalvars
            assgn = get_var_slot_assignment(lin_var)::Tuple
            var_sols[lin_var] = insert_node_here!(compact1,
                NewInstruction(
                    Expr(:invoke, nothing, Intrinsics.state, assgn...),
                    Incidence(lin_var),
                    line))
        end
    end

    function insert_solved_var_here!(compact1, var, curval, line)
        if result.var_kind[var] != Owned
            return
        end
        insert_node_here!(compact1, NewInstruction(Expr(:call, solved_variable, var, curval), Nothing, line))
        vint = invview(result.var_to_diff)[var]
        if vint !== nothing && (vint in key.diff_states) && !(var in diff_states_in_callee)
            (eqnum, _) = get_var_slot_assignment(vint)
            insert_node_here!(compact1, NewInstruction(Expr(:call, contribution, eqnum, StateDiff, curval), Nothing, line))
        end
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
                @assert isa(spec, NTuple{6, Any})
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
                    incT = result.total_incidence[this_eq]
                    var = invview(var_eq_matching)[this_eq]
                    curval = insert_node_here!(compact1, NewInstruction(eqinst; stmt=Expr(:call, getfield, this_call, idx), type=Any))
                    push!(eqs[this_eq][2], NewSSAValue(curval.id))
                end
            else
                var = invview(var_eq_matching)[eq]

                incT = result.total_incidence[eq]
                anynonlinear = !is_fully_state_linear(incT, key.param_vars)
                nonlinearssa = nothing
                if anynonlinear
                    if isa(var, Int) && isa(vars[var], SolvedVariable)
                        nonlinearssa = schedule_nonlinear!(compact1, key.param_vars, var_eq_matching, ir, vars[var].ssa, ssa_rename; vars=var_sols, schedule_missing_var!)
                    else
                        for eqcallssa in eqs[eq][2]
                            if !isa(eqcallssa, NewSSAValue)
                                inst = ir[eqcallssa]
                                this_nonlinearssa = schedule_nonlinear!(compact1, key.param_vars, var_eq_matching, ir, inst[:stmt].args[3], ssa_rename; vars=var_sols, schedule_missing_var!)
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
                    curval = ir_mul_const!(compact1, line, 1/thiscoeff, curval)
                    var_sols[var] = curval
                    insert_solved_var_here!(compact1, var, curval, line)
                else
                    curval = nonlinearssa
                    (curval, thiscoeff) = schedule_incidence!(compact1, var_eq_matching, curval, incT, -1, line; vars=var_sols, schedule_missing_var!)
                    insert_node_here!(compact1, NewInstruction(Expr(:call, Intrinsics.contribution, eq, Explicit, curval), Nothing, line))
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
                    nonlinearssa = schedule_nonlinear!(compact1, key.param_vars, var_eq_matching, ir, ir[eqcallssa][:stmt].args[3], ssa_rename; vars=var_sols, schedule_missing_var!)
                end
            end
            push!(eq_resids.args, nonlinearssa === nothing ? 0.0 : nonlinearssa)
        end

        eq_resid_ssa = isempty(out_eqs) ? () :
            insert_node_here!(compact1, NewInstruction(eq_resids, Tuple,
                ir[SSAValue(length(ir.stmts))][:line]))

        insert_node_here!(compact1, NewInstruction(ReturnNode(eq_resid_ssa), Union{},
            ir[SSAValue(length(ir.stmts))][:line]))

        push!(irs, finish(compact1))
    end

    if ir_sicm === nothing
        src = nothing
        sig = Tuple
        debuginfo = Core.DebugInfo(:sicm)
    else
        widen_extra_info!(ir_sicm)
        src = ir_to_src(ir_sicm)
        sig = Tuple{map(CC.widenconst, ir_sicm.argtypes)...}
        debuginfo = src.debuginfo
    end

    sicm_ci = cache_dae_ci!(interp, ci, src, debuginfo, sig, SICMSpec(key))

    result.sicm_cache[key] = sicm_ci
    result.tearing_cache[key] = TornIR(ir_sicm, irs)

    return sicm_ci
end
