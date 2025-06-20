const VectorViewType = SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}
const VectorIntViewType = SubArray{Int, 1, Vector{Int}, Tuple{UnitRange{Int}}, true}

"""
    struct RHSSpec

Cache partition for the RHS
"""
struct RHSSpec
    key::TornCacheKey
    ordinal::Int
end

function Base.StackTraces.show_custom_spec_sig(io::IO, owner::RHSSpec, linfo::CodeInstance, frame::Base.StackTraces.StackFrame)
    print(io, "RHS Partition #$(owner.ordinal) for ")
    mi = Base.get_ci_mi(linfo)
    return Base.StackTraces.show_spec_sig(io, mi.def, mi.specTypes)
end

function handle_contribution!(ir::Compiler.IRCode, settings::Settings, source, inst::Compiler.Instruction, kind, slot, arg_range, red)
    pos = SSAValue(inst.idx)
    @assert Int(LastStateKind) < Int(kind) <= Int(LastEquationStateKind)
    which = Argument(arg_range[Int(kind)])
    prev = insert_node!(ir, pos, NewInstruction(inst; stmt=Expr(:call, Base.getindex, which, slot), type=Float64))
    sum = insert_node!(ir, pos, NewInstruction(inst; stmt=Expr(:call, +, prev, red), type=Float64))
    replace_call!(ir, pos, Expr(:call, Base.setindex!, which, sum, slot), settings, source)
end

function compute_slot_ranges(caller_state::TransformationState, info::MappingInfo, callee_key, var_assignment, eq_assignment)
    # Compute the ranges for this child's states in the parent range.
    # We rely upon earlier stages of the pipeline having put these adjacent to each other
    # and in order. We could just trust that, but because it's a little bit tricky, here
    # we go through explicitly and check.
    state_ranges = UnitRange{Int}[0:-1 for _ in 1:Int(LastEquationStateKind)]

    function assign_slot!(kind, slotidx)
        currange = state_ranges[kind]
        if first(currange) == 0
            state_ranges[kind] = slotidx:slotidx
        else
            @assert last(currange) == slotidx-1
            state_ranges[kind] = first(currange):slotidx
        end
    end

    for callee_var = 1:length(info.result.var_to_diff)
        varclassification(info.result, callee_var) == External && continue
        caller_map = info.mapping.var_coeffs[callee_var]
        isa(caller_map, Const) && continue
        caller_var = only(rowvals(caller_map.row))-1
        varkind(caller_state, caller_var) == Intrinsics.Continuous || continue

        callee_kind = classify_var(info.result.var_to_diff, callee_key, callee_var)
        callee_kind === nothing && continue
        (kind, slotidx) = var_assignment[caller_var]
        @assert callee_kind == kind

        assign_slot!(kind, slotidx)
        if kind == AssignedDiff
            assign_slot!(StateDiff, slotidx)
        end
    end

    for (callee_eq, eq) in enumerate(info.mapping.eqs)
        eqa = eq_assignment[eq]
        eqa === nothing && continue
        (kind, slotidx) = eqa
        kind == Explicit || continue

        assign_slot!(kind, slotidx)
    end

    @assert state_ranges[StateDiff] == state_ranges[AssignedDiff]
    #@assert state_ranges[Explicit] == state_ranges[Algebraic]

    return state_ranges
end

function rhs_finish!(result::DAEIPOResult, ci::CodeInstance, key::TornCacheKey, world::UInt, settings::Settings, ordinal::Int, indexT=Int)
    structure = make_structure_from_ipo(result)
    tstate = TransformationState(result, structure, copy(result.total_incidence))
    return rhs_finish!(tstate, ci, key, world, settings, ordinal, indexT)
end

function rhs_finish!(
    state::TransformationState,
    ci::CodeInstance,
    key::TornCacheKey,
    world::UInt,
    settings::Settings,
    ordinal::Int,
    indexT=Int)

    @assert !settings.skip_optimizations
    (; result, structure) = state
    result_ci = find_matching_ci(ci->isa(ci.inferred, RHSSpec) && ci.inferred.key == key && ci.inferred.ordinal == ordinal, ci.def, world)
    if result_ci !== nothing
        return result_ci
    end

    allow_unassigned = false

    var_eq_matching = matching_for_key(state, key)
    (slot_assignments, var_assignment, eq_assignment) = assign_slots(state, key, var_eq_matching)

    torn_ci = find_matching_ci(ci->isa(ci.owner, TornIRSpec) && ci.owner.key == key, ci.def, world)
    torn = torn_ci.inferred
    rhs_ms = nothing
    old_daef_mi = nothing
    assigned_slots = falses(length(result.total_incidence))

    cis = Vector{CodeInstance}()
    for (ir_ordinal, ir) in enumerate(torn.ir_seq)
        # Read in from the last level before any DAE or ODE-specific `ir_levels`
        # We assume this is named `tearing_schedule!`
        ir = copy(ir)
        empty!(ir.argtypes)
        push!(ir.argtypes, Tuple)  # SICM State
        push!(ir.argtypes, Tuple)  # in vars
        if in(settings.mode, (ODE, ODENoInit))
            slotnames = [:sicm_state, :vars, :in_u_mm, :in_u_unassgn, :in_alg_derv, :in_alg, :out_du_mm, :out_eq, :t]
        elseif in(settings.mode, (DAE, DAENoInit))
            slotnames = [:sicm_state, :vars, :in_u_mm, :in_u_unassgn, :in_du_unassgn, :in_alg, :out_du_mm, :out_eq, :t]
        else
            slotnames = nothing # XXX: define slotnames for `InitUncompress`
        end

        arg_range = 3:8
        @assert length(arg_range) == Int(LastEquationStateKind)
        for arg in arg_range
            push!(ir.argtypes, VectorViewType)
        end

        t = Argument(last(arg_range)+1)
        push!(ir.argtypes, Float64)  #  t

        processed_variables = BitSet()

        diff_states_in_callee = BitSet()

        for i = 1:length(ir.stmts)
            inst = ir[SSAValue(i)]
            stmt = inst[:stmt]
            info = inst[:info]

            if isa(info, Compiler.ConstCallInfo) && any(result->isa(result, Compiler.SemiConcreteResult), info.results)
                # Drop any semi-concrete results from the DAE-interpreter. We will redo
                # them with the native interpreter to avoid getting suboptimal codegen.
                ir[SSAValue(i)][:info] = info.call
                ir[SSAValue(i)][:flag] |= Compiler.IR_FLAG_REFINED
            end

            if isexpr(stmt, :invoke) && isa(stmt.args[1], Tuple)
                info::MappingInfo
                callee_ci = stmt.args[1][1]
                closure_env = stmt.args[2]
                in_vars = stmt.args[3]
                if isa(callee_ci, MethodInstance)
                    callee_ci = Compiler.get(Compiler.code_cache(interp), callee_ci, nothing)
                end

                @assert callee_ci !== nothing

                spec_data = stmt.args[1]
                callee_key = spec_data[2]
                callee_ordinal = spec_data[end]::Int
                callee_result = structural_analysis!(callee_ci, world, settings)
                callee_daef_ci = rhs_finish!(callee_result, callee_ci, callee_key, world, settings, callee_ordinal)
                # Allocate a continuous block of variables for all callee alg and diff states

                empty!(stmt.args)
                push!(stmt.args, callee_daef_ci)
                push!(stmt.args, closure_env)
                push!(stmt.args, in_vars)

                # Ordering from tearing is (AssignedDiff, UnassignedDiff, Algebraic, Explicit)
                slot_ranges = compute_slot_ranges(state, info, callee_key, var_assignment, eq_assignment)
                for (arg, range) in zip(arg_range, slot_ranges)
                    push!(stmt.args, insert_node!(ir, SSAValue(i),
                        NewInstruction(inst;
                        stmt=Expr(:call, view, Argument(arg), range),
                        type=VectorViewType)))
                end

                # TODO: Track whether the system is autonomous
                push!(stmt.args, t)
            end

            if is_equation_call(stmt, ir)
                display(ir)
                error()
            elseif is_known_invoke(stmt, variable, ir)
                varnum = idnum(ir.stmts.type[i])
                kind = varkind(state, varnum)
                if kind == Intrinsics.Epsilon
                    replace_call!(ir, SSAValue(i), 0., settings, @__SOURCE__)
                    continue
                end
                @assert kind == Intrinsics.Continuous

                assgn = var_assignment[varnum]
                if assgn == nothing
                    display(StateSelection.MatchedSystemStructure(result, structure, var_eq_matching))
                    @sshow varnum
                    error("Variable left over in IR that doesn't have an assignment")
                end
                (kind, slot) = assgn
                @assert 1 <= Int(kind) <= Int(LastStateKind)
                which = Argument(arg_range[Int(kind)])
                replace_call!(ir, SSAValue(i), Expr(:call, Base.getindex, which, slot), settings, @__SOURCE__)
                inst[:type] = Float64
            elseif is_known_invoke_or_call(stmt, InternalIntrinsics.contribution!, ir)
                eq = stmt.args[end-2]::Int
                kind = stmt.args[end-1]::EquationStateKind
                (eqkind, slot) = eq_assignment[eq]::Pair
                @assert eqkind == kind
                red = stmt.args[end]
                handle_contribution!(ir, settings, @__SOURCE__(), inst, kind, slot, arg_range, red)
            elseif is_known_invoke(stmt, equation, ir)
                # Equation - used, but only as an arg to equation call, which will all get
                # eliminated by the end of this loop, so we can delete this statement, as
                # long as we don't touch the type yet.
                ir[SSAValue(i)][:inst] = Intrinsics.placeholder_equation
            elseif is_solved_variable(stmt)
                var = stmt.args[end-1]
                vint = invview(structure.var_to_diff)[var]
                if vint !== nothing && key.diff_states !== nothing && (vint in key.diff_states) && !(var in diff_states_in_callee)
                    handle_contribution!(ir, settings, @__SOURCE__(), inst, StateDiff, var_assignment[vint][2], arg_range, stmt.args[end])
                else
                    ir[SSAValue(i)] = nothing
                end
            else
                replace_if_intrinsic!(ir, settings, SSAValue(i), nothing, nothing, Argument(1), t, var_assignment)
            end
        end

        # Just before the end of the function
        spec = RHSSpec(key, ir_ordinal)
        daef_ci = rhs_ir_finish!(ir, ci, settings, spec, slotnames)
        push!(cis, daef_ci)
    end

    return cis[ordinal]
end

function rhs_ir_finish!(ir::IRCode, ci::CodeInstance, settings::Settings, spec::RHSSpec, slotnames)
    ir = Compiler.compact!(ir)
    resize!(ir.cfg.blocks, 1)
    empty!(ir.cfg.blocks[1].succs)

    widen_extra_info!(ir)
    Compiler.verify_ir(ir)
    src = ir_to_src(ir, settings; slotnames)

    abi = Tuple{ir.argtypes...}
    daef_ci = cache_dae_ci!(ci, src, src.debuginfo, abi, spec)
    ccall(:jl_add_codeinst_to_jit, Cvoid, (Any, Any), daef_ci, src)

    return daef_ci
end
