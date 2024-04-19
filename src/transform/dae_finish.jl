"""
    dae_finish!()

Converts scheduled IR into IR sutiable to be passed to either DAESolver or an ODESolver
with singular mass-matrix support.
"""
@breadcrumb "ir_levels" function dae_finish!(
    state, var_eq_matching, isdae = true;
    allow_unassigned=false, mass_matrix_eltype=Float64
)
    (; graph, var_to_diff) = state.structure
    debug_config = DebugConfig(state)
    p_type = parameter_type(get_sys(state))
    
    # Read in from the last level before any DAE or ODE-specific `ir_levels`
    # We assume this is named `tearing_schedule!`
    ir = compact!(copy(state.ir))
    empty!(ir.argtypes)
    push!(ir.argtypes, Tuple{})  #OpaqueClosure
    push!(ir.argtypes, AbstractVector{<:Real})  # out
    isdae && push!(ir.argtypes, AbstractVector{<:Real})  # du,
    push!(ir.argtypes, AbstractVector{<:Real})  # u,
    push!(ir.argtypes, p_type)  # p
    push!(ir.argtypes, Real)  #  t

    record_ir!(state, "compact!", ir)

    # Rewrite this into a form suitable for passing to a DAE Solver, i.e.
    # f(out, du, u, p, t)
    # Name the arguments for ease of reference
    if isdae
        out,du,u,p,t = Argument.(2:6)
    else
        du = nothing  # no du for ODE
        out,u,p,t = Argument.(2:5)
    end

    var_assignment, eq_assignment, differential_vars, var_num, neqs, dummy_map = assign_vars_and_eqs(
        MatchedSystemStructure(state.structure, var_eq_matching), isdae)

    mmI = Int64[]
    mmJ = Int64[]
    mmV = mass_matrix_eltype[]

    for i = 1:var_num
        push!(mmI, i); push!(mmJ, i); push!(mmV, one(mass_matrix_eltype))
    end

    assigned_slots = falses(neqs)

    processed_variables = Set{Int}()
    residual_eq_ssa = Union{Nothing, Vector{SSAValue}}[nothing for i = 1:neqs]
    for i = 1:length(ir.stmts)
        stmt = ir.stmts[i][:inst]
        info = ir[SSAValue(i)][:info]

        if isa(info, CC.ConstCallInfo) && any(result->isa(result, CC.SemiConcreteResult), info.results)
            # Drop any semi-concrete results from the DAE-interpreter. We will redo
            # them with the native interpreter to avoid getting suboptimal codegen.
            ir[SSAValue(i)][:info] = info.call
            ir[SSAValue(i)][:flag] |= CC.IR_FLAG_REFINED
        end

        if is_known_invoke(stmt, variable, ir) || is_known_invoke_or_call(stmt, state_ddt, ir)
            # Receive the variable number based on the incidence -- we know it will be there
            varnum = idnum(ir.stmts.type[i])

            # Ensure that we only process each variable once
            if varnum ∈ processed_variables
                record_ir!(state, "error", ir)
                throw(UnsupportedIRException("Duplicate variable ($(varnum))", ir))
            end
            push!(processed_variables, varnum)

            (slot, in_du) = var_assignment[varnum]
            if slot == 0
                if allow_unassigned
                    # Statements like 0.0 * a drop incidence on `a`, but may
                    # not be legal to drop in the compiler. We insert a GlobalRef
                    # here that marks this and that can be varied to verify that
                    # the final result indeed does not depend on this variable.
                    ir.stmts[i][:inst] = GlobalRef(DAECompiler.Intrinsics, :_VARIABLE_UNASSIGNED)
                else
                    @assert false "Tried to emit IR for unused slot"
                end
            else
                replace_call!(ir, SSAValue(i), Expr(:call, Base.getindex, in_du ? du : u, slot))
            end
        elseif is_equation_call(stmt, ir)
            eq = idnum(argextype(_eq_function_arg(stmt), ir))
            red = _eq_val_arg(stmt)
            slot = eq_assignment[eq]
            if slot == 0
                record_ir!(state, "error", ir)
                throw(UnsupportedIRException(
                    "Expected equation ($eq) to have been solved and removed from the system at %$i",
                    ir))
            end
            if residual_eq_ssa[slot] === nothing
                residual_eq_ssa[slot] = SSAValue[]
            end
            push!(residual_eq_ssa[slot], red)
            ir[SSAValue(i)][:inst] = nothing
        elseif is_known_invoke(stmt, equation, ir)
            # Equation - used, but only as an arg to equation call, which will all get
            # eliminated by the end of this loop, so we can delete this statement, as
            # long as we don't touch the type yet.
            ir[SSAValue(i)][:inst] = Intrinsics.placeholder_equation
        elseif is_solved_variable(stmt)
            var = stmt.args[end-1]
            vint = invview(var_to_diff)[var]
            if vint !== nothing && var_eq_matching[vint] == SelectedState()
                (slotnum, b) = var_assignment[vint]
                @assert b == false
                if isdae
                    differential_vars[slotnum] = true
                    dustate = insert_node!(ir, i, NewInstruction(Expr(:call, Base.getindex,
                        du, slotnum), Any))
                    sub = insert_node!(ir, i, NewInstruction(Expr(:call, -, stmt.args[end], dustate), Any))
                    # By convention, the equation slot for the implicit equation matches the
                    # variable slot for the integrated variable.
                    eqnum = slotnum
                    new_call = Expr(:call, Base.setindex!, out, sub, eqnum)
                else
                    new_call = Expr(:call, Base.setindex!, out, stmt.args[end], slotnum)
                end
                replace_call!(ir, SSAValue(i), new_call)
                @assert !assigned_slots[slotnum]
                assigned_slots[slotnum] = true
                ir.stmts.type[i] = Any
            else
                # Solved algebric variable, not used in this lowering
                ir[SSAValue(i)][:inst] = nothing
            end
        else
            replace_if_intrinsic!(ir, SSAValue(i), du, u, p, t, var_assignment)
        end
    end

    # Just before the end of the function
    idx = length(ir.stmts)
    function ir_add!(a, b)
        ni = NewInstruction(Expr(:call, +, a, b), Any, ir[SSAValue(idx)][:line])
        insert_node!(ir, idx, ni)
    end

    # Generate sum of residual equations
    # TODO: We could just sum the output array in place (but needs to be benchmarked
    # and initialized)
    for (slot, ssas) in enumerate(residual_eq_ssa)
        ssas === nothing && continue
        red = foldl(ir_add!, ssas)
        ni = NewInstruction(Expr(:call, Base.setindex!, out, red, slot), Any, ir[SSAValue(idx)][:line])
        insert_node!(ir, idx, ni)
        @assert !assigned_slots[slot]
        assigned_slots[slot] = true
    end

    # Generate implicit equations between selected states
    for v = 1:ndsts(graph)
        vdiff = var_to_diff[v]
        vdiff === nothing && continue

        if var_eq_matching[v] !== SelectedState() || var_eq_matching[vdiff] !== SelectedState()
            # Solved variables were already handled above
            continue
        end
        if isdae
            (vslot, vb) = var_assignment[v]
            (vdiffslot, vbdiff) = var_assignment[vdiff]
            differential_vars[vdiffslot] = true
            @assert vb == vbdiff == false
            # By convention, the equation slot for the implicit equation matches the
            # variable slot for the integrated variable.
            eqnum = vslot
            dustate = insert_node!(ir, idx, NewInstruction(
                Expr(:call, Base.getindex, du, vslot),
                Any))
            ustate = insert_node!(ir, idx, NewInstruction(
                Expr(:call, Base.getindex, u, vdiffslot),
                Any))
            sub = insert_node!(ir, idx, NewInstruction(
                Expr(:call, -, ustate, dustate),
                Any))
            insert_node!(ir, idx, NewInstruction(
                Expr(:call, Base.setindex!, out, sub, eqnum),
                Any))
        else
            (vslot, vb) = var_assignment[v]
            (vdiffslot, vbdiff) = var_assignment[vdiff]
            dustate = insert_node!(ir, idx, NewInstruction(
                Expr(:call, Base.getindex, u, vdiffslot),
                Any))
            insert_node!(ir, idx, NewInstruction(
                Expr(:call, Base.setindex!, out, dustate, vslot),
                Any))
        end
        @assert !assigned_slots[vslot]
        assigned_slots[vslot] = true
    end

    if !isempty(dummy_map)
        @assert !isdae
        for (diff, var) in dummy_map
            varref = insert_node!(ir, idx, NewInstruction(Expr(:call, Base.getindex, u, var), Any))
            insert_node!(ir, idx, NewInstruction(Expr(:call, Base.setindex!, out, varref, diff), Any))
            @assert !assigned_slots[diff]
            assigned_slots[diff] = true
        end
    end

    if !all(assigned_slots)
        error("Slot assignment inconsistency")
    end

    ir = compact!(ir)
    ir = store_args_for_replay!(ir, debug_config, "RHS")
    widen_extra_info!(ir)

    goldclass_sig = if isdae
        Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, p_type, Float64}
    else
        Tuple{Vector{Float64}, Vector{Float64}, p_type, Float64}
    end

    F! = JITOpaqueClosure{:RHS, goldclass_sig}() do arg_types...
        # TODO: Turning off compilesig_invokes is not great here, because it might give us non-legal :invokes.
        opt_params = OptimizationParams(compilesig_invokes=false, preserve_local_sources=true)
        compile_overload(ir, state, arg_types; opt_params)
    end

    if isdae
        return (; F!, differential_vars, neqs, var_assignment, eq_assignment)
    else
        mass_matrix = sparse(mmI, mmJ, mmV, neqs, neqs)
        return (; F!, neqs, var_assignment, eq_assignment, dummy_map, mass_matrix)
    end
end
