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

function ir_to_cache(ir::IRCode)
    isva = false
    slotnames = nothing
    ir = CC.copy(ir)
    # if the user didn't specify a definition MethodInstance or filename Symbol to use for the debuginfo, set a filename now
    ir.debuginfo.def === nothing && (ir.debuginfo.def = :var"generated IR for OpaqueClosure")
    nargtypes = length(ir.argtypes)
    nargs = nargtypes-1
    sig = Base.Experimental.compute_oc_signature(ir, nargs, isva)
    rt = Base.Experimental.compute_ir_rettype(ir)
    src = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
    if slotnames === nothing
        src.slotnames = fill(:none, nargtypes)
    else
        length(slotnames) == nargtypes || error("mismatched `argtypes` and `slotnames`")
        src.slotnames = slotnames
    end
    src.slotflags = fill(zero(UInt8), nargtypes)
    src.slottypes = copy(ir.argtypes)
    src = CC.ir_to_codeinf!(src, ir)
    return src
end

const VectorViewType = SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}

function cache_dae_ci!(old_ci, src, debuginfo, owner)
    daef_ci = CC.engine_reserve(old_ci.def, owner)
    ccall(:jl_fill_codeinst,  Cvoid, (Any, Any, Any, Any, Int32, UInt, UInt, UInt32, Any, Any, Any),
        daef_ci, Tuple{}, Union{}, nothing, Int32(0),
        UInt(1)#=ci.min_world=#, old_ci.max_world,
        old_ci.ipo_purity_bits, nothing, nothing, CC.empty_edges)
    ccall(:jl_update_codeinst, Cvoid, (Any, Any, Int32, UInt, UInt, UInt32, Any, UInt8, Any, Any),
        daef_ci, src, Int32(0), UInt(1)#=ci.min_world=#, old_ci.max_world, old_ci.ipo_purity_bits,
        nothing, 0x0, debuginfo, CC.empty_edges)
    ccall(:jl_mi_cache_insert, Cvoid, (Any, Any), old_ci.def, daef_ci)
    return daef_ci
end

function dae_finish_ipo!(
        interp,
        ci::CodeInstance,
        key::TornCacheKey,
        ordinal::Int,
        indexT=Int)

    result = CC.traverse_analysis_results(ci) do @nospecialize result
        return result isa Union{DAEIPOResult, UncompilableIPOResult} ? result : nothing
    end

    if haskey(result.dae_finish_cache, key)
        ms = result.dae_finish_cache[key]
        while !isa(ms.data, RHSSpec) || ms.data.ordinal != ordinal
            ms = ms.next
        end
        return ms
    end

    allow_unassigned = false

    torn = result.tearing_cache[key]
    rhs_ms = nothing
    old_daef_mi = nothing
    assigned_slots = falses(length(result.total_incidence))

    cis = Vector{CodeInstance}()
    for (ir_ordinal, ir) in enumerate(torn.ir_seq)
        ir = torn.ir_seq[ir_ordinal]

        # Read in from the last level before any DAE or ODE-specific `ir_levels`
        # We assume this is named `tearing_schedule!`
        ir = copy(ir)
        empty!(ir.argtypes)
        push!(ir.argtypes, Tuple)  # SICM State
        push!(ir.argtypes, Tuple)  # in vars

        arg_range = 3:8
        out_du_mm, out_eq, in_u_mm, in_u_unassgn, in_du_unassgn, in_alg =
            Argument.(arg_range)
        for arg in arg_range
            push!(ir.argtypes, VectorViewType)
        end

        t = Argument(last(arg_range)+1)
        push!(ir.argtypes, Float64)  #  t


        processed_variables = BitSet()
        var_assignment = Vector{Union{Nothing, Int}}(nothing, length(result.var_to_diff))

        diff_states_in_callee = BitSet()

        for i = 1:length(ir.stmts)
            inst = ir[SSAValue(i)]
            stmt = inst[:stmt]
            info = inst[:info]

            if isa(info, CC.ConstCallInfo) && any(result->isa(result, CC.SemiConcreteResult), info.results)
                # Drop any semi-concrete results from the DAE-interpreter. We will redo
                # them with the native interpreter to avoid getting suboptimal codegen.
                ir[SSAValue(i)][:info] = info.call
                ir[SSAValue(i)][:flag] |= CC.IR_FLAG_REFINED
            end

            if isexpr(stmt, :invoke) && isa(stmt.args[1], Tuple)
                info::MappingInfo
                callee_mi = stmt.args[1][1]
                closure_env = stmt.args[2]
                in_vars = stmt.args[3]
                callee_ci = CC.get(CC.code_cache(interp), callee_mi, nothing)

                @assert callee_ci !== nothing

                spec_data = stmt.args[1]
                callee_key = stmt.args[1][2]
                callee_ordinal = stmt.args[1][end]::Int
                callee_daef_cis = dae_finish_ipo!(interp, callee_ci, callee_key, callee_ordinal)
                # Allocate a continuous block of variables for all callee alg and diff states

                empty!(stmt.args)
                push!(stmt.args, callee_daef_cis[1])
                push!(stmt.args, closure_env)
                push!(stmt.args, in_vars)

                # Ordering from tearing is (AssignedDiff, UnassignedDiff, Algebraic, Explicit)
                for (arg, range_idx) in zip(arg_range, (1, 4, 1, 2, 2, 3))
                    push!(stmt.args, insert_node!(ir, SSAValue(i),
                        NewInstruction(inst;
                        stmt=Expr(:call, view, Argument(arg), spec_data[2+range_idx]),
                        type=VectorViewType)))
                end

                # TODO: Track whether the system is autonomous
                push!(stmt.args, t)
            end

            if is_known_invoke(stmt, variable, ir) || is_known_invoke_or_call(stmt, state_ddt, ir) || is_equation_call(stmt, ir)
                display(ir)
                error()
            elseif is_known_invoke_or_call(stmt, Intrinsics.state, ir)
                kind = stmt.args[end]::StateKind
                slot = stmt.args[end-1]
                which = kind == AssignedDiff        ? in_u_mm :
                        kind == UnassignedDiff      ? in_u_unassgn :
                        kind == AlgebraicDerivative ? in_du_unassgn :
                        kind == Algebraic           ? in_alg : error()
                replace_call!(ir, SSAValue(i), Expr(:call, Base.getindex, which, slot))
            elseif is_known_invoke_or_call(stmt, Intrinsics.contribution, ir)
                slot = stmt.args[end-2]::Int
                kind = stmt.args[end-1]::EquationKind
                red = stmt.args[end]
                which = kind == StateDiff ? out_du_mm :
                        kind == Explicit  ? out_eq : error()
                prev = insert_node!(ir, SSAValue(i), NewInstruction(inst; stmt=Expr(:call, Base.getindex, which, slot), type=Float64))
                sum = insert_node!(ir, SSAValue(i), NewInstruction(inst; stmt=Expr(:call, +, prev, red), type=Float64))
                replace_call!(ir, SSAValue(i), Expr(:call, Base.setindex!, which, sum, slot))
            elseif is_known_invoke(stmt, equation, ir)
                # Equation - used, but only as an arg to equation call, which will all get
                # eliminated by the end of this loop, so we can delete this statement, as
                # long as we don't touch the type yet.
                ir[SSAValue(i)][:inst] = Intrinsics.placeholder_equation
            elseif is_solved_variable(stmt)
                # Not used in this lowering
                ir[SSAValue(i)] = nothing
           else
                replace_if_intrinsic!(ir, SSAValue(i), nothing, nothing, Argument(1), t, var_assignment)
            end
        end

        # Just before the end of the function
        idx = length(ir.stmts)
        function ir_add!(a, b)
            ni = NewInstruction(Expr(:call, +, a, b), Any, ir[SSAValue(idx)][:line])
            insert_node!(ir, idx, ni)
        end
        ir = compact!(ir)

        widen_extra_info!(ir)
        src = ir_to_src(ir)

        abi = Tuple{Tuple, Tuple, (VectorViewType for _ in arg_range)..., Float64}
        owner = Core.ABIOverwrite(abi, RHSSpec(key, ir_ordinal))
        daef_ci = cache_dae_ci!(ci, src, src.debuginfo, owner)

        global nrhscompiles += 1
        push!(cis, daef_ci)
    end

    result.dae_finish_cache[key] = cis

    return cis
end

function ir_to_src(ir::IRCode)
    isva = false
    slotnames = nothing
    ir.debuginfo.def === nothing && (ir.debuginfo.def = :var"generated IR for OpaqueClosure")
    nargtypes = length(ir.argtypes)
    nargs = nargtypes-1
    sig = Base.Experimental.compute_oc_signature(ir, nargs, isva)
    rt = Base.Experimental.compute_ir_rettype(ir)
    src = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
    if slotnames === nothing
        src.slotnames = Symbol[Symbol("arg$i") for i = 1:nargtypes]
    else
        length(slotnames) == nargtypes || error("mismatched `argtypes` and `slotnames`")
        src.slotnames = slotnames
    end
    src.nargs = length(ir.argtypes)
    src.isva = false
    src.slotflags = fill(zero(UInt8), nargtypes)
    src.slottypes = copy(ir.argtypes)
    src = CC.ir_to_codeinf!(src, ir)
    return src
end

function dae_factory_gen(world::UInt, source::LineNumberNode, _, @nospecialize(fT))
    sys_ipo = IRODESystem(Tuple{fT}; world, ipo_analysis_mode=true);

    result = getfield(sys_ipo, :result)
    interp = getfield(sys_ipo, :interp)
    codeinst = CC.get(CC.code_cache(interp), getfield(sys_ipo, :mi), nothing)

    # For the top-level problem, all external vars are state-invariant, and we do no other fissioning
    param_vars = BitSet(1:result.nexternalvars)

    structure = make_structure_from_ipo(result)
    complete!(structure)

    varwhitelist = StateSelection.computed_highest_diff_variables(structure)

    for param in param_vars
        varwhitelist[param] = false
    end

    # Max match is the (unique) tearing result given the choice of states
    var_eq_matching = complete(maximal_matching(structure.graph, Union{Unassigned, SelectedState};
        dstfilter = var->varwhitelist[var]))

    var_eq_matching = partial_state_selection_graph!(structure, var_eq_matching)

    diff_vars = BitSet()
    alg_vars = BitSet()

    for (v, match) in enumerate(var_eq_matching)
        v in param_vars && continue
        if match === SelectedState()
            push!(diff_vars, v)
        elseif match === unassigned
            push!(alg_vars, v)
        end
    end

    key = TornCacheKey(diff_vars, alg_vars, param_vars, Vector{Pair{BitSet, BitSet}}())

    DAECompiler.tearing_schedule!(interp, codeinst, key)
    ir_factory = dae_factory_gen(interp, codeinst, key)
    src = ir_to_src(ir_factory)
    src.ssavaluetypes = length(src.code)
    src.min_world = @atomic codeinst.min_world
    src.max_world = @atomic codeinst.max_world
    src.edges = codeinst.edges

    return src
end

function dae_factory_gen(interp, ci::CodeInstance, key)
    result = CC.traverse_analysis_results(ci) do @nospecialize result
        return result isa Union{DAEIPOResult, UncompilableIPOResult} ? result : nothing
    end

    torn_ir = result.tearing_cache[key]

    (;ir_sicm) = torn_ir

    ir_factory = copy(result.ir)
    pushfirst!(ir_factory.argtypes, Tuple{})
    compact = IncrementalCompact(ir_factory)

    local line
    if ir_sicm !== nothing
        line = result.ir[SSAValue(1)][:line]
        sicm = insert_node_here!(compact,
            NewInstruction(Expr(:invoke, result.sicm_cache[key], (Argument(i+1) for i = 1:length(result.ir.argtypes))...), Tuple, line))
    else
        sicm = ()
    end

    argt = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters, Float64}

    daef_cis = dae_finish_ipo!(interp, ci, key, 1)

    # Create a small opaque closure to adapt from SciML ABI to our own internal
    # ABI

    numstates = zeros(Int, Int(LastEquationKind))

    all_states = Int[]
    for var = 1:length(result.var_to_diff)
        kind = classify_var(result.var_to_diff, key, var)
        kind == nothing && continue
        numstates[kind] += 1
        (kind != AlgebraicDerivative) && push!(all_states, var)
    end

    ir_oc = copy(result.ir)
    empty!(ir_oc.argtypes)
    push!(ir_oc.argtypes, Tuple)
    push!(ir_oc.argtypes, Vector{Float64})
    push!(ir_oc.argtypes, Vector{Float64})
    push!(ir_oc.argtypes, Vector{Float64})
    push!(ir_oc.argtypes, SciMLBase.NullParameters)
    push!(ir_oc.argtypes, Float64)

    oc_compact = IncrementalCompact(ir_oc)

    # Zero the output
    line = ir_oc[SSAValue(1)][:line]
    insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, zero!, Argument(2)), VectorViewType, line))

    # out_du_mm, out_eq, in_u_mm, in_u_unassgn, in_du_unassgn, in_alg
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]
    out_du_mm = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, view, Argument(2), 1:nassgn), VectorViewType, line))
    out_eq = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, view, Argument(2), (nassgn+1):ntotalstates), VectorViewType, line))

    in_du_unassgn = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, view, Argument(3), (nassgn+1):(nassgn+numstates[UnassignedDiff])), VectorViewType, line))
    du_assgn = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, view, Argument(3), 1:nassgn), VectorViewType, line))

    in_u_mm = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, view, Argument(4), 1:nassgn), VectorViewType, line))
    in_u_unassgn = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, view, Argument(4), (nassgn+1):(nassgn+numstates[UnassignedDiff])), VectorViewType, line))
    in_alg = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, view, Argument(4), (nassgn+numstates[UnassignedDiff]+1):ntotalstates), VectorViewType, line))

    # Call DAECompiler-generated RHS with internal ABI
    oc_sicm = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, getfield, Argument(1), 1), Tuple, line))
    insert_node_here!(oc_compact,
        NewInstruction(Expr(:invoke, daef_cis[1], oc_sicm, (), out_du_mm, out_eq, in_u_mm, in_u_unassgn, in_du_unassgn, in_alg, Argument(6)), Nothing, line))

    # Manually apply mass matrix
    bc = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, Base.Broadcast.broadcasted, -, out_du_mm, du_assgn), Any, line))
    insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, Base.Broadcast.materialize!, out_du_mm, bc), Nothing, line))

    # Return
    insert_node_here!(oc_compact, NewInstruction(ReturnNode(nothing), Union{}, line))

    ir_oc = finish(oc_compact)
    oc = Core.OpaqueClosure(ir_oc)

    line = result.ir[SSAValue(1)][:line]

    oc_source_method = oc.source
    # Sketchy, but not clear that we have something better for the time being
    oc_ci = oc_source_method.specializations.cache
    @atomic oc_ci.max_world = @atomic ci.max_world
    @atomic oc_ci.min_world = 1 # @atomic ci.min_world

    new_oc = insert_node_here!(compact, NewInstruction(Expr(:new_opaque_closure,
        argt, Union{}, Nothing, oc_source_method, sicm), Core.OpaqueClosure, line), true)

    differential_states = Bool[v in key.diff_states for v in all_states]

    daef = insert_node_here!(compact, NewInstruction(Expr(:call, DAEFunction, new_oc),
    DAEFunction, line), true)

    # TODO: Ideally, this'd be in DAEFunction
    daef_and_diff = insert_node_here!(compact, NewInstruction(
        Expr(:call, tuple, daef, differential_states),
        Tuple, line), true)

    insert_node_here!(compact, NewInstruction(ReturnNode(daef_and_diff), Core.OpaqueClosure, line), true)

    ir_factory = finish(compact)
    global nfactorycompiles += 1

    return ir_factory
end

"""
    dae_factory(f)

Given Julia function `f` compatible with DAECompiler's model representation, return a `DAEFunction`
suitable for use with DAEProblem. The DAEFunction will be specific to the parameterization of `f`.
To obtain a new parameterization, re-run this function. The runtime complexity of this function is
at most equivalent to one ordinary evaluation of `f`, but this function may have significant
compile-time cost (cached as usual for Julia code).
"""
function dae_factory end

function refresh()
    @eval function dae_factory(f)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, dae_factory_gen))
    end
end
refresh()
