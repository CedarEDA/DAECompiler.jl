"""
    sciml_dae_split_u!(compact, arg, numstates)

Given an IR value `arg` that corresponds to `u` in SciML's DAE ABI, split it into component pieces for
the DAECompiler internal ABI.
"""
function sciml_dae_split_u!(compact, line, arg, numstates)
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]

    u_mm = insert_node_here!(compact,
        NewInstruction(Expr(:call, view, arg, 1:nassgn), VectorViewType, line))
    u_unassgn = insert_node_here!(compact,
        NewInstruction(Expr(:call, view, arg, (nassgn+1):(nassgn+numstates[UnassignedDiff])), VectorViewType, line))
    alg = insert_node_here!(compact,
        NewInstruction(Expr(:call, view, arg, (nassgn+numstates[UnassignedDiff]+1):ntotalstates), VectorViewType, line))

    return (u_mm, u_unassgn, alg)
end

"""
    sciml_dae_split_du!(compact, arg, numstates)

Given an IR value `arg` that corresponds to `du` in SciML's DAE ABI, split it into component pieces for
the DAECompiler internal ABI.
"""
function sciml_dae_split_du!(compact, line, arg, numstates)
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]

    in_du_unassgn = insert_node_here!(compact,
        NewInstruction(Expr(:call, view, arg, (nassgn+1):(nassgn+numstates[UnassignedDiff])), VectorViewType, line))
    du_assgn = insert_node_here!(compact,
        NewInstruction(Expr(:call, view, arg, 1:nassgn), VectorViewType, line))

    return (in_du_unassgn, du_assgn)
end

function make_daefunction(f)
    DAEFunction(f)
end

function make_daefunction(f, initf)
    DAEFunction(f; initialization_data = SciMLBase.OverrideInitData(NonlinearProblem((args...)->nothing, nothing, nothing), nothing, initf, nothing, nothing))
end

"""
    dae_factory_gen(ci, key)

Generate the `factory` function for CodeInstance `ci`, returning a DAEFunction.
The resulting function is roughly:

```
function factory(settings, f)
    # Run all parts of `f` that do not depend on state
    state_invariant_pieces = f_state_invariant()
    f! = %new_opaque_closure(f_rhs, state_invariant_pieces)
    DAEFunction(f!), differential_vars
end
```

"""
function dae_factory_gen(state::TransformationState, ci::CodeInstance, key::TornCacheKey, world::UInt, edges::SimpleVector, init_key::Union{TornCacheKey, Nothing})
    result = state.result
    torn_ci = find_matching_ci(ci->isa(ci.owner, TornIRSpec) && ci.owner.key == key, ci.def, world)
    torn_ir = torn_ci.inferred

    (;ir_sicm) = torn_ir

    ir_factory = copy(result.ir)
    pushfirst!(ir_factory.argtypes, Settings)
    pushfirst!(ir_factory.argtypes, typeof(factory))
    compact = IncrementalCompact(ir_factory)

    local line
    if ir_sicm !== nothing
        sicm_ci = find_matching_ci(ci->isa(ci.owner, SICMSpec) && ci.owner.key == key, ci.def, world)
        @assert sicm_ci !== nothing

        line = result.ir[SSAValue(1)][:line]
        #insert_node_here!(compact, NewInstruction(Expr(:call, println, "Trace: A"), Cvoid, line))
        sicm = insert_node_here!(compact,
            NewInstruction(Expr(:call, invoke, Argument(3), sicm_ci, (Argument(i+1) for i = 2:length(result.ir.argtypes))...), Tuple, line))
        #insert_node_here!(compact, NewInstruction(Expr(:call, println, "Trace: B"), Cvoid, line))
    else
        sicm = ()
    end

    argt = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters, Float64}

    daef_ci = rhs_finish!(state, ci, key, world, 1, edges)

    # Create a small opaque closure to adapt from SciML ABI to our own internal
    # ABI

    numstates = zeros(Int, Int(LastEquationStateKind))

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

    (in_du_unassgn, du_assgn) = sciml_dae_split_du!(oc_compact, line, Argument(3), numstates)
    (in_u_mm, in_u_unassgn, in_alg) = sciml_dae_split_u!(oc_compact, line, Argument(4), numstates)

    # Call DAECompiler-generated RHS with internal ABI
    oc_sicm = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, getfield, Argument(1), 1), Tuple, line))

    # N.B: The ordering of arguments should match the ordering in the StateKind enum
    insert_node_here!(oc_compact,
        NewInstruction(Expr(:invoke, daef_ci, oc_sicm, (), in_u_mm, in_u_unassgn, in_du_unassgn, in_alg, out_du_mm, out_eq, Argument(6)), Nothing, line))

    # TODO: We should not have to recompute this here
    var_eq_matching = matching_for_key(result, key, state.structure)
    (slot_assignments, var_assignment, eq_assignment) = assign_slots(state, key, var_eq_matching)

    # Manually apply mass matrix and implicit equations between selected states
    for v = 1:ndsts(state.structure.graph)
        vdiff = state.structure.var_to_diff[v]
        vdiff === nothing && continue

        if var_eq_matching[v] !== SelectedState() || var_eq_matching[vdiff] !== SelectedState()
            # Solved variables were already handled above
            continue
        end

        (kind, slot) = var_assignment[v]
        (dkind, dslot) = var_assignment[vdiff]
        @assert kind == AssignedDiff
        @assert dkind in (AssignedDiff, UnassignedDiff)

        v_val = insert_node_here!(oc_compact,
            NewInstruction(Expr(:call, Base.getindex, dkind == AssignedDiff ? in_u_mm : in_u_unassgn, dslot), Any, line))
        insert_node_here!(oc_compact,
            NewInstruction(Expr(:call, Base.setindex!, out_du_mm, v_val, slot), Any, line))
    end

    bc = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, Base.Broadcast.broadcasted, -, out_du_mm, du_assgn), Any, line))
    insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, Base.Broadcast.materialize!, out_du_mm, bc), Nothing, line))

    # Return
    insert_node_here!(oc_compact, NewInstruction(ReturnNode(nothing), Union{}, line))

    ir_oc = Compiler.finish(oc_compact)
    oc = Core.OpaqueClosure(ir_oc)

    line = result.ir[SSAValue(1)][:line]

    oc_source_method = oc.source
    # Sketchy, but not clear that we have something better for the time being
    oc_ci = oc_source_method.specializations.cache
    @atomic oc_ci.max_world = @atomic ci.max_world
    @atomic oc_ci.min_world = 1 # @atomic ci.min_world

    new_oc = insert_node_here!(compact, NewInstruction(Expr(:new_opaque_closure,
        argt, Union{}, Nothing, true, oc_source_method, sicm), Core.OpaqueClosure, line), true)

    differential_states = Bool[v in key.diff_states for v in all_states]

    if init_key !== nothing
        initf = init_uncompress_gen!(compact, result, ci, init_key, key, world, edges)
        daef = insert_node_here!(compact, NewInstruction(Expr(:call, make_daefunction, new_oc, initf),
            DAEFunction, line), true)
    else
        daef = insert_node_here!(compact, NewInstruction(Expr(:call, make_daefunction, new_oc),
        DAEFunction, line), true)
    end

    # TODO: Ideally, this'd be in DAEFunction
    daef_and_diff = insert_node_here!(compact, NewInstruction(
        Expr(:call, tuple, daef, differential_states),
        Tuple, line), true)

    insert_node_here!(compact, NewInstruction(ReturnNode(daef_and_diff), Core.OpaqueClosure, line), true)

    ir_factory = Compiler.finish(compact)

    return ir_factory
end
