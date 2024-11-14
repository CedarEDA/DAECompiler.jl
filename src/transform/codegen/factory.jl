
"""
    dae_factory_gen(ci, key)

Generate the `factory` funciton for CodeInstance `ci`, returning a DAEFunction.
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
function dae_factory_gen(result::DAEIPOResult, ci::CodeInstance, key::TornCacheKey, world::UInt)
    torn_ir = result.tearing_cache[key]

    (;ir_sicm) = torn_ir

    ir_factory = copy(result.ir)
    pushfirst!(ir_factory.argtypes, Settings)
    pushfirst!(ir_factory.argtypes, typeof(factory))
    compact = IncrementalCompact(ir_factory)

    local line
    if ir_sicm !== nothing
        line = result.ir[SSAValue(1)][:line]
        #insert_node_here!(compact, NewInstruction(Expr(:call, println, "Trace: A"), Cvoid, line))
        sicm = insert_node_here!(compact,
            NewInstruction(Expr(:call, invoke, Argument(3), result.sicm_cache[key], (Argument(i+1) for i = 2:length(result.ir.argtypes))...), Tuple, line))
        #insert_node_here!(compact, NewInstruction(Expr(:call, println, "Trace: B"), Cvoid, line))
    else
        sicm = ()
    end

    argt = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters, Float64}

    daef_ci = dae_finish_ipo!(result, ci, key, world, 1)

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
        NewInstruction(Expr(:invoke, daef_ci, oc_sicm, (), out_du_mm, out_eq, in_u_mm, in_u_unassgn, in_du_unassgn, in_alg, Argument(6)), Nothing, line))

    # Manually apply mass matrix
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

    daef = insert_node_here!(compact, NewInstruction(Expr(:call, DAEFunction, new_oc),
    DAEFunction, line), true)

    # TODO: Ideally, this'd be in DAEFunction
    daef_and_diff = insert_node_here!(compact, NewInstruction(
        Expr(:call, tuple, daef, differential_states),
        Tuple, line), true)

    insert_node_here!(compact, NewInstruction(ReturnNode(daef_and_diff), Core.OpaqueClosure, line), true)

    ir_factory = Compiler.finish(compact)

    return ir_factory
end
