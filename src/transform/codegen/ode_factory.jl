"""
    sciml_ode_split_u!(compact, arg, numstates)

Given an IR value `arg` that corresponds to `u` in SciML's ODE ABI, split it into component pieces for
the DAECompiler internal ABI.
"""
function sciml_ode_split_u!(compact, line, arg, numstates)
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]

    u_mm = @insert_node_here compact line view(arg, 1:nassgn)::VectorViewType
    u_unassgn = @insert_node_here compact line view(arg, (nassgn+1):(nassgn+numstates[UnassignedDiff]))::VectorViewType
    alg = @insert_node_here compact line view(arg, (nassgn+numstates[UnassignedDiff]+1):ntotalstates)::VectorViewType

    return (u_mm, u_unassgn, alg)
end

function make_odefunction(f)
    ODEFunction(f)
end

function make_odefunction(f, initf)
    ODEFunction(f; initialization_data = SciMLBase.OverrideInitData(NonlinearProblem((args...)->nothing, nothing, nothing), nothing, initf, nothing, nothing))
end

"""
    ode_factory_gen(ci, key)

Generate the `factory` function for CodeInstance `ci`, returning a ODEFunction.
The resulting function is roughly:

```
function factory(settings, f)
    # Run all parts of `f` that do not depend on state
    state_invariant_pieces = f_state_invariant()
    f! = %new_opaque_closure(f_rhs, state_invariant_pieces)
    ODEFunction(f!)
end
```

"""
function ode_factory_gen(result::DAEIPOResult, ci::CodeInstance, key::TornCacheKey, world::UInt, init_key::Union{TornCacheKey, Nothing})
    @ccall jl_safe_printf("$key\n"::Cstring)::Cvoid
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

    odef_ci = rhs_finish!(result, ci, key, world, 1)

    # Create a small opaque closure to adapt from SciML ABI to our own internal
    # ABI

    numstates = zeros(Int, Int(LastEquationStateKind))

    all_states = Int[]
    for var = 1:length(result.var_to_diff)
        kind = classify_var(result.var_to_diff, key, var)
        kind == nothing && continue
        @ccall jl_safe_printf("$kind\n"::Cstring)::Cvoid
        numstates[kind] += 1
        (kind != AlgebraicDerivative) && push!(all_states, var)
    end

    ir_oc = copy(result.ir)
    empty!(ir_oc.argtypes)
    argt = Tuple{Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters, Float64}
    push!(ir_oc.argtypes, Tuple)
    append!(ir_oc.argtypes, fieldtypes(argt))

    oc_compact = IncrementalCompact(ir_oc)

    # Zero the output
    line = ir_oc[SSAValue(1)][:line]

    @insert_node_here oc_compact line zero!(_2)::VectorViewType

    # out_du_mm, out_eq, in_u_mm, in_u_unassgn, in_alg
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]
    out_du_mm = @insert_node_here oc_compact line view(_2, 1:nassgn)::VectorViewType
    out_eq = @insert_node_here oc_compact line view(_2, (nassgn+1):ntotalstates)::VectorViewType

    (in_u_mm, in_u_unassgn, in_alg) = sciml_ode_split_u!(oc_compact, line, Argument(3), numstates)

    # Call DAECompiler-generated RHS with internal ABI
    oc_sicm = @insert_node_here oc_compact line getfield(_1, 1)::Tuple

    # N.B: The ordering of arguments should match the ordering in the StateKind enum
    @insert_node_here oc_compact line (:invoke)(odef_ci, oc_sicm, (), in_u_mm, in_u_unassgn, in_alg, out_du_mm, out_eq, _5)::Nothing

    # Manually apply mass matrix
    # bc = insert_node_here!(oc_compact,
    #     NewInstruction(Expr(:call, Base.Broadcast.broadcasted, -, out_du_mm, du_assgn), Any, line))
    # insert_node_here!(oc_compact,
    #     NewInstruction(Expr(:call, Base.Broadcast.materialize!, out_du_mm, bc), Nothing, line))

    # Return
    @insert_node_here oc_compact line (return)::Union{}

    ir_oc = Compiler.finish(oc_compact)
    oc = Core.OpaqueClosure(ir_oc)

    line = result.ir[SSAValue(1)][:line]

    oc_source_method = oc.source
    # Sketchy, but not clear that we have something better for the time being
    oc_ci = oc_source_method.specializations.cache
    @atomic oc_ci.max_world = @atomic ci.max_world
    @atomic oc_ci.min_world = 1 # @atomic ci.min_world

    new_oc = @insert_node_here compact line (:new_opaque_closure)(argt, Union{}, Nothing, true, oc_source_method, sicm)::Core.OpaqueClosure true

    if init_key !== nothing
        initf = init_uncompress_gen!(compact, result, ci, init_key, key, world)
        odef = @insert_node_here compact line make_odefunction(new_oc, initf)::ODEFunction true
    else
        odef = @insert_node_here compact line make_odefunction(new_oc)::ODEFunction true
    end

    @insert_node_here compact line (return odef)::Core.OpaqueClosure true

    ir_factory = Compiler.finish(compact)

    return ir_factory
end
