"""
    sciml_ode_split_u!(compact, arg, numstates)

Given an IR value `arg` that corresponds to `u` in SciML's ODE ABI, split it into component pieces for
the DAECompiler internal ABI.
"""
function sciml_ode_split_u!(compact, line, arg, numstates)
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic] + numstates[AlgebraicDerivative]

    u_mm = @insert_node_here compact line view(arg,
        1:numstates[AssignedDiff])::VectorViewType
    u_unassgn = @insert_node_here compact line view(arg,
        (numstates[AssignedDiff] + 1):(numstates[AssignedDiff] + numstates[UnassignedDiff]))::VectorViewType
    alg = @insert_node_here compact line view(arg,
        (numstates[AssignedDiff] + numstates[UnassignedDiff] + 1):(numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]))::VectorViewType
    alg_derv = @insert_node_here compact line view(arg,
        (numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic] + 1):ntotalstates)::VectorViewType

    return (u_mm, u_unassgn, alg, alg_derv)
end

function generate_ode_mass_matrix(nd, na)
    n = nd + na
    mass_matrix = zeros(Float64, n, n)
    for i in 1:nd
        mass_matrix[i, i] = 1.0
    end
    return mass_matrix
end

function make_odefunction(f, mass_matrix = LinearAlgebra.I, initf = nothing)
    ODEFunction(f; mass_matrix, initialization_data = (initf === nothing ? nothing : initialization_data_ode(initf)))
end

function initialization_data_ode(initf)
    return SciMLBase.OverrideInitData(NonlinearProblem((args...)->nothing, nothing, nothing), nothing, initf, nothing, nothing, Val{false}())
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
function ode_factory_gen(state::TransformationState, ci::CodeInstance, key::TornCacheKey, world::UInt, settings::Settings, init_key::Union{TornCacheKey, Nothing})
    result = state.result
    torn_ci = find_matching_ci(ci->isa(ci.owner, TornIRSpec) && ci.owner.key == key, ci.def, world)
    torn_ir = torn_ci.inferred

    (;ir_sicm) = torn_ir

    ir_factory = copy(ci.inferred.ir)
    pushfirst!(ir_factory.argtypes, Settings)
    pushfirst!(ir_factory.argtypes, typeof(factory))
    compact = IncrementalCompact(ir_factory)

    local line
    if ir_sicm !== nothing
        sicm_ci = find_matching_ci(ci->isa(ci.owner, SICMSpec) && ci.owner.key == key, ci.def, world)
        @assert sicm_ci !== nothing

        line = result.ir[SSAValue(1)][:line]
        param_list = flatten_parameter!(Compiler.fallback_lattice, compact, ci.inferred.ir.argtypes[1:end], argn->Argument(2+argn), line)
        sicm = insert_node_here!(compact,
            NewInstruction(Expr(:call, invoke, param_list, sicm_ci), Tuple, line))
    else
        sicm = ()
    end

    odef_ci = rhs_finish!(state, ci, key, world, settings, 1)

    # Create a small opaque closure to adapt from SciML ABI to our own internal ABI

    numstates = zeros(Int, Int(LastEquationStateKind))

    all_states = Int[]
    for var = 1:length(result.var_to_diff)
        varkind(state, var) == Intrinsics.Continuous || continue
        kind = classify_var(result.var_to_diff, key, var)
        kind == nothing && continue
        numstates[kind] += 1
        (kind != AlgebraicDerivative) && push!(all_states, var)
    end

    ir_oc = copy(ci.inferred.ir)
    empty!(ir_oc.argtypes)
    argt = Tuple{Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters, Float64}
    push!(ir_oc.argtypes, Tuple)
    append!(ir_oc.argtypes, fieldtypes(argt))
    Compiler.verify_ir(ir_oc)

    oc_compact = IncrementalCompact(ir_oc)
    self = Argument(1)
    du = Argument(2)
    u = Argument(3)
    p = Argument(4)
    t = Argument(5)

    line = ir_oc[SSAValue(1)][:line]

    # Zero the output
    @insert_node_here oc_compact line zero!(du)::VectorViewType

    nassgn = numstates[AssignedDiff]
    nunassgn = numstates[UnassignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic] + numstates[AlgebraicDerivative]

    (in_u_mm, in_u_unassgn, in_alg, in_alg_derv) = sciml_ode_split_u!(oc_compact, line, u, numstates)
    out_du_mm = @insert_node_here oc_compact line view(du, 1:nassgn)::VectorViewType
    out_du_unassgn = @insert_node_here oc_compact line view(du, (nassgn+1):(nassgn+nunassgn))::VectorViewType
    out_eq = @insert_node_here oc_compact line view(du, (nassgn+nunassgn+1):ntotalstates)::VectorViewType

    # Call DAECompiler-generated RHS with internal ABI
    oc_sicm = @insert_node_here oc_compact line getfield(self, 1)::Tuple

    # N.B: The ordering of arguments should match the ordering in the StateKind enum
    @insert_node_here oc_compact line (:invoke)(odef_ci, oc_sicm, (), in_u_mm, in_u_unassgn, in_alg_derv, in_alg, out_du_mm, out_eq, t)::Nothing

    # Assign the algebraic derivatives to the their corresponding variables
    bc = insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, Base.Broadcast.broadcasted, identity, in_alg_derv), Any, line))
    insert_node_here!(oc_compact,
        NewInstruction(Expr(:call, Base.Broadcast.materialize!, out_du_unassgn, bc), Nothing, line))

    # Return
    @insert_node_here oc_compact line (return)::Union{}

    ir_oc = Compiler.finish(oc_compact)
    maybe_rewrite_debuginfo!(ir_oc, settings)
    Compiler.verify_ir(ir_oc)
    oc = Core.OpaqueClosure(ir_oc)

    line = result.ir[SSAValue(1)][:line]

    oc_source_method = oc.source
    # Sketchy, but not clear that we have something better for the time being
    oc_ci = oc_source_method.specializations.cache
    @atomic oc_ci.max_world = @atomic ci.max_world
    @atomic oc_ci.min_world = 1 # @atomic ci.min_world

    new_oc = @insert_node_here compact line (:new_opaque_closure)(argt, Union{}, Nothing, true, oc_source_method, sicm)::Core.OpaqueClosure true

    nd = numstates[AssignedDiff] + numstates[UnassignedDiff]
    na = numstates[Algebraic] + numstates[AlgebraicDerivative]
    mass_matrix = na == 0 ? GlobalRef(LinearAlgebra, :I) : @insert_node_here compact line generate_ode_mass_matrix(nd, na)::Matrix{Float64}
    initf = init_key !== nothing ? init_uncompress_gen!(compact, result, ci, init_key, key, world, settings) : nothing
    odef = @insert_node_here compact line make_odefunction(new_oc, mass_matrix, initf)::ODEFunction true

    odef_and_n = @insert_node_here compact line tuple(odef, nd + na)::Tuple true
    @insert_node_here compact line (return odef_and_n)::Core.OpaqueClosure true

    ir_factory = Compiler.finish(compact)
    Compiler.verify_ir(ir_factory)

    return ir_factory
end
