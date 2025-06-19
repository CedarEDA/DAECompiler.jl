"""
    sciml_dae_split_u!(compact, arg, numstates)

Given an IR value `arg` that corresponds to `u` in SciML's DAE ABI, split it into component pieces for
the DAECompiler internal ABI.
"""
function sciml_dae_split_u!(compact, line, settings, arg, numstates)
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]

    u_mm = @insert_instruction compact line settings view(arg, 1:nassgn)::VectorViewType
    u_unassgn = @insert_instruction compact line settings view(arg, (nassgn+1):(nassgn+numstates[UnassignedDiff]))::VectorViewType
    alg = @insert_instruction compact line settings view(arg, (nassgn+numstates[UnassignedDiff]+1):ntotalstates)::VectorViewType

    return (u_mm, u_unassgn, alg)
end

"""
    sciml_dae_split_du!(compact, arg, numstates)

Given an IR value `arg` that corresponds to `du` in SciML's DAE ABI, split it into component pieces for
the DAECompiler internal ABI.
"""
function sciml_dae_split_du!(compact, line, settings, arg, numstates)
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]

    in_du_assgn = @insert_instruction compact line settings view(arg, 1:nassgn)::VectorViewType
    in_du_unassgn = @insert_instruction compact line settings view(arg, (nassgn+1):(nassgn+numstates[UnassignedDiff]))::VectorViewType

    return (in_du_assgn, in_du_unassgn)
end

function make_daefunction(f)
    DAEFunction(f)
end

function make_daefunction(f, initf)
    DAEFunction(f; initialization_data = SciMLBase.OverrideInitData(NonlinearProblem((args...)->nothing, nothing, nothing), nothing, initf, nothing, nothing, Val{false}()))
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
function dae_factory_gen(state::TransformationState, ci::CodeInstance, key::TornCacheKey, world::UInt, settings::Settings, init_key::Union{TornCacheKey, Nothing})
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
        param_list = flatten_parameter!(Compiler.fallback_lattice, compact, ci.inferred.ir.argtypes[1:end], argn->Argument(2+argn), line, settings)
        sicm = @insert_instruction compact line settings invoke(param_list, sicm_ci)::Tuple
    else
        sicm = ()
    end

    argt = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, SciMLBase.NullParameters, Float64}

    daef_ci = rhs_finish!(state, ci, key, world, settings, 1)

    # Create a small opaque closure to adapt from SciML ABI to our own internal
    # ABI

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
    push!(ir_oc.argtypes, Tuple)
    push!(ir_oc.argtypes, Vector{Float64})
    push!(ir_oc.argtypes, Vector{Float64})
    push!(ir_oc.argtypes, Vector{Float64})
    push!(ir_oc.argtypes, SciMLBase.NullParameters)
    push!(ir_oc.argtypes, Float64)

    oc_compact = IncrementalCompact(ir_oc)

    # Zero the output
    line = ir_oc[SSAValue(1)][:line]
    @insert_instruction oc_compact line settings zero!(Argument(2))::VectorViewType

    # out_du_mm, out_eq, in_u_mm, in_u_unassgn, in_du_unassgn, in_alg
    nassgn = numstates[AssignedDiff]
    ntotalstates = numstates[AssignedDiff] + numstates[UnassignedDiff] + numstates[Algebraic]
    out_du_mm = @insert_instruction oc_compact line settings view(Argument(2), 1:nassgn)::VectorViewType
    out_eq = @insert_instruction oc_compact line settings view(Argument(2), (nassgn+1):ntotalstates)::VectorViewType

    (in_du_assgn, in_du_unassgn) = sciml_dae_split_du!(oc_compact, line, settings, Argument(3), numstates)
    (in_u_mm, in_u_unassgn, in_alg) = sciml_dae_split_u!(oc_compact, line, settings, Argument(4), numstates)

    # Call DAECompiler-generated RHS with internal ABI
    oc_sicm = @insert_instruction oc_compact line settings getfield(Argument(1), 1)::Core.OpaqueClosure

    # N.B: The ordering of arguments should match the ordering in the StateKind enum
    @insert_instruction oc_compact line settings (:invoke)(daef_ci, oc_sicm, (), in_u_mm, in_u_unassgn, in_du_unassgn, in_alg, out_du_mm, out_eq, Argument(6))::Nothing

    # TODO: We should not have to recompute this here
    var_eq_matching = matching_for_key(state, key)
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

        v_val = @insert_instruction oc_compact line settings getindex(dkind == AssignedDiff ? in_u_mm : in_u_unassgn, dslot)::Any
        @insert_instruction oc_compact line settings setindex!(out_du_mm, v_val, slot)::Any
    end

    bc = @insert_instruction oc_compact line settings Base.Broadcast.broadcasted(-, out_du_mm, in_du_assgn)::Any
    @insert_instruction oc_compact line settings Base.Broadcast.materialize!(out_du_mm, bc)::Nothing

    # Return
    @insert_instruction oc_compact line settings (return nothing)::Union{}

    ir_oc = Compiler.finish(oc_compact)
    maybe_rewrite_debuginfo!(ir_oc, settings)
    resize!(ir_oc.cfg.blocks, 1)
    empty!(ir_oc.cfg.blocks[1].succs)
    Compiler.verify_ir(ir_oc)
    oc = Core.OpaqueClosure(ir_oc)

    line = result.ir[SSAValue(1)][:line]

    oc_source_method = oc.source
    # Sketchy, but not clear that we have something better for the time being
    oc_ci = oc_source_method.specializations.cache
    @atomic oc_ci.max_world = @atomic ci.max_world
    @atomic oc_ci.min_world = 1 # @atomic ci.min_world

    new_oc = @insert_instruction compact line settings (:new_opaque_closure)(argt, Union{}, Nothing, true, oc_source_method, sicm)::Core.OpaqueClosure true

    differential_states = Bool[v in key.diff_states for v in all_states]

    if init_key !== nothing
        initf = init_uncompress_gen!(compact, result, ci, init_key, key, world, settings)
        daef = @insert_instruction compact line settings make_daefunction(new_oc, initf)::DAEFunction true
    else
        daef = @insert_instruction compact line settings make_daefunction(new_oc)::DAEFunction true
    end

    # TODO: Ideally, this'd be in DAEFunction
    daef_and_diff = @insert_instruction compact line settings tuple(daef, differential_states)::Tuple true

    @insert_instruction compact line settings (return daef_and_diff)::Tuple true

    ir_factory = Compiler.finish(compact)
    resize!(ir_factory.cfg.blocks, 1)
    empty!(ir_factory.cfg.blocks[1].succs)
    Compiler.verify_ir(ir_factory)

    slotnames = [[:factory, :settings]; Symbol.(:arg, 1:(length(ir_factory.argtypes) - 2))]
    return ir_factory, slotnames
end
