function torn_cache_key(tstate::TransformationState, settings::Settings)
    (; result, structure) = tstate
    nvars = length(result.varkinds)
    neqs = length(result.eqkinds)
    param_vars = BitSet(1:result.nexternalargvars)
    diff_vars = BitSet()
    alg_vars = BitSet()
    explicit_eqs = BitSet(1:neqs) # make all equations explicit
    var_schedule = Pair{BitSet, BitSet}[]
    for var in 1:nvars
        # TODO: Derive this information via equation uses through `result.total_incidence`.
        if structure.var_to_diff[var] !== nothing
            # This variable has a derivative.
            push!(diff_vars, var)
        elseif invview(structure.var_to_diff)[var] === nothing
            # This variable doesn't have and is not a derivative.
            push!(alg_vars, var)
        end
    end

    diff_key = TornCacheKey(diff_vars, alg_vars, param_vars, explicit_eqs, var_schedule)

    return diff_key
end

function rhs_finish_noopt!(
    state::TransformationState,
    ci::CodeInstance,
    key::TornCacheKey,
    world::UInt,
    settings::Settings,
    indexT=Int)

    (; result, structure) = state
    result_ci = find_matching_ci(ci->isa(ci.inferred, RHSSpec) && ci.inferred.key == key && ci.inferred.ordinal == 0, ci.def, world)
    if result_ci !== nothing
        return result_ci
    end

    ir = copy(result.ir)
    slotnames = [:captures, :vars, :out, :du, :u, :out_indices, :du_indices, :u_indices, :t]
    argtypes = [Tuple, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, VectorIntViewType, VectorIntViewType, VectorIntViewType, Float64]
    append!(empty!(ir.argtypes), argtypes)
    captures, vars, out, du, u, out_indices, du_indics, u_indices, t = Argument.(eachindex(slotnames))
    @assert length(slotnames) == length(ir.argtypes)

    equations = Pair{SSAValue, Eq}[]
    compact = IncrementalCompact(ir)

    for ((_, i), _) in compact
        ssaidx = SSAValue(i)
        inst = compact[ssaidx]
        stmt = inst[:stmt]
        type = inst[:type]
        line = inst[:line]

        if is_known_invoke(stmt, Intrinsics.equation, compact)
            push!(equations, ssaidx => type::Eq)
            inst[:stmt] = nothing
        elseif is_known_invoke(stmt, Intrinsics.ddt, compact)
            var = invview(structure.var_to_diff)[idnum(type)]
            getdu = Expr(:call, getindex, du, var)
            replace_call!(compact, ssaidx, getdu, settings, @__SOURCE__)
            inst[:type] = Float64
        elseif is_equation_call(stmt, compact)
            callee, value = stmt.args[2], stmt.args[3]
            i = findfirst(x -> first(x) == callee, equations)::Int
            eq = last(equations[i])
            call = Expr(:call, setindex!, out, value, eq.id)
            replace_call!(compact, ssaidx, call, settings, @__SOURCE__)
        elseif is_known_invoke_or_call(stmt, variable, compact)
            var = idnum(type)
            call = Expr(:call, getindex, u, var)
            replace_call!(compact, ssaidx, call, settings, @__SOURCE__)
            inst[:type] = Float64
        elseif is_known_invoke_or_call(stmt, sim_time, compact)
            inst[:stmt] = t
        # TODO: process flattened variables
        # TODO: process other intrinsics (epsilon, etc)
        # else
        #     replace_if_intrinsic!(compact, settings, ssaidx, nothing, nothing, nothing, t, nothing)
        end
        type = inst[:type]
        if isa(type, Incidence) || isa(type, Eq)
            inst[:type] = widenconst(type)
        end
    end

    spec = RHSSpec(key, 0)
    daef_ci = rhs_ir_finish!(Compiler.finish(compact), ci, settings, spec, slotnames)
    return daef_ci
end

function sciml_to_internal_abi_noopt!(ir::IRCode, state::TransformationState, internal_ci::CodeInstance, settings::Settings)
    slotnames = [:captures, :out, :du, :u, :p, :t]
    captures, out, du, u, p, t = Argument.(eachindex(slotnames))

    empty!(ir.argtypes)
    push!(ir.argtypes, Tuple) # opaque closure captures
    append!(ir.argtypes, fieldtypes(SCIML_ABI))

    compact = IncrementalCompact(ir)
    line = ir[SSAValue(1)][:line]

    internal_oc = @insert_instruction_here compact line settings getfield(captures, 1)::Core.OpaqueClosure
    # TODO: Compute proper indices.
    neqs = length(state.structure.eq_to_diff)
    out_indices = @insert_instruction_here compact line settings view(out, 1:neqs)::VectorIntViewType
    du_indices = @insert_instruction_here compact line settings view(du, 1:neqs)::VectorIntViewType
    u_indices = @insert_instruction_here compact line settings view(u, 1:neqs)::VectorIntViewType
    # TODO: Provide actual external variables.
    vars = @insert_instruction_here compact line settings getindex(Float64)::Vector{Float64}
    @insert_instruction_here compact line settings (:invoke)(internal_ci, internal_oc, vars, out, du, u, out_indices, du_indices, u_indices, t)::Nothing
    @insert_instruction_here compact line settings (return nothing)::Union{}

    ir = Compiler.finish(compact)
    maybe_rewrite_debuginfo!(ir, settings)
    resize!(ir.cfg.blocks, 1)
    empty!(ir.cfg.blocks[1].succs)
    Compiler.verify_ir(ir)

    return Core.OpaqueClosure(ir; slotnames)
end
