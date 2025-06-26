function expand_residuals(state::TransformationState, key::TornCacheKey, compressed, u, du, t)
    (; result, structure) = state
    expanded = Float64[]
    i = 1
    # TODO: Remove this and the `key` argument if unused
    # var_eq_matching = matching_for_key(state, key)
    # (slot_assignments, var_assignment, _) = assign_slots(state, key, var_eq_matching)
    for (eq, incidence) in enumerate(result.total_incidence)
        if !is_var_part_known_linear(incidence)
            sign = get_linear_residual_sign(result, eq)
            push!(expanded, sign * compressed[i])
            i += 1
            continue
        end

        residual = 0.0
        sign = get_linear_residual_sign(result, eq)
        for (coeff, var) in zip(nonzeros(incidence.row), rowvals(incidence.row))
            var -= 1
            if var == 0
                value = t
            else
                vint = invview(structure.var_to_diff)[var]
                (slot, source) = vint === nothing ? (var, u) : (vint, du)
                vint !== nothing && (i += 1)
                value = source[slot]
            end
            residual += value * coeff
        end
        constant_term = isa(incidence.typ, Const) ? incidence.typ.val::Float64 : 0.0
        push!(expanded, constant_term + sign * residual)
    end
    return expanded
end

function get_linear_residual_sign(result::DAEIPOResult, eq::Int)
    # If a linear solved term appears with a positive coefficient,
    # the residual will be taken as the negative of the value provided to `always!`.
    # For example: ẋ₁ - x₁x₂ = 0
    #                    -ẋ₁ = -x₁x₂
    #                     ẋ₁ = -x₁x₂/-1
    #                     ẋ₁ = x₁x₂
    #                      0 = x₁x₂ - ẋ₁   <-- residual
    incidence = result.total_incidence[eq]
    for (coeff, var) in zip(nonzeros(incidence.row), rowvals(incidence.row))
        var - 1 === eq || continue
        isa(coeff, Float64) || continue
        return coeff ≤ 0 ? -1 : 1
    end
    return 1
end

function expand_residuals(f, residuals, u, du, t)
    result = @code_structure result=true f()
    structure = make_structure_from_ipo(result)
    state = TransformationState(result, structure)
    key, _ = top_level_state_selection!(state)
    return expand_residuals(state, key, residuals, u, du, t)
end

function extract_removed_states(state::TransformationState, key::TornCacheKey, torn::TornIR, u, du, t)
    (; result, structure) = state
    # TODO: handle multiple partitions
    torn_ir = only(torn.ir_seq)
    removed_states = Int[]
    for (i, inst) in enumerate(torn_ir.stmts)
        stmt = inst[:stmt]
        is_solved_variable(stmt) || continue
        var = stmt.args[2]::Int
        vint = invview(structure.var_to_diff)[var]
        vint === nothing || key.diff_states === nothing || !in(vint, key.diff_states) || continue
        push!(removed_states, var)
    end
    return removed_states
end

"""
    compute_residual_vectors(f, u, du; t = rand())

Compute residual vectors with the optimized and unoptimized code generation.
For a consistent `u` and `du` pair (in particular with respect
to the equations defining state derivatives), both residuals should be equal.
If not, it may indicate a bug in the code generation process and should be addressed.

If a state derivative is used in more than one equation, `u` and `du` must
be provided such that the selected equation that determines this derivative
holds; otherwise, residuals for equations involving the value of this state
derivative may differ between the unoptimized and optimized versions.
"""
function compute_residual_vectors(f, u, du; t = rand(), mode=DAE, world=Base.tls_world_age())
    @assert mode === DAE # TODO: support ODEs
    settings = Settings(; mode)
    ci = _code_ad_by_type(Tuple{typeof(f)}; world)
    result = @code_structure result=true mode=mode world=world f()
    structure = make_structure_from_ipo(result)
    state = TransformationState(result, structure)
    key, _ = top_level_state_selection!(state)
    tearing_schedule!(state, ci, key, world, settings)
    torn_ci = find_matching_ci(ci->isa(ci.owner, TornIRSpec) && ci.owner.key == key, ci.def, world)
    torn_ir = torn_ci.inferred
    removed_states = extract_removed_states(state, key, torn_ir, u, du, t)

    residuals = zeros(length(u))
    p = SciMLBase.NullParameters()
    indices = filter(!in(removed_states), eachindex(u))
    u_compressed = u[indices]
    du_compressed = du[indices]
    residuals_compressed = zeros(length(residuals) - length(removed_states))

    our_prob = DAECProblem(f, (1,) .=> 1., insert_stmt_debuginfo = true)
    sciml_prob = DiffEqBase.get_concrete_problem(our_prob, true)
    f_compressed! = sciml_prob.f.f
    f_compressed!(residuals_compressed, du_compressed, u_compressed, p, t)

    our_prob = DAECProblem(f, (1,) .=> 1., insert_stmt_debuginfo = true, skip_optimizations = true)
    sciml_prob = DiffEqBase.get_concrete_problem(our_prob, true)
    f_original! = sciml_prob.f.f
    f_original!(residuals, du, u, p, t)

    expanded = expand_residuals(f, residuals_compressed, u, du, t)
    return residuals, expanded
end
