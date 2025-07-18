function expand_residuals(state::TransformationState, key::TornCacheKey, states, compressed, u, du, t)
    (; result, structure) = state
    expanded = Float64[]
    i = 1
    var_eq_matching = matching_for_key(state, key)
    for (eq, incidence) in enumerate(result.total_incidence)
        uses_replacement_variable = any(x -> in(x.by + 1, rowvals(incidence.row)), result.replacement_map.variables)
        if !is_const_plus_var_known_linear(incidence) || uses_replacement_variable
            sign = infer_residual_sign(result, eq, var_eq_matching)
            uses_replacement_variable && (sign *= -1) # XXX: this may be too simple
            push!(expanded, sign * compressed[i])
            i += 1
            continue
        end

        residual = 0.0
        for (coeff, var) in zip(nonzeros(incidence.row), rowvals(incidence.row))
            var -= 1
            if var == 0
                value = t
            else
                is_diff = is_differential_variable(structure, var)
                source = ifelse(is_diff, du, u)
                # XXX: that's probably incorrect but has done the correct thing so far
                var === invview(var_eq_matching)[eq] && !is_diff && (i += 1)
                state = states[var]
                @assert state ≠ -1 "Reading from a state vector for a variable that has no corresponding state"
                value = source[state]
            end
            residual += value * coeff
        end
        constant_term = incidence.typ.val::Float64
        push!(expanded, constant_term + residual)
    end
    return expanded
end

function infer_residual_sign(result::DAEIPOResult, eq::Int, var_eq_matching)
    # If a linear solved term appears with a positive coefficient,
    # the residual will be taken as the negative of the value provided to `always!`.
    # For example: ẋ₁ - x₁x₂ = 0
    #                    -ẋ₁ = -x₁x₂
    #                     ẋ₁ = -x₁x₂/-1
    #                     ẋ₁ = x₁x₂
    #                      0 = x₁x₂ - ẋ₁   <-- residual
    incidence = result.total_incidence[eq]
    var = invview(var_eq_matching)[eq]
    isa(var, Int) || return 1
    coeff = incidence.row[var + 1]
    isa(coeff, Float64) || return -1
    return -sign(coeff)
end

function is_differential_variable(structure::DAESystemStructure, var)
    structure.var_to_diff[var] !== nothing && return false
    return invview(structure.var_to_diff)[var] !== nothing && return true
    @assert false
end

function extract_removed_variables(state::TransformationState, key::TornCacheKey, torn::TornIR)
    (; result, structure) = state
    # TODO: handle multiple partitions
    torn_ir = only(torn.ir_seq)
    removed_vars = Int[]
    for (i, inst) in enumerate(torn_ir.stmts)
        stmt = inst[:stmt]
        is_solved_variable(stmt) || continue
        var = stmt.args[2]::Int
        vint = invview(structure.var_to_diff)[var]
        vint === nothing || key.diff_states === nothing || !in(vint, key.diff_states) || continue
        push!(removed_vars, var)
    end
    return removed_vars
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
function compute_residual_vectors(f, u, du; t = 1.0, mode=DAE, world=Base.tls_world_age())
    @assert mode === DAE # TODO: support ODEs
    settings = Settings(; mode, insert_stmt_debuginfo = true)
    tt = Base.signature_type(f, ())
    ci = _code_ad_by_type(tt; world)
    result = @code_structure result=true mode=settings.mode insert_stmt_debuginfo=settings.insert_stmt_debuginfo world=world f()
    structure = make_structure_from_ipo(result)
    state = TransformationState(result, structure)
    key, _ = top_level_state_selection!(state)
    tearing_schedule!(state, ci, key, world, settings)
    torn_ci = find_matching_ci(ci->isa(ci.owner, TornIRSpec) && ci.owner.key == key, ci.def, world)
    torn_ir = torn_ci.inferred

    our_prob = DAECProblem(f, (1,) .=> 1.; settings.insert_stmt_debuginfo)
    sciml_prob = DiffEqBase.get_concrete_problem(our_prob, true)
    f_compressed! = sciml_prob.f.f

    our_prob = DAECProblem(f, (1,) .=> 1.; settings.insert_stmt_debuginfo, skip_optimizations = true)
    sciml_prob = DiffEqBase.get_concrete_problem(our_prob, true)
    f_original! = sciml_prob.f.f

    residuals = zeros(length(u))
    p = SciMLBase.NullParameters()
    states = map_variables_to_states(state)
    removed_variables = extract_removed_variables(state, key, torn_ir)
    removed_states = filter(≠(-1), states[removed_variables])
    compressed_states = filter(x -> !in(x, removed_states) && x ≠ -1, states)
    state_compression = unique(compressed_states)
    u_compressed = u[state_compression]
    du_compressed = du[state_compression]

    n = length(state.result.eqkinds)
    residuals_compressed = zeros(n)
    f_compressed!(residuals_compressed, du_compressed, u_compressed, p, t)
    f_original!(residuals, du, u, p, t)

    expanded = expand_residuals(state, key, states, residuals_compressed, u, du, t)
    @assert issorted(result.replacement_map.variables, by = x -> x.equation)
    for (; equation) in result.replacement_map.variables
        insert!(residuals, equation, 0.0)
    end

    return residuals, expanded
end
