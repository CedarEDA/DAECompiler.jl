using StateSelection.BipartiteGraphs: 𝑠neighbors, BipartiteEdge


struct TransformationState <: StateSelection.TransformationState{DAEIPOResult}
    result::DAEIPOResult
    structure::DAESystemStructure
    total_incidence::Vector{Incidence}
end

function StateSelection.linear_subsys_adjmat!(state::TransformationState)
    graph = state.structure.graph
    eadj = Vector{Int}[]
    cadj = Vector{Int}[]
    linear_equations = Vector{Int}()
    for (i, inc) in enumerate(state.total_incidence)
        isa(inc, Const) && continue
        any(x->x === nonlinear || !isinteger(x), nonzeros(inc.row)) && continue
        (isa(inc.typ, Const) && iszero(inc.typ.val)) || continue
        # Skip any equations involving `t` for now
        # TODO: We may want to adjust our ILS to include equations with `t` in them.
        (1 ∉ rowvals(inc.row)) || continue
        this_eadj = Vector{Int}()
        this_cadj = Vector{Int}()
        for (var, coeff) in zip(rowvals(inc.row), nonzeros(inc.row))
            push!(this_eadj, var - 1)
            push!(this_cadj, coeff)
        end
        p = sortperm(this_eadj)
        push!(eadj, this_eadj[p])
        push!(cadj, this_cadj[p])
        push!(linear_equations, i)
    end
    return StateSelection.SparseMatrixCLIL(
                            nsrcs(graph), ndsts(graph),
                            linear_equations, eadj, cadj)
end

function StateSelection.eq_derivative!(state::TransformationState, eq)
    s = state.structure
    eq_diff = StateSelection.eq_derivative_graph!(s, eq)
    for var in collect(𝑠neighbors(s.graph, eq))
        diffvar = s.var_to_diff[var]
        # If this is solvable, it is linear, so no longer occurs in the derivative
        if !(BipartiteEdge(eq, var) in s.solvable_graph)
            add_edge!(s.graph, eq_diff, var)
            # TODO: We know this is linear in the derivative, but we don't
            # necessarily know that the coefficient is a constant and even if
            # it is, there's no guarantee we can determine that. For now,
            # don't add it the solvable graph. Eventually we could try to
            # reach back into the system to figure this out.
        else
            add_edge!(s.solvable_graph, eq_diff, s.var_to_diff[var])
        end
        add_edge!(s.graph, eq_diff, s.var_to_diff[var])
    end
    push!(state.total_incidence, structural_inc_ddt(s.var_to_diff, nothing, nothing, state.total_incidence[eq]))
end

function StateSelection.var_derivative!(state::TransformationState, var)
    return StateSelection.var_derivative_graph!(state.structure, var)
end

function baseeq(result, structure, eq)
    while eq > length(result.eqkinds)
        eq = invview(structure.eq_to_diff)[eq]
    end
    return eq
end

function basevar(result, structure, var)
    while var > length(result.varkinds)
        var = invview(structure.var_to_diff)[var]
    end
    return var
end

function eqkind(result, structure, eq)
    return result.eqkinds[baseeq(result, structure, eq)]
end
eqkind(state::TransformationState, eq) = eqkind(state.result, state.structure, eq)

function eqclassification(result, structure, eq)
    return result.eqclassification[baseeq(result, structure, eq)]
end

function ssrm!(state::TransformationState)
    ils = StateSelection.structural_singularity_removal!(state)

    s = state.structure
    for (ei, e) in enumerate(ils.nzrows)
        for g in (s.graph, s.solvable_graph)
            StateSelection.set_neighbors!(g, e, ils.row_cols[ei])
        end
        state.total_incidence[e] = Incidence(Const(0.), IncidenceVector(MAX_EQS, map(x->x+1, ils.row_cols[ei]), Union{Float64, NonLinear}[Float64(x) for x in ils.row_vals[ei]]))
    end
end

varkind(result, structure, var) = result.varkinds[basevar(result, structure, var)]
varkind(state, var) = varkind(state.result, state.structure, var)
varclassification(result, structure, var) = result.varkinds[basevar(result, structure, var)]
varclassification(state, var) = varclassification(state.result, state.structure, var)

function structural_transformation!(state::TransformationState)
    first = true
    # This loop is required to handle situations where additional structural signularities arise in the differentiated
    # equations. We could probably do lot better here by not constantly recomputing the datastructures.
    while true
        neq_before = length(state.structure.eq_to_diff)
        var_eq_matching = StateSelection.pantelides!(state;
            varfilter = var->varkind(state, var) == Intrinsics.Continuous && !(var <= state.result.nexternalvars),
            eqfilter  = eq->eqkind(state, eq) == Intrinsics.Always)

        differentiated_any = neq_before != length(state.structure.eq_to_diff)
        if differentiated_any || first
            ssrm!(state)
            first = false
            continue
        end

        return StateSelection.complete(var_eq_matching, nsrcs(state.structure.graph))
    end
end

using StateSelection: Unassigned, SelectedState, unassigned

struct StateInvariant; end
StateSelection.BipartiteGraphs.overview_label(::Type{StateInvariant}) = ('P', "State Invariant / Parameter", :red)

struct InOut
    ordinal::Int
end
StateSelection.BipartiteGraphs.overview_label(::Type{InOut}) = ('#', "IPO in var / out eq", :green)
StateSelection.BipartiteGraphs.overview_label(io::InOut) = (string(io.ordinal), "IPO in var / out eq", :green)

const IPOMatches = Union{Unassigned, SelectedState, StateInvariant, InOut}
const IPOMatching = StateSelection.Matching{IPOMatches}

function top_level_state_selection!(tstate)
    (; result, structure) = tstate
    # For the top-level problem, all external vars are state-invariant, and we do no other fissioning
    param_vars = BitSet(1:result.nexternalvars)

    highest_diff_max_match = structural_transformation!(tstate)

    StateSelection.complete!(structure)

    ## Part 1: Perform the selection of differential states and subsequent tearing of the
    #          non-linear problem at every time step.

    var_eq_matching = convert(IPOMatching, StateSelection.partial_state_selection_graph!(structure, highest_diff_max_match))

    diff_vars = BitSet()
    alg_vars = BitSet()
    explicit_eqs = BitSet()

    for (v, match) in enumerate(var_eq_matching)
        if v in param_vars
            @assert match === unassigned
            var_eq_matching[v] = StateInvariant()
            continue
        end
        if match === SelectedState()
            push!(diff_vars, v)
        elseif match === unassigned
            push!(alg_vars, v)
        end
    end

    for (eq, match) in enumerate(invview(var_eq_matching))
        match === unassigned || continue
        push!(explicit_eqs, eq)
    end


    diff_key = TornCacheKey(diff_vars, alg_vars, param_vars, explicit_eqs, Vector{Pair{BitSet, BitSet}}())
    @assert matching_for_key(result, diff_key, structure) == var_eq_matching

    varfilter(var) = varkind(result, structure, var) == Intrinsics.Continuous && !(var <= result.nexternalvars)

    ## Part 2: Perform the selection of differential states and subsequent tearing of the
    #          non-linear problem at every time step.
    init_var_eq_matching = StateSelection.complete(StateSelection.maximal_matching(structure.graph, IPOMatches;
        dstfilter = varfilter, srcfilter = eq->eqkind(result, structure, eq) in (Intrinsics.Always, Intrinsics.Initial)), nsrcs(structure.graph))
    init_var_eq_matching = convert(IPOMatching, StateSelection.pss_graph_modia!(structure, init_var_eq_matching))

    init_state_vars = BitSet()
    init_explicit_eqs = BitSet()
    for (v, match) in enumerate(init_var_eq_matching)
        if v in param_vars
            @assert match === unassigned
            init_var_eq_matching[v] = StateInvariant()
            continue
        end
        varfilter(v) || continue
        if match === unassigned
            push!(init_state_vars, v)
        end
    end

    for (eq, match) in enumerate(invview(init_var_eq_matching))
        match === unassigned || continue
        push!(init_explicit_eqs, eq)
    end

    init_key = TornCacheKey(nothing, init_state_vars, param_vars, init_explicit_eqs, Vector{Pair{BitSet, BitSet}}())
    @assert matching_for_key(result, init_key, structure) == init_var_eq_matching

    (diff_key, init_key)
end
