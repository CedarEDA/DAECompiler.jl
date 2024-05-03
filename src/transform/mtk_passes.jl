using StateSelection: pantelides!, dummy_derivative_graph!, partial_state_selection_graph!, complete!
using StateSelection: MatchedSystemStructure
using StateSelection.CLIL

function alias_elimination!(state::IRTransformationState)
    ils = StateSelection.alias_eliminate_graph!(state)
    s = state.structure
    for g in (s.graph, s.solvable_graph)
        for (ei, e) in enumerate(ils.nzrows)
            set_neighbors!(g, e, ils.row_cols[ei])
        end
    end
    return ils
end

struct BadSystemException <: Exception
    neqs::Int
    nvars::Int
    ir::IRCode
    names::Vector{Union{Nothing, Symbol}}
    vals::Vector{SSAValue}
end

function Base.print(io::IO, exc::BadSystemException)
    print(io, "BadSystemException(...)")
end

function Base.showerror(io::IO, exc::BadSystemException, bt; backtrace=true)
    # Do not print our own backtrace
    Base.showerror(io, exc)
end

function Base.showerror(io::IO, exc::BadSystemException)
    (; neqs, nvars) = exc
    if neqs == nvars
        print(io, "The system is structurally singular.\n")
    else
        print(io, "The system is unbalanced. There are ", nvars, " highest order differentiated variable(s) and "
            , neqs, " equation(s).\n")
    end
    for (name, val) in zip(exc.names, exc.vals)
        if neqs == nvars
            if name === nothing
                print(io, "This variable may be problematic:")
            else
                print(io, "Variable `")
                printstyled(io, name, bold=true)
                print(io, "` may be problematic:")
            end
        elseif neqs > nvars
            if name === nothing
                print(io, "This equation was potentially redundant:")
            else
                print(io, "Equation `")
                printstyled(io, name, bold=true)
                print(io, "` was potentially redundant:")
            end
        else
            if name === nothing
                print(io, "This variable was potentially unused:")
            else
                print(io, "Variable `")
                printstyled(io, name, bold=true)
                print(io, "` was potentially unused:")
            end
        end
        Base.show_backtrace(io, get_inline_backtrace(exc.ir, val))
        println(io)
    end
end

# TODO: Merge the checking parts of this with MTK, and only change the error
# reporting parts.
function StateSelection.check_consistency(state::IRTransformationState, _)
    neqs = StateSelection.n_concrete_eqs(state)
    (; graph, var_to_diff) = state.structure
    highest_vars = StateSelection.computed_highest_diff_variables(state.structure)
    n_highest_vars = 0
    for (v, h) in enumerate(highest_vars)
        h || continue
        isempty(𝑑neighbors(graph, v)) && continue
        n_highest_vars += 1
    end
    is_balanced = n_highest_vars == neqs

    eq_num2name = Dict(_eq_names_flattened(state))
    find_equation_names(idxs) = get.(Ref(eq_num2name), idxs, nothing)

    if neqs > 0 && !is_balanced
        (eqs, vars) = find_eqs_vars(state)

        varwhitelist = var_to_diff .== nothing
        var_eq_matching = maximal_matching(graph, eq -> true, v -> varwhitelist[v]) # not assigned
        # Just use `error_reporting` to do conditional
        iseqs = n_highest_vars < neqs
        if iseqs
            eq_var_matching = invview(complete(var_eq_matching, nsrcs(graph))) # extra equations
            bad_idxs = findall(isequal(unassigned), @view eq_var_matching[1:nsrcs(graph)])
            names = find_equation_names(bad_idxs)
        else
            
            bad_idxs = findall(isequal(unassigned), var_eq_matching)
            names = Union{Nothing, Symbol}[
                findfirst(x->x.var == idx,
                         getfield(get_sys(state), :result).names) for idx in bad_idxs]
        end

        throw(BadSystemException(neqs, n_highest_vars, copy(state.ir), names,
            iseqs ? map(x->x[1], eqs)[bad_idxs] : vars[bad_idxs] ))
    end

    # This is defined to check if Pantelides algorithm terminates. For more
    # details, check the equation (15) of the original paper.
    extended_graph = (@StateSelection.set graph.fadjlist = Vector{Int}[graph.fadjlist;
        map(collect, edges(var_to_diff))])
    extended_var_eq_matching = maximal_matching(extended_graph)

    unassigned_var = []
    for (vj, eq) in enumerate(extended_var_eq_matching)
        if eq === unassigned && vj ∈ 𝑑vertices(graph) && !isempty(𝑑neighbors(graph, vj))
            push!(unassigned_var, vj)
        end
    end

    if !isempty(unassigned_var) || !is_balanced
        (_, vars) = find_eqs_vars(state)

        throw(BadSystemException(neqs, n_highest_vars, copy(state.ir),
            find_equation_names(unassigned_var), vars[unassigned_var]))
    end
end

_eq_names_flattened(state::IRTransformationState) = _eq_names_flattened(getfield(get_sys(state), :result).names)
function _eq_names_flattened(lvl_children, prefix=:▫)
    ret = Pair{Int, Symbol}[]
    for (name, sublvl) in lvl_children
        sublvl_prefix = Symbol(prefix, :., name)
        if !isnothing(sublvl.children)
            append!(ret, _eq_names_flattened(sublvl.children, sublvl_prefix))
        elseif !isnothing(sublvl.eq)
            # NB: if we need this for other named things, it is trivial to generalize this to take the fieldname as a argument
            push!(ret, sublvl.eq => sublvl_prefix)
        end
    end
    return ret
end

"""
    structural_simplify(sys::IRODESystem) -> tsys::TransformedIRODESystem

Perform structural simplifications on `sys::IRODESystem` and transforms it into `tsys::TransformedIRODESystem`.
"""
function structural_simplify(sys::IRODESystem)
    state = IRTransformationState(sys)
    debug_config = DebugConfig(state)
    @may_timeit debug_config "mtk_passes" begin
        record_mss!(state, "initial", state.structure)

        @may_timeit debug_config "alias_elimination!" begin
            ils = DAECompiler.alias_elimination!(state)
        end
        complete!(state.structure)
        record_mss!(state, "post_alias_elimination", state.structure)

        StateSelection.check_consistency(state, nothing)

        @may_timeit debug_config "pantelides!" begin
            var_eq_matching = pantelides!(state)
            var_eq_matching = complete(var_eq_matching, maximum(m for m in var_eq_matching.match if isa(m, Int); init=0))
        end
        record_mss!(state, "post_pantelides", MatchedSystemStructure(state.structure, var_eq_matching))

        linear_eqs = index_lowering_ad!(state, ils)
        record_mss!(state, "post_ad", MatchedSystemStructure(state.structure, var_eq_matching))

        @may_timeit debug_config "partial_state_selection_graph!" begin
            var_eq_matching = partial_state_selection_graph!(state.structure, var_eq_matching)
        end
        record_mss!(state, "post_pss", MatchedSystemStructure(state.structure, var_eq_matching))

        @may_timeit debug_config "tearing_schedule!" begin
            tearing_schedule!(state, linear_eqs, ils, var_eq_matching)
        end

        return TransformedIRODESystem(state, var_eq_matching)
    end
end
