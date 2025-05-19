using StateSelection.BipartiteGraphs: 𝑑neighbors, 𝑑vertices, 𝑠vertices

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

function maybe_realpath(path::AbstractString)
    try
        return realpath(path)
    catch
        # If the path doesn't exist (e.g. `runtime.jl` doesn't exist in our JuliaHub builds)
        # just return the normal path.
        return path
    end
end

function get_inline_backtrace(ir::IRCode, v::SSAValue)
    frames = Base.StackTrace();
    runtime_jl_path = maybe_realpath(joinpath(dirname(pathof(@__MODULE__)), "runtime.jl"))

    frames = Base.StackTrace();
    for lineinfo in Base.IRShow.buildLineInfoNode(ir.debuginfo, nothing, v.id)
        btpath = maybe_realpath(expanduser(string(lineinfo.file)))
        if btpath != runtime_jl_path
            frame = Base.StackFrame(lineinfo.method, lineinfo.file, lineinfo.line)
            push!(frames, frame)
        end
    end
    return frames
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

function StateSelection.n_concrete_eqs(state::TransformationState, filter)
    neqs = count(e -> !isempty(𝑠neighbors(state.structure.graph, e)) && filter(e), 𝑠vertices(state.structure.graph))
end

# TODO: Why is this different from the one in `StateSelection`?
function StateSelection.check_consistency(state::TransformationState, _)
    neqs = StateSelection.n_concrete_eqs(state, always_eq_filter(state))
    (; graph, var_to_diff) = state.structure
    varfilter = continuous_vars_filter(state)
    highest_vars = StateSelection.computed_highest_diff_variables(state.structure, varfilter)
    n_highest_vars = 0
    for (v, h) in enumerate(highest_vars)
        h || continue
        isempty(𝑑neighbors(graph, v)) && continue
        n_highest_vars += 1
    end
    is_balanced = n_highest_vars == neqs

    if neqs > 0 && !is_balanced
        (eqs, vars) = find_eqs_vars(state)

        varwhitelist = var_to_diff .== nothing
        var_eq_matching = StateSelection.maximal_matching(graph, srcfilter=eq -> true, dstfilter=v -> varwhitelist[v]) # not assigned
        # Just use `error_reporting` to do conditional
        iseqs = n_highest_vars < neqs
        if iseqs
            eq_var_matching = invview(StateSelection.complete(var_eq_matching, nsrcs(graph))) # extra equations
            bad_idxs = findall(isequal(unassigned), @view eq_var_matching[1:nsrcs(graph)])
            names = [nothing for _ in bad_idxs]
        else

            bad_idxs = findall(isequal(unassigned), var_eq_matching)
            # TODO: Restore once the scope sytem is fixed
            names = Union{Nothing, Symbol}[nothing for idx in bad_idxs]
        end

        return BadSystemException(neqs, n_highest_vars, copy(state.result.ir), names,
            iseqs ? map(x->x[1], eqs)[bad_idxs] : vars[bad_idxs] )
    end

    # This is defined to check if Pantelides algorithm terminates. For more
    # details, check the equation (15) of the original paper.
    extended_graph = (@StateSelection.set graph.fadjlist = Vector{Int}[graph.fadjlist;
        map(collect, edges(var_to_diff))])
    extended_var_eq_matching = StateSelection.maximal_matching(extended_graph; dstfilter=varfilter)

    unassigned_var = []
    for (vj, eq) in enumerate(extended_var_eq_matching)
        if eq === unassigned && vj ∈ 𝑑vertices(graph) && !isempty(𝑑neighbors(graph, vj)) && varfilter(vj)
            push!(unassigned_var, vj)
        end
    end

    if !isempty(unassigned_var) || !is_balanced
        (_, vars) = find_eqs_vars(state)

        return BadSystemException(neqs, n_highest_vars, copy(state.result.ir),
            [nothing for bad_idxs in unassigned_var], vars[unassigned_var])
    end
    return nothing
end
