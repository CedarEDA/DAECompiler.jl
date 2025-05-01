function index_lowering_ad!(state, ils)
    (; var_to_diff, eq_to_diff, graph, solvable_graph) = state.structure
    linear_eqs = Dict(e=>(ei, 0) for (ei, e) in enumerate(ils.nzrows))

    # Figure out which equations we need to differentiate
    # TODO: Should have some nicer interface in MTK
    diff_ssas = Pair{SSAValue,Int}[]
    for i = 1:length(eq_to_diff)
        # If this is a linear equation, we cannot differentiate it, because
        # alias elimination changed the equation on us, but didn't update the
        # IR. We codegen it directly below.
        islinear = haskey(linear_eqs, i)
        if invview(eq_to_diff)[i] === nothing && eq_to_diff[i] !== nothing && !isempty(𝑠neighbors(graph, eq_to_diff[i]))
            level = 1
            diff = eq_to_diff[i]
            islinear && (linear_eqs[diff] = (linear_eqs[i][1], level))
            while (diff = eq_to_diff[diff]) !== nothing
                level += 1
                islinear && (linear_eqs[diff] = (linear_eqs[i][1], level))
            end
            if !islinear
                for ssa in eqs[i][2]
                    push!(diff_ssas, ssa => level)
                end
            end
        end
    end
end