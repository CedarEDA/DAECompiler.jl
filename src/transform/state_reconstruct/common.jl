"makes sure var and obs arguments to `compile_batched_reconstruct_func` are good"
function check_variable_specification_preconditions(tsys, vars, obs)
    # Only allow sorted variable specifications; this avoids us compiling different reconstruction
    # functions for simple permutations in the order in which we ask for variables.
    compact = IncrementalCompact(copy(tsys.state.ir))
    (_, state_vars) = find_eqs_vars(tsys.state, compact)
    for (name, set, max) in (("variable", vars, length(state_vars)), ("observable", obs, tsys.state.nobserved))
        if !isempty(set)
            if !issorted(set) || length(unique(set)) != length(set)
                throw(ArgumentError("$(name)s specification must be sorted and unique: $(set)"))
            end
            if minimum(set) < 1
                throw(ArgumentError("Asking for non-existant $(name) $(minimum(set)) (must be greater than 0)"))
            end
            if maximum(set) > max
                throw(ArgumentError("Asking for non-existant $(name) $(maximum(set)) (highest known: $(max)"))
            end
        end
    end
end


"inserts return values"
function conclude_reconstruct_like!(ir, return_values, du, u, p, t, var_assignment)
    ir = compact!(ir)
    # insert return value
    for idx in 1:length(ir.stmts)
        ssa = SSAValue(idx)
        if ir[ssa][:inst] isa ReturnNode
            ret_ssa = insert_node!(
                ir, ssa,
                NewInstruction(Expr(:call, Core.tuple, return_values...), Any),
                #==attach_after=# false
            )
            ir[ssa][:inst] = ReturnNode(ret_ssa)
            ir[ssa][:type] = Any
        else
            replace_if_intrinsic!(ir, ssa, du, u, p, t, var_assignment)
            ir[ssa][:type] = widenconst(ir[ssa][:type])
        end
    end
    return ir
end

function filter_reconstruction_output_ssas(ir)
    ssas = SSAValue[]
    for ii in 1:length(ir.stmts)
        stmt = ir[SSAValue(ii)][:inst]
        if is_known_invoke(stmt, observed!, ir) || is_solved_variable(stmt)
            push!(ssas, SSAValue(ii))
        end
    end
    return ssas
end