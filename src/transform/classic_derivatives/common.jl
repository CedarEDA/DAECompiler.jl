@inline zero!(x::A) where A<:AbstractArray = fill!(x, zero(eltype(A)))::A
@inline zero!(x::Array{Float64}) = fill!(x, 0.0)

Base.@propagate_inbounds function addindex!(J::AbstractArray, val, inds::Vararg{Int, N}) where N
    J[inds...] += val
    return nothing
end


"Do the things needed to wrap up a function like `jac` or `tgrad` out of the `compact` for the ir into something callable"
function construct_derivative_function(
    function_name, compact, transformed_sys, var_assignment, debug_config, isdae; has_γ=false,
)
    is_tgrad = function_name == :tgrad
    @may_timeit debug_config "construct_derivative_function" begin
        if isdae
            if has_γ
                J, du, u, p, γ, t = Argument.(2:7)
            else
                J, du, u, p, t = Argument.(2:6)
            end
        else
            J, u, p, t = Argument.(2:5)
            du = nothing
        end

        for ((_, idx), stmt) in compact
            ssa = SSAValue(idx)
            if stmt isa ReturnNode
                # Actually return J instead
                compact[ssa][:inst] = ReturnNode(J)
                compact[ssa][:type] = Any
            else
                replace_if_intrinsic!(compact, ssa, du, u, p, t, var_assignment)
                compact[ssa][:type] = widenconst(compact[ssa][:type])
            end
        end
        ir = finish(compact)

        ir = compact!(ir)  # This shouldn't be needed, but without it, it segfaults. See https://github.com/JuliaComputing/DAECompiler.jl/pull/390#issuecomment-1558505420
        record_ir!(transformed_sys.state, "differentiated", ir)

        p_type = parameter_type(get_sys(transformed_sys))
        goldclass_sig = if isdae
            if has_γ
                Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}, p_type, Float64, Float64}
            else
                Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}, p_type, Float64}
            end
        else
            # tgrad doesn't exist for DAEs, and it writes a vector out, not a Matrix.
            if is_tgrad
                Tuple{Vector{Float64}, Vector{Float64}, p_type, Float64}
            else
                Tuple{Matrix{Float64}, Vector{Float64}, p_type, Float64}
            end
        end
        F! = JITOpaqueClosure{function_name, goldclass_sig}() do arg_types...
            state_type = eltype(arg_types[isdae ? 3 : 2])
            breadcrumb_name = "state_type=$(state_type)"
            with_breadcrumb("ir_levels", breadcrumb_name) do
                opt_params = OptimizationParams(inlining=is_tgrad)  # don't inline for jacobians (paramjac, jac, only for tgrad)
                return compile_overload(ir, transformed_sys.state, arg_types; opt_params)
            end
        end
        return F!
    end
end


function filter_output_ssas(input_ir)
    # First pull out all the SSAs we want to take the derivative of.
    # These are exactly the ones that `equation!` or `solved_variable` occur on
    diff_ssas = SSAValue[]
    for ii in 1:length(input_ir.stmts)
        inst = input_ir[SSAValue(ii)]
        stmt = inst[:inst]
        if is_equation_call(stmt, input_ir) || is_solved_variable(stmt)
            push!(diff_ssas, SSAValue(ii))
        end
    end
    return diff_ssas
end



"""
    determine_jacobian_row_and_bundle(ir, stmt, eq_assignment, var_to_diff, var_eq_matching, var_assignment)

from a equation_call or a solved_variable determine the corresponding row in the jacobian,
and the ssavalue for the bundle that should contain the corresponding primal and derivative
"""
function determine_jacobian_row_and_bundle(ir, stmt, eq_assignment, var_to_diff, var_eq_matching, var_assignment)
    if is_solved_variable(stmt)
        var = stmt.args[end-1]
        vint = invview(var_to_diff)[var]
        if vint === nothing || var_eq_matching[vint] !== SelectedState()
            # Solved algebric variable, not used in this lowering
            return nothing, nothing # Not selected, nothing to do.
        end
        # By convention, for the DAE case, we match the index these equations
        # with that of the variable for which it is generated.
        row, in_du = var_assignment[vint]
        @assert !in_du

        bundle = stmt.args[end]
        return row, bundle
    else
        @assert is_equation_call(stmt, ir)
        # Expr(:invoke, (mi, equation, bundle)), or Expr(:call, equation, bundle)
        eq = idnum(argextype(_eq_function_arg(stmt), ir))  # equation number

        # this is the sigular part of mass matrix, so row comes from the equation
        row = eq_assignment[eq]  # row to write from this equation
        if iszero(row)
            @warn "failed to identify jacobian row for equation" row eq (ssa, stmt)
            return nothing, nothing
        end
        bundle = _eq_val_arg(stmt)
        return row, bundle
    end
end