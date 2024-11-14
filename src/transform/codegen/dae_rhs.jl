const VectorViewType = SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int}}, true}

function dae_finish_ipo!(
    result::DAEIPOResult,
    ci::CodeInstance,
    key::TornCacheKey,
    world::UInt,
    ordinal::Int,
    indexT=Int)

    if haskey(result.dae_finish_cache, key)
        return result.dae_finish_cache[key][ordinal]
    end

    @show ci.def

    allow_unassigned = false

    @show result.tearing_cache
    torn = result.tearing_cache[key]
    rhs_ms = nothing
    old_daef_mi = nothing
    assigned_slots = falses(length(result.total_incidence))

    cis = Vector{CodeInstance}()
    for (ir_ordinal, ir) in enumerate(torn.ir_seq)
        ir = torn.ir_seq[ir_ordinal]

        # Read in from the last level before any DAE or ODE-specific `ir_levels`
        # We assume this is named `tearing_schedule!`
        ir = copy(ir)
        empty!(ir.argtypes)
        push!(ir.argtypes, Tuple)  # SICM State
        push!(ir.argtypes, Tuple)  # in vars

        arg_range = 3:8
        out_du_mm, out_eq, in_u_mm, in_u_unassgn, in_du_unassgn, in_alg =
            Argument.(arg_range)
        for arg in arg_range
            push!(ir.argtypes, VectorViewType)
        end

        t = Argument(last(arg_range)+1)
        push!(ir.argtypes, Float64)  #  t


        processed_variables = BitSet()
        var_assignment = Vector{Union{Nothing, Int}}(nothing, length(result.var_to_diff))

        diff_states_in_callee = BitSet()

        for i = 1:length(ir.stmts)
            inst = ir[SSAValue(i)]
            stmt = inst[:stmt]
            info = inst[:info]

            if isa(info, Compiler.ConstCallInfo) && any(result->isa(result, Compiler.SemiConcreteResult), info.results)
                # Drop any semi-concrete results from the DAE-interpreter. We will redo
                # them with the native interpreter to avoid getting suboptimal codegen.
                ir[SSAValue(i)][:info] = info.call
                ir[SSAValue(i)][:flag] |= Compiler.IR_FLAG_REFINED
            end

            if isexpr(stmt, :invoke) && isa(stmt.args[1], Tuple)
                info::MappingInfo
                callee_ci = stmt.args[1][1]
                closure_env = stmt.args[2]
                in_vars = stmt.args[3]
                if isa(callee_ci, MethodInstance)
                    callee_ci = Compiler.get(Compiler.code_cache(interp), callee_ci, nothing)
                end

                @assert callee_ci !== nothing

                spec_data = stmt.args[1]
                callee_key = stmt.args[1][2]
                callee_ordinal = stmt.args[1][end]::Int
                callee_result = structural_analysis!(callee_ci, world)
                callee_daef_ci = dae_finish_ipo!(callee_result, callee_ci, callee_key, world, callee_ordinal)
                # Allocate a continuous block of variables for all callee alg and diff states

                @show callee_daef_ci.rettype

                empty!(stmt.args)
                push!(stmt.args, callee_daef_ci)
                push!(stmt.args, closure_env)
                push!(stmt.args, in_vars)

                # Ordering from tearing is (AssignedDiff, UnassignedDiff, Algebraic, Explicit)
                for (arg, range_idx) in zip(arg_range, (1, 4, 1, 2, 2, 3))
                    push!(stmt.args, insert_node!(ir, SSAValue(i),
                        NewInstruction(inst;
                        stmt=Expr(:call, view, Argument(arg), spec_data[2+range_idx]),
                        type=VectorViewType)))
                end

                # TODO: Track whether the system is autonomous
                push!(stmt.args, t)
            end

            if is_known_invoke(stmt, variable, ir) || is_equation_call(stmt, ir)
                display(ir)
                error()
            elseif is_known_invoke_or_call(stmt, InternalIntrinsics.state, ir)
                kind = stmt.args[end]::StateKind
                slot = stmt.args[end-1]
                which = kind == AssignedDiff        ? in_u_mm :
                        kind == UnassignedDiff      ? in_u_unassgn :
                        kind == AlgebraicDerivative ? in_du_unassgn :
                        kind == Algebraic           ? in_alg : error()
                replace_call!(ir, SSAValue(i), Expr(:call, Base.getindex, which, slot))
            elseif is_known_invoke_or_call(stmt, InternalIntrinsics.contribution!, ir)
                slot = stmt.args[end-2]::Int
                kind = stmt.args[end-1]::EquationStateKind
                red = stmt.args[end]
                which = kind == StateDiff ? out_du_mm :
                        kind == Explicit  ? out_eq : error()
                prev = insert_node!(ir, SSAValue(i), NewInstruction(inst; stmt=Expr(:call, Base.getindex, which, slot), type=Float64))
                sum = insert_node!(ir, SSAValue(i), NewInstruction(inst; stmt=Expr(:call, +, prev, red), type=Float64))
                replace_call!(ir, SSAValue(i), Expr(:call, Base.setindex!, which, sum, slot))
            elseif is_known_invoke(stmt, equation, ir)
                # Equation - used, but only as an arg to equation call, which will all get
                # eliminated by the end of this loop, so we can delete this statement, as
                # long as we don't touch the type yet.
                ir[SSAValue(i)][:inst] = Intrinsics.placeholder_equation
            elseif is_solved_variable(stmt)
                # Not used in this lowering
                ir[SSAValue(i)] = nothing
        else
                replace_if_intrinsic!(ir, SSAValue(i), nothing, nothing, Argument(1), t, var_assignment)
            end
        end

        # Just before the end of the function
        idx = length(ir.stmts)
        function ir_add!(a, b)
            ni = NewInstruction(Expr(:call, +, a, b), Any, ir[SSAValue(idx)][:line])
            insert_node!(ir, idx, ni)
        end
        ir = Compiler.compact!(ir)

        widen_extra_info!(ir)
        src = ir_to_src(ir)

        abi = Tuple{Tuple, Tuple, (VectorViewType for _ in arg_range)..., Float64}
        daef_ci = cache_dae_ci!(ci, src, src.debuginfo, abi, RHSSpec(key, ir_ordinal))
        ccall(:jl_add_codeinst_to_jit, Cvoid, (Any, Any), daef_ci, src)

        push!(cis, daef_ci)
    end

    result.dae_finish_cache[key] = cis

    return cis[ordinal]
end
