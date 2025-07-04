struct UnoptimizedKey end

function Base.StackTraces.show_custom_spec_sig(io::IO, owner::UnoptimizedKey, linfo::CodeInstance, frame::Base.StackTraces.StackFrame)
    print(io, "Unoptimized transformed IR for ")
    mi = Base.get_ci_mi(linfo)
    return Base.StackTraces.show_spec_sig(io, mi.def, mi.specTypes)
end

function rhs_finish_noopt!(
    state::TransformationState,
    ci::CodeInstance,
    key::UnoptimizedKey,
    world::UInt,
    settings::Settings,
    equation_to_residual_mapping = 1:length(state.structure.eq_to_diff),
    variable_to_state_mapping = map_variables_to_states(state);
    opaque_closure)

    (; result, structure) = state
    result_ci = find_matching_ci(ci -> ci.owner === key, ci.def, world)
    if result_ci !== nothing
        return result_ci
    end

    ir = copy(result.ir)
    src = ci.inferred::AnalyzedSource
    argrange = 1:src.nargs
    # Original arguments.
    slotnames = src.slotnames[argrange]
    argtypes = remove_variable_and_equation_annotations(ir.argtypes)
    if opaque_closure
        slotnames[1] = :captures
        argtypes[1] = Tuple
    end
    # Additional ABI arguments.
    push!(slotnames, :out, :du, :u, :residuals, :states, :t)
    push!(argtypes, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Int}, Vector{Int}, Float64)

    @assert length(slotnames) == length(argtypes)
    append!(empty!(ir.argtypes), argtypes)
    captures, args..., out, du, u, residuals, states, t = Argument.(eachindex(slotnames))

    equations = Pair{Union{Argument, SSAValue}, Eq}[]
    for arg in args
        index = [arg.n]
        i = findfirst(==(index), result.argmap.equations)
        i !== nothing && push!(equations, arg => Eq(i))
    end
    callee_to_caller_eq_map = invert_eq_callee_mapping(result.eq_callee_mapping)
    compact = IncrementalCompact(ir)

    for ((old, i), _) in compact
        ssaidx = SSAValue(i)
        inst = compact[ssaidx]
        stmt = inst[:stmt]
        type = inst[:type]
        line = inst[:line]

        if i == 1
            # @insert_instruction_here(compact, nothing, settings, println("Residuals: ", residuals)::Any)
            # @insert_instruction_here(compact, nothing, settings, println("States: ", states)::Any)
        end

        if is_known_invoke_or_call(stmt, Intrinsics.variable, compact)
            var = idnum(type)
            index = @insert_instruction_here(compact, line, settings, getindex(states, var)::Int)
            value = @insert_instruction_here(compact, line, settings, getindex(u, index)::Float64)
            replace_uses!(compact, (old, inst) => value)
            # @insert_instruction_here(compact, line, settings, println("Variable (", var, "): ", value)::Float64)
        elseif is_known_invoke(stmt, Intrinsics.ddt, compact)
            var = idnum(type)
            index = @insert_instruction_here(compact, line, settings, getindex(states, var)::Int)
            value = @insert_instruction_here(compact, line, settings, getindex(du, index)::Float64)
            replace_uses!(compact, (old, inst) => value)
            # @insert_instruction_here(compact, line, settings, println("Variable derivative (", var, " := ", invview(structure.var_to_diff)[var], "′): ", value)::Any)
        elseif is_known_invoke(stmt, Intrinsics.equation, compact)
            push!(equations, ssaidx => type::Eq)
        elseif is_equation_call(stmt, compact)
            callee, value = stmt.args[2], stmt.args[3]
            i = findfirst(x -> first(x) == callee, equations)::Int
            eq = last(equations[i])
            index = @insert_instruction_here(compact, line, settings, getindex(residuals, eq.id)::Int)
            previous = @insert_instruction_here(compact, line, settings, getindex(out, index)::Float64)
            accumulated = @insert_instruction_here(compact, line, settings, +(previous, value)::Float64)
            ret = @insert_instruction_here(compact, line, settings, setindex!(out, accumulated, index)::Any)
            replace_uses!(compact, (old, inst) => ret)
            # @insert_instruction_here(compact, line, settings, println("Residuals (index = ", index, ", value = ", value, "): ", residuals)::Any)
        elseif is_known_invoke_or_call(stmt, Intrinsics.sim_time, compact)
            inst[:stmt] = t
        elseif is_known_invoke_or_call(stmt, Intrinsics.epsilon, compact)
            inst[:stmt] = 0.0
        elseif isexpr(stmt, :invoke)
            info = inst[:info]::MappingInfo
            callee_ci, args = stmt.args[1]::CodeInstance, @view stmt.args[2:end]
            callee_result = structural_analysis!(callee_ci, world, settings)
            callee_structure = make_structure_from_ipo(callee_result)
            callee_state = TransformationState(callee_result, callee_structure)

            caller_eqs = get(Vector{Int}, callee_to_caller_eq_map, StructuralSSARef(old))
            callee_residuals = equation_to_residual_mapping[caller_eqs]
            caller_variables = map(info.mapping.var_coeffs) do coeff
                isa(coeff, Incidence) || return -1
                nnz(coeff.row) == 1 || return -1
                idnum(coeff)
            end
            callee_states = [get(variable_to_state_mapping, i, -1) for i in caller_variables]

            callee_daef_ci = rhs_finish_noopt!(callee_state, callee_ci, UnoptimizedKey(), world, settings, callee_residuals, callee_states; opaque_closure = false)
            call = @insert_instruction_here(compact, line, settings, (:invoke)(callee_daef_ci, args...,
                out,
                du,
                u,
                @insert_instruction_here(compact, line, settings, Base.vect(callee_residuals...)::Vector{Int}),
                @insert_instruction_here(compact, line, settings, Base.vect(callee_states...)::Vector{Int}),
                t)::type)
            replace_uses!(compact, (old, inst) => call)
            isa(type, Eq) && push!(equations, call => type)
        end

        type = inst[:type]
        if isa(type, Incidence) || isa(type, Eq)
            inst[:type] = widenconst(type)
        end
    end

    daef_ci = rhs_finish_ir!(Compiler.finish(compact), ci, settings, key, slotnames)
    # @sshow daef_ci.inferred
    return daef_ci
end

function map_variables_to_states(state::TransformationState)
    (; structure) = state
    diff_to_var = invview(structure.var_to_diff)
    states = Int[]
    prev_state = 0
    for var in continuous_variables(state)
        ref = is_differential_variable(structure, var) ? diff_to_var[var] : var
        state = @something(get(states, ref, nothing), prev_state += 1)
        push!(states, state)
    end
    return states
end

function replace_uses!(compact, ((old, inst), new))
    inst[:stmt] = nothing
    compact.ssa_rename[old] = new
end

function sciml_to_internal_abi_noopt!(ir::IRCode, state::TransformationState, internal_ci::CodeInstance, settings::Settings)
    slotnames = [:captures, :out, :du, :u, :p, :t]
    captures, out, du, u, p, t = Argument.(eachindex(slotnames))

    empty!(ir.argtypes)
    push!(ir.argtypes, Tuple) # opaque closure captures
    append!(ir.argtypes, fieldtypes(SCIML_ABI))

    compact = IncrementalCompact(ir)
    line = ir[SSAValue(1)][:line]

    internal_oc = @insert_instruction_here(compact, line, settings, getfield(captures, 1)::Core.OpaqueClosure)
    neqs = length(state.structure.eq_to_diff)
    nvars = length(state.structure.var_to_diff)
    residuals = @insert_instruction_here(compact, line, settings, getindex(Int, 1:neqs...)::Vector{Int})
    states = @insert_instruction_here(compact, line, settings, getindex(Int, map_variables_to_states(state)...)::Vector{Int})
    @insert_instruction_here(compact, line, settings, (:invoke)(internal_ci, internal_oc, out, du, u, residuals, states, t)::Nothing)
    @insert_instruction_here(compact, line, settings, (return nothing)::Union{})

    ir = Compiler.finish(compact)
    maybe_rewrite_debuginfo!(ir, settings)
    resize!(ir.cfg.blocks, 1)
    empty!(ir.cfg.blocks[1].succs)
    Compiler.verify_ir(ir)

    return Core.OpaqueClosure(ir; slotnames)
end
