using .Intrinsics: Scope
using Compiler: bbidxiter, isexpr, is_known_call

function find_matching_ci(predicate, mi::MethodInstance, world::UInt)
    if isdefined(mi, :cache)
        result_ci = mi.cache
        while true
            if result_ci.min_world <= world && world <= result_ci.max_world && predicate(result_ci)
                return result_ci
            end
            isdefined(result_ci, :next) || break
            result_ci = result_ci.next
        end
    end

    return nothing
end

function structural_analysis!(ci::CodeInstance, world::UInt, settings::Settings)
    # Check if we have aleady done this work - if so return the cached result
    result_ci = find_matching_ci(ci->ci.owner == StructureCache(settings), ci.def, world)
    if result_ci !== nothing
        return result_ci.inferred
    end

    result = _structural_analysis!(ci, world, settings)
    # TODO: The world bounds might have been narrowed
    cache_dae_ci!(ci, result, nothing, nothing, StructureCache(settings))

    return result
end

struct EqVarState
    var_to_diff
    varclassification
    varkinds
    total_incidence
    eqclassification
    eqkinds
    eq_callee_mapping
end

function _structural_analysis!(ci::CodeInstance, world::UInt, settings::Settings)
    # Variables
    var_to_diff = DiffGraph(0)
    varclassification = VarEqClassification[]
    varssa = Union{SSAValue, Argument}[]
    varkinds = Intrinsics.VarKind[]
    function add_variable!(i::Union{SSAValue, Argument})
        v = add_vertex!(var_to_diff)
        push!(varclassification, isa(i, Argument) ? External : Owned)
        push!(varssa, i)
        push!(varkinds, Intrinsics.RemovedVar)
        return v
    end

    # Equations
    eqclassification = VarEqClassification[]
    eqssas = Union{SSAValue, Argument}[]
    eqkinds = Intrinsics.EqKind[]
    function add_equation!(i::Union{SSAValue, Argument})
        push!(eqssas, i)
        push!(eqclassification, isa(i, Argument) ? External : Owned)
        push!(eqkinds, Intrinsics.RemovedEq)
        return length(eqssas)
    end

    # Scopes
    nsysmscopes = 0
    function add_scope!(i::Union{SSAValue, Argument})
        nsysmscopes += 1
        return nsysmscopes
    end

    # Get the IR
    ir = copy(ci.inferred.ir)

    # IR Warnings - this tracks cases of malformed optional information. The system
    # is still malformed, but something is likely wrong. These will get printed if
    # a system that requires this function is run, but will not be fatal.
    warnings = BadDAECompilerInputException[]

    compact = IncrementalCompact(ir)
    argmap = ArgumentMap(ir.argtypes)
    nexternalargvars = length(argmap.variables)
    nexternaleqs = length(argmap.equations)
    if !settings.skip_optimizations
        state = FlatteningState(compact, settings, argmap)
        arg_replacements = flatten_arguments!(state)
        if arg_replacements === nothing
            return UncompilableIPOResult(warnings, UnsupportedIRException("Unhandled argument types", Compiler.finish(compact)))
        end
        argtypes = Any[Incidence(ir.argtypes[i], i) for i = 1:nexternalargvars]
    else
        argtypes = annotate_variables_and_equations(ir.argtypes, argmap)
        arg_replacements = nothing
    end

    for i = 1:nexternalargvars
        # TODO: Need to handle different var kinds for IPO
        # TODO: Don't use `Argument` when we don't flatten, maybe something
        # like an `ArgumentView` with composite indices from the `ArgumentMap`.
        add_variable!(Argument(i))
    end
    for i = 1:nexternaleqs
        # Not technically an argument, but let's use it for now
        add_equation!(Argument(i))
    end

    # Allocate variable and equation numbers of any incoming arguments
    refiner = StructuralRefiner(world, settings, var_to_diff, varkinds, varclassification, eqkinds, eqclassification)

    # Go through the IR, annotating each intrinsic with an appropriate taint
    # source lattice element.
    externally_refined = BitSet()
    for ((old_idx, i), stmt) in compact
        urs = userefs(stmt)
        compact[SSAValue(i)] = nothing
        if arg_replacements !== nothing
            for ur in urs
                if isa(ur[], Argument)
                    repl = arg_replacements[ur[].n]
                    ur[] = repl
                end
            end
        end
        stmt = urs[]
        compact[SSAValue(i)] = stmt
        if isexpr(stmt, :invoke)
            if is_known_invoke(stmt, variable, compact)
                v = add_variable!(SSAValue(i))
                compact[SSAValue(i)][:type] = Incidence(v)
                push!(externally_refined, i)
            elseif is_known_invoke(stmt, equation, compact)
                compact[SSAValue(i)][:type] = Eq(add_equation!(SSAValue(i)))
                push!(externally_refined, i)
            elseif is_known_invoke(stmt, sim_time, compact)
                compact[SSAValue(i)][:type] = Incidence(0)
                push!(externally_refined, i)
            else
                compact[SSAValue(i)][:flag] |= Compiler.IR_FLAG_REFINED
            end
        elseif isexpr(stmt, :call)
            if is_known_call(stmt, Core.current_scope, compact)
                # N.B.: We make the assumption here that all current_scope that
                # was inside EnterScope within the same function has already
                # been folded by SROA, so the only thing left are those that
                # refer to the function's entry scope.
                compact[SSAValue(i)][:type] = cur_scope_lattice
                compact[SSAValue(i)][:flag] |= Compiler.IR_FLAG_REFINED
                push!(externally_refined, i)
                continue
            end
        elseif isexpr(stmt, :code_coverage_effect) || isexpr(stmt, :throw_undef_if_not)
            continue
        elseif isa(stmt, ReturnNode)
            # Nothing to do for the moment
            continue
        elseif isexpr(stmt, :foreigncall)
            # Could be bad, but we'll just ignore it for now.
            # One of the testcases wants to allocate an array
            continue
        elseif isa(stmt, GlobalRef)
            continue
        elseif isexpr(stmt, :new)
            newT = argextype(stmt.args[1], compact)
            if isa(newT, Const) && newT.val === Intrinsics.ScopeIdentity
                # Allocate the identity now. After inlining, we're guaranteed that
                # every Expr(:new) uniquely corresponds to a scope identity, so this
                # is legal here (but not before)
                compact[SSAValue(i)][:stmt] = Intrinsics.ScopeIdentity()
                compact[SSAValue(i)][:flag] |= Compiler.IR_FLAG_REFINED
            end
            continue
        elseif isexpr(stmt, :splatnew)
            continue
        elseif isexpr(stmt, :boundscheck)
            compact[SSAValue(i)][:type] = Incidence(Bool)
            compact[SSAValue(i)][:flag] |= Compiler.IR_FLAG_REFINED
        elseif isa(stmt, PhiNode)
            # Take into account control-dependent taint
            compact[SSAValue(i)][:flag] |= Compiler.IR_FLAG_REFINED
        elseif isa(stmt, PiNode)
            continue
        elseif isa(stmt, GotoIfNot) || isa(stmt, GotoNode)
            continue
        elseif stmt === nothing
            continue
        else
            @sshow stmt
        end
    end
    ir = Compiler.finish(compact)

    # Perform the actual dataflow analysis
    mi = Compiler.get_ci_mi(ci)
    (nargs, isva) = isa(mi.def, Method) ? (mi.def.nargs, mi.def.isva) : (0, false)
    method_info = Compiler.SpecInfo(nargs, isva, #=propagate_inbounds=#true, nothing)
    irsv = Compiler.IRInterpretationState(refiner, method_info, ir, mi, argtypes,
                                    world, ci.min_world, ci.max_world)
    ultimate_rt, _ = Compiler.ir_abstract_constant_propagation(refiner, irsv; externally_refined)

    if ultimate_rt === Union{}
        return UncompilableIPOResult(warnings, FunctionErrorsException())
    end

    # For easier debugging, delete all the statements that are dead, but don't renumber things
    for (i, bb) in enumerate(ir.cfg.blocks)
        (length(bb.preds) == 0 && i != 1) || continue
        for i in bb.stmts
            ir[SSAValue(i)] = nothing
            ir[SSAValue(i)][:type] = Union{}
        end
    end

    # Go through each variable we previously identified and record the (post-propagation scope and kind)
    # For this we inspect the `kind` and `scope` arguments from `variable(kind, scope)` invokes, then
    # we turn them into `nothing` when we are done.
    for (ivar, ssa) in enumerate(varssa)
        isa(ssa, SSAValue) || continue
        inst = ir[ssa][:inst]::Union{Nothing, Expr}
        inst === nothing && continue # variable unused and was deleted
        @assert is_known_invoke(inst, variable, ir)
        type = ir[ssa][:type]
        var_num = idnum(type)::Int
        @assert ivar == var_num

        kind = argextype(inst.args[3], ir)
        if !isa(kind, Const) || !isa(kind.val, Intrinsics.VarKind) || kind.val === Intrinsics.RemovedVar
            return UncompilableIPOResult(warnings, UnsupportedIRException("Saw invalid variable kind (`$kind`)  for variable $(var_num) (SSA $ssa)", ir))
        end
        varkinds[var_num] = kind.val
        dv = var_num
        while true
            dv = var_to_diff[dv]
            dv === nothing && break
            varkinds[dv] = kind.val
        end

        scope = argextype(inst.args[4], ir)
        if (!isa(scope, Const) || !isa(scope.val, Intrinsics.AbstractScope)) && !is_valid_partial_scope(scope)
            push!(warnings,
                UnsupportedIRException(
                    "Saw non-constant name (`$scope`) for variable $(var_num) (SSA $ssa)",
                    ir))
        elseif isa(scope, Const) && scope.val === Scope()
            # Explicitly unnamed
        else
            record_scope!(ir, warnings, names, scope, ScopeDictEntry(true, var_num))
        end

        if !settings.skip_optimizations
            # Delete - we've recorded this into our our side table, we don't need to
            # keep it around in the IR
            inst.args[3] = nothing
            inst.args[4] = nothing
        end
    end

    # Do the same for equations
    for (ieq, ssa) in enumerate(eqssas)
        isa(ssa, Argument) && continue
        inst = ir[ssa][:inst]::Union{Nothing, Expr}
        inst === nothing && continue # equation unused and was deleted
        @assert is_known_invoke(inst, equation, ir)
        type = ir[ssa][:type]
        eq_num = idnum(type)::Int
        @assert ieq == eq_num

        kind = argextype(inst.args[3], ir)
        if !isa(kind, Const) || !isa(kind.val, Intrinsics.EqKind) || kind.val === Intrinsics.RemovedEq
            return UncompilableIPOResult(warnings, UnsupportedIRException("Saw invalid equation kind (`$kind`)  for equation $(eq_num) (SSA $ssa)", ir))
        end
        eqkinds[eq_num] = kind.val

        scope = argextype(inst.args[4], ir)
        if (!isa(scope, Const) || !isa(scope.val, Intrinsics.AbstractScope)) && !is_valid_partial_scope(scope)
            push!(warnings,
                UnsupportedIRException(
                    "Saw non-constant name (`$scope`) for equation $(eq_num) (SSA $ssa)",
                    ir))
        elseif isa(scope, Const) && scope.val === Scope()
            # Explicitly unnamed
        else
            record_scope!(ir, warnings, names, scope, ScopeDictEntry(false, eq_num))
        end

        if !settings.skip_optimizations
            # Delete - we've recorded this into our our side table, we don't need to
            # keep it around in the IR
            inst.args[3] = nothing
            inst.args[4] = nothing
        end
    end

    # Now record the association of (::equation)() calls with the equations that they originate from
    total_incidence = Vector{Any}(undef, length(eqkinds))

    # Now go through and incorporate the structural information from any interior calls
    eq_callee_mapping = Vector{Union{Nothing, Vector{Pair{StructuralSSARef, Int}}}}(nothing, length(eqkinds))
    handler_info = Compiler.compute_trycatch(ir)
    ncallees = 0
    compact = IncrementalCompact(ir)
    opaque_eligible = isempty(total_incidence) && all(==(External), varclassification)
    for ((old_idx, i), stmt) in compact
        stmt === nothing && continue
        # No need to process error paths - even if they were to contain intrinsics, such intrinsics would have
        # no effect.
        compact[SSAValue(i)][:type] === Union{} && continue
        isexpr(stmt, :invoke) || continue
        is_known_invoke(stmt, variable, compact) && continue
        is_known_invoke(stmt, equation, compact) && continue
        is_known_invoke(stmt, InternalIntrinsics.external_equation, compact) && continue
        is_known_invoke(stmt, sim_time, compact) && continue
        is_known_invoke(stmt, ddt, compact) && continue
        if is_equation_call(stmt, compact)
            eqeq = argextype(stmt.args[2], compact)

            if !isa(eqeq, Eq)
                return UncompilableIPOResult(warnings, UnsupportedIRException("Equation call at $(SSAValue(i)) has unknown equation reference.", ir))
            end
            ieq = eqeq.id

            eqssaval = stmt.args[3]
            if !isa(eqssaval, SSAValue) && !isa(eqssaval, Argument)
                if !iszero(eqssaval)
                    return UncompilableIPOResult(warnings, UnsupportedIRException(
                        "Equation call for $ieq at $ssa is set to $eqssaval. The system is unsolvable.", ir))
                end
                continue
            end

            inc = argextype(eqssaval, compact)
            if !isa(inc, Incidence)
                return UncompilableIPOResult(warnings, UnsupportedIRException("Expected incidence analysis to produce result for $eqssaval, got $inc", ir))
            end
            if isassigned(total_incidence, ieq)
                total_incidence[ieq] += inc
            else
                total_incidence[ieq] = inc
            end
            continue
        end

        inst = compact[SSAValue(i)]
        stmtype = inst[:type]
        stmtflags = inst[:flag]
        line = inst[:line]

        info = inst[:info]
        callee_codeinst = stmt.args[1]
        if isa(info, MappingInfo)
            (; result, mapping) = info
        else
            result = structural_analysis!(callee_codeinst, Compiler.get_inference_world(refiner), settings)

            if isa(result, UncompilableIPOResult)
                # TODO: Stack trace?
                return result
            end

            callee_argtypes = Any[argextype(stmt.args[i], compact) for i in 2:length(stmt.args)]
            mapping = CalleeMapping(Compiler.optimizer_lattice(refiner), callee_argtypes, callee_codeinst, result)
            inst[:info] = info = MappingInfo(info, result, mapping)
        end

        if result.opaque_eligible
            compact[SSAValue(i)] = nothing
            compact[SSAValue(i)] = Expr(:call, stmt.args[2:end]...)
            continue
        else
            opaque_eligible = false
        end

        if !settings.skip_optimizations
            # Rewrite to flattened ABI
            compact[SSAValue(i)] = nothing
            compact.result_idx -= 1
            callee_argtypes = callee_codeinst.inferred.ir.argtypes
            callee_argmap = ArgumentMap(callee_argtypes)
            args = @view(stmt.args[2:end])
            𝕃 = Compiler.optimizer_lattice(refiner)
            new_args = flatten_arguments_for_callee!(compact, callee_argmap, callee_argtypes, args, line, settings, 𝕃)
            new_call = insert_instruction_here!(compact, settings, @__SOURCE__,
            NewInstruction(Expr(:invoke, (StructuralSSARef(compact.result_idx), callee_codeinst), new_args...), stmtype, info, line, stmtflags))
            compact.ssa_rename[compact.idx - 1] = new_call
            ssa = StructuralSSARef(new_call.id)
        else
            ssa = StructuralSSARef(i)
        end

        cms = CallerMappingState(result, refiner.var_to_diff, refiner.varclassification, refiner.varkinds, eqclassification, eqkinds)
        err = add_internal_equations_to_structure!(refiner, cms, total_incidence, eq_callee_mapping,
            ssa, result, mapping)
        if err !== true
            return UncompilableIPOResult(warnings, UnsupportedIRException(err, ir))
        end
    end

    eqvars = EqVarState(var_to_diff, varclassification, varkinds,
        total_incidence, eqclassification, eqkinds, eq_callee_mapping)

    # Replace non linear return by a new variable and return that variable
    if !opaque_eligible && !settings.skip_optimizations
        last_ssa = SSAValue(compact.result_idx - 1)
        ret_stmt_inst = compact[last_ssa]
        ret_stmt = ret_stmt_inst[:stmt]
        @assert isa(ret_stmt, ReturnNode)
        line = ret_stmt_inst[:line]
        Compiler.delete_inst_here!(compact)

        (new_ret, ultimate_rt) = rewrite_ipo_return!(Compiler.typeinf_lattice(refiner), compact, line, settings, ret_stmt.val, ultimate_rt, eqvars)
        insert_instruction_here!(compact, settings, @__SOURCE__, NewInstruction(ReturnNode(new_ret), ultimate_rt, Compiler.NoCallInfo(), line, Compiler.IR_FLAG_REFINED), reverse_affinity = true)
    elseif isa(ultimate_rt, Type)
        # If we don't have any internal variables (in which case we might have to to do a more aggressive rewrite), strengthen the incidence
        # by demoting to full incidence over the argument variables. Incidence is not allowed to propagate through global mutable state, so
        # the incidence of the return type is bounded by the incidence of the arguments in this case.
        ultimate_rt = Incidence(ultimate_rt,
            IncidenceVector(MAX_EQS, Int[1:length(varclassification)+1;], IncidenceValue[nonlinear for _ in 1:length(varclassification)+1]))
    end

    ir = Compiler.finish(compact)

    var_to_diff = StateSelection.complete(var_to_diff)

    names = OrderedDict{Any, ScopeDictEntry}()
    return DAEIPOResult(ir, opaque_eligible, ultimate_rt, argtypes, argmap,
        nexternalargvars,
        nsysmscopes,
        nexternaleqs,
        ncallees,
        var_to_diff,
        varclassification,
        total_incidence, eqclassification, eq_callee_mapping,
        names,
        varkinds,
        eqkinds,
        warnings)
end

function rewrite_ipo_return!(𝕃, compact::IncrementalCompact, line, settings, ssa, ultimate_rt::Any, eqvars::EqVarState)
    if isa(ultimate_rt, Eq)
        return Pair{Any, Any}(ssa, ultimate_rt)
    end

    if isa(ultimate_rt, PartialStruct)
        new_fields = Any[]
        new_types = Any[]
        for i = 1:length(ultimate_rt.fields)
            ssa_type = Compiler.getfield_tfunc(𝕃, ultimate_rt, Const(i))
            ssa_field = insert_instruction_here!(compact, settings, @__SOURCE__,
                NewInstruction(Expr(:call, getfield, ssa, i), ssa_type, Compiler.NoCallInfo(), line, Compiler.IR_FLAG_REFINED), reverse_affinity = true)

            (new_field, new_type) = rewrite_ipo_return!(𝕃, compact, line, settings, ssa_field, ssa_type, eqvars)
            push!(new_fields, new_field)
            push!(new_types, new_type)
        end
        newT = Compiler.PartialStruct(ultimate_rt.typ, new_types)
        if widenconst(ultimate_rt) <: Tuple
            retssa = insert_instruction_here!(compact, settings, @__SOURCE__,
                NewInstruction(Expr(:call, tuple, new_fields...), newT, Compiler.NoCallInfo(), line, Compiler.IR_FLAG_REFINED), reverse_affinity = true)
        else
            T = insert_instruction_here!(compact, settings, @__SOURCE__,
                NewInstruction(Expr(:call, typeof, ssa), Type, Compiler.NoCallInfo(), line, Compiler.IR_FLAG_REFINED), reverse_affinity = true)

            retssa = insert_instruction_here!(compact, settings, @__SOURCE__,
                NewInstruction(Expr(:new, T, new_fields...), newT, Compiler.NoCallInfo(), line, Compiler.IR_FLAG_REFINED), reverse_affinity = true)
        end
        return Pair{Any, Any}(retssa, newT)
    end

    if !isa(ultimate_rt, Incidence) || (nnz(ultimate_rt.row) <= 1 && only(nonzeros(ultimate_rt.row)) != nonlinear)
        return Pair{Any, Any}(ssa, ultimate_rt)
    end

    nonlinrepl = add_vertex!(eqvars.var_to_diff)
    push!(eqvars.varclassification, External)
    push!(eqvars.varkinds, Intrinsics.Continuous)

    new_var_ssa = insert_instruction_here!(compact, settings, @__SOURCE__,
        NewInstruction(Expr(:invoke, nothing, variable), Incidence(nonlinrepl), Compiler.NoCallInfo(), line, Compiler.IR_FLAG_REFINED); reverse_affinity = true)

    eq_incidence = ultimate_rt - Incidence(nonlinrepl)
    push!(eqvars.total_incidence, eq_incidence)
    push!(eqvars.eq_callee_mapping, nothing)
    push!(eqvars.eqclassification, Owned)
    push!(eqvars.eqkinds, Intrinsics.Always)
    new_eq = length(eqvars.total_incidence)

    new_eq_ssa = insert_instruction_here!(compact, settings, @__SOURCE__,
        NewInstruction(Expr(:invoke, nothing, equation), Eq(new_eq), Compiler.NoCallInfo(), line, Compiler.IR_FLAG_REFINED); reverse_affinity = true)

    eq_val_ssa = insert_instruction_here!(compact, settings, @__SOURCE__,
        NewInstruction(Expr(:call, InternalIntrinsics.assign_var, new_var_ssa, ssa), eq_incidence, Compiler.NoCallInfo(), line, Compiler.IR_FLAG_REFINED); reverse_affinity = true)

    eq_call_ssa = insert_instruction_here!(compact, settings, @__SOURCE__,
        NewInstruction(Expr(:invoke, nothing, new_eq_ssa, eq_val_ssa), Nothing, Compiler.NoCallInfo(), line, Compiler.IR_FLAG_REFINED); reverse_affinity = true)

    T = widenconst(ultimate_rt)
    # TODO: We don't have a way to express that the return value is directly this variable for arbitrary types
    return Pair{Any, Any}(new_var_ssa, T === Float64 ? Incidence(nonlinrepl) : Incidence(T, nonlinrepl))
end

function add_internal_equations_to_structure!(refiner::StructuralRefiner, cms::CallerMappingState, total_incidence,
        eq_callee_mapping::Vector{Union{Nothing, Vector{Pair{StructuralSSARef, Int}}}}, thisssa::StructuralSSARef, callee_result::DAEIPOResult, callee_mapping::CalleeMapping)
    for i in 1:length(callee_mapping.var_coeffs)
        if !isassigned(callee_mapping.var_coeffs, i)
            compute_missing_coeff!(callee_mapping.var_coeffs, cms, i)
        end
    end

    eq_offset = length(total_incidence)
    if isempty(callee_result.total_incidence) && isempty(callee_result.names)
        return true
    end

    for eq = 1:length(callee_result.eqclassification)
        mapped_eq = callee_mapping.eqs[eq]
        mapped_eq == 0 && continue
        if !isassigned(callee_result.total_incidence, eq)
            # This equation has no contributions in the callee - the only reason
            # we're here is because it leaked an explicit reference.
            continue
        end
        mapped_inc = apply_linear_incidence(Compiler.typeinf_lattice(refiner), callee_result.total_incidence[eq], cms, callee_mapping)
        if isassigned(total_incidence, mapped_eq)
            total_incidence[mapped_eq] = tfunc(Val(Core.Intrinsics.add_float),
                total_incidence[mapped_eq],
                mapped_inc)
        else
            total_incidence[mapped_eq] = mapped_inc
        end
        if eq_callee_mapping[mapped_eq] === nothing
            eq_callee_mapping[mapped_eq] = []
        end
        push!(eq_callee_mapping[mapped_eq], thisssa=>eq)
    end

    for ieq in 1:length(callee_result.total_incidence)
        callee_mapping.eqs[ieq] == 0 || continue
        callee_result.eqclassification[ieq] === External && continue
        isassigned(callee_result.total_incidence, ieq) || continue
        inc = callee_result.total_incidence[ieq]
        extinc = apply_linear_incidence(Compiler.typeinf_lattice(refiner), inc, cms, callee_mapping)
        if !isa(extinc, Incidence) && !isa(extinc, Const)
            return "Failed to map internal incidence for equation $ieq (internal result $inc) - got $extinc while processing $thisssa"
        end
        push!(total_incidence, extinc)
        push!(eq_callee_mapping, [thisssa=>ieq])
        push!(cms.caller_eqclassification, CalleeInternal)
        push!(cms.caller_eqkinds, callee_result.eqkinds[ieq])
        callee_mapping.eqs[ieq] = length(total_incidence)
    end

    return true
end
