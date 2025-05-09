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

function structural_analysis!(ci::CodeInstance, world::UInt)
    # Check if we have aleady done this work - if so return the cached result
    result_ci = find_matching_ci(ci->ci.owner == StructureCache(), ci.def, world)
    if result_ci !== nothing
        return result_ci.inferred
    end

    result = _structural_analysis!(ci, world)
    # TODO: The world bounds might have been narrowed
    cache_dae_ci!(ci, result, nothing, nothing, StructureCache())

    return result
end

function _structural_analysis!(ci::CodeInstance, world::UInt)
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
    eqssas = Pair{Union{SSAValue, Argument}, Vector{SSAValue}}[]
    eqkinds = Intrinsics.EqKind[]
    function add_equation!(i::Union{SSAValue, Argument})
        push!(eqssas, i=>SSAValue[])
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

    # Allocate variable and equation numbers of any incoming arguments
    refiner = StructuralRefiner(world, var_to_diff, varkinds, varclassification)
    argtypes = Any[make_argument_lattice_elem(Compiler.typeinf_lattice(refiner), Argument(i), argt, add_variable!, add_equation!, add_scope!) for (i, argt) in enumerate(ir.argtypes)]
    nexternalvars = length(var_to_diff)
    nexternaleqs = length(eqssas)

    # (::equation)(...) calls
    eqcallssas = SSAValue[]

    # IR Warnings - this tracks cases of malformed optional information. The system
    # is still malformed, but something is likely wrong. These will get printed if
    # a system that requires this function is run, but will not be fatal.
    warnings = UnsupportedIRException[]

    # Go through the IR, annotating each intrinsic with an appropriate taint
    # source lattice element.
    externally_refined = BitSet()
    for (bb, i) in bbidxiter(ir)
        stmt = ir.stmts[i][:inst]
        if isexpr(stmt, :invoke)
            if is_known_invoke(stmt, variable, ir)
                v = add_variable!(SSAValue(i))
                ir.stmts[i][:type] = Incidence(v)
                push!(externally_refined, i)
            elseif is_known_invoke(stmt, equation, ir)
                ir.stmts[i][:type] = Eq(add_equation!(SSAValue(i)))
                push!(externally_refined, i)
            elseif is_known_invoke(stmt, sim_time, ir)
                ir.stmts[i][:type] = Incidence(0)
                push!(externally_refined, i)
            elseif is_equation_call(stmt, ir, #=allow_call=#false)
                push!(eqcallssas, SSAValue(i))
            end
        elseif isexpr(stmt, :call)
            if is_known_call(stmt, Core.current_scope, ir)
                # N.B.: We make the assumption here that all current_scope that
                # was inside EnterScope within the same function has already
                # been folded by SROA, so the only thing left are those that
                # refer to the function's entry scope.
                ir.stmts[i][:type] = cur_scope_lattice
                ir.stmts[i][:flag] |= Compiler.IR_FLAG_REFINED
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
            newT = argextype(stmt.args[1], ir)
            if isa(newT, Const) && newT.val === Intrinsics.ScopeIdentity
                # Allocate the identity now. After inlining, we're guaranteed that
                # every Expr(:new) uniquely corresponds to a scope identity, so this
                # is legal here (bug not before)
                ir.stmts[i][:stmt] = Intrinsics.ScopeIdentity()
                ir.stmts[i][:flag] |= CC.IR_FLAG_REFINED
            end
            continue
        elseif isexpr(stmt, :splatnew)
            continue
        elseif isexpr(stmt, :boundscheck)
            ir.stmts[i][:type] = Incidence(Bool)
            ir.stmts[i][:flag] |= Compiler.IR_FLAG_REFINED
        elseif isa(stmt, PhiNode)
            # Take into account control-dependent taint
            ir.stmts[i][:flag] |= Compiler.IR_FLAG_REFINED
        elseif isa(stmt, PiNode)
            continue
        elseif isa(stmt, GotoIfNot) || isa(stmt, GotoNode)
            continue
        elseif stmt === nothing
            continue
        else
            @show stmt
        end
    end

    # Perform the actual dataflow analysis
    mi = Compiler.get_ci_mi(ci)
    (nargs, isva) = isa(mi.def, Method) ? (mi.def.nargs, mi.def.isva) : (0, false)
    method_info = Compiler.SpecInfo(nargs, isva, #=propagate_inbounds=#true, nothing)
    irsv = Compiler.IRInterpretationState(refiner, method_info, ir, mi, argtypes,
                                    world, ci.min_world, ci.max_world)
    ultimate_rt, _ = Compiler.ir_abstract_constant_propagation(refiner, irsv; externally_refined)

    if ultimate_rt === Union{}
        return UncompilableIPOResult(warnings, UnsupportedIRException("Function was discovered to unconditionally error", ir))
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

        # Delete - we've recorded this into our our side table, we don't need to
        # keep it around in the IR
        inst.args[3] = nothing
        inst.args[4] = nothing
    end

    # Do the same for equations
    for (ieq, (ssa, _)) in enumerate(eqssas)
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

        # Delete - we've recorded this into our our side table, we don't need to
        # keep it around in the IR
        inst.args[3] = nothing
        inst.args[4] = nothing
    end

    # Now record the association of (::equation)() calls with the equations that they originate from
    total_incidence = Vector{Any}(undef, length(eqssas))
    for ssa in eqcallssas
        eqcall = ir[ssa][:inst]
        eqcall === nothing && continue # equation call was on a dead branch - deleted
        eqeq = argextype(eqcall.args[2], ir)

        if !isa(eqeq, Eq)
            return UncompilableIPOResult(warnings, UnsupportedIRException("Equation call at $ssa has unknown equation reference.", ir))
        end
        ieq = eqeq.id

        eqssaval = eqcall.args[3]
        if !isa(eqssaval, SSAValue) && !isa(eqssaval, Argument)
            if !iszero(eqssaval)
                return UncompilableIPOResult(warnings, UnsupportedIRException(
                    "Equation call for $ieq at $ssa is set to $eqssaval. The system is unsolvable.", ir))
            end
            continue
        end

        inc = argextype(eqssaval, ir)
        if !isa(inc, Incidence)
            return UncompilableIPOResult(warnings, UnsupportedIRException("Expected incidence analysis to produce result for $eqssaval, got $inc", ir))
        end
        if isassigned(total_incidence, ieq)
            total_incidence[ieq] += inc
        else
            total_incidence[ieq] = inc
        end

        push!(eqssas[ieq][2], ssa)
    end

    # Now go through and incorporate the structural information from any interior calls
    eq_callee_mapping = Vector{Union{Nothing, Vector{Pair{SSAValue, Int}}}}(nothing, length(eqssas))
    handler_info = Compiler.compute_trycatch(ir)
    ncallees = 0
    for i = 1:length(ir.stmts)
        inst = ir[SSAValue(i)]
        stmt = inst[:stmt]
        stmt === nothing && continue
        # No need to process error paths - even if they were to contain intrinsics, such intrinsics would have
        # no effect.
        inst[:type] === Union{} && continue
        isexpr(stmt, :invoke) || continue
        is_known_invoke(stmt, variable, ir) && continue
        is_known_invoke(stmt, equation, ir) && continue
        is_known_invoke(stmt, sim_time, ir) && continue
        is_known_invoke(stmt, ddt, ir) && continue
        is_equation_call(stmt, ir) && continue

        info = inst[:info]
        if isa(info, MappingInfo)
            (; result, mapping) = info
        else
            callee_codeinst = stmt.args[1]
            result = structural_analysis!(callee_codeinst, Compiler.get_inference_world(refiner))

            if isa(result, UncompilableIPOResult)
                # TODO: Stack trace?
                return result
            end

            callee_argtypes = Compiler.collect_argtypes(refiner, stmt.args, Compiler.StatementState(nothing, false), irsv)[2:end]
            mapping = CalleeMapping(Compiler.optimizer_lattice(refiner), callee_argtypes, result)
            inst[:info] = MappingInfo(info, result, mapping)
        end

        err = add_internal_equations_to_structure!(refiner, eqkinds, eqclassification, total_incidence, eq_callee_mapping, SSAValue(i),
            result, mapping)
        if err !== true
            return UncompilableIPOResult(warnings, UnsupportedIRException(err, ir))
        end
    end

    nimplicitoutpairs = 0
    var_to_diff = StateSelection.complete(var_to_diff)
    ultimate_rt, nimplicitoutpairs = process_ipo_return!(Compiler.typeinf_lattice(refiner), ultimate_rt, eqclassification, varclassification,
        var_to_diff, total_incidence, eq_callee_mapping)

    names = OrderedDict{Any, ScopeDictEntry}()
    return DAEIPOResult(ir, ultimate_rt, argtypes,
        nexternalvars,
        nsysmscopes,
        nexternaleqs,
        ncallees,
        nimplicitoutpairs,
        var_to_diff,
        varclassification,
        total_incidence, eqclassification, eq_callee_mapping,
        names,
        varkinds,
        eqkinds,
        warnings)
end

function process_ipo_return!(𝕃, ultimate_rt::Incidence, eqclassification, varclassification, var_to_diff, total_incidence, eq_callee_mapping)
    nonlinrepl = nothing
    nimplicitoutpairs = 0
    function get_nonlinrepl()
        if nonlinrepl === nothing
            nonlinrepl = add_vertex!(var_to_diff)
            push!(varclassification, External)
            push!(varkinds, Intrinsics.RemovedVar)
            nimplicitoutpairs += 1
        end
        return nonlinrepl
    end
    new_row = _zero_row()
    new_eq_row = _zero_row()
    for (v_offset, coeff) in zip(rowvals(ultimate_rt.row), nonzeros(ultimate_rt.row))
        v = v_offset - 1
        if v != 0 && varclassification[v] != External && coeff == nonlinear
            get_nonlinrepl()
            new_eq_row[v_offset] = nonlinear
        else
            new_row[v_offset] = coeff
            while v != 0 && v !== nothing
                varclassification[v] = External
                v = invview(var_to_diff)[v]
            end
        end
    end
    #=
    if ultimate_rt.typ === Float64
        get_nonlinrepl()
    end
    =#
    if nonlinrepl !== nothing
        new_eq_row[get_nonlinrepl()+1] = -1.
        new_row[get_nonlinrepl()+1] = 1.
        ultimate_rt = Incidence(Const(0.0), new_row, ultimate_rt.eps)
        push!(total_incidence, Incidence(ultimate_rt.typ, new_eq_row, BitSet()))
        push!(eq_callee_mapping, nothing)
        push!(eqclassification, Owned)
        push!(eqkinds, Intrinsics.Always)
    end

    return ultimate_rt, nimplicitoutpairs
end

function add_internal_equations_to_structure!(refiner::StructuralRefiner, eqkinds::Vector{Intrinsics.EqKind}, eqclassification::Vector{VarEqClassification}, total_incidence,
        eq_callee_mapping::Vector{Union{Nothing, Vector{Pair{SSAValue, Int}}}}, thisssa::SSAValue, callee_result::DAEIPOResult, callee_mapping::CalleeMapping)
    for i in (callee_result.nexternalvars+1):length(callee_mapping.var_coeffs)
        if !isassigned(callee_mapping.var_coeffs, i)
            compute_missing_coeff!(callee_mapping.var_coeffs, callee_result, refiner.var_to_diff, refiner.varclassification, refiner.varkinds, i)
        end
    end

    eq_offset = length(total_incidence)
    if isempty(callee_result.total_incidence) && isempty(callee_result.names)
        return true
    end

    for eq = 1:length(callee_result.eqclassification)
        mapped_eq = callee_mapping.eqs[eq]
        mapped_eq == 0 && continue
        mapped_inc = apply_linear_incidence(Compiler.typeinf_lattice(refiner), callee_result.total_incidence[eq], callee_result, refiner.var_to_diff, refiner.varclassification, eqclassification, callee_mapping)
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

    for (ieq, inc) in enumerate(callee_result.total_incidence[(callee_result.nexternaleqs+1):end])
        callee_mapping.eqs[ieq] == 0 || continue
        extinc = apply_linear_incidence(Compiler.typeinf_lattice(refiner), inc, callee_result, refiner.var_to_diff, refiner.varclassification, eqclassification, callee_mapping)
        if !isa(extinc, Incidence) && !isa(extinc, Const)
            return "Failed to map internal incidence for equation $ieq (internal result $inc) - got $extinc while processing $thisssa"
        end
        push!(total_incidence, extinc)
        push!(eq_callee_mapping, [thisssa=>ieq])
        push!(eqclassification, CalleeInternal)
        push!(eqkinds, callee_result.eqkinds[ieq])
        callee_mapping.eqs[ieq] = length(total_incidence)
    end

    return true
end

function process_ipo_return!(𝕃, ultimate_rt::Type, eqclassification, varclassification, var_to_diff, total_incidence, eq_callee_mapping)
    # If we don't have any internal variables (in which case we might have to to do a more aggressive rewrite), strengthen the incidence
    # by demoting to full incidence over the argument variables. Incidence is not allowed to propagate through global mutable state, so
    # the incidence of the return type is bounded by the incidence of the arguments in this case.
    if !all(==(External), varclassification)
        return ultimate_rt, 0
    end
    # TODO: Keep track of whether we have any time dependence?
    return Incidence(ultimate_rt, IncidenceVector(MAX_EQS, Int[1:length(varclassification)+1;], Union{Float64, NonLinear}[nonlinear for _ in 1:length(varclassification)+1])), 0
end

function process_ipo_return!(𝕃, ultimate_rt::Eq, eqclassification, args...)
    eqclassification[ultimate_rt.id] = External
    return ultimate_rt, 0
end
process_ipo_return!(𝕃, ultimate_rt::Union{Type, PartialScope, PartialKeyValue, Const}, args...) = ultimate_rt, 0
function process_ipo_return!(𝕃, ultimate_rt::PartialStruct, args...)
    nimplicitoutpairs = 0
    fields = Any[]
    for f in ultimate_rt.fields
        (rt, n) = process_ipo_return!(𝕃, f, args...)
        nimplicitoutpairs += n
        push!(fields, rt)
    end
    return PartialStruct(𝕃, ultimate_rt.typ, fields), nimplicitoutpairs
end
