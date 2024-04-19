using Core: CodeInfo
using Core.Compiler: IRCode, NewInstruction, InliningState,
    adce_pass!, compact!, insert_node!, quoted, sroa_pass!, ssa_inlining_pass!
using ModelingToolkit: MatchedSystemStructure
using SparseArrays
using StaticArraysCore: StaticArraysCore
using DiffEqCallbacks
using OrdinaryDiffEq

struct TransformedIRODESystem
    state::IRTransformationState
    var_eq_matching::Matching

    epsjac_cache::Base.RefValue{Function}
end

TransformedIRODESystem(sys, var_eq_matching) = TransformedIRODESystem(sys, var_eq_matching, Base.RefValue{Function}())
TransformedIRODESystem(sys::IRODESystem) = DAECompiler.structural_simplify(sys)

function Base.show(io::IO, ::MIME"text/plain", tsys::TransformedIRODESystem)
    print(io, "TransformedIRODESystem for ")
    show(io, MIME"text/plain"(), get_sys(tsys))
end
function Base.show(io::IO, tsys::TransformedIRODESystem)
    print(io, "TransformedIRODESystem for ")
    show(io, get_sys(tsys))
end

ChainRulesCore.zero_tangent(::TransformedIRODESystem) = ChainRulesCore.NoTangent()

"""
    get_transformed_sys(obj)

Extracts the `TransformedIRODESystem` from a SciML object, such as a `SciMLSolution` or a `ODEFunction` etc.
"""
function get_transformed_sys end
get_transformed_sys(sol::SciMLSolution) = get_transformed_sys(sol.prob)
get_transformed_sys(prob::SciMLBase.AbstractDEProblem) = get_transformed_sys(prob.f)
get_transformed_sys(f::Union{ODEFunction, DAEFunction}) = f.sys
get_transformed_sys(integrator::SciMLBase.DEIntegrator) = get_transformed_sys(integrator.f)


"""
    get_sys(obj)

Extracts the system from a SciML object, such as a `SciMLSolution` or a `ODEFunction` etc, or from a `TransformedIRODESystem`.
The system is a `ModelingToolkit.AbstractODESystem`, and in the DAECompiler case is normally a `IRODESystem`.
"""
get_sys(obj) = get_sys(get_transformed_sys(obj))
get_sys(tsys::TransformedIRODESystem) = get_sys(tsys.state)
get_sys(state::IRTransformationState) = state.sys


function ModelingToolkit.MatchedSystemStructure(tsys::TransformedIRODESystem)
    MatchedSystemStructure(tsys.state.structure, tsys.var_eq_matching)
end

DebugConfig(tsys::TransformedIRODESystem) = DebugConfig(tsys.state)

function assign_vars_and_eqs(tsys::TransformedIRODESystem, isdae::Bool)
    return assign_vars_and_eqs(MatchedSystemStructure(tsys.state.structure, tsys.var_eq_matching), isdae)
end

function assign_vars_and_eqs(mss::MatchedSystemStructure, isdae::Bool)
    (; structure, var_eq_matching) = mss
    (; graph, var_to_diff) = structure

    var_assignment = Pair{Int, Bool}[0=>false for i = 1:ndsts(graph)]
    dummy_map = Pair{Int, Int}[]
    differential_vars = Bool[]
    var_num = 0

    assigned_eqs = BitSet()

    # First assign slots to all selected states
    for i = 1:ndsts(graph)
        if var_eq_matching[i] == SelectedState()
            var_num += 1
            var_assignment[i] = var_num => false
            if isdae
                diff_matching = var_eq_matching[var_to_diff[i]]
                if diff_matching !== unassigned
                    # This generates an implicit equation for this selected state.
                    # In particular, if this variable is matched to an equation,
                    # then we need to re-ify that equation.
                    # If the variable is selected, we need to add the implicit
                    # link between the derivatives as an equation.
                    # Re-use the variable index for the equation to increase alignment with the ODE case.
                    push!(assigned_eqs, var_num)
                end
                if diff_matching != SelectedState()
                    var_assignment[var_to_diff[i]] = var_num => true
                end
            end
            push!(differential_vars, true)
        end
    end

    diff_var_num = var_num

    # For mass-matrix DAE case, move remaining du variables into u
    if !isdae
        for i = 1:ndsts(graph)
            if var_eq_matching[i] == SelectedState() && var_eq_matching[var_to_diff[i]] == unassigned
                var_num += 1
                push!(dummy_map, var_assignment[i][1] => var_num)
                var_assignment[var_to_diff[i]] = var_num => false
            end
        end
    end

    # Now assign slots to everything else
    # slot is the index to the numerical array
    for i = 1:ndsts(graph)
        isempty(𝑑neighbors(graph, i)) && continue
        if var_eq_matching[i] === unassigned && var_assignment[i] == (0=>false)
            var_num += 1
            var_assignment[i] = var_num => false
            push!(differential_vars, false)
        end
    end

    if isdae
        available_eqslots = setdiff!(BitSet(1:var_num), assigned_eqs)
    else
        eqnum = diff_var_num
    end

    eq_assignment = nothing
    # Assign `out` slots (or slots with all zero rows in the mass matrix)
    eq_assignment = Int[0 for _ = 1:nsrcs(graph)]
    for i = 1:nsrcs(graph)
        isempty(𝑠neighbors(graph, i)) && continue
        if invview(var_eq_matching)[i] == unassigned
            if isdae
                thiseq = popfirst!(available_eqslots)
            else
                eqnum += 1
                thiseq = eqnum
            end
            eq_assignment[i] = thiseq
        end
    end

    if isdae
        @assert isempty(available_eqslots)
    end

    (; var_assignment, eq_assignment, differential_vars, var_num = (isdae ? var_num : diff_var_num), neqs=(isdae ? var_num : eqnum), dummy_map)
end

function unoptimized_matching(state)
    (;var_to_diff, graph) = state.structure
    var_eq_matching = complete(Matching{Union{Unassigned, SelectedState}}(ndsts(graph)), ndsts(graph))
    for i = 1:ndsts(graph)
        if var_to_diff[i] !== nothing && !isempty(𝑑neighbors(graph, var_to_diff[i]))
            var_eq_matching[i] = SelectedState()
        end
    end
    var_eq_matching
end

_parameter_type(sys::IRODESystem) = only(getfield(sys, :mi).specTypes.parameters)

function parameter_type(sys::IRODESystem)
    T = _parameter_type(sys)
    if Base.issingletontype(T)
        return Nothing
    end
    return T
end

function normalize_parameters(sys::IRODESystem, @nospecialize p)
    T = _parameter_type(sys)
    # normalize away various different signlton types and represent them all with `nothing` avoiding extra compiling
    # similarly, some parts of SciML-verse uses [] where there are no parameters, while we don't it is also safe to normalize that here
    if Base.issingletontype(T) && (isa(p, T) || (isa(p, Array) && isempty(p)))
        return nothing
    end
    return p
end

# May be overloaded
function default_parameters(@nospecialize T::Type)
    if isa(T, DataType) && isdefined(T, :instance)
        return T.instance
    end
    error("Specifying parameterization is required for non-singleton model $T")
end

function arg1_from_sys(sys::IRODESystem)
    return default_parameters(parameter_type(sys))
end

function SciMLBase.DAEProblem(sys::IRODESystem, du0, u0, tspan, p=arg1_from_sys(sys); kwargs...)
    tsys = TransformedIRODESystem(sys)
    if du0 !== nothing && isempty(du0)
        du0 = nothing
    end
    if u0 !== nothing && isempty(u0)
        u0 = nothing
    end
    return DAEProblem(tsys, du0, u0, tspan, p; kwargs...)
end

# Resolve Ambiguity
function SciMLBase.ODEProblem(sys::IRODESystem, u0::StaticArraysCore.StaticArray, tspan, p=arg1_from_sys(sys); kwargs...)
    return @invoke ODEProblem(sys::IRODESystem, u0::Any, tspan, p; kwargs...)
end


function SciMLBase.ODEProblem(sys::IRODESystem, u0, tspan, p=arg1_from_sys(sys); kwargs...)
    tsys = TransformedIRODESystem(sys)
    if u0 !== nothing && isempty(u0)
        u0 = nothing
    end
    return ODEProblem(tsys, u0, tspan, p; kwargs...)
end

@breadcrumb "ir_levels" function SciMLBase.DAEProblem(tsys::TransformedIRODESystem, du0, u0, tspan, p=arg1_from_sys(tsys.get_sys(state)); kwargs...)
    debug_config = DebugConfig(tsys)
    @may_timeit debug_config "DAEProblem" begin
        return build_problem(tsys, du0, u0, tspan, p, true; kwargs...)
    end
end

@breadcrumb "ir_levels" function SciMLBase.ODEProblem(tsys::TransformedIRODESystem, u0, tspan, p=arg1_from_sys(tsys.get_sys(state)); kwargs...)
    debug_config = DebugConfig(tsys)
    @may_timeit debug_config "ODEProblem" begin
        return build_problem(tsys, nothing, u0, tspan, p, false; kwargs...)
    end
end

function build_problem(tsys::TransformedIRODESystem, du0, u0, tspan, p, isdae; jac=false, paramjac=false, kwargs...)
    debug_config = DebugConfig(tsys)
    state_type = isnothing(u0) ? Float64 : eltype(u0)
    (; state, var_eq_matching) = tsys
    p = normalize_parameters(get_sys(state), p)
    # Collect this IR because `dae_finish!()` mutates it
    if jac || paramjac
        ir_for_differentiation = prepare_ir_for_differentiation(state.ir, tsys; keep_epsilon=false)
    end

    # Compile these functions first, to pre-empt `dae_finish!` mutating `tsys.state.ir`
    if state.vcc_nzc > 0
        @may_timeit debug_config "compile_vector_continuous_callback_func" begin
            vcc_condition = compile_vector_continuous_callback_func(tsys, isdae)
        end
    end
    if state.ic_nzc > 0
        @may_timeit debug_config "compile_iterative_callback_func" begin
            ic_condition = compile_iterative_callback_func(tsys, isdae)
        end
    end
    function build_callbacks()
        callbacks = Any[]
        if state.ic_nzc > 0
            # Use an IterativeCallback to calculate pre-known but non-constant discontinuities
            push!(callbacks, IterativeCallback(ic_condition, Returns(nothing); save_positions=(false, false)))
        end
        if state.vcc_nzc > 0
            # Use a VectorContinuousCallback to dynamically calculate unknown discontinuities
            push!(callbacks, VectorContinuousCallback(vcc_condition, Returns(nothing), state.vcc_nzc))
        end
        # XXX: These callbacks are state-ful so if using an EnsembleProblem, they must be recreated
        #      for each worker in the `prob_func`.
        return CallbackSet(callbacks...)
    end
    tsys.state.callback_func = build_callbacks

    # TODO: With an additional post-AD inlining pass, we should be able to get rid of
    # unused states and removed the allow_unassigned here.
    @may_timeit debug_config "dae_finish!" begin
        finish_result = dae_finish!(state, var_eq_matching, isdae; allow_unassigned=true, mass_matrix_eltype=state_type)
        if isdae
            (; F!, neqs, var_assignment, eq_assignment, differential_vars) = finish_result
            dummy_map = nothing
            mass_matrix = nothing
        else
            (; F!, neqs, var_assignment, eq_assignment, dummy_map, mass_matrix) = finish_result
            differential_vars = nothing
        end
    end

    # `jac_prototype` is a matrix with a structure showing the nonzero elements of the Jacobian.
    if !isdae
        @may_timeit debug_config "jac_prototype" begin
            jac_prototype = jacobian_prototype(state.structure.graph, state.structure.var_to_diff, var_eq_matching, var_assignment, eq_assignment, neqs)
        end
    else
        jac_prototype = nothing
    end

    if jac
        @may_timeit debug_config "construct_jacobian" begin
            jac = construct_jacobian(ir_for_differentiation, tsys, var_assignment, eq_assignment, dummy_map, neqs; isdae)
        end
        # Only ODEProblems use `tgrad`
        if !isdae
            @may_timeit debug_config "construct_tgrad" begin
                tgrad = construct_tgrad(ir_for_differentiation, tsys, var_assignment, eq_assignment)
            end
        end
    else
        jac = nothing
        tgrad = nothing
    end

    # We only support `paramjac` on ODE forms for now
    if !isdae && paramjac
        @may_timeit debug_config "construct_paramjac" begin
            paramjac = construct_paramjac(ir_for_differentiation, tsys, var_assignment, eq_assignment)
        end
    else
        paramjac = nothing
    end

    if isdae
        f! = DAEFunction{true, SciMLBase.FullSpecialize}(F!; sys=tsys, jac, jac_prototype,
                                                         observed=DAEReconstructedObserved(tsys))
    else
        if neqs != 0
            f! = ODEFunction(F!; sys=tsys, jac_prototype, jac, tgrad, paramjac,
                             mass_matrix, observed=ODEReconstructedObserved(tsys))
        else
            f! = ODEFunction(F!; sys=tsys, jac_prototype, jac, tgrad, paramjac,
                             observed=ODEReconstructedObserved(tsys))
        end
    end

    if isdae && du0 === nothing
        du0 = 1e-7 .* randn(neqs)
    end
    if u0 === nothing
        u0 = 1e-7 .* randn(neqs)
    end
    if neqs == 0
        u0 = du0 = nothing
    else
        if neqs != length(u0)
            throw(ArgumentError("Provided u0 (length $(length(u0))) != number of states ($(neqs))"))
        end
        if isdae && neqs != length(du0)
            throw(ArgumentError("Provided du0 (length $(length(du0))) != number of states ($(neqs))"))
        end
    end
    if isdae
        return DAEProblem{true}(f!, du0, u0, tspan, p; differential_vars, callback=tsys.state.callback_func(), kwargs...)
    else
        return ODEProblem{true}(f!, u0, tspan, p; callback=tsys.state.callback_func(), kwargs...)
    end
end

"""
    jacobian_prototype(graph, var_to_diff, var_eq_matching, var_assignment, eq_assignment, neqs)

Determine the `jac_prototype` from the structure of the problem.
The `jac_prototype` is a sparse matrix of similar shape and element type to the jacobian,
with non-zeros whereever the jacobian has nonzeros.
See the [`ODEProblem` documentation](https://docs.sciml.ai/DiffEqDocs/latest/types/ode_types/#SciMLBase.ODEProblem).
"""
function jacobian_prototype(graph, var_to_diff, var_eq_matching, var_assignment, eq_assignment, neqs; jac_eltype=Float64)
    dig = DiCMOBiGraph{true}(graph, var_eq_matching)
    I = Int[]; J = Int[]; V = jac_eltype[]
    for i in 1:ndsts(graph)
        var_eq_matching[i] == SelectedState() || continue
        row = var_assignment[i][1]
        du_assgn = var_assignment[var_to_diff[i]]
        if du_assgn != (0 => false)
            push!(I, row); push!(J, du_assgn[1]); push!(V, NaN)
            row = du_assgn[1]
        end
        vars = StructuralTransformations.neighborhood(dig, var_to_diff[i], Inf; dir = :in)
        for j in vars
            (var_assignment[j][1] == 0) && continue
            col = var_assignment[j][1]
            push!(I, row); push!(J, col); push!(V, NaN)
        end
    end
    for (eq, assgn) in enumerate(eq_assignment)
        assgn == 0 && continue
        allvars = Int[]
        # TODO: Make a Graphs.jl method that does this
        for var in 𝑠neighbors(graph, eq)
            append!(allvars, StructuralTransformations.neighborhood(dig, var, Inf; dir = :in))
        end
        vars = unique(allvars)
        for j in vars
            (var_assignment[j][1] == 0) && continue
            push!(I, assgn); push!(J, var_assignment[j][1]); push!(V, NaN)
        end
    end
    # TODO: remove the collect here. UMFPACk currently is missing the ability to pass check=false to ldiv
    return collect(sparse(I, J, V, neqs, neqs))
end
