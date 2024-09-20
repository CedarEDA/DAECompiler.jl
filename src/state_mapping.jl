using SciMLBase, SymbolicIndexingInterface

struct ScopeRef{T, ST}
    sys::T
    scope::Scope{ST}
end
Base.Broadcast.broadcastable(ref::ScopeRef) = Ref(ref)  # broadcast as scalar

IRODESystem(sr::ScopeRef) = getfield(sr, :sys)

SymbolicIndexingInterface.symbolic_type(::ScopeRef) = ScalarSymbolic()
SymbolicIndexingInterface.symbolic_type(::Type{<:ScopeRef}) = ScalarSymbolic()

SymbolicIndexingInterface.is_independent_variable(sys::TransformedIRODESystem, sym::ScopeRef) = false

function SymbolicIndexingInterface.is_variable(sys::TransformedIRODESystem, sym::ScopeRef)
    ssys = get_sys(sys)
    ssys === getfield(sym, :sys) || return false
    _sym_to_index(ssys, sym) isa Dict && return false
    return !isnothing(SciMLBase.sym_to_index(sym, sys))
end

SymbolicIndexingInterface.variable_index(sys::TransformedIRODESystem, sym::ScopeRef) = SciMLBase.sym_to_index(sym, sys)

SymbolicIndexingInterface.is_parameter(sys::TransformedIRODESystem, sym::ScopeRef) = false

function SymbolicIndexingInterface.is_observed(sys::TransformedIRODESystem, sym::ScopeRef)
    ssys = get_sys(sys)
    ssys === getfield(sym, :sys) && SciMLBase.sym_to_index(sym, sys) === nothing
end
SymbolicIndexingInterface.is_time_dependent(sys::TransformedIRODESystem) = true
SymbolicIndexingInterface.constant_structure(sys::TransformedIRODESystem) = false

#SymbolicIndexingInterface.variable_symbols(tsys::TransformedIRODESystem) = ()
# TODO: do this properly once we have syntax for time derivatives of variables.
# unitl then, we can't give a symbolic name to every state.
function add_recursive_vars!(syms, lvl_children::AbstractDict, sys)
    for (sym, child) in lvl_children
        if !isnothing(child.children)
            add_recursive_vars!(syms, child.children, getproperty(sys, sym))
        elseif !isnothing(child.var)  # only record vars, not other names
            push!(syms, getproperty(sys, sym))
        end
    end
    return syms
end

function SymbolicIndexingInterface.variable_symbols(tsys::TransformedIRODESystem)
    sys = get_sys(tsys)
    syms = add_recursive_vars!(ScopeRef[], StructuralAnalysisResult(sys).names, sys)
    nvars = count(x->isa(x, StateSelection.SelectedState), tsys.var_eq_matching)
    out = [Symbol() for _ in 1:nvars]
    for sym in syms
        is_variable(tsys, sym) || continue
        out[variable_index(tsys, sym)] = Symbol(sym)
    end
    out
    #[sym for sym in syms if is_variable(tsys, sym)]
end

SymbolicIndexingInterface.independent_variable_symbols(sys::TransformedIRODESystem) = ()

function SymbolicIndexingInterface.parameter_symbols(sys::TransformedIRODESystem)
    fieldnames(getfield(get_sys(sys), :mi).specTypes.types[1])
end

function sym_stack(scope::Union{Scope, GenScope})
    stack = Any[]
    while !isa(scope, Scope) || isdefined(scope, :parent)
        if isa(scope, GenScope)
            push!(stack, Gen(scope.identity, scope.sc.name))
            scope = scope.sc.parent
        else
            push!(stack, scope.name)
            scope = scope.parent
        end
    end
    return stack
end

_sym_to_index(sr::ScopeRef) = _sym_to_index(IRODESystem(sr), sr)
function _sym_to_index(sys::IRODESystem, sr::ScopeRef)
    scope = getfield(sr, :scope)
    stack = sym_stack(scope)
    strct = NameLevel(StructuralAnalysisResult(sys).names)
    for s in reverse(stack)
        strct = strct.children[s]
    end
    return strct
end

function SciMLBase.sym_to_index(sr::ScopeRef, transformed_sys::TransformedIRODESystem)
    unopt_idx = _sym_to_index(get_sys(transformed_sys), sr)
    if unopt_idx.var === nothing && unopt_idx.obs === nothing
        sname = String(getfield(sr, :scope).name)
        if unopt_idx.children !== nothing && !isempty(unopt_idx.children)
            suggest = first(unopt_idx)[1]
            error("$sname is not a concrete index. Did you mean $sname.var\"$suggest\"")
        else
            error("$sname is not a variable or observed.")a
        end
    end
    if unopt_idx.var === nothing
        # Observed gets handled elsewhere
        return nothing
    end
    unopt_idx = unopt_idx.var
    # Check if this index is still here
    (; var_assignment) = assign_vars_and_eqs(MatchedSystemStructure(transformed_sys), true)

    assgn = var_assignment[unopt_idx]
    if assgn == (0 => false)
        return nothing
    end

    if assgn[2]
        # Variable is now in `du` (e.g. because it got aliased)
        return nothing
    end

    return assgn[1]
end


function SciMLBase.sym_to_index(sr::ScopeRef{IRODESystem}, A::SciMLSolution)
    transformed_sys = get_transformed_sys(A)
    @assert IRODESystem(sr) == get_sys(transformed_sys)
    return SciMLBase.sym_to_index(sr, transformed_sys)
end

function SciMLBase.sym_to_index(sr::ScopeRef, A::SciMLBase.DEIntegrator)
    transformed_sys = get_transformed_sys(A)
    @assert IRODESystem(sr) == get_sys(transformed_sys)
    return SciMLBase.sym_to_index(sr, transformed_sys)
end

function Base.getproperty(sys::IRODESystem, name::Symbol)
    haskey(StructuralAnalysisResult(sys).names, name) || throw(Base.UndefRefError())
    return ScopeRef(sys, Scope(Scope(), name))
end

function Base.propertynames(sr::ScopeRef)
    scope = getfield(sr, :scope)
    stack = sym_stack(scope)
    strct = NameLevel(StructuralAnalysisResult(IRODESystem(sr)).names)
    for s in reverse(stack)
        strct = strct.children[s]
        strct.children === nothing && return keys(Dict{Symbol, Any}())
    end
    return keys(strct.children)
end

function Base.getproperty(sr::ScopeRef{IRODESystem}, name::Symbol)
    scope = getfield(sr, :scope)
    stack = sym_stack(scope)
    strct = NameLevel(StructuralAnalysisResult(IRODESystem(sr)).names)
    for s in reverse(stack)
        strct = strct.children[s]
        strct.children === nothing && throw(Base.UndefRefError())
    end
    if !haskey(strct.children, name)
        throw(Base.UndefRefError())
    end
    ScopeRef(IRODESystem(sr), Scope(getfield(sr, :scope), name))
end

function Base.show(io::IO, scope::Scope)
    if !isdefined(scope, :parent)
        print(io, '▫')
    else
        show(io, scope.parent)
        print(io, '.', scope.name)
    end
end
Base.show(io::IO, sr::ScopeRef) = show(io, getfield(sr, :scope))

Base.propertynames(sys::IRODESystem) = keys(StructuralAnalysisResult(sys).names)

# Observed
const ReconstructCache = Dict{Tuple{Vector{Int64},Vector{Int64}},Function}
const DerivativeCache = Dict{Tuple{Vector{Int64},Vector{Int64},Bool},Function}

struct DAEReconstructedObserved
    sys
    tsys
    cache::ReconstructCache
    derivative_cache::DerivativeCache
    DAEReconstructedObserved(tsys) = new(get_sys(tsys), tsys, (@new_cache ReconstructCache()), (@new_cache DerivativeCache()))
end

struct ODEReconstructedObserved
    sys
    tsys
    cache::ReconstructCache
    derivative_cache::DerivativeCache
    time_derivative_cache::ReconstructCache
    ODEReconstructedObserved(tsys) = new(get_sys(tsys), tsys, (@new_cache ReconstructCache()), (@new_cache DerivativeCache()), (@new_cache ReconstructCache()))
end

# Split `syms` into `vars` and `obs` and sort them in preparation for
# passing them off to `compile_batched_reconstruct_func()`
function split_and_sort_syms(sys, syms)
    vars = Int64[]
    obs = Int64[]
    for level in _sym_to_index.(Ref(sys), syms)
        if level.obs !== nothing
            @assert level.var === nothing
            push!(obs, level.obs)
        else
            push!(vars, level.var)
        end
    end
    return sort(vars), sort(obs)
end

"""
    join_syms(
        sys, syms, vars, obs,
        (var_inds, obs_inds)=split_and_sort_syms(sys, syms)
    )

The user just asked for syms, return the values regardless of whether
they were variables or observed, in order they were requested.

`vars` and `obs` are the data to be joined and ordered per the ordering in `syms`.
The must be currently order in first axis by `var_inds` and `obs_inds` resepectively.
"""
function join_syms(sys,
    syms, vars::AbstractMatrix, obs::AbstractMatrix,
    (var_inds, obs_inds)=split_and_sort_syms(sys, syms)
)
    length(var_inds) == size(vars, 1) || throw(DimensionMismatch("wrong number of vars"))
    length(obs_inds) == size(obs, 1) || throw(DimensionMismatch("wrong number of obs"))
    length(obs_inds) + length(var_inds) == length(syms) || throw(DimensionMismatch("wrong number of syms, vars or obs"))
    size(vars, 2) == size(obs, 2) || throw(DimensionMismatch("second dim of obs and vars must match"))

    out = similar(vars, (size(vars, 1) + size(obs, 1), size(vars, 2)))
    out_idx = 1
    vars_idx = 1
    obs_idx = 1
    for (out_idx, sym) in enumerate(syms)
        level = DAECompiler._sym_to_index(sys, sym)
        if level.obs !== nothing
            @assert level.var === nothing
            loc=findfirst(==(level.obs), obs_inds)
            out[out_idx, :] .= obs[loc, :]
            obs_idx += 1
        else
            loc=findfirst(==(level.var), var_inds)
            out[out_idx, :] .= vars[loc, :]
            vars_idx += 1
        end
    end
    return out
end

"""
    batch_reconstruct(ro, syms, dus, us, p, ts)

Because SciMLBase does not yet have a batch reconstruction API [0], we define
one here.  This method will JIT compile a specialized reconstruction function
for the particular `syms` being requested.  The return value is a
`length(syms)` by `length(ts)` matrix.
"""
function batch_reconstruct(ro, syms, dus, us, p, ts)
    sys = get_sys(ro.tsys)
    vars, obs = split_and_sort_syms(sys, syms)
    debug_config = DebugConfig(ro.tsys)

    # First, look up the appropriate reconstruction function from our cache:
    reconstruct = get!(ro.cache, (vars, obs)) do
        @may_timeit debug_config "compile_batched_reconstruct_func" begin
            f = compile_batched_reconstruct_func(ro.tsys, vars, obs, isa(ro, DAEReconstructedObserved))
        end
        return f
    end

    out_vars = similar(us[1], (length(vars), length(ts)))
    out_obs  = similar(us[1], (length(obs), length(ts)))

    # Use `vars_tmp` and `obs_tmp` until we no longer have special goldclass support for `Vector{Float64}`
    # but can directly use `@view(out_vars[:, t_idx])` in our `reconstruct()` invocation with no dynamic
    # dispatch penalty.
    vars_tmp = similar(us[1], (length(vars),))
    obs_tmp = similar(us[1], (length(obs),))

    # TODO: We should get du somehow: https://github.com/SciML/SciMLBase.jl/issues/295
    for t_idx in eachindex(ts)
        t = ts[t_idx]
        u = us[t_idx]
        if isa(ro, DAEReconstructedObserved)
            du = dus[t_idx]
            u_vars, u_obs = reconstruct(vars_tmp, obs_tmp, du, u, p, t)
        else
            u_vars, u_obs = reconstruct(vars_tmp, obs_tmp, u, p, t)
        end
        out_vars[:, t_idx] .= vars_tmp
        out_obs[:, t_idx] .= obs_tmp
    end
    return join_syms(sys, syms, out_vars, out_obs, (vars, obs))
end

function batch_reconstruct(sol::SciMLBase.AbstractODESolution,
                           syms::Vector{<:ScopeRef},
                           ts=nothing)
    if ts === nothing
        us = sol.u
        dus = zero.(us)
        ts = sol.t
    else
        # SciML doesn't handle empty states very well, let's work around
        # that by noticing that `sol.u`'s are all empty, and just constructing
        # our own "interpolated empties":
        if isempty(first(sol.u))
            us = [[] for t in ts]
            dus = [[] for t in ts]
        else
            us = sol(ts).u
            dus = zero.(us)
        end
    end
    return batch_reconstruct(sol.prob.f.observed, syms, dus, us, sol.prob.p, ts)
end
function batch_reconstruct(sol::SciMLBase.AbstractODESolution,
                           sym::ScopeRef,
                           ts = nothing)
    return batch_reconstruct(sol, [sym], ts)
end

function (ro::DAEReconstructedObserved)(sym::ScopeRef, u, p, t)
    # TODO: More upstream work is required before we can provide `du` everywhere,
    #       so for now incorrectly provide zeros in some cases.
    du = zero(u)
    return batch_reconstruct(ro, [sym], [du], [u], p, [t])[1]
end

function (ro::DAEReconstructedObserved)(sym::ScopeRef, du, u, p, t)
    if du === nothing
        # TODO: More upstream work is required before we can provide `du` everywhere,
        #       so for now incorrectly provide zeros in some cases.
        du = zero(u)
    end
    return batch_reconstruct(ro, [sym], [du], [u], p, [t])[1]
end

function (ro::ODEReconstructedObserved)(sym::ScopeRef, u, p, t)
    return batch_reconstruct(ro, [sym], [], [u], p, [t])[1]
end

function (ro::DAEReconstructedObserved)(sym::Vector{<:ScopeRef}, u, p, t)
    # TODO: More upstream work is required before we can provide `du` everywhere,
    #       so for now incorrectly provide zeros in some cases.
    du = zero(u)
    return vec(batch_reconstruct(ro, sym, [du], [u], p, [t]))
end

function (ro::DAEReconstructedObserved)(sym::Vector{<:ScopeRef}, du, u, p, t)
    if du === nothing
        # TODO: More upstream work is required before we can provide `du` everywhere,
        #       so for now incorrectly provide zeros in some cases.
        du = zero(u)
    end
    return vec(batch_reconstruct(ro, sym, [du], [u], p, [t]))
end

function (ro::ODEReconstructedObserved)(sym::Vector{<:ScopeRef}, u, p, t)
    return vec(batch_reconstruct(ro, sym, [], [u], p, [t]))
end

"""
    reconstruct_time_deriv(sol::SciMLBase.AbstractODESolution, syms, ts=sol.t)

For each variable/observed in `syms` computes it's derivative with respect to time at each time step in `ts`
"""
function reconstruct_time_deriv(sol::ODESolution, syms, ts=sol.t)
    # We only allow ODESolution not AbstractODESolution as we do not suport DAESolutions
    transformed_sys = get_transformed_sys(sol)
    sys = get_sys(transformed_sys)
    var_inds, obs_inds = split_and_sort_syms(sys, syms)

    dreconstruct_dtime! = get!(sol.prob.f.observed.time_derivative_cache, (var_inds, obs_inds)) do
        construct_reconstruction_time_derivative(transformed_sys, var_inds, obs_inds, false;)
    end

    state_type = eltype(eltype((sol.u)))
    dvar_dt = Matrix{state_type}(undef, (length(var_inds), length(ts)))
    dobs_dt = Matrix{state_type}(undef, (length(obs_inds), length(ts)))
    dvar = Vector{state_type}(undef, length(var_inds))
    dobs = Vector{state_type}(undef, length(obs_inds))
    for i in eachindex(ts)
        tᵢ = ts[i]
        # Prem-Opt: we could do idxs=1:sol.prob.f.numindvar on next two lines
        # for ODEForwardSensitivityFunction, since we will never read beyond those in dreconstruct
        uₜ = sol(tᵢ)
        du_dtₜ = sol(tᵢ, Val{1})
        dreconstruct_dtime!(
            dvar, dobs,
            du_dtₜ, uₜ, sol.prob.p, tᵢ
        )
        dvar_dt[:, i] .= dvar
        dobs_dt[:, i] .= dobs
    end
    return join_syms(sys, syms, dvar_dt, dobs_dt, (var_inds, obs_inds))
end
