module DAECompilerSciMLSensitivityExt

using SciMLSensitivity
using SciMLBase
using DAECompiler: DAECompiler, IRODESystem, TransformedIRODESystem, arg1_from_sys, @breadcrumb
using SciMLSensitivity: SciMLSensitivity, ODEForwardSensitivityProblem, ForwardSensitivity, extract_local_sensitivities
using ChainRulesCore

function SciMLSensitivity.ODEForwardSensitivityProblem(sys::IRODESystem, u0, tspan, arg1=arg1_from_sys(sys); kwargs...)
    tsys = TransformedIRODESystem(sys)
    if u0 !== nothing && isempty(u0)
        u0 = nothing
    end
    ODEForwardSensitivityProblem(tsys, u0, tspan, arg1; kwargs...)
end

@breadcrumb "ir_levels" function SciMLSensitivity.ODEForwardSensitivityProblem(tsys::TransformedIRODESystem, u0, tspan,
        arg1=arg1_from_sys(get_sys(tsys));
        kwargs...)

    prob = SciMLBase.ODEProblem(
        tsys::TransformedIRODESystem, u0, tspan, arg1;
        kwargs..., jac=true, paramjac=true
    )
    sensealg = ForwardSensitivity(autodiff=false, autojacvec=false, autojacmat=false)
    return ODEForwardSensitivityProblem(prob.f, prob.u0, prob.tspan, arg1, sensealg; kwargs...)
end

DAECompiler.get_transformed_sys(f::SciMLSensitivity.ODEForwardSensitivityFunction) = DAECompiler.get_transformed_sys(f.f)

"""
    reconstruct_sensitivities(sol::SciMLBase.AbstractODESolution, syms, ts=sol.t)

Returns the the derivatives of each given `variable`/`observed!` in `syms` with respect to each parameter of the solved system.

Similar to `SciMLSensitivity.extract_local_sensitivities`, but only returning the sensitivities (not the primal values)
Returns a collection of matrixes, one per parameter,
with one column per time step in `ts` and one one row per `variable`/`observed!` in `syms`
"""
function DAECompiler.reconstruct_sensitivities(sol::SciMLBase.AbstractODESolution, syms::Vector{<:DAECompiler.ScopeRef}, ts=sol.t)
    us, du_dparams = extract_local_sensitivities(sol, ts)
    var_inds, obs_inds = DAECompiler.split_and_sort_syms(syms)
    
    transformed_sys = DAECompiler.get_transformed_sys(sol)
    dreconstruct! = get!(sol.prob.f.observed.derivative_cache, (var_inds, obs_inds, false)) do
        DAECompiler.compile_batched_reconstruct_derivatives(transformed_sys, var_inds, obs_inds, false, false;)
    end
    
    num_params = length(du_dparams)
    dout_vars_per_param = [similar(us, (length(var_inds), length(ts))) for _ in 1:num_params]
    dout_obs_per_param  = [similar(us, (length(obs_inds), length(ts))) for _ in 1:num_params]
    dvars_du = similar(us, length(var_inds), sol.prob.f.numindvar) # (v, u)
    dvars_dp = similar(us, length(var_inds), num_params)           # (v, p)
    dobs_du = similar(us, length(obs_inds), sol.prob.f.numindvar)  # (o, u)
    dobs_dp = similar(us, length(obs_inds), num_params)            # (o, p)
    for i in eachindex(ts)
        tᵢ = ts[i]
        uₜ = us[:, i] #u x 1
        dreconstruct!(dvars_du, dvars_dp, dobs_du, dobs_dp, uₜ, sol.prob.p, tᵢ) # sizes: (v, u), (v, n), (o, u), (o, n)
        for k = 1:num_params
            du_dparamₜ = du_dparams[k][:, i]  # u x 1
            dvars_dp[:, k] += dvars_du * du_dparamₜ # v x 1
            dobs_dp[:, k] += dobs_du * du_dparamₜ # o x 1
            dout_vars_per_param[k][:, i] .= dvars_dp[:, k]
            dout_obs_per_param[k][:, i] .= dobs_dp[:, k]
        end
    end

    return map(dout_vars_per_param, dout_obs_per_param) do dout_vars, dout_obs
        DAECompiler.join_syms(syms, dout_vars, dout_obs, (var_inds, obs_inds))
    end
end

# we put this here rather than in extra_rules.jl as actually this rule belongs to this package
# rather than being type-piracy
function ChainRulesCore.frule((_, ṡsol, _, ṫ), ::typeof(DAECompiler.batch_reconstruct), ssol, refs, t)
    y = DAECompiler.batch_reconstruct(ssol, refs, t)
    ẏ = zero(y)
    if !iszero(ṡsol)
        dout_dparams = DAECompiler.reconstruct_sensitivities(ssol, refs, t)
        ṗarams = ṡsol.prob.p
        ẏ .+= sum(ṗarams .* dout_dparams)  # dout_dparams are scaled to pertubation 1, we need to rescale based on input pertubation
    end
    if !iszero(ṫ)
        # transpose `ṫ` as `reconstruct_time_deriv` always returns a 1 row matrix
        ẏ .+= ṫ' .* DAECompiler.reconstruct_time_deriv(ssol, refs, t)
    end
    return y, ẏ
end

end
