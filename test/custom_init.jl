module custom_init
using DAECompiler
using DAECompiler.Intrinsics
using DAECompiler.Intrinsics: state_ddt
using OrdinaryDiffEq
using Test

# This reduces to 3 states. two algebraic, and one not
function foo()
    (;a2, n1, n2) = variables()
    equation!(a2^2-10.0)
    equation!(state_ddt(n1) - n2)
    # this +n2 is very important since otherwise the solution has no solution since n2^2>0
    equation!(n2^2 + n2 - sin(sim_time()))
end

sys = IRODESystem(Tuple{typeof(foo)});
u0 = [5.0, 6.0, 7.0]
tspan = (0.0, 3)
prob = ODEProblem(sys, u0, tspan, nothing; jac=true)

@testset "Basic" begin
    sol = solve(prob, Rodas5P(), initializealg=CustomBrownFullBasicInit())
    @test sol.retcode == ReturnCode.Success
end

@testset "No jac_prototype" begin
    (;f, sys, jac, tgrad, paramjac, mass_matrix, observed) = prob.f
    # By default we always include a jac_prototype, but if you go through ODEForwardSensitivityFunction it loses it. See https://github.com/SciML/SciMLSensitivity.jl/issues/886
    # We replicate that here:
    func_no_jac_proto = ODEFunction(f; sys, jac_prototype=nothing, jac, tgrad, paramjac, mass_matrix, observed)
    prob_no_jac_proto = ODEProblem{true}(func_no_jac_proto, u0, tspan, nothing)
    sol = solve(prob_no_jac_proto, Rodas5P(); initializealg=CustomBrownFullBasicInit())
    # Above would have errored if we were not correctly handling the initialization.
    @test sol.retcode == ReturnCode.Success
end

end
