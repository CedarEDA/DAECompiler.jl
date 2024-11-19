module eps

using Test
using DAECompiler
using DAECompiler.Intrinsics
using DAECompiler.Intrinsics: state_ddt
using SciMLBase, OrdinaryDiffEqRosenbrock

@testset "expeps" begin
    function expeps()
        (;x) = variables()
        ε = epsilon(:ε)
        equation!(x - state_ddt(x) + ε*x)
    end
    sys = IRODESystem(Tuple{typeof(expeps)})
    prob = ODEProblem(sys, [1.], (0., 1.); jac=true)
    sol = solve(prob, Rosenbrock23())

    @test get_transformed_sys(sol).state.neps == 1
    @test get_transformed_sys(sol).state.names[:ε].eps == 1

    epsjac! = get_epsjac(sol)
    eps = epsjac!.([Matrix{Float64}(undef, 1, 1) for _ in sol.u], sol.u, expeps, sol.t)
    @test map(vec, eps) == sol.u
end

@testset "time linked" begin
    function time_linked()
        (; x, y) = variables()
        ε = epsilon(:ε)
        equation!(sin(y) - sin(sim_time()) + ε*sim_time())
        equation!(x - state_ddt(x))
    end
    sys = IRODESystem(Tuple{typeof(time_linked)})
    prob = ODEProblem(sys, nothing, (0., 1.); jac=true)
    sol = solve(prob, Rosenbrock23())

    @test get_transformed_sys(sol).state.neps == 1
    @test get_transformed_sys(sol).state.names[:ε].eps == 1

    epsjac! = get_epsjac(sol)
    eps_vals = epsjac!.([Matrix{Float64}(undef, 2, 1) for _ in sol.u], sol.u, time_linked, sol.t)

    # ε and x are independent
    @test all(iszero(dx) for (dx, _) in eps_vals)

    # y == t == ε
    @assert isapprox(sol[2, :], sol.t, rtol=1e-6) # make sure the system solved as expected
    @test last.(eps_vals) ≈ sol.t rtol=1e-6
end

end
