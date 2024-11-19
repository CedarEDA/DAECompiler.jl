module sensitivity
using Test
using DAECompiler
using SciMLSensitivity
using OrdinaryDiffEqRosenbrock
using DAECompiler.Intrinsics: variable, variables, equation, equation!, state_ddt, sim_time, observed!
import FiniteDifferences

const _fdm = FiniteDifferences.central_fdm(5, 1; max_range=1e-1)

@testset "Lorenz" begin
    # Basically just like Lorenz1split from thbe lorenz.jl file,
    # but subtyping AbstractVector so it can be used as a parameter
    struct Lorenz1splitA{T} <: AbstractVector{T}
        σ::T
        ρ::T
        β::T
        unused::T
    end
    Base.size(::Lorenz1splitA) = (4,)
    Base.getindex(x::Lorenz1splitA, i::Int) = getfield(x, i)

    # x, dx/dt, y, z, a, u
    function (l::Lorenz1splitA)()
        (; x, y, z, a, b) = variables()
        let e1 = equation(),
            e2 = equation(),
            e3 = equation(),
            e4 = equation(),
            e5 = equation()

            observed!(0+1x, :one_x)
            observed!(2x, :two_x)
            observed!(10x, :ten_x)
            observed!(l.ρ^2 * x^2, :ρ_x)
            xy = x-y
            e1(b); e1(xy);
            e2(a); e2(-b); e2(-xy);
            e3(state_ddt(x)); e3(-2l.σ*b);
            e4(state_ddt(y)); e4(-x*(3l.ρ-z)); e4(y);
            e5(state_ddt(z)); e5(-x * y); e5(5l.β*z)
        end
    end

    foo = Lorenz1splitA(10.0, 28.0, 8.0/3.0, -0.0)
    foo()

    sys = IRODESystem(Tuple{typeof(foo)}, debug_config=Dict(:store_ir_levels=>true, :verify_ir_levels=>true))

    sprob = ODEForwardSensitivityProblem(sys, nothing, (0.0, 3.0), foo)
    sol = solve(sprob, Rodas5P(autodiff=false), initializealg=CustomBrownFullBasicInit(), reltol = 1e-10, abstol = 1e-10)

    # Test SciMLSensitivity.extract_local_sensitivities works as expected
    (u, (dσ, dρ, dβ, d_unused)) = extract_local_sensitivities(sol)
    # Basic sensibility test, that we got an output:
    ntime = size(u, 2)
    @test ntime >= 2
    @test size(u) == (3, ntime)
    @test size(dσ) == (3, ntime)
    @test size(dρ) == (3, ntime)
    @test size(dβ) == (3, ntime)
    @test size(d_unused) == (3, ntime)
    @test all(!isnan, u)
    @test all(!isnan, dσ)
    @test all(!isnan, dρ)
    @test all(!isnan, dβ)
    @test all(iszero, d_unused)

    # Test DAECompiler.reconstruct_sensitivities
    (dσ_dsyms, dρ_dsyms, dβ_dsyms, d_unused_dsyms) = reconstruct_sensitivities(sol, [sys.y, sys.x, sys.one_x, sys.two_x, sys.ten_x, sys.ρ_x])

    @test size(dσ_dsyms) == (6, length(sol.t))
    @test size(dρ_dsyms) == (6, length(sol.t))
    @test size(dβ_dsyms) == (6, length(sol.t))
    @test size(d_unused_dsyms) == (6, length(sol.t))

    for dparam_dsym in (dσ_dsyms, dρ_dsyms, dβ_dsyms)
        @test all(iszero, dparam_dsym[1:5, 1])  # first column is always zero for any observed!
                                                # that does not depend on params directly
        dparam_dsym = dparam_dsym[1:5, 2:end]   # skip first as it is zero

        @test all(==(1), dparam_dsym[3, :]./dparam_dsym[2, :])  # one_x/x
        @test all(≈(2), dparam_dsym[4, :]./dparam_dsym[2, :])  # two_x/x
        @test all(≈(10), dparam_dsym[5, :]./dparam_dsym[2, :])  # ten_x/x
    end

    @test all(iszero, d_unused_dsyms)

    # test passing times work
    times = sol.t[5:6]
    (dσ_dsyms_2, dρ_dsyms_2, dβ_dsyms_2, d_unused_dsyms_2) = reconstruct_sensitivities(sol, [sys.y, sys.x], times)
    @test dσ_dsyms_2 == dσ_dsyms[1:2, 5:6]
    @test dρ_dsyms_2 == dρ_dsyms[1:2, 5:6]
    @test dβ_dsyms_2 == dβ_dsyms[1:2, 5:6]
    @test d_unused_dsyms_2 == d_unused_dsyms[1:2, 5:6]


    # test passing a single time index works
    (dσ_dsyms_3, dρ_dsyms_3, dβ_dsyms_3, d_unused_dsyms_3) = reconstruct_sensitivities(sol, [sys.y, sys.x], 3)
    @test dρ_dsyms_3 ≈ dρ_dsyms[1:2, [3]]
    @test dβ_dsyms_3 ≈ dβ_dsyms[1:2, [3]]
    @test dσ_dsyms_3 ≈ dσ_dsyms[1:2, [3]]
    @test d_unused_dsyms_3 ≈ d_unused_dsyms[1:2, [3]]

    # test that passing a single time point
    t3 = sol.t[4]
    (dσ_dsyms_4, dρ_dsyms_4, dβ_dsyms_4, d_unused_dsyms_4) = reconstruct_sensitivities(sol, [sys.y, sys.x], t3)
    @test dρ_dsyms_4 == dρ_dsyms[1:2, [4]]
    @test dβ_dsyms_4 == dβ_dsyms[1:2, [4]]
    @test dσ_dsyms_4 == dσ_dsyms[1:2, [4]]
    @test d_unused_dsyms_3 == d_unused_dsyms[1:2, [4]]

    # Compare solution to FiniteDifferences approximation
    saveat = sol.t
    function wrapped_sol(p)
        sprob = ODEForwardSensitivityProblem(sys, u[:,1], (0.0, 3.0), p)
        sol = solve(sprob, Rodas5P(autodiff=false); saveat, initializealg=CustomBrownFullBasicInit(), reltol=1e-10, abstol=1e-10)
        return DAECompiler.batch_reconstruct(sol, [sys.y, sys.x, sys.one_x, sys.two_x, sys.ten_x, sys.ρ_x])
    end
    foo_vec, from_vec = FiniteDifferences.to_vec(foo)
    fdobs_dp = FiniteDifferences.jacobian(_fdm, wrapped_sol ∘ from_vec, foo_vec)[1]

    @test ≈(dσ_dsyms, reshape(fdobs_dp[:,1], size(dσ_dsyms)); rtol=1e-3, atol=1e-6)
    @test ≈(dρ_dsyms, reshape(fdobs_dp[:,2], size(dρ_dsyms)); rtol=1e-3, atol=1e-6)
    @test ≈(dβ_dsyms, reshape(fdobs_dp[:,3], size(dβ_dsyms)); rtol=1e-3, atol=1e-6)
    @test ≈(d_unused_dsyms, reshape(fdobs_dp[:,4], size(d_unused_dsyms)); rtol=1e-3, atol=1e-6)
end

end # module
