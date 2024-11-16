using Test
using DAECompiler, SciMLBase, Sundials, OrdinaryDiffEqBDF
using DAECompiler: equation!, state_ddt, variables

struct Robertson{T}
    k1::T
    k2::T
    k3::T
end

function (p::Robertson)()
    (; x, y, z) = variables()

    equation!.((
        -state_ddt(x) + -p.k1*x + p.k2*y*z,
        -state_ddt(y) + p.k1*x - p.k2*y*z - p.k3*y^2,
        x + y + z - 1,
    ))
end

# Ensure that we hit all the important points regions of behavior for this system:
@testset "Robertson" begin
    # TODO: Figure out a good way to determine what `u0` should be
    u0 = [1.0, 0.0]
    tspan = (0.0, 1e3)
    rober = Robertson(0.04, 10.0^4, 3*10.0^7)
    sys = IRODESystem(Tuple{typeof(rober)});
    daeprob = DAEProblem(sys, zero(u0), u0, tspan, rober);
    daesol = solve(daeprob, IDA())

    odeprob = ODEProblem(sys, u0, tspan, rober);
    odesol = solve(odeprob, FBDF(autodiff=false), abstol=1e-7)

    for (sol, name) in ((daesol, "DAEProblem"), (odesol, "ODEProblem"))
        @testset "$(name)" begin
            # At the beginning, `x` is 1, `y` and `z` are zero:
            idxs = [sys.x, sys.y, sys.z]
            @test sol(0.0; idxs) ≈ [1.0, 0.0, 0.0]

            # There is a small hump very early on, in `y`, between `0.002` and `0.2`:
            @test all(sol.(LinRange(0.002, 0.2, 10); idxs=sys.y) .> 3.5e-5)

            # Assert that outside of that hump, `y` is small:
            @test all(sol.(LinRange(tspan[1], 0.001, 100); idxs=sys.y) .< 3.5e-5)
            @test all(sol.(LinRange(1.0, tspan[2], 100); idxs=sys.y) .< 3.5e-5)

            # Assert after a certain period of time, `x` is small and `z` is large:
            @test all(sol.(LinRange(300.0, tspan[2], 100); idxs=sys.x) .< 0.5)
            @test all(sol.(LinRange(300.0, tspan[2], 100); idxs=sys.z) .> 0.5)
        end
    end
end
