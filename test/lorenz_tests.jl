module lorenz

using Test
using DAECompiler, SciMLBase, OrdinaryDiffEqRosenbrock, OrdinaryDiffEqBDF, Sundials
include(joinpath(Base.pkgdir(DAECompiler), "test", "testutils.jl"))
include(joinpath(Base.pkgdir(DAECompiler), "test", "lorenz.jl"))

# DAE IR
u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 100.0)
x = Lorenz1ts(10.0, 28.0, 8.0/3.0)
sys = IRODESystem(Tuple{typeof(x)});
daeprob = DAEProblem(sys, zero(u0), u0, tspan, x);
daesol = solve(daeprob, IDA())
@test daesol.retcode == ReturnCode.Success
@test all(x->abs(x) < 100, daesol)
@test length(daeprob.kwargs[:callback].continuous_callbacks) == 1
# Test that we get _two_ points that are at 50.0, due to `insert_pre_discontinuity_points`
function test_n_close_points(t₀, ts, N)
    @test all(sort(abs.(ts .- t₀))[1:N] .< 1e-12)
end
function test_no_close_points(t₀, ts)
    @test sort(abs.(ts .- t₀))[1] .> 1e-12
end
test_n_close_points(50, daesol.t, 2)

x2 = Lorenz1ts(10.0, 29.0, 8.0/3.0)
daeprob2 = remake(daeprob, p=x2)
daesol2 = solve(daeprob, IDA())
@test daesol2.retcode == ReturnCode.Success
@test all(x->abs(x) < 100, daesol2)

# Split equation test
u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 100.0)
x = Lorenz1split(10.0, 28.0, 8.0/3.0)
sys = IRODESystem(Tuple{typeof(x)});
daeprob = DAEProblem(sys, zero(u0), u0, tspan, x);
daesol = solve(daeprob, IDA())
@test daesol.retcode == ReturnCode.Success
@test all(x->abs(x) < 100, daesol)

# ODE IR events
# No DAE due to https://github.com/SciML/OrdinaryDiffEq.jl/issues/1887
# No IDA due to its vector type
x3 = Lorenz1cb(10.0, 28.0, 8.0/3.0)
sys = IRODESystem(Tuple{typeof(x3)});
prob = ODEProblem(sys, u0, tspan, x3);
sol = solve(prob, Rodas5P(autodiff=false))
@test sol.retcode == ReturnCode.Success
@test all(x->abs(x) < 100, sol)
@test length(prob.kwargs[:callback].continuous_callbacks) === 1
@test length(prob.kwargs[:callback].discrete_callbacks) === 1

# Test that our callback points (and the pre-discontinuity points
# inserted by `insert_pre_discontinuity_points`) are present
for idx in 0:3
    test_n_close_points(5 + 10*idx, sol.t, 2)
end
test_n_close_points(13.37, sol.t, 2)
test_no_close_points(2*13.37, sol.t)

test_n_close_points(50/3, sol.t, 1)
# Check RHS can be invoked with both Vector and slice view of a vector
@test prob.f.f(copy(u0), u0, x3, first(tspan)) isa Any
@test prob.f.f(@view(copy(u0)[1:end]), @view(u0[1:end]), x3, first(tspan))  isa Any



using StaticArrays

su0 = @MVector Float64[1.0, 0.0, 0.0]
sys = IRODESystem(Tuple{typeof(x)});
prob = ODEProblem(sys, su0, tspan, x);
sol = solve(prob, FBDF(autodiff=false))
@test sol.retcode == ReturnCode.Success
@test all(x->abs(x) < 100, sol)
end
