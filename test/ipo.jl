module ipo

using Test
using DAECompiler
using DAECompiler.Intrinsics
using Sundials
using SciMLBase
using OrdinaryDiffEq

#= Basic IPO: We need to read the incidence of the contained `-` =#
@noinline function onecall!()
    x = continuous()
    always!(ddt(x) - x)
end

onecall!()
sol = solve(DAECProblem(onecall!, (1,) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], exp.(sol.t)))
sol = solve(ODECProblem(onecall!, (1,) .=> 1.), Rodas5(autodiff=false))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], exp.(sol.t)))

#= + Contained Equations =#
function twocall!()
    onecall!(); onecall!();
    return nothing
end

twocall!()
dae_sol = solve(DAECProblem(twocall!, (1, 2) .=> 1.), IDA())
ode_sol = solve(ODECProblem(twocall!, (1, 2) .=> 1.), Rodas5(autodiff=false))
for (sol, i) in Iterators.product((dae_sol, ode_sol), 1:2)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[i, :], exp.(sol.t)))
end

#= + NonLinear =#
@noinline function sin!()
    x = continuous()
    always!(ddt(x) - sin(x))
end
function sin2!()
    sin!(); sin!();
    return nothing
end
dae_sol = solve(DAECProblem(sin2!, (1, 2) .=> 1.), IDA())
ode_sol = solve(ODECProblem(sin2!, (1, 2) .=> 1.), Rodas5(autodiff=false))
for (sol, i) in Iterators.product((dae_sol, ode_sol), 1:2)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[i, :], 2*acot.(exp.(-sol.t).*cot(1/2))))
end

#= + SICM =#
struct sicm!
    arg::Float64
end

@noinline function (this::sicm!)()
    x = continuous()
    always!(ddt(x) - this.arg)
end

struct sicm2!
    a::Float64
    b::Float64
end

function (this::sicm2!)()
    sicm!(this.a)(); sicm!(this.b)();
    return nothing
end
dae_sol = solve(DAECProblem(sicm2!(1., 1.), (1, 2) .=> 1.), IDA())
ode_sol = solve(ODECProblem(sicm2!(1., 1.), (1, 2) .=> 1.), Rodas5(autodiff=false))
for (sol, i) in Iterators.product((dae_sol, ode_sol), 1:2)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[i, :], 1. .+ sol.t))
end


end
