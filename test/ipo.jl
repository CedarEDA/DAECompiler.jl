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

#==== Contained Equations ============#
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

#============== NonLinear ============#
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

#============== SICM ============#
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

#========== NonLinear SICM ===========#
struct nlsicm!
    arg::Float64
end

@noinline function (this::nlsicm!)()
    x = continuous()
    always!(ddt(x) - sin(this.arg))
end

struct nlsicm2!
    a::Float64
    b::Float64
end

function (this::nlsicm2!)()
    nlsicm!(this.a)(); nlsicm!(this.b)();
    return nothing
end
dae_sol = solve(DAECProblem(nlsicm2!(1., 1.), (1, 2) .=> 1.), IDA())
ode_sol = solve(ODECProblem(nlsicm2!(1., 1.), (1, 2) .=> 1.), Rodas5(autodiff=false))
for (sol, i) in Iterators.product((dae_sol, ode_sol), 1:2)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[i, :], 1. .+ sin(1.)*sol.t))
end

#============== Ping Pong ============#
@noinline function ping(a, b, c, d)
    always!(b - sin(a))
    always!(d - sin(c))
end

@noinline function pong(a, b, c, d)
    always!(b - asin(a))
    always!(ddt(d) - asin(c))
end

function pingpong()
    # N.B.: Deliberate not using variables, which requires
    # scope, etc handling
    a = continuous()
    b = continuous()
    c = continuous()
    d = continuous()
    ping(a, b, c, d)
    pong(b, c, d, a)
end
dae_sol = solve(DAECProblem(pingpong, (1,) .=> 0.1), IDA())
ode_sol = solve(ODECProblem(pingpong, (1,) .=> 0.1), Rodas5(autodiff=false))

# asin(sin) are inverses in [-pi/2, pi/2]
for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 0.1exp.(sol.t)))
end

#========== Implicit External ===========#
@noinline intro() = ddt(continuous())
@noinline outro!(x) = always!(x-1)

implicit() = outro!(intro())

dae_sol = solve(DAECProblem(implicit, (1,) .=> 1), IDA())
ode_sol = solve(ODECProblem(implicit, (1,) .=> 1), Rodas5(autodiff=false))

for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 1 .+ sol.t))
end

#========== Structured SICM ===========#
@noinline function add4((a, b, c, d)::NTuple{4, Float64})
    always!((a + b) + (c + d))
end

function structured_sicm()
    x = continuous()
    add4((1., 2., ddt(x), -x))
end

dae_sol = solve(DAECProblem(structured_sicm, (1,) .=> 1), IDA())
ode_sol = solve(ODECProblem(structured_sicm, (1,) .=> 1), Rodas5(autodiff=false))

for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], 3 .- 2exp.(sol.t)))
end

end
