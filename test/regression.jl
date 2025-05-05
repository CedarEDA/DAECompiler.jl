module Regression

using Test
using DAECompiler
using DAECompiler.Intrinsics
using Sundials
using SciMLBase
using OrdinaryDiffEq

const cf = Base.copysign_float
const ief = Core.ifelse
const -ᵢ = Core.Intrinsics.sub_float
const oif = Core.Intrinsics.or_int

function tfb1()
    x = continuous()
    b = (x < 43200.)
    v = ief(b, x, cf(0., x))
    always!(ddt(x) -ᵢ v)
end

sol = solve(DAECProblem(tfb1, (1,) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

function tfb1_ipo()
    x = continuous()
    always!(ddt(x) - (x < 43200.)*x)
end

sol = solve(DAECProblem(tfb1_ipo, (1,) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

function tfb2()
    x = continuous()

    fx = Float64(-100)
    b = oif(fx < x,
        ((fx == x) & oif((fx == Float64(typemax(Int64))), (-100 < unsafe_trunc(Int64,fx)) )))

    always!(ddt(x) - b*x)
end
sol = solve(DAECProblem(tfb2, (1,) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

function tfb2_ipo()
    x = continuous()
    always!(ddt(x) - (-100 < x)*x)
end
sol = solve(DAECProblem(tfb2_ipo, (1,) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

function tfb3()
    x = continuous()
    r = Base.fma_emulated(x,x,x)
    always!(ddt(x) - r)
end
sol = solve(DAECProblem(tfb3, (1,) .=> 1., (0, 0.1)), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], -exp.(sol.t)./(exp.(sol.t) .- 2)))

function tfb4()
    x = continuous()
    r = Core.ifelse(Core.Intrinsics.have_fma(Float64), Base.fma_float(x,x,x), Base.fma_emulated(x,x,x))
    always!(ddt(x) - r)
end
sol = solve(DAECProblem(tfb4, (1,) .=> 1., (0, 0.1)), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], -exp.(sol.t)./(exp.(sol.t) .- 2)))

function tfb5()
    x = continuous()
    always!(ddt(x) - log(1. + sim_time()))
end
sol = solve(DAECProblem(tfb5, (1,) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], (sol.t .+ 1) .* log.(sol.t .+ 1) .+ 1 .- sol.t))

end