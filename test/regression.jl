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

end