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

function tfb1()
    x = continuous()
    b = (x < 43200.)
    v = ief(b, x, cf(0., x))
    always!(ddt(x) -ᵢ v)
end

sol = solve(DAECProblem(tfb1, (1,) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

end