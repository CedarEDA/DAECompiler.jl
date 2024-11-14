module Basic

using Test
using DAECompiler
using DAECompiler.Intrinsics
using Sundials
using SciMLBase

#= Simplest possible smoke test; one variable, one equation =#
@noinline function oneeq!()
    x = continuous()
    # sub_float here to not have to test any IPO functionality for plain `x!`
    always!(Base.sub_float(ddt(x), x))
end

oneeq!()
sol = solve(DAECProblem(oneeq!, (1,) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

end