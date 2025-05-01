module ipo

using Test
using DAECompiler
using DAECompiler.Intrinsics
using Sundials
using SciMLBase

#= Basic IPO: We need to read the incidence of the contained `-` =#
@noinline function onecall!()
    x = continuous()
    always!(ddt(x) - x)
end

onecall!()
sol = solve(DAECProblem(onecall!, (1,) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], exp.(sol.t)))

#= + Contained Equations =#
function twocall!()
    onecall!(); onecall!();
    return nothing
end

twocall!()
sol = solve(DAECProblem(twocall!, (1, 2) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[1, :], exp.(sol.t)))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol[2, :], exp.(sol.t)))

end
