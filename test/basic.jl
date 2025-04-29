module Basic

using Test
using DAECompiler
using DAECompiler.Intrinsics
using Sundials
using SciMLBase
using OrdinaryDiffEq

# We don't want to test IPO here, so directly call the intrinsic that DAECompiler models
const -ᵢ = Core.Intrinsics.sub_float

#= Simplest possible smoke test; one variable, one equation =#
@noinline function oneeq!()
    x = continuous()
    always!(ddt(x) -ᵢ x)
end

oneeq!()
sol = solve(DAECProblem(oneeq!, (1,) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

#= + Initial Condition =#
@noinline function oneeq_ic!()
    x = continuous()
    always!(ddt(x) -ᵢ x)
    initial!(x -ᵢ 1.0)
end

oneeq_ic!()
@test DAECompiler.factory(Val(DAECompiler.Settings(;mode=DAECompiler.InitUncompress)), oneeq_ic!)((;u0=Float64[])) == [1.0]
# TODO: Sundials is broken and doesn't respect the custom initialization (https://github.com/SciML/Sundials.jl/issues/469)
sol = solve(DAECProblem(oneeq_ic!), DFBDF(autodiff=false))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

end