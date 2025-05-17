module Basic

using Test
using DAECompiler
using DAECompiler.Intrinsics
using Sundials
using SciMLBase
using OrdinaryDiffEq

function unblanaced_too_many_eqs()
    x = continuous(#= :x =#)
    always!(x, #= :foo=#)
    always!(x^2 #= Scope(Scope(Scope(), :Bar), :qux) =#)
end
@test_throws ["The system is unbalanced.",
    "There are 1 highest order differentiated variable(s) and 2 equation(s).",
    r"(▫.Bar.qux)|(▫.foo)",
    "was potentially redundant:"] solve(ODECProblem(unblanaced_too_many_eqs, (1,) .=> 1.), Rodas5(autodiff=false))

function structurally_singular()
    x = continuous(#= :x =#)
    u1 = continuous(#= :u1 =#)
    u2 = continuous(#= :u2 =#)
    always!.((x + sin(u1 + u2),
        ddt(x) - (x + 1.0),
        cos(x) - 2.0))
end
@test_throws "The system is structurally singular.\nThis variable may be problematic:" solve(ODECProblem(structurally_singular, (1,) .=> 1.), Rodas5(autodiff=false))

end