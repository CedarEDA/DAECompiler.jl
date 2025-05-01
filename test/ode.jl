using Test
using DAECompiler
using DAECompiler.Intrinsics
using DAECompiler: refresh
using SciMLBase
using OrdinaryDiffEq

const -ᵢ = Core.Intrinsics.sub_float

@noinline function oneeq!()
    x = continuous()
    always!(ddt(x) -ᵢ x)
end

oneeq!()
refresh(); sol = solve(ODECProblem(oneeq!, (1,) .=> 1.), Rodas5())
# @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

# # For reference, we want to codegen something equivalent to that

# function f!(du, u, p, t)
#     du .= u
# end

# odef = ODEFunction(f!; mass_matrix = ones(1, 1))
# prob = ODEProblem(odef, [1.0], (0.0, 1.0))
# sol = solve(prob, Rodas5())
