module ssm

using Test
using DAECompiler
using DAECompiler.Intrinsics
using Sundials
using SciMLBase
using OrdinaryDiffEq

const +ᵢ = Core.Intrinsics.add_float
const -ᵢ = Core.Intrinsics.sub_float

function ssrm2()
    a = continuous()
    b = continuous()
    abdot = ddt(a +ᵢ b)
    always!(a -ᵢ abdot)
    always!(b +ᵢ abdot)
end

ssrm2()
# DFBDF doens't like the degenerate case
@test isempty(DAECompiler.factory(ssrm2)[2])
@test_broken solve(DAECProblem(ssrm2, ()), DFBDF())

# This is example (7.30) from Taihei Oki "Computing Valuations of Determinants via Combinatorial Optimization: Applications to Differential Equations".
# The system is index 4 and requires iterating pantelides/ssm.
function ssrm4()
    x₁ = continuous()
    x₂ = continuous()
    x₃ = continuous()
    x₄ = continuous()
    ẋ₁ = ddt(x₁)
    ẋ₂ = ddt(x₂)
    ẋ₃ = ddt(x₃)
    ẋ₄ = ddt(x₄)
    ẍ₁ = ddt(ẋ₁)
    ẍ₂ = ddt(ẋ₂)
    ẍ₃ = ddt(ẋ₃)
    always!((ẍ₁+ẍ₂)-(ẋ₁+ẋ₂)+x₄)
    always!((ẍ₁+ẍ₂)+x₃)
    always!(x₂+ẍ₃+ẋ₄)
    always!(x₃+ẋ₄)
end

ssrm4()
# We expect state selection here to pick (x₁, x₄, ẋ₁)
# The system simplifies to ẍ₁ = ẋ₄ = ẋ₁ - x₄
init = (1.,0.,1.)
function analytic(init, t)
    c = init[3] - init[2]
    ẋ₁ = init[3] + c*t
    x₄ = init[2] + c*t
    x₁ = init[1] + init[3]*t + 1/2*c*t^2
    return (x₁, x₄, ẋ₁)
end
sol = solve(DAECProblem(ssrm4, (1,2,3) .=> init), DFBDF(autodiff=false))
@test isapprox(sol[:, :]', mapreduce(t->[analytic(init, t)...]', vcat, sol.t), atol=1e-4)

end
