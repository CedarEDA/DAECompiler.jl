module ssm

using Test
using DAECompiler
using DAECompiler.Intrinsics
using Sundials
using SciMLBase
using OrdinaryDiffEq

const +ᵢ = Core.Intrinsics.add_float
const -ᵢ = Core.Intrinsics.sub_float

function ssm2()
    a = continuous()
    b = continuous()
    abdot = ddt(a +ᵢ b)
    always!(a -ᵢ abdot)
    always!(b +ᵢ abdot)
end

ssm2()

# TODO: Currently broken
# solve(DAECProblem(ssm2, (1,) .=> 1.), DFBDF(autodiff=false))

# This is example (7.30) from Taihei Oki "Computing Valuations of Determinants via Combinatorial Optimization: Applications to Differential Equations".
# The system is index 4 and requires iterating pantelides/ssm.
function ssm4()
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
    always!(ẍ₁+ẍ₂-(ẋ₁+ẋ₂)+x₄)
    always!(ẍ₁+ẍ₂+x₃)
    always!(x₂+ẍ₃+ẋ₄)
    always!(x₃+ẋ₄)
end

ssm4()

# TODO: Currently broken
# solve(DAECProblem(ssm4, (1,) .=> 1.), DFBDF(autodiff=false))
end