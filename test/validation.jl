using DAECompiler
using DAECompiler: refresh, expand_residuals

using Test
using DAECompiler
using DAECompiler.Intrinsics
using SciMLBase
using OrdinaryDiffEq

const *ᵢ = Core.Intrinsics.mul_float
const +ᵢ = Core.Intrinsics.add_float
const -ᵢ = Core.Intrinsics.sub_float

@noinline function f()
    x₁ = continuous() # selected
    x₂ = continuous() # selected
    x₃ = continuous() # algebraic, optimized away
    x₄ = continuous() # algebraic
    always!(ddt(x₁) -ᵢ x₁ *ᵢ x₂)
    always!(ddt(x₂) -ᵢ 3.0)
    always!(x₃ -ᵢ x₁) # optimized away, not part of the DAE problem.
    always!(x₄ *ᵢ x₄ -ᵢ ddt(x₁))
end

f()

# Setup SciML inputs.
u = [3.0, 1.0, 100.0, 4.0]
du = [3.0, 0.0, 0.0, 0.0]
residuals = zeros(4)
p = SciMLBase.NullParameters()
t = 1.0

# Retrieve the compressed inputs
# TODO: hardcoded for this example, we'll want to automate this
dropped_equations = 1 # the third was dropped, along with the third variable
du_compressed = du[[1, 2, 4]]
u_compressed = u[[1, 2, 4]]
residuals_compressed = zeros(length(residuals) - dropped_equations)

@testset "Validation" begin
    refresh() # TODO: remove before merge
    our_prob = DAECProblem(f, (1,) .=> 1., insert_stmt_debuginfo = true)
    sciml_prob = DiffEqBase.get_concrete_problem(our_prob, true);
    f_compressed = sciml_prob.f.f
    f_compressed(residuals_compressed, du_compressed, u_compressed, p, t)
    @test residuals_compressed == [0.0, 3.0, 13.0]

    refresh() # TODO: remove before merge
    our_prob = DAECProblem(f, (1,) .=> 1., insert_stmt_debuginfo = true, skip_optimizations = true)
    sciml_prob = DiffEqBase.get_concrete_problem(our_prob, true);
    f_original = sciml_prob.f.f
    f_original(residuals, du, u, p, t)
    @test residuals == [0.0, -3.0, 97.0, 13.0]

    residuals_recovered = expand_residuals(f, residuals_compressed, u, du, t)
    @test residuals_recovered ≈ residuals
end
