using DAECompiler
using DAECompiler: refresh

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
du = [2.0, 0.0, 0.0, 0.0]
residuals = zeros(4)
p = SciMLBase.NullParameters()
t = 1.0

# Retrieve the compressed inputs
# TODO: hardcoded for this example, we'll want to automate this
dropped_equations = 1 # the third was dropped, along with the third variable
du_compressed = du[[1, 2, 4]]
u_compressed = u[[1, 2, 4]]
residuals_compressed = zeros(length(residuals) - dropped_equations)

refresh()
our_prob = DAECProblem(f, (1,) .=> 1., insert_stmt_debuginfo = true)
sciml_prob = DiffEqBase.get_concrete_problem(our_prob, true);
f_compressed = sciml_prob.f.f
f_compressed(residuals_compressed, du_compressed, u_compressed, p, t)
# XXX: `ddt(x₁)` gets substituted by `x₁ *ᵢ x₂` for the last equation after scheduling.
# XXX: I believe the sign differences with `residuals` below are due to solving for
# the corresponding variable in a variable-equation matching pair, therefore the negation
# will depend on whether the solved variable appears with a positive or negative factor.
# For example: ẋ₁ - x₁x₂ = 0
#                    -ẋ₁ = -x₁x₂
#                     ẋ₁ = -x₁x₂/-1
#                     ẋ₁ = x₁x₂
#                      0 = x₁x₂ - ẋ₁   <-- residual
# Therefore, if a linear solved term appears with a positive coefficient,
# the residual will be taken as the negative of the value provided to `always!`.
# Empirical evidence validates this conjecture.
@test residuals_compressed == [1.0, 3.0, 13.0]

refresh()
our_prob = DAECProblem(f, (1,) .=> 1., insert_stmt_debuginfo = true, skip_optimizations = true)
sciml_prob = DiffEqBase.get_concrete_problem(our_prob, true);
f_original = sciml_prob.f.f
f_original(residuals, du, u, p, t)
@test residuals == [-1.0, -3.0, 97.0, 14.0]
# -> the third equation is removed, so this entry in `f_original` can't be reliably tested,
# unless added back somehow with `expand_residual` in a way that reflects the variable solve
# (otherwise if we solve it wrong we won't catch the issue).

# residuals_recovered = expand_residual(residuals_compressed, ...)
# @test residuals_recovered ≈ residuals
