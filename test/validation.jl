using DAECompiler
using DAECompiler: refresh, compute_residual_vectors

using Test
using DAECompiler
using DAECompiler.Intrinsics

const *ᵢ = Core.Intrinsics.mul_float
const +ᵢ = Core.Intrinsics.add_float
const -ᵢ = Core.Intrinsics.sub_float

@noinline function onecall!()
    x = continuous()
    always!(ddt(x) - x)
end

function multiple_linear_equations!()
    x₁ = continuous() # selected
    x₂ = continuous() # selected
    x₃ = continuous() # algebraic, optimized away
    x₄ = continuous() # algebraic
    always!(ddt(x₁) -ᵢ x₁ *ᵢ x₂)
    always!(ddt(x₂) -ᵢ 3.0)
    always!(x₃ -ᵢ x₁) # optimized away, not part of the DAE problem.
    always!(x₄ *ᵢ x₄ -ᵢ ddt(x₁))
end

@noinline function sin!()
    x = continuous()
    always!(ddt(x) - sin(x))
end

@noinline function neg_sin!()
    x = continuous()
    always!(sin(x) - ddt(x))
end

function twocall!()
    onecall!(); onecall!();
    return nothing
end

function sin2!()
    sin!(); sin!();
    return nothing
end

@noinline new_equation() = always()
function external_equation!()
    x = continuous()
    eq = new_equation()
    eq(ddt(x) - 1.0)
end

@noinline function flattening_inner!((a, b))
    x = continuous()
    always!(ddt(x) - (a + b))
end
function flattening!()
    x = continuous()
    flattening_inner!((1.0, x))
    always!(ddt(x) - 2.0)
end

@noinline apply_equation(eq, residual) = eq(residual)
function equation_argument!()
    x = continuous()
    equation = new_equation()
    apply_equation(equation, x - ddt(x))
end

@noinline new_equation_and_variable() = (always(), continuous())
@noinline apply_equation_on_ddtx_minus_one(eq, x) = apply_equation(eq, ddt(x) - 1.)
function equation_used_multiple_times!()
    (eq, x) = new_equation_and_variable()
    apply_equation_on_ddtx_minus_one(eq, x)
    apply_equation_on_ddtx_minus_one(eq, x)
end

function equation_with_callable!()
    x = continuous()
    callable = @noinline Returns(x)
    always!(ddt(callable()) - 3.0)
end

@noinline apply_equation!(lhs, rhs) = always!(lhs - rhs)
function nonlinear_argument!()
    x = continuous()
    apply_equation!(ddt(x), sin(x))
end

struct WithParameter{N} end
@noinline (::WithParameter{N})(eq, x) where {N} = eq(ddt(x) - N)
function callable_with_type_parameter!()
    eq, x = new_equation_and_variable()
    WithParameter{3}()(eq, x)
end

@testset "Validation" begin
    refresh() # TODO: remove before merge

    u = [2.0]
    du = [3.0]
    residuals, expanded_residuals = compute_residual_vectors(onecall!, u, du)
    @test residuals ≈ [1.0]
    @test residuals ≈ expanded_residuals

    u = [3.0, 1.0, 100.0, 4.0]
    du = [3.0, 0.0, 0.0, 0.0]
    residuals, expanded_residuals = compute_residual_vectors(multiple_linear_equations!, u, du)
    @test residuals ≈ [0.0, -3.0, 97.0, 13.0]
    @test residuals ≈ expanded_residuals

    u = [2.0]
    du = [3.0]
    residuals, expanded_residuals = compute_residual_vectors(sin!, u, du)
    @test residuals ≈ du .- sin.(u)
    @test residuals ≈ expanded_residuals

    u = [2.0]
    du = [3.0]
    residuals, expanded_residuals = compute_residual_vectors(neg_sin!, u, du)
    @test residuals ≈ sin.(u) .- du
    @test residuals ≈ expanded_residuals

    # IPO

    u = [2.0]
    du = [3.0]
    residuals, expanded_residuals = compute_residual_vectors(() -> onecall!(), u, du)
    @test residuals ≈ [1.0]
    @test residuals ≈ expanded_residuals

    u = [2.0, 4.0]
    du = [3.0, 7.0]
    residuals, expanded_residuals = compute_residual_vectors(twocall!, u, du)
    @test residuals ≈ [1.0, 3.0]
    @test residuals ≈ expanded_residuals

    u = [2.0, 4.0]
    du = [1.0, 1.0]
    residuals, expanded_residuals = compute_residual_vectors(sin2!, u, du)
    @test all(>(0), residuals)
    @test residuals ≈ expanded_residuals

    u = [2.0]
    du = [4.0]
    residuals, expanded_residuals = compute_residual_vectors(nonlinear_argument!, u, du)
    @test residuals ≈ du .- sin.(u)
    @test residuals ≈ expanded_residuals

    u = [0.0]
    du = [2.0]
    residuals, expanded_residuals = compute_residual_vectors(external_equation!, u, du)
    @test residuals ≈ [1.0]
    @test residuals ≈ expanded_residuals

    u = [2.0]
    du = [1.0]
    residuals, expanded_residuals = compute_residual_vectors(equation_argument!, u, du)
    @test residuals ≈ [1.0]
    @test residuals ≈ expanded_residuals

    u = [2.0, 4.0]
    du = [1.0, 1.0]
    residuals, expanded_residuals = compute_residual_vectors(flattening!, u, du)
    @test residuals ≈ [-1.0, -2.0]
    @test residuals ≈ expanded_residuals

    u = [2.0]
    du = [4.0]
    residuals, expanded_residuals = compute_residual_vectors(equation_used_multiple_times!, u, du)
    @test residuals ≈ [6.0]
    @test residuals ≈ expanded_residuals

    u = [2.0]
    du = [4.0]
    residuals, expanded_residuals = compute_residual_vectors(equation_with_callable!, u, du)
    @test residuals ≈ [1.0]
    @test residuals ≈ expanded_residuals

    u = [2.0]
    du = [3.0]
    residuals, expanded_residuals = compute_residual_vectors(callable_with_type_parameter!, u, du)
    @test residuals ≈ [0.0]
    @test residuals ≈ expanded_residuals
end;
