using DAECompiler
using DAECompiler: refresh, compute_residual_vectors

using Test
using DAECompiler
using DAECompiler.Intrinsics
using SciMLBase
using OrdinaryDiffEq

const *ᵢ = Core.Intrinsics.mul_float
const +ᵢ = Core.Intrinsics.add_float
const -ᵢ = Core.Intrinsics.sub_float

function f()
    x₁ = continuous() # selected
    x₂ = continuous() # selected
    x₃ = continuous() # algebraic, optimized away
    x₄ = continuous() # algebraic
    always!(ddt(x₁) -ᵢ x₁ *ᵢ x₂)
    always!(ddt(x₂) -ᵢ 3.0)
    always!(x₃ -ᵢ x₁) # optimized away, not part of the DAE problem.
    always!(x₄ *ᵢ x₄ -ᵢ ddt(x₁))
end

@noinline function onecall!()
    x = continuous()
    always!(ddt(x) - x)
end

function twocall!()
    onecall!(); onecall!();
    return nothing
end

@testset "Validation" begin
    refresh() # TODO: remove before merge

    u = [2.0]
    du = [3.0]
    residuals, expanded_residuals = compute_residual_vectors(onecall!, u, du; t = 1.0)
    @test residuals ≈ [1.0]
    @test residuals ≈ expanded_residuals

    u = [3.0, 1.0, 100.0, 4.0]
    du = [3.0, 0.0, 0.0, 0.0]
    residuals, expanded_residuals = compute_residual_vectors(f, u, du; t = 1.0)
    @test residuals ≈ [0.0, -3.0, 97.0, 13.0]
    @test residuals ≈ expanded_residuals

    # IPO

    u = [2.0]
    du = [3.0]
    residuals, expanded_residuals = compute_residual_vectors(() -> onecall!(), u, du; t = 1.0)
    @test residuals ≈ [1.0]
    @test residuals ≈ expanded_residuals

    u = [2.0, 4.0]
    du = [3.0, 7.0]
    residuals, expanded_residuals = compute_residual_vectors(twocall!, u, du; t = 1.0)
    @test residuals ≈ [1.0, 3.0]
    @test residuals ≈ expanded_residuals
end;
