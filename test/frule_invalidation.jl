module frule_invalidation

module frule_invalidation_util

export UTIL_EX, correct_res

const UTIL_EX = quote

using DAECompiler, ModelingToolkit, SciMLBase, OrdinaryDiffEq, Sundials
using DAECompiler.Intrinsics
using DAECompiler.Intrinsics: state_ddt

struct Lorenz1{use_custom_op}
    σ::Float64
    ρ::Float64
    β::Float64
end

x::Number ⊖ y::Number = x ⊟ y
x::Number ⊟ y::Number = x - y # invalidation target

function (lorenz::Lorenz1{use_custom_op})() where use_custom_op
    (; x, y, z, a, u) = variables()
    ∸ = use_custom_op ? (⊖) : (-)
    equation!.((
        u ∸ (y ∸ x),       # test tearing
        a ∸ (u ∸ (y ∸ x)), # test a == 0
        state_ddt(x) ∸ (lorenz.σ * u),
        state_ddt(y) ∸ (x * (lorenz.ρ - z) - y),
        state_ddt(z) ∸ (x * y - lorenz.β * z)
    ))
end

function jac_lorenz1(use_custom_op=true)
    x = Lorenz1{use_custom_op}(10.0, 28.0, 8.0/3.0)
    sys = IRODESystem(Tuple{typeof(x)});
    jac! = DAECompiler.construct_jacobian(sys, isdae=false)
    return jac!(zeros(10,10), [1.0:10.0;], x, 1.0)
end

const correct_res = jac_lorenz1(#=use_custom_op=#false)

end # const UTIL_EX = quote

end # module frule_invalidation_util


module frule_invalidation_changed

using ..frule_invalidation_util
using Test, ChainRulesCore

@eval (@__MODULE__) $UTIL_EX

const Ωs1, Ωs2 = Number[], Number[]

# the first (wrong) frule (essentially same as the one for `x + y`)
function ChainRulesCore.frule((_, Δ1, Δ2), ::typeof(⊟), x::Number, y::Number)
    Ω = x ⊟ y
    push!(Ωs1, Ω)
    return (Ω, (ChainRulesCore.muladd(true, Δ2, true * Δ1)))
end

empty!(Ωs1); empty!(Ωs2);
@test jac_lorenz1() ≠ correct_res
@test !isempty(Ωs1) && isempty(Ωs2)

# the second (correct) frule (essentially same as the one for `x - y`)
function ChainRulesCore.frule((_, Δ1, Δ2), ::typeof(⊟), x::Number, y::Number)
    Ω = x ⊟ y
    push!(Ωs2, Ω)
    return (Ω, (ChainRulesCore.muladd(-1, Δ2, true * Δ1)))
end

empty!(Ωs1); empty!(Ωs2);
@test jac_lorenz1() == correct_res
@test isempty(Ωs1) && !isempty(Ωs2)

end # module frule_invalidation_changed


module frule_invalidation_noexist

using ..frule_invalidation_util
using Test, ChainRulesCore

@eval (@__MODULE__) $UTIL_EX

const Ωs = Number[]

empty!(Ωs)
@test jac_lorenz1() == correct_res
@test isempty(Ωs)

# now define a custom frule for `⊟`
function ChainRulesCore.frule((_, Δ1, Δ2), ::typeof(⊟), x::Number, y::Number)
    Ω = x ⊟ y
    push!(Ωs, Ω)
    return (Ω, (ChainRulesCore.muladd(-1, Δ2, true * Δ1)))
end

empty!(Ωs)
@test jac_lorenz1() == correct_res
@test !isempty(Ωs)

end # module frule_invalidation_noexist

end # module frule_invalidation
