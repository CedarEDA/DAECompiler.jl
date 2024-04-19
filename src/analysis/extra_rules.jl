using ForwardDiff
using ChainRulesCore

ForwardDiff.can_dual(::Type{<:ChainRulesCore.AbstractZero}) = true

# ChainRules expects the tangent type of a <: Number type to be itself
# <: Number. This rule makes that happen for ForwardDiff
function ChainRulesCore.frule((_, da, db), T::Type{ForwardDiff.Dual{Tag}} where Tag, a::Number, b::Number)
    _da = isa(da, ZeroTangent) ? zero(a) : da
    _db = isa(db, ZeroTangent) ? zero(b) : db
    (T(a, b), T(_da, _db))
end

function ChainRulesCore.frule((_, da, db), T::Type{ForwardDiff.Dual{Tag}} where Tag, a::Number, b::NTuple{N}) where {N}
    y = T(a, b)
    _da = isa(da, ZeroTangent) ? zero(a) : da
    ∂y = T(_da, ntuple(i->isa(db[i], ZeroTangent) ? zero(a) : db[i], Val{N}()))
    return y, ∂y
end

function ChainRulesCore.frule((_, dv, dp), T::Type{<:ForwardDiff.Dual}, v::Number, p::ForwardDiff.Partials)
    y = T(v, p)
    ∂y = T(dv, ensure_ForwardDiffPartials(dp))
    return y, ∂y
end

ensure_ForwardDiffPartials(dp::ForwardDiff.Partials) = dp
ensure_ForwardDiffPartials(dp::Tangent{P}) where P<:ForwardDiff.Partials = P(ChainRulesCore.backing(dp.values))

# Disable chainrule for getindex, since it messes with our SROA elimination
function ChainRulesCore.frule(::Any, ::typeof(getindex), a::Tuple, i::Union{Int64, Int32})
    nothing
end


# Right now the ODESolutionisn't supported by zero_tangent as it is recursive
# Just don't try and use a structural tangent for it.
ChainRulesCore.zero_tangent(::ODESolution) = ChainRulesCore.ZeroTangent()
