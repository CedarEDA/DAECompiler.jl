function diff_bundle(bundle::Diffractor.UniformBundle{N, B, U}) where {N, B, U}
    return Diffractor.UniformBundle{N-1}(bundle.tangent.val, bundle.tangent)
end

function diff_bundle(bundle::Diffractor.TaylorBundle{N}) where {N}
    return Diffractor.TaylorBundle{N-1}(bundle.tangent.coeffs[1], bundle.tangent.coeffs[2:end])
end