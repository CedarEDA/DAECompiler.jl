@inline zero!(x::A) where A<:AbstractArray = fill!(x, zero(eltype(A)))::A
@inline zero!(x::Array{Float64}) = fill!(x, 0.0)
