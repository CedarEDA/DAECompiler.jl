using Diffractor: ZeroBundle, ATB, TangentBundle, primal, partial, TaylorBundle, UniformBundle, TaylorTangent
using ChainRulesCore: Tangent, ZeroTangent
struct BatchOfBundles{N, P, NT <: Tuple}
    # This can be massively generalized, but lets keep specific and simple for now
    primal::P
    batched_partials::NT

    @eval @inline function BatchOfBundles(primal::P, partials) where {P}
        N = length(partials)
        T = BatchOfBundles{N, P, typeof(partials)}
        return $(Expr(:new, :T, :primal, :partials))
    end

    # we just have this as variadic because it is easier to construct the IR for it if we don't first need to make a tuple
    # This one is for `variable` intrinsics
    @eval @inline function BatchOfBundles{N}(primal::P, partials::Vararg{Any, N}) where {N, P}
        T = BatchOfBundles{N, P, typeof(partials)}
        return $(Expr(:new, :T, :primal, :partials))
    end
end

@eval @inline function BatchOfBundles(bundles::Tuple{Vararg{ATB{1}}})
    the_primal = Diffractor.primal(first(bundles))
    P = typeof(the_primal)
    # For debugging, you may uncomment the following lines, however
    # they inhibit many important optimizations, so we leave them commented for now.
    #if !all(x -> isequal(the_primal, Diffractor.primal(x)), bundles)
    #    throw(DomainError(Diffractor.primal.(bundles), "Primals must match"))
    #end
    the_partials::Tuple = map(x->Diffractor.partial(x, 1), bundles)
    return BatchOfBundles(the_primal, the_partials)
end

Base.size(::BatchOfBundles{N}) where N = (N,)
function Base.getindex(bob::BatchOfBundles, ii::Int)
    partial = bob.batched_partials[ii]
    return Diffractor.bundle(bob.primal, partial)
end

Diffractor.primal(bob::BatchOfBundles) = bob.primal

extract_partial(bob::BatchOfBundles, ii::Int) = Diffractor.first_partial(bob[ii])
extract_partial(atb::ATB, ii) =  Diffractor.first_partial(atb)  # It doesn't vary with input as it doesn't depend on any BoB, so replictate for all indexes
extract_partial(x::Real, ii) = zero(x)   # It is just a literal, thus no derivative was computed. zero for all indexes


"selects the ith bundle from a BoB x, or just returns the a bundle if x is just a bundle"
_extract_bundle(x::ATB{1}, _) = x
_extract_bundle(x::BatchOfBundles, i) = x[i]

function bobmap_impl(@nospecialize(args), M::Int)
    compute_bundles = map(1:M) do i
        this_args = map(1:length(args)) do j
            if args[j] <: ATB{1}
                return :(args[$j])
            elseif args[j] <: BatchOfBundles
                return :(args[$j][$i])
            else
                # We don't know which it is at compile time for some reason
                # The world is an awful place.
                @assert args[j] isa Union
                return :(_extract_bundle(args[$j], $i))
            end
        end
        :(this($(this_args...)))
    end
    rest_partials = map(compute_bundles[2:end]) do cb
        Expr(:call, :partial, cb, 1)
    end
    quote
        bundle1 = $(compute_bundles[1])
        BatchOfBundles{$M}(primal(bundle1), partial(bundle1, 1), $(rest_partials...))
    end
end

@generated function (this::Diffractor.∂☆{N})(args::Union{BatchOfBundles{M}, ATB{N}}...) where {N, M}
    return bobmap_impl(args, M)
end

@generated function (this::Diffractor.∂☆new{N})(args::Union{BatchOfBundles{M}, ATB{N}}...) where {N, M}
    return bobmap_impl(args, M)
end


# TODO: make this fully infer
basis_bob(primal) = BatchOfBundles(primal, basis_tangents(primal))


basis_tangents(::N) where N<:Number = (oneunit(N),)
function basis_tangents(x::Vector{T}) where T<:Real
    return ntuple(length(x)) do ii
        ele=zero(x)
        ele[ii] = oneunit(T)
        return ele
    end
end

tangentify(T::Type{<:Number}, val) = val
function tangentify(T::Type{<:Tuple}, val_tup)
    fieldcount(T) == 0 && return NoTangent()
    Tangent{T, typeof(val_tup)}(val_tup)
end
function tangentify(T, val_tup)
    fieldcount(T) == 0 && return NoTangent()
    val_nt = NamedTuple{fieldnames(T)}(val_tup)
    Tangent{T, typeof(val_nt)}(val_nt)
end

# Note: this is very similar to ChainRulesCore.zero_tangent but operating only on type, not value
deep_zero(T::Type{<:Number}) = zero(T)
deep_zero(T::Type{<:Array}) = error("Array within tuples/structs not supported")  # would need also reference to value to handle then
deep_zero(T::Type{Tuple{}}) = NoTangent()
Base.@assume_effects :foldable function deep_zero(T::Type{<:Tuple})
    tup = ntuple(Val{fieldcount(T)}()) do ii
        deep_zero(fieldtype(T, ii))
    end
    return Tangent{T, typeof(tup)}(tup)
end
Base.@assume_effects :foldable function deep_zero(T)  # general struct
    fieldcount(T) == 0 && return NoTangent()
    tup = ntuple(Val{fieldcount(T)}()) do ii
        deep_zero(fieldtype(T, ii))
    end
    nt = NamedTuple{fieldnames(T)}(tup)
    return Tangent{T, typeof(nt)}(nt)
end

determine_num_tangents(::Type{<:Number}) = 1
function determine_num_tangents(@nospecialize(T))
    iszero(fieldcount(T)) && return 0
    return sum(1:fieldcount(T)) do ii
        determine_num_tangents(fieldtype(T, ii))
    end
end

@generated function basis_tangents(x)
    rec_one(::Type{<:Array}) = error("Array within tuples/structs not supported")
    rec_one(T::Type{<:Number}) = oneunit(T)
    function rec_one(T)
        fieldcount(T) == 0 && return tuple()
        ret = []
        for one_at in 1:fieldcount(T)
            one_fieldT = fieldtype(T, one_at)
            for tangent_here in rec_one(one_fieldT)
                val_tup = ntuple(Val{fieldcount(T)}()) do ii
                    fieldT = fieldtype(T, ii)
                    if ii == one_at
                        tangent_here
                    else
                        deep_zero(fieldT)
                    end
                end
                push!(ret, tangentify(T, val_tup))
            end
        end
        return Tuple(ret)
    end

    return rec_one(x)
end
