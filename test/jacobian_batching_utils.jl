module jacobian_batching
using Test
using DAECompiler: basis_tangents, basis_bob, BatchOfBundles, deep_zero
using Diffractor: TaylorBundle, primal, first_partial
using InteractiveUtils: @code_typed
using ChainRulesCore

@testset "basis_tangents" begin
    @testset "Numbers" begin
        @test basis_tangents(201.0) === (1.0,)
    end

    @testset "Numeric Vector" begin
        @test basis_tangents([10.0, 20.0, 30.0]) == (
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        )
    end

    @testset "tuple" begin
        @test basis_tangents(tuple()) == ()
        @test basis_tangents((10.0,)) == (Tangent{Tuple{Float64}}(1.0,),)
        @test basis_tangents((10.0, 20.0)) == (
            Tangent{Tuple{Float64,Float64}}(1.0, 0.0),
            Tangent{Tuple{Float64,Float64}}(0.0, 1.0),
        )

        @test basis_tangents((10.0, 20.0, 30.0)) == (
            Tangent{Tuple{Float64, Float64, Float64}}(1.0, 0.0, 0.0),
            Tangent{Tuple{Float64, Float64, Float64}}(0.0, 1.0, 0.0),
            Tangent{Tuple{Float64, Float64, Float64}}(0.0, 0.0, 1.0),
        )
    end

    @testset "mixed tuple" begin
        @test basis_tangents((sin, 20.0)) == (
            Tangent{Tuple{typeof(sin),Float64}}(NoTangent(), 1.0),
        )

        @test basis_tangents(((10.0, 20.0),30.0)) == (
            Tangent{Tuple{Tuple{Float64, Float64}, Float64}}(Tangent{Tuple{Float64, Float64}}(1.0, 0.0), 0.0),
            Tangent{Tuple{Tuple{Float64, Float64}, Float64}}(Tangent{Tuple{Float64, Float64}}(0.0, 1.0), 0.0),
            Tangent{Tuple{Tuple{Float64, Float64}, Float64}}(Tangent{Tuple{Float64, Float64}}(0.0, 0.0), 1.0),
        )

        @test basis_tangents((30.0, (10.0, 20.0))) == (
            Tangent{Tuple{Float64, Tuple{Float64, Float64}}}(1.0, Tangent{Tuple{Float64, Float64}}(0.0, 0.0)),
            Tangent{Tuple{Float64, Tuple{Float64, Float64}}}(0.0, Tangent{Tuple{Float64, Float64}}(1.0, 0.0)),
            Tangent{Tuple{Float64, Tuple{Float64, Float64}}}(0.0, Tangent{Tuple{Float64, Float64}}(0.0, 1.0)),
        )
    end

    @testset "Singleton Structs" begin
        struct Singleton end

        @test basis_tangents(nothing) == ()
        @test basis_tangents(sin) == ()
        @test basis_tangents(Singleton()) == ()
    end

    @testset "structs" begin
        @test basis_tangents(1.5=>2.5) == (
            Tangent{Pair{Float64, Float64}}(first = 1.0, second = 0.0),
            Tangent{Pair{Float64, Float64}}(first = 0.0, second = 1.0)
        )

        struct Foo
            x::Float64
            bar::Pair{Float64, Float64}
        end
        @test basis_tangents(Foo(1.5, 2.5=>3.5)) == (
            Tangent{Foo}(x=1.0, bar=Tangent{Pair{Float64, Float64}}(first = 0.0, second = 0.0)),
            Tangent{Foo}(x=0.0, bar=Tangent{Pair{Float64, Float64}}(first = 1.0, second = 0.0)),
            Tangent{Foo}(x=0.0, bar=Tangent{Pair{Float64, Float64}}(first = 0.0, second = 1.0))
        )

        struct Qux
            x::Float64
            bar::Pair{Float64, Float64}
            y::Float64
        end
        basis_tangents(Qux(1.5, 2.5=>3.5, 4.5)) ==  (
            Tangent{Qux}(x = 1.0, bar = Tangent{Pair{Float64, Float64}}(first = 0.0, second = 0.0), y = 0.0),
            Tangent{Qux}(x = 0.0, bar = Tangent{Pair{Float64, Float64}}(first = 1.0, second = 0.0), y = 0.0),
            Tangent{Qux}(x = 0.0, bar = Tangent{Pair{Float64, Float64}}(first = 0.0, second = 1.0), y = 0.0),
            Tangent{Qux}(x = 0.0, bar = Tangent{Pair{Float64, Float64}}(first = 0.0, second = 0.0), y = 1.0)
        )
    end

end

@testset "deep_zero" begin
    NT = @NamedTuple{params::@NamedTuple{c::Float64, r::Float64}}

    # zero tangent should be correct
    zero_tangent = deep_zero(NT)
    @test zero_tangent === Tangent{
        @NamedTuple{params::@NamedTuple{c::Float64, r::Float64}}
    }(params=Tangent{@NamedTuple{c::Float64, r::Float64}}(c=0.0, r=0.0))

    # return type should be fully-concrete (and correct)
    RT = (@code_typed deep_zero(NT)).second
    @test RT === typeof(zero_tangent)
end

@testset "basis_bob" begin
    struct Demo3{T}
        x1::T
        x2::T
        x3::T
    end
    p3 = Demo3(1.5, 2.5, 3.5)
    bob3 = basis_bob(p3)
    @test primal(bob3) == p3
    @test length(bob3.batched_partials) == 3
    @test bob3[1] isa TaylorBundle{1, Demo3{Float64}}
    @test primal(bob3[1]) == p3
    @test first_partial(bob3[1]) == Tangent{Demo3{Float64}}(x1 = 1.0, x2 = 0.0, x3 = 0.0)


    struct Demo1Nest3{T}
        x1::T
        x2::Demo3{T}
    end
    p1n3 = Demo1Nest3(0.5, Demo3(1.5, 2.5, 3.5))
    bob1n3 = basis_bob(p1n3)
    @test primal(bob1n3) == p1n3
    @test length(bob1n3.batched_partials) == 4
    @test bob1n3[1] isa TaylorBundle{1, Demo1Nest3{Float64}}
    @test primal(bob1n3[1]) == p1n3
end  # module

@testset "BatchOfBundles constructor" begin
    @test BatchOfBundles(1.5, (0.0, 1.0)) isa BatchOfBundles{2, Float64, Tuple{Float64, Float64}}
end

end  # module