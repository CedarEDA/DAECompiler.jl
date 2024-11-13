module Compiler_and_Lattice

import Compiler
const CC = Compiler
using DAECompiler
using DAECompiler: aggressive_incidence_join, Incidence, nonlinear, tfunc, Const
using Test

@testset "idnum" begin
    @test DAECompiler.idnum(Incidence(4)) == 4
    @test DAECompiler.idnum(DAECompiler.Eq(4)) == 4
end

const 𝕃 = CC.optimizer_lattice(DAECompiler.DAEInterpreter(Base.get_world_counter()))

@testset "aggressive_incidence_join" begin
    f4 = tfunc(Val(Core.Intrinsics.mul_float), Incidence(4), Incidence(4))
    f4l5 = tfunc(Val(Core.Intrinsics.add_float), f4, Incidence(5))
    f4f5 = tfunc(Val(Core.Intrinsics.mul_float), f4, Incidence(5))
    l4l5 = tfunc(Val(Core.Intrinsics.add_float), Incidence(4), Incidence(5))
    f4f5_b = tfunc(Val(Core.Intrinsics.mul_float), f4, l4l5)
    @test f4f5 == f4f5_b  # this should realy be in a seperate testset, but we need these for testing incidence join

    # actual tests of aggressive_incidence_join
    struct Foo
        x
    end

    type_differs = Core.PartialStruct(𝕃, Foo, Any[Incidence(4),
        Core.PartialStruct(𝕃, Foo, Any[Core.PartialStruct(𝕃, Tuple{Float64}, Any[Incidence(5)])])
    ])
    @test aggressive_incidence_join(Const(0.0), type_differs.fields) == f4f5

    type_same = Core.PartialStruct(𝕃, Foo, Any[l4l5,
        Core.PartialStruct(𝕃, Foo, Any[Core.PartialStruct(𝕃, Tuple{Float64}, Any[l4l5])])
    ])
    @test aggressive_incidence_join(Const(0.0), type_same.fields) == l4l5

    type_overlaps = Core.PartialStruct(𝕃, Foo, Any[Incidence(5),
        Core.PartialStruct(𝕃, Foo, Any[Core.PartialStruct(𝕃, Tuple{Float64}, Any[l4l5])])
    ])
    @test aggressive_incidence_join(Const(0.0), type_overlaps.fields) == f4l5

    type_overlaps_rev = Core.PartialStruct(𝕃, Foo, Any[l4l5,
        Core.PartialStruct(𝕃, Foo, Any[Core.PartialStruct(𝕃, Tuple{Float64}, Any[Incidence(5)])])
    ])
    @test aggressive_incidence_join(Const(0.0), type_overlaps_rev.fields) == f4l5
end

@test CC.tmeet(𝕃, DAECompiler.Eq(1), DAECompiler.equation) === DAECompiler.Eq(1)

end  # module
