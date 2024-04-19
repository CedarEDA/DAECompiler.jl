module RunMSL
using DAECompiler

# Load all the MSL test dependencies
using SciMLBase
using ModelingToolkitStandardLibrary
using OrdinaryDiffEq
using SafeTestsets
using Test

include(joinpath(Base.pkgdir(DAECompiler), "test", "MSL", "modeling_toolkit_helper.jl"))

# Run the tests
const msl_dir = Base.pkgdir(ModelingToolkitStandardLibrary)
msl_test(name) = joinpath(msl_dir, "test", name)
@testset "MSL" begin
    # Blocks
    @safetestset "Blocks: math" include($(@eval msl_test("Blocks/math.jl")))
    @safetestset "Blocks: nonlinear" include($(@eval msl_test("Blocks/nonlinear.jl")))
    @safetestset "Blocks: continuous" include($(@eval msl_test("Blocks/continuous.jl")))
    @safetestset "Blocks: sources" include($(@eval msl_test("Blocks/sources.jl")))
    @safetestset "Blocks: analysis points" include($(@eval msl_test("Blocks/test_analysis_points.jl")))

    # Electrical
    @safetestset "Analog Circuits" include($(@eval msl_test("Electrical/analog.jl")))
    @safetestset "Digital Circuits" include($(@eval msl_test("Electrical/digital.jl")))
    @safetestset "Chua Circuit Demo" include($(@eval msl_test("chua_circuit.jl")))

    # Thermal
    @safetestset "Thermal Circuits" include($(@eval msl_test("Thermal/thermal.jl")))
    @safetestset "Thermal Demo" include($(@eval msl_test("Thermal/demo.jl")))

    # Magnetic
    @safetestset "Magnetic" include($(@eval msl_test("Magnetic/magnetic.jl")))

    # Mechanical
    # 1 test is broken here as something is going wrong in Bareiss that we don't understand casing an assertion to fail.
    @safetestset "Mechanical Rotation" include($(@eval msl_test("Mechanical/rotational.jl")))
    @safetestset "Mechanical Translation" include($(@eval msl_test("Mechanical/translational.jl")))
    @safetestset "Mechanical Translation Modelica" include($(@eval msl_test("Mechanical/translational_modelica.jl")))
    # We do not run this test as it is incomplete, and doesn't test anything we do. (but does test something we do not do before-hand)
    # In future version of MSL it might be completed, if so we should enable it.
    # @safetestset "Multi-body" include($(@eval msl_test("Mechanical/multibody.jl")))

    # Hydraulic
    @safetestset "Hydraulic IsothermalCompressible" include($(@eval msl_test("Hydraulic/isothermal_compressible.jl")))

    # MultiDomain
    @safetestset "MultiDomain" include($(@eval msl_test("multi_domain.jl")))
end # @testset

end # module
