using Test

@testset "state_mapping.jl" include("state_mapping.jl")
@testset "interpreter.jl" include("interpreter.jl")
@testset "compiler_and_lattice.jl" include("compiler_and_lattice.jl")
@testset "JITOpaqueClosures.jl" include("JITOpaqueClosures.jl")
@testset "robertson.jl" include("robertson.jl")
@testset "ipo.jl" include("ipo.jl")
@testset "lorenz.jl" include("lorenz_tests.jl")
@testset "pendulum.jl" include("pendulum_tests.jl")
@testset "mass_matrix.jl" include("mass_matrix.jl")
@testset "custom_init.jl" include("custom_init.jl")
@testset "custom_inline.jl" include("custom_inline.jl")
@testset "control.jl" include("control.jl")
#@testset "dynamic_state_error.jl" include("dynamic_state_error.jl")
@testset "index_lowering_ad.jl" include("index_lowering_ad.jl")
@testset "ddt.jl" include("ddt.jl")
@testset "implied_alias.jl" include("implied_alias.jl")
@testset "regression.jl" include("regression.jl")
@testset "transform_common.jl" include("transform_common.jl")
@testset "reconstruct.jl" include("reconstruct.jl")
@testset "reconstruct_time_derivative.jl" include("reconstruct_time_derivative.jl")
@testset "tearing_schedule.jl" include("tearing_schedule.jl")
@testset "jacobian_batching_utils.jl" include("jacobian_batching_utils.jl")
@testset "jacobian.jl" include("jacobian.jl")
@testset "tgrad.jl" include("tgrad.jl")
@testset "paramjac.jl" include("paramjac.jl")
@testset "periodic_callback.jl" include("periodic_callback.jl")
@testset "statemapping_derivatives.jl" include("statemapping_derivatives.jl")
@testset "invalidation.jl" include("invalidation.jl")
@testset "frule_invalidation.jl" include("frule_invalidation.jl")
@testset "debug_config.jl" include("debug_config.jl")
@testset "warnings.jl" include("warnings.jl")
@testset "sensitivity.jl" include("sensitivity.jl")
@testset "sensitivity_rccircuit.jl" include("sensitivity_rccircuit.jl")
@testset "epsilon.jl" include("epsilon.jl")
@testset "cthulhu.jl" include("cthulhu.jl")
@testset "mtk_components.jl" include("mtk_components.jl")

# must be last to minimize risks from monkeypatching
@testset "MSL" include("MSL/run_msl_tests.jl")
