using DAECompiler, SciMLBase

include(joinpath(Base.pkgdir(DAECompiler), "test/lorenz.jl"))

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 100.0)
l = Lorenz1(10.0, 28.0, 8.0/3.0)

# Construct `sys` with the given debug flags
sys = IRODESystem(Tuple{typeof(l)}; debug_config = (;
    # Store all of our IR levels.
    # You can run with `verify_ir_levels` enabled as well,
    # but note that it increases runtime dramatically.
    store_ir_levels = true,
    verify_ir_levels = true));

# Construct the DAEProblem, which will perform system structure analysis,
# and the DAE compiler pipeline.
daeprob = DAEProblem(sys, zero(u0), u0, tspan, l)

# Summarize the generated IR
DAECompiler.summarize_ir_levels(daeprob)
