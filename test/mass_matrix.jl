module mass_matrix

# DAECompiler version of https://github.com/SciML/ModelingToolkit.jl/blob/master/test/mass_matrix.jl

using Test
using DAECompiler
using DAECompiler.Intrinsics
using DAECompiler.Intrinsics: state_ddt
using SciMLBase, OrdinaryDiffEqRosenbrock

struct MMTest1
    k::NTuple{3, Float64}
end

function (k::MMTest1)()
    (x, y, z) = ntuple(_->variable(), 3)
    equation!.((
        state_ddt(x) + k.k[1] * x - k.k[3] * y * sin(z),
        state_ddt(y) - k.k[1] * x + k.k[3] * y * sin(z) + k.k[2] * y^2,
        x + y + sin(z) - 1
    ))
end

sys = IRODESystem(Tuple{MMTest1});
prob_mm = ODEProblem(sys, [1.0, 0.0, 0.0], (0.0, 1e5), MMTest1((0.04, 3e7, 1e4)));
@test prob_mm.f.mass_matrix == [1. 0. 0.; 0. 1. 0.; 0. 0. 0.]
sol = solve(prob_mm, Rodas5(autodiff=false), reltol = 1e-8, abstol = 1e-8)

# TODO: In each of these tests, we should only have two ODE states.
# However, at present, tearing produces 3. This should be improved.
struct ParallelCapacitorSinISource1
    I::Float64
end

const C = 1e-4
const R = 1e3
function (PCIS::ParallelCapacitorSinISource1)()
    VC1, VC2 = ntuple(_->variable(), 2)
    I1 = C * state_ddt(VC1)
    I2 = C * state_ddt(VC2)
    equation!.((
        PCIS.I*sin(sim_time()) - (I1 + I2),
        VC1 + I1*R - (VC2 + I2*R)))
end

sys = IRODESystem(Tuple{ParallelCapacitorSinISource1});
prob_mm = ODEProblem(sys, [0., 0., 0.], (0.0, 10.), ParallelCapacitorSinISource1(1e-3));
sol = solve(prob_mm, Rodas5(autodiff=false), reltol = 1e-8, abstol = 1e-8)

struct ParallelCapacitorSinISource2
    R::NTuple{2, Float64}
    I::Float64
end

function (PCIS::ParallelCapacitorSinISource2)()
    VC1, VC2 = ntuple(_->variable(), 2)
    R1, R2 = PCIS.R
    I1 = C * state_ddt(VC1)
    I2 = C * state_ddt(VC2)
    equation!.((
        PCIS.I*sin(sim_time()) - (I1 + I2),
        VC1 + I1*R1 - (VC2 + I2*R2)))
end

sys = IRODESystem(Tuple{ParallelCapacitorSinISource2});
prob_mm = ODEProblem(sys, [0., 0., 0.], (0.0, 10.), ParallelCapacitorSinISource2((1e4, 3e4), 1e-3));
sol = solve(prob_mm, Rodas5(autodiff=false), reltol = 1e-8, abstol = 1e-8)

struct ParallelCapacitorSinISource3
    R::NTuple{2, Float64}
    C::NTuple{2, Float64}
    I::Float64
end

function (PCIS::ParallelCapacitorSinISource3)()
    VC1, VC2 = ntuple(_->variable(), 2)
    R1, R2 = PCIS.R
    I1 = PCIS.C[1] * state_ddt(VC1)
    I2 = PCIS.C[2] * state_ddt(VC2)
    equation!.((
        PCIS.I*sin(sim_time()) - (I1 + I2),
        VC1 + I1*R1 - (VC2 + I2*R2)))
end
sys = IRODESystem(Tuple{ParallelCapacitorSinISource3});
prob_mm = ODEProblem(sys, [0.0, 0.0, 0.0, 0.0], (0.0, 10.), ParallelCapacitorSinISource3((1e4, 3e4), (1e-4, 2e-4), 1e-3));
sol = solve(prob_mm, Rodas5(autodiff=false), reltol = 1e-8, abstol = 1e-8)

end # module mass_matrix
