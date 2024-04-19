module implied_alias

using Test
using SciMLBase
using DAECompiler
using DAECompiler.Intrinsics
using OrdinaryDiffEq

include("testutils.jl")

function implied_alias_1()
    (x, y, z) = ntuple(_->variable(), 3)
    equation!(x - y + 2)
    equation!(state_ddt(y))
    equation!(z - state_ddt(x))
end

# TODO: MTK currently selects both x and y as states. This is suboptimal.
# It should only select `y` and make `x` and `dx` observables.
function implied_alias_2()
    (x, y, z) = ntuple(_->variable(), 3)
    equation!(x - y + sin(sim_time()))
    equation!(state_ddt(y))
    equation!(z - state_ddt(x))
end

for implied_alias in (implied_alias_1, implied_alias_2)
    sys = IRODESystem(Tuple{typeof(implied_alias)})
    for sol in solve_dae_and_ode(sys, [0.], [1.], (0, 1e-5))
        @test all(==(1.0), sol[1,:])
    end
end

end # module implied_alias
