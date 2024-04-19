module dynamic_state_error

using Test
using DAECompiler
using DAECompiler.Intrinsics
using SciMLBase
using OrdinaryDiffEq

function dynamic_state()
    v = variable()
    dv = state_ddt(v)
    equation!(dv + (v - 5.0)/10.)
    if sim_time() > 1.0
        equation!(variable() - v)
    end
    return nothing
end

daeprob = DAEProblem(IRODESystem(Tuple{typeof(dynamic_state)}), [0.0], [0.0], (0, 2.), dynamic_state)
@test_throws DAECompiler.DynamicStateError solve(daeprob, DFBDF(autodiff=false))
odeprob = ODEProblem(IRODESystem(Tuple{typeof(dynamic_state)}), [0.0], (0, 2.), dynamic_state)
@test_throws DAECompiler.DynamicStateError solve(odeprob, FBDF(autodiff=false))

end # module dynamic_state_error
