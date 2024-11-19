module testutils

using OrdinaryDiffEqBDF, Test, Sundials
using DAECompiler
export solve_dae_and_ode, solve_dae, solve_ode, test_ddx_eq_x

function solve_dae(jac::Bool, sys::IRODESystem, args...; kwargs...)
    return solve(DAEProblem(sys, args...; jac), IDA(); kwargs...)
end

function solve_dae(args...; kwargs...)
    return (solve_dae(#=jac=#false, args...; kwargs...),
            solve_dae(#=jac=#true, args...; kwargs...))
end

function solve_dae(entry_func, jac::Bool, args...; kwargs...)
    sys = IRODESystem(entry_func)
    return solve_dae(jac, sys, args...; kwargs...)
end

function solve_ode(jac::Bool, sys::IRODESystem, args...; kwargs...)
    return solve(ODEProblem(sys, args...; jac=false), FBDF(autodiff=false); kwargs...)
end

function solve_ode(args...; kwargs...)
    return (solve_ode(#=jac=#false, args...; kwargs...),
            solve_ode(#=jac=#true, args...; kwargs...))
end

function solve_ode(entry_func, jac::Bool, args...; kwargs...)
    return solve_ode(jac, ODESystem(entry_func), args...; kwargs...)
end

function solve_dae_and_ode(sys, du, u, tspan, args...; kwargs...)
    (solve_dae(sys, du, u, tspan, args...; kwargs...)...,
     solve_ode(sys, u, tspan, args...; kwargs...)...)
end

function test_ddx_eq_x(entry_func, args...)
    let sys = IRODESystem(entry_func)
        for sol in solve_dae_and_ode(sys, [0.], [1.], (0., 1.), args...)
            @test all(isapprox.(sol[sys.x], exp.(sol.t), atol=1e-2))
        end
    end
end

end # module testutils

using .testutils
import DAECompiler.Intrinsics: state_ddt
