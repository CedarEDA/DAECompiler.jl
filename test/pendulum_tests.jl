module pendulum

using Test
using DAECompiler, SciMLBase, OrdinaryDiffEq
using StateSelection: MatchedSystemStructure
using DAECompiler: assign_vars_and_eqs

include("pendulum.jl")

tspan = (0.0, 100.0)
invsqrt2 = 1. / sqrt(2.0)
test_systems = Tuple{Any, Vector{Float64}, Any, Any}[]

# Pendulum starting at pi / 4
# TODO: These initializations need to be remapped correctly to the transformed
#       system. For now, they are just zero-padded so almost certainly wrong.

# Fails, reporting SingularException
# push!(test_systems, (Pendulum(1.0, 1.0),
                     # [invsqrt2, -invsqrt2, 20. * invsqrt2, π/4, 0.0, 0.0, 0.0],
                     # :DAE,
                     # DFBDF))

# Fails, reporting that system is unstable
# push!(test_systems, (Pendulum(1.0, 1.0),
                     # [invsqrt2, -invsqrt2, 20. * invsqrt2, π/4, 0.0, 0.0, 0.0, 0.0],
                     # :ODE,
                     # FBDF))

# Fails, reporting SingularException
# push!(test_systems, (FirstOrderPendulum(1.0, 1.0),
                     # [invsqrt2, 0.0, -invsqrt2, 0.0, 20. * invsqrt2, π/4, 0.0],
                     # :DAE,
                     # DFBDF))

# Fails, reporting that system is unstable
# push!(test_systems, (FirstOrderPendulum(1.0, 1.0),
                     # [invsqrt2, 0.0, -invsqrt2, 0.0, 20. * invsqrt2, π/4, 0.0, 0.0],
                     # :ODE,
                     # FBDF))

# Currently broken - requires dynamic pivoting
# push!(test_systems, (SingularJacobianPendulum(1.0, 1.0),
                     # [invsqrt2, -invsqrt2, 20. * invsqrt2],
                     # :DAE,
                     # DFBDF))

# push!(test_systems, (SingularJacobianPendulum(1.0, 1.0),
                     # [invsqrt2, -invsqrt2, 20. * invsqrt2],
                     # :ODE,
                     # FBDF))

sols = Any[]

for (p, u0, problem, Solver) in test_systems
    sys = IRODESystem(Tuple{typeof(p)});
    if problem == :ODE
        prob = ODEProblem(sys, u0, tspan, p);
    elseif problem == :DAE
        prob = DAEProblem(sys, zero(u0), u0, tspan, p);
    end
    sol = solve(prob, Solver(autodiff=false))
    @test all(x->abs(x) < 100, sol)

    push!(sols, sol)
end

end

pendulum.sols
