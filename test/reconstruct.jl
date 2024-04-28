module reconstruct

using Test

using DAECompiler, StateSelection, SciMLBase, OrdinaryDiffEq, ForwardDiff
using DAECompiler.Intrinsics
include(joinpath(Base.pkgdir(DAECompiler), "test", "lorenz.jl"))


function check_state(du, u, du1, u1, differential_vars)
    @test length(du) == length(u) == length(du1) == length(u1)
    for i = 1:length(u)
        @test u[i] == u1[i]
        if differential_vars[i]
            @test du[i] == du1[i]
        end
    end
end

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 100.0)
x = Lorenz1(10.0, 28.0, 8.0/3.0)
sys = IRODESystem(Tuple{typeof(x)});
let state = IRTransformationState(sys)
    var_eq_matching = DAECompiler.unoptimized_matching(state)
    tsys = TransformedIRODESystem(state, var_eq_matching)
    compress = DAECompiler.compile_state_compression_func(tsys, true)
    reconstruct = DAECompiler.compile_batched_reconstruct_func(tsys, collect(1:length(var_eq_matching)), collect(1:state.nobserved), true)
    (; differential_vars) = DAECompiler.assign_vars_and_eqs(tsys, true)

    du0 = zeros(5)
    u0 = [1.0, 0.0, 0.0, 0.0, -1.0]
    out_vars = similar(u0, length(var_eq_matching))
    out_obs = similar(u0, length(state.nobserved))
    reconstruct(out_vars, out_obs, du0, u0, x, 0.0)
    recompressed_u = similar(u0)
    recompressed_du = similar(du0)
    compress(recompressed_du, recompressed_u, out_vars)
    check_state(recompressed_du, recompressed_u, du0, u0, differential_vars)
end

let tsys = TransformedIRODESystem(sys)
    compress = DAECompiler.compile_state_compression_func(tsys, true)
    reconstruct = DAECompiler.compile_batched_reconstruct_func(tsys, collect(1:length(tsys.var_eq_matching)), collect(1:tsys.state.nobserved), true)
    _, _, differential_vars, _ = DAECompiler.assign_vars_and_eqs(tsys, true)
    # For optimized system, round-trip will only work for points on the manifold
    @test DAECompiler.num_selected_states(tsys, true) == 3
    u0 = [1.0, 0.0, 0.0]
    du0 = [-10., 28, 0.]
    (daef!,) = DAECompiler.dae_finish!(tsys.state, tsys.var_eq_matching, true; allow_unassigned=false)
    out = fill(NaN, 3)
    daef!(out, du0, u0, x, 0.0)
    @test all(==(0.0), out)

    out_vars = similar(u0, length(tsys.var_eq_matching))
    out_obs = similar(u0, length(tsys.state.nobserved))
    reconstruct(out_vars, out_obs, du0, u0, x, 0.0)
    recompressed_u = similar(u0)
    recompressed_du = similar(du0)
    compress(recompressed_du, recompressed_u, out_vars)
    check_state(recompressed_du, recompressed_u, du0, u0, differential_vars)
end

# Create a no-state system, then test that reconstruction works on it
struct SinSystem
end
function (::SinSystem)()
    (;x) = variables()
    equation!(x - sin(sim_time()))
end
x = SinSystem()
sys = IRODESystem(Tuple{typeof(x)});
prob = ODEProblem(sys, nothing, (0.0, 2π), x; jac=true);
sol = solve(prob, Rodas5P())

# Test that even though we have no states, we can reconstruct the time-dependent behavior
ts = range(0, 2π, 1000)
y = DAECompiler.batch_reconstruct(sol, [get_sys(sol).x], ts)
@test isapprox(y, sin.(ts)')
@test length(first(sol.u)) == 0
@test DAECompiler.num_selected_states(prob) == 0

# Test reconstruction of observables with derivatives.
# Issues #141 #666
using Sundials
function ddt_nonlinear()
    (;x,y) = variables()
    equation!(ddt(x*x)/2x - x)
    equation!(ddt(x*x)/2x - y)
end
sys = IRODESystem(ddt_nonlinear)
sol = solve(DAEProblem(sys, [0.], [1.], (0., 1.)), IDA())
@test all(isapprox.(sol[sys.x], exp.(sol.t), atol=1e-2)) # passes
@test all(isapprox.(sol[sys.y], exp.(sol.t), atol=1e-2)) # broken

end # module reconstruct
