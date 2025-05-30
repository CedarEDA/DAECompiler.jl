module Basic

using Test
using DAECompiler
using DAECompiler.Intrinsics
using Sundials
using SciMLBase
using OrdinaryDiffEq

# We don't want to test IPO here, so directly call the intrinsic that DAECompiler models
const *ᵢ = Core.Intrinsics.mul_float
const +ᵢ = Core.Intrinsics.add_float
const -ᵢ = Core.Intrinsics.sub_float

#= Simplest possible smoke test; one variable, one equation =#
@noinline function oneeq!()
    x = continuous()
    always!(ddt(x) -ᵢ x)
end

oneeq!()
sol = solve(DAECProblem(oneeq!, (1,) .=> 1.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))
sol = solve(ODECProblem(oneeq!, (1,) .=> 1.), Rodas5(autodiff=false))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

# Cover the `debuginfo` rewrite.
sol = solve(DAECProblem(oneeq!, (1,) .=> 1.), IDA(), insert_stmt_debuginfo = true)
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

#= + parameterized =#
struct parameterized
    param::Float64
end
function (this::parameterized)()
    x = continuous()
    always!(ddt(x) -ᵢ getfield(this, :param))
end

dae_sol = solve(DAECProblem(parameterized(1.0), (1,) .=> 1.), IDA())
ode_sol = solve(ODECProblem(parameterized(1.0), (1,) .=> 1.), Rodas5(autodiff=false))
for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], 1. .+ sol.t))
end

#= + non-linear =#
@noinline function oneeq_nl!()
    x = continuous()
    t = sim_time()
    always!(ddt(x) -ᵢ t *ᵢ t)
end

oneeq_nl!()
sol = solve(DAECProblem(oneeq_nl!, (1,) .=> 0.), IDA())
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], (t->1/3*t^3).(sol.t)))

#= Initial Condition =#
@noinline function oneeq_ic!()
    x = continuous()
    always!(ddt(x) -ᵢ x)
    initial!(x -ᵢ 1.0)
end

oneeq_ic!()
@test DAECompiler.factory(Val(DAECompiler.Settings(;mode=DAECompiler.InitUncompress)), oneeq_ic!)((;u0=Float64[])) == [1.0]
# TODO: Sundials is broken and doesn't respect the custom initialization (https://github.com/SciML/Sundials.jl/issues/469)
sol = solve(DAECProblem(oneeq_ic!), DFBDF(autodiff=false))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))
sol = solve(ODECProblem(oneeq_ic!), Rodas5(autodiff=false))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(sol.t)))

#= Pantelides =#
function pantelides()
    a = continuous()
    b = continuous()
    always!(a -ᵢ sim_time())
    always!(ddt(a) -ᵢ ddt(b))
end

pantelides()
sol = solve(DAECProblem(pantelides, (1,) .=> 0.), DFBDF(autodiff=false))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], sol.t))
sol = solve(ODECProblem(pantelides, (1,) .=> 0.), Rodas5(autodiff=false))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], sol.t))

#= Structural Singularity Removal =#
function ssrm()
    a = continuous()
    b = continuous()
    abdot = ddt(a +ᵢ b)
    always!(a -ᵢ abdot)
    always!(b -ᵢ abdot)
end

ssrm()
sol = solve(DAECProblem(ssrm, (1,) .=> 1.), DFBDF(autodiff=false))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(0.5sol.t)))
sol = solve(ODECProblem(ssrm, (1,) .=> 1.), Rodas5(autodiff=false))
@test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], exp.(0.5sol.t)))

#= Pantelides from init =#
function pantelides_from_init()
    x = continuous()
    y = continuous()
    t = sim_time()
    always!(y -ᵢ t *ᵢ t)
    always!(ddt(x) -ᵢ y)
    initial!(x -ᵢ ddt(ddt(y)))
end

pantelides_from_init()
# Incompletely implemented
@test_broken solve(DAECProblem(pantelides_from_init, (1,) .=> 1.), DFBDF(autodiff=false))

#= SICM variables =#
struct sicm_vars
    param::Float64
end
function (this::sicm_vars)()
    x = continuous()
    p = continuous()
    always!(p -ᵢ this.param)
    always!(ddt(x) -ᵢ p)
end
dae_sol = solve(DAECProblem(sicm_vars(1.0), (1,) .=> 1.), IDA())
ode_sol = solve(ODECProblem(sicm_vars(1.0), (1,) .=> 1.), Rodas5(autodiff=false))
for sol in (dae_sol, ode_sol)
    @test all(map((x,y)->isapprox(x[], y, atol=1e-2), sol.u[:, 1], 1. .+ sol.t))
end


end
