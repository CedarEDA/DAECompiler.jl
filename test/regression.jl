module regression

using Test
using DAECompiler
using DAECompiler.Intrinsics
using SciMLBase
using ForwardDiff
import Compiler
const CC = Compiler

include(joinpath(Base.pkgdir(DAECompiler), "test/testutils.jl"))

# Test setproperty! inlining
function regression_setproperty()
    var = variable()
    r = Base.RefValue{Float64}(0.)
    finalizer(r->equation!(r[]), r)
    r.x += var
    r.x += 2.0
end

let sys = IRODESystem(Tuple{typeof(regression_setproperty)});
    daeprob = DAEProblem(sys, Float64[], Float64[], (0.0, 1e-5));
    odeprob = ODEProblem(sys, Float64[], (0.0, 1e-5));
    @test daeprob.u0 == odeprob.u0 == nothing
end

let sys = IRODESystem(Tuple{typeof(regression_setproperty)});
    interp = getfield(sys, :interp)
    ⼕🥬 = CC.typeinf_lattice(interp)
    ⼕🥬opt = CC.optimizer_lattice(interp)
    # Test tmeet for non-Float64
    @test CC.widenconst(CC.tmeet(
        CC.typeinf_lattice(interp),
        DAECompiler.Incidence(ForwardDiff.Partials{7, Float64}),
        ForwardDiff.Partials{7, Float64})) == ForwardDiff.Partials{7, Float64}

    # Test tmeet for Unions
    @test CC.tmeet(⼕🥬,
        DAECompiler.Incidence(Union{Int, Float64}), Int) ==
            DAECompiler.Incidence(Int)


    # tmerge that results in a union
    let T = CC.tmerge(⼕🥬,
            CC.Const(0.0),
            CC.PartialStruct(⼕🥬, Tuple{Float64, Float64}, Any[DAECompiler.Incidence(1), DAECompiler.Incidence(1)]))
        @test isa(T, DAECompiler.Incidence)
        @test T.typ == Union{Float64, Tuple{Float64, Float64}}
    end

    # tmerge nested struct that results in union
    let T = CC.tmerge(⼕🥬,
                CC.PartialStruct(⼕🥬, Tuple{Float64, Tuple{Float64}},
                    Any[DAECompiler.Incidence(1),
                        CC.PartialStruct(⼕🥬, Tuple{Float64}, Any[DAECompiler.Incidence(2)])]),
                Core.Const(1.0))
        @test isa(T, DAECompiler.Incidence)
        @test T.typ == Union{Float64, Tuple{Float64, Tuple{Float64}}}
    end


    # tmerge for partial structs
    let T = CC.tmerge(⼕🥬,
            CC.PartialStruct(⼕🥬, NTuple{3, Float64}, Any[DAECompiler.Incidence(1), DAECompiler.Incidence(2), Core.Const(0.0)]),
            CC.PartialStruct(⼕🥬, NTuple{3, Float64}, Any[DAECompiler.Incidence(2), DAECompiler.Incidence(1), Core.Const(0.0)]))
        @test isa(T, CC.PartialStruct) && isa(T.fields[1], DAECompiler.Incidence)
    end

    # tmerge for consts
    let T = CC.tmerge(⼕🥬,
            CC.Const(1.0),
            CC.Const(2.0))
        @test isa(T, DAECompiler.Incidence) && isempty(T)
    end

    # tmerge shouldn't lose unions
    let T = CC.tmerge(⼕🥬, DAECompiler.Incidence(Int64), Float64)
        @test T == Union{Int64, Float64}
    end

    # Time shouldn't disappear into constants
    @test !CC.:⊑(⼕🥬, DAECompiler.Incidence(0), Core.Const(0.0))
    let T = CC.tmerge(⼕🥬, DAECompiler.Incidence(0), Core.Const(0.0))
        @test isa(T, DAECompiler.Incidence)
    end

    # tmerge should lose PartialStruct
    PT = Core.PartialStruct(⼕🥬, Tuple{Float64, Float64}, Any[Core.Const(1.0), Float64])
    let T = CC.tmerge(⼕🥬, Union{}, PT)
        @test T === PT
    end
end

# Test for missing opaque_implicit_equation
function sub_ode()
    (x, y) = ntuple(_->variable(), 2)
    dy = state_ddt(y)
    dx = state_ddt(x)
    equation!(dy - 1.)
    equation!(dy - dx)
end
let sys = IRODESystem(Tuple{typeof(sub_ode)});
    for sol in solve_dae_and_ode(sys, [0.0, 0.0], [1.0, 1.0], (0.0, 1e-5))
        # This should presumably be exact.
        @test sol[1, :] == sol[2, :]
    end
end

# Issue #114
function f114()
    (x,y) = variable.((:x, :y))
    equation!(x-0.)
    equation!(state_ddt(y) - x - 1.)
end

let sys = IRODESystem(Tuple{typeof(f114)});
    for sol in solve_dae_and_ode(sys, [1.], [1.0], (0.0, 1e-5))
        @test all(sol[sys.x] .== 0.)
    end
end

# Bug in incidence analysis for phi nodes
@noinline complicated(x) = Base.inferencebarrier(x)::NTuple{3, Float64}
function phi_complicated()
    x, y, z = map(variable, (:x, :y, :z))
    if x > 0.
        t = (y, z, 0.)
    else
        t = (z, y, 0.)
    end
    (a, b, _) = complicated(t)
    equation!(x - sim_time() + 2)
    equation!(a - sim_time() + 3)
    equation!(b - sim_time() + 4)
end

let sys = IRODESystem(Tuple{typeof(phi_complicated)});
    for sol in solve_dae_and_ode(sys, [0., 0.], [1.0, 1.0], (0., 1.), phi_complicated)
        @test sol[sys.y][end] ≈ -3.
    end
end

# Issue #167
using ForwardDiff
using ForwardDiff: Dual
function var_alias_not_in_system()
    v = variable(:v)
    d = Dual(v, variable(:dv))
    if sim_time() > 1.
            x = sim_time()
    else
            x = 2sim_time()
    end
    dd = d - x
    equation!(state_ddt(v) - variable(:dvdt))
    equation!(dd.value)
    equation!(dd.partials[1])
end

let sys = IRODESystem(Tuple{typeof(var_alias_not_in_system)});
    for sol in solve_dae_and_ode(sys, nothing, nothing, (0., 2.));
        @test all(x->(x ≈ 1. || x ≈ 2.), sol[sys.dvdt])
    end
end

function var_alias_in_system()
    v = variable(:v)
    d = Dual(v, variable(:dv))
    if sim_time() > 1.
            x = sim_time()
    else
            x = 2sim_time()
    end
    dd = d - x
    equation!(state_ddt(v) - state_ddt(variable(:v2)))
    equation!(dd.value)
    equation!(dd.partials[1])
end
let sys = IRODESystem(Tuple{typeof(var_alias_in_system)});
    @test_nowarn for sol in solve_dae_and_ode(sys, [1.], [0.], (0., 1.))
        @test sol[sys.v2][end] ≈ 2.0
    end
end

function error_in_system()
    v = variable(:v)
    error("this should break us!")
    equation!(v + 5)
end

# Ensure that we have something in our `ir` in that exception:
try
    IRODESystem(Tuple{typeof(error_in_system)})
catch e
    @test isa(e, DAECompiler.UnsupportedIRException)
    ir = e.ir
    @test occursin("unreachable", string(ir))
end

function issue_291()
    f = variable(:f)
    g = variable(:g)
    fdot = state_ddt(f)
    fdotdot = state_ddt(fdot)
    equation!(fdot - fdotdot)
    equation!(g - fdotdot)
end
sys = IRODESystem(Tuple{typeof(issue_291)})
# TODO this is currently broken
#daesol, odesol = solve_dae_and_ode(sys, nothing, nothing, (0.,1.))
# test == rather than approx because they are both nothing.
#@test odesol[sys.f][end] == daesol[sys.f][end]

# Issue #274
function bar274()
    x = variable(:x)
    y = variable(:y)
    z = variable(:z)
    equation!(y-z)
    equation!(z + state_ddt(z))
    equation!(y^3 - x^3)
end
sys = IRODESystem(Tuple{typeof(bar274)});
@test_nowarn for sol in solve_dae_and_ode(sys, [0., 0.], [1., 1.], (0., 1.), reltol=1e-9, abstol=1e-12)
    # z should be equal to `exp(-t)` because we have `equation!(z + state_ddt(z))`
    # which implies that `state_ddt(z) = -z`
    @test all(sol[sys.z] .≈ exp.(-sol.t))
end

function bar274_part2()
    x = variable(:x)
    equation!(x + state_ddt(state_ddt(x)))
end
sys = IRODESystem(Tuple{typeof(bar274_part2)});
@test_nowarn for sol in solve_dae_and_ode(sys, [0., 0.], [1., 0.], (0., 1.), reltol=1e-9, abstol=1e-12)
    @test all(sol[sys.x] .≈ cos.(sol.t))
end

# Test duplicated state_ddt for the same variable
function bar311()
    h = variable(:h)
    i = variable(:i)
    equation!(h - state_ddt(state_ddt(h)))
    equation!(i - state_ddt(h))
end

@test_throws DAECompiler.UnsupportedIRException IRODESystem(Tuple{typeof(bar311)})

# Alias Elimination stress test from MTK
function alias_elimination_stress()
    sys = variables(:sys); sys2 = variables(:sys2)
    equation!.((
        state_ddt(sys2.x) + sys.x,
        sys2.y - sys.y,
        state_ddt(sys.x) + sys.x,
        sys.y - sys.x
    ))
end
sys = IRODESystem(Tuple{typeof(alias_elimination_stress)});

# TODO: Support symbolic assignment of initial conditions
du0 = [-1.,  1.]
u0 =  [ 0., -1.]

@test_nowarn for sol in solve_dae_and_ode(sys, du0, u0, (0., 1.))
    @test maximum(abs.(sol[sys.sys2.x] .- (sol[sys.sys2.x][1] .+ sol[sys.sys.x][1].*(exp.(-sol.t) .- 1.0)))) <= 1e-3
end

# Test a complicated linear system with a pantelides'ed linear equation
function linear_system()
    (; a, b, c, d) = variables()
    ċ = state_ddt(c)
    equation!(state_ddt(d)-25.)
    equation!(a + c - d)
    equation!(state_ddt(b) - ċ + a)
    equation!(state_ddt(a) - state_ddt(ċ))
end

# From Mathematica:
function linear_system_closed_form(t, (C1, C2, C3, C4))
    a_t = 25 + C1 - 1/32 * exp(-t) * (C1 + C3 - C4)
    b_t = -((5+t)*C1) + C2
    c_t = -25 + 25*t - C1 + 1/32 * exp(-t) * (C1 + C3 - C4) + C4
    d_t = 25*t + C4

    return (a_t, b_t, c_t, d_t)
end

Cs = (1., 2., 3., 4.)
# TODO: This depends on the exact state selection. That's not a great situation.
# We could potentially ask the system what states are chosen and read off the
# initial condition from the above.
db0 = -Cs[1]
ddb0 = 0.
dc0 = 25 - 1/32 * (Cs[1] + Cs[3] - Cs[4])
ddc0 = 1/32 * (Cs[1] + Cs[3] - Cs[4])
dd0 = 25.
u0 = [linear_system_closed_form(0., Cs)[2:end]...; dc0]
du0 = [db0; dc0; dd0; ddc0]

let sys = IRODESystem(Tuple{typeof(linear_system)})
    for sol in solve_dae_and_ode(sys, du0, u0, (0., 1.))
        @test all(collect.(linear_system_closed_form.(sol.t, Ref(Cs))) .≈ sol[[sys.a, sys.b, sys.c, sys.d]])
    end
end

# Test Propagation through SROA refinement
struct Immut50285
    x::Any
end

struct SroaRefineTest
    p::Float64
end
function (sroatest::SroaRefineTest)()
    x = variable(:x)
    equation!(state_ddt(x) - Immut50285(sroatest.p).x * x)
end
let sroatest = SroaRefineTest(1.)
    test_ddx_eq_x(sroatest, sroatest)
end

using Base.ScopedValues
const used_to_be_sls = ScopedValue(0.)

struct SroaRefineSLS end
function (p::SroaRefineSLS)()
    with(used_to_be_sls => 1.) do
        x = variable(:x)
        equation!(state_ddt(x) - used_to_be_sls[] * x)
    end
end
test_ddx_eq_x(SroaRefineSLS())

struct SemiConcreteEvalInvokes end
@inline function (::SemiConcreteEvalInvokes)()
    x = variable(:x)
    function semi_concrete_eval_me(val, name)
        return (first(val), Symbol(name))
    end
    equation!(semi_concrete_eval_me((state_ddt(x) - x, 1.0), "name")...)
end
# First, test that this solves correctly
test_ddx_eq_x(SemiConcreteEvalInvokes())

# Second, test that there are no `Symbol()` invokes
let ir_levels = DAECompiler.IRCodeRecords()
    sys = IRODESystem(SemiConcreteEvalInvokes(); debug_config=(; ir_levels))
    unopt_ir = ir_levels[1].ir
    for stmt in unopt_ir.stmts
        # Assert that there are no `Symbol(::String)` statements that match our "name" invocation above.
        # The `@test` is somewhat nonsensical, it's just here to alert the user this test failed.
        if DAECompiler.is_known_invoke_or_call(stmt[:inst], Symbol, unopt_ir) && stmt[:inst].args[3] == "name"
            ft = CC.argextype(stmt[:inst].args[2], unopt_ir)
            @test CC.singleton_type(ft) != Symbol
        end
    end
end

# Test that a non-tainted second derivative gets the correct order during index
# reduction (CedarSim#530)
struct OneParam
    p::Float64
end
function (p::OneParam)()
    (; x, y, z) = variables()
    ẋ = state_ddt(x)
    ẍ = state_ddt(ẋ)
    equation!(y - p.p*x)
    equation!(state_ddt(state_ddt(y)) - z*ẍ)
    equation!(x - ẍ)
end
let sys = IRODESystem(Tuple{OneParam}; debug_config=(;store_ir_levels=true, store_ss_levels=true));
    daeprob = DAEProblem(sys, [0., 0., 0.], [1., 0., 1.], (0., 1.), OneParam(1.));
    sol = solve(daeprob, IDA())
    @test all(sol[sys.z] .≈ 1.0)
end

# Test that singularity_root!() with a non-incident value is an error
using DAECompiler: UnsupportedIRException
function non_incident_singularity()
    (;x) = variables()
    singularity_root!(5.0)
    equation!(x - 5.0)
end
function too_few_args_singularity()
    (;x) = variables()
    singularity_root!()
    equation!(x - 5.0)
end
function too_many_args_singularity()
    (;x) = variables()
    singularity_root!(x, 5.0)
    equation!(x - 5.0)
end
#@test_throws UnsupportedIRException IRODESystem(Tuple{typeof(non_incident_singularity)});
#@test_throws UnsupportedIRException IRODESystem(Tuple{typeof(too_few_args_singularity)});
#@test_throws UnsupportedIRException IRODESystem(Tuple{typeof(too_many_args_singularity)});

# propertynames
function pnames()
    (;x, y) = variables(Scope(Scope(), :a))
    equation!(ddt(x)-x)
    equation!(ddt(y)-y)
end

let sys = IRODESystem(pnames)
    @test Set(propertynames(sys.a)) == Set((:x, :y))
end

end # module regression
