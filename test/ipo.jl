module ipo

using Test
using DAECompiler
using DAECompiler.Intrinsics
using DAECompiler.Intrinsics: state_ddt
using SciMLBase, OrdinaryDiffEq, Sundials

include(joinpath(Base.pkgdir(DAECompiler), "test", "testutils.jl"))

debug_config = (; verify_ir_levels=true)

#= Test basic IPO functionality with repeated-noinlined function =#
@noinline function x!()
    x = variable()
    equation!(state_ddt(x) - x)
end
function x2!()
    x!(); x!();
    return nothing
end

sys = IRODESystem(Tuple{typeof(x2!)}; debug_config, ipo_analysis_mode=false);
sol = solve(DAEProblem(sys, [0., 0.], [1., 1.], (0., 1.)), IDA())
sol_ipo = solve(DAECProblem(x2!, (1, 2) .=> 1.), IDA())
@test sol_ipo(sol.t).u ≈ sol.u

#=================== + NonLinear =============================#
@noinline function x_sin!()
    x = variable()
    equation!(state_ddt(x) - sin(x))
end
function x2_sin!()
    x_sin!(); x_sin!();
    return nothing
end

sys_ipo = IRODESystem(Tuple{typeof(x2_sin!)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_sin!)}; debug_config, ipo_analysis_mode=false);

@test length(getfield(sys_ipo, :result).var_to_diff) ==
      length(getfield(sys, :result).var_to_diff)

sol_ipo = solve(DAECProblem(x2_sin!, (1, 2) .=> 1.), IDA())
sol = solve(DAEProblem(sys, [0., 0.], [1., 1.], (0., 1.)), IDA())

@test sol_ipo(sol.t).u ≈ sol.u

#=================== + SICM =============================#
struct x_sicm!
    arg::Float64
end

@noinline function (this::x_sicm!)()
    x = variable()
    equation!(state_ddt(x) - this.arg)
end

struct x_sicm2!
    a::Float64
    b::Float64
end

function (this::x_sicm2!)()
    x_sicm!(this.a)(); x_sicm!(this.b)();
    return nothing
end

sys_ipo = IRODESystem(Tuple{x_sicm2!}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{x_sicm2!}; debug_config, ipo_analysis_mode=false);

sol_ipo = solve(DAECProblem(x_sicm2!(1.0, 2.0), (1, 2) .=> 1.), IDA())
sol = solve(DAEProblem(sys, [0., 0.], [1., 1.], (0., 1.), x_sicm2!(1.0, 2.0)), IDA())

@test sol_ipo(sol.t).u ≈ sol.u

#=================== + NonLinear SICM =============================#
struct x_nl_sicm!
    arg::Float64
end

@noinline function (this::x_nl_sicm!)()
    x = variable()
    equation!(state_ddt(x) - sin(this.arg))
end

struct x_nl_sicm2!
    a::Float64
    b::Float64
end

function (this::x_nl_sicm2!)()
    x_nl_sicm!(this.a)(); x_nl_sicm!(this.b)();
    return nothing
end

sys_ipo = IRODESystem(Tuple{x_nl_sicm2!}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{x_nl_sicm2!}; debug_config, ipo_analysis_mode=false);

sol_ipo = solve(DAECProblem(x_nl_sicm2!(1.0, 2.0), (1, 2) .=> 1.), IDA())
sol = solve(DAEProblem(sys, [0., 0.], [1., 1.], (0., 1.), x_nl_sicm2!(1.0, 2.0)), IDA())

@test sol_ipo(sol.t).u ≈ sol.u

#======================== + Ping Pong =============================#
@noinline function ping(a, b, c, d)
    equation!(b - sin(a))
    equation!(d - sin(c))
end

@noinline function pong(a, b, c, d)
    equation!(b - sin(a))
    equation!(state_ddt(d) - c)
end

function pingpong()
    # N.B.: Deliberate not using variables, which requires
    # scope, etc handling
    a = variable()
    b = variable()
    c = variable()
    d = variable()
    ping(a, b, c, d)
    pong(b, c, d, a)
end

sys_ipo = IRODESystem(Tuple{typeof(pingpong)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(pingpong)}; debug_config, ipo_analysis_mode=false);

sol_ipo = solve(DAECProblem(pingpong, (1,) .=> 1.), IDA())
sol = solve(DAEProblem(sys, [0.], [1.], (0.), pingpong), IDA())

@test sol_ipo(sol.t).u ≈ sol.u

#=================== + Implicit External ==========================#
@noinline x_intro() = state_ddt(variable())
@noinline x_outro(x) = equation!(x-1)

x_implicit() = x_outro(x_intro())
sys_ipo = IRODESystem(Tuple{typeof(x_implicit)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x_implicit)}; debug_config, ipo_analysis_mode=false);

@test length(getfield(sys_ipo, :result).var_to_diff) ==
      length(getfield(sys, :result).var_to_diff)

sol_ipo = solve(DAECProblem(x_implicit, (1,) .=> 1.), IDA())
sol = solve(DAEProblem(sys, [0.], [1.], (0.), x_implicit), IDA())

@test sol_ipo(sol.t).u ≈ sol.u

#=================== + Non-linear Implicit External ==========================#
@noinline x_intro_nl()  = tanh(state_ddt(variable()))
@noinline x_outro_nl(x) = equation!(x-0.5)

x_implicit_nl() = x_outro_nl(x_intro_nl())
sys_ipo = IRODESystem(Tuple{typeof(x_implicit_nl)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x_implicit_nl)}; debug_config, ipo_analysis_mode=false);

sol_ipo = solve(DAECProblem(x_implicit_nl, (1,) .=> 1.), IDA())
sol = solve(DAEProblem(sys, [0.], [1.], (0.), x_implicit_nl), IDA())

@test map(x->x[1], sol_ipo(sol_ipo.t).u) ≈ (1. .+ atanh(0.5)*sol_ipo.t)
@test sol_ipo(sol.t).u ≈ sol.u

#============================= + External Eq =================================#
@noinline x_intro_eq()  = equation()
@noinline function x_outro_eq(eq)
    v = variable()
    eq(state_ddt(v) - v)
end

x_implicit_eq() = x_outro_eq(x_intro_eq())
sys_ipo = IRODESystem(Tuple{typeof(x_implicit_eq)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x_implicit_eq)}; debug_config, ipo_analysis_mode=false);

sol_ipo = solve(DAECProblem(x_implicit_eq, (1,) .=> 1.), IDA())
sol = solve(DAEProblem(sys, [0.], [1.], (0.), x_implicit_eq), IDA())

@test map(x->x[1], sol_ipo(sol_ipo.t).u) ≈ exp.(sol_ipo.t) atol=0.1
@test_broken sol_ipo(sol.t).u ≈ sol.u

#============================= + External Eq NL =================================#
@noinline x_intro_eq_nl()  = equation()
@noinline function x_outro_eq_nl(eq)
    v = variable()
    eq(state_ddt(v) - sin(v))
end

x_implicit_eq_nl() = x_outro_eq_nl(x_intro_eq_nl())
sys_ipo = IRODESystem(Tuple{typeof(x_implicit_eq_nl)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x_implicit_eq_nl)}; debug_config, ipo_analysis_mode=false);

if Base.__has_internal_change(v"1.12-alpha", :methodspecialization)
    sol_ipo = solve(DAECProblem(x_implicit_eq_nl, (1,) .=> 1.), IDA())
    sol = solve(DAEProblem(sys, [0.], [1.], (0.), x_implicit_eq_nl), IDA())

    @test map(x->x[1], sol_ipo(sol_ipo.t).u) ≈ 2acot.(exp.(log(cot(0.5)) .- sol_ipo.t)) atol=0.1
    @test sol_ipo(sol.t).u ≈ sol.u
end

#============================= External Eq Multi =================================#
@noinline x_intro_eq_var()  = (variable(), equation())
@noinline x_outro_eq_var(var, eq) = eq(ddt(var) - 1.)

function x_implicit_eq_var()
    (var, eq) = x_intro_eq_var()
    x_outro_eq_var(var, eq)
    x_outro_eq_var(var, eq)
end
sys_ipo = IRODESystem(Tuple{typeof(x_implicit_eq_var)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x_implicit_eq_var)}; debug_config, ipo_analysis_mode=false);

sol_ipo = solve(DAECProblem(x_implicit_eq_var, (1,) .=> 1.), IDA())
sol = solve(DAEProblem(sys, [0.], [1.], (0.), x_implicit_eq_var), IDA())

# 2dx/dt = 2
@test map(x->x[1], sol_ipo(sol_ipo.t).u) ≈ (1. .+ sol.t) atol=0.1
@test sol_ipo(sol.t).u ≈ sol.u

#============================= External Eq Multi (2) =================================#
@noinline x_intro_eq_var2()  = (state_ddt(variable()), equation())
@noinline x_outro_eq_var2(var, eq) = eq(var)

function x_implicit_eq_var2()
    (var, eq) = x_intro_eq_var2()
    x_outro_eq_var2(-var, eq)
    x_outro_eq_var2(1., eq)
    x_outro_eq_var2(1., eq)
end
sys_ipo = IRODESystem(Tuple{typeof(x_implicit_eq_var2)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x_implicit_eq_var2)}; debug_config, ipo_analysis_mode=false);

sol_ipo = solve(DAECProblem(x_implicit_eq_var2, (1,) .=> 1.), IDA())
sol = solve(DAEProblem(sys, [0.], [1.], (0.), x_implicit_eq_var2), IDA())

@test map(x->x[1], sol_ipo(sol_ipo.t).u) ≈ (1. .+ 2sol_ipo.t) atol=0.1
@test sol_ipo(sol.t).u ≈ sol.u

#=================== + Scope handling =============================#
@noinline function x_scope!(scope)
    x = variable(scope)
    equation!(state_ddt(x) - x + epsilon(scope), scope)
end
function x2_scope!()
    x_scope!(Scope(Scope(), :x1));
    x_scope!(Scope(Scope(), :x2));
    x_scope!(Scope(Scope(Scope(), :x3), :x4));
    return nothing
end

sys_ipo = IRODESystem(Tuple{typeof(x2_scope!)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_scope!)}; debug_config, ipo_analysis_mode=false);

let ipo_result = getfield(sys_ipo, :result), nonipo_result = getfield(sys, :result)
    @test length(ipo_result.var_to_diff) == length(nonipo_result.var_to_diff)
    @test length(ipo_result.names) == length(nonipo_result.names)
    @test isa(sys_ipo.x3.x4, DAECompiler.ScopeRef)
end

#=================== + GenScope =============================#
@noinline function x_gen!(scope)
    scope = GenScope(scope, :g)
    x = variable(scope)
    equation!(state_ddt(x) - x + epsilon(scope), scope)
end
function x2_gen!()
    x_gen!(Scope(Scope(), :x1)); x_gen!(Scope(Scope(), :x1));
    return nothing
end

sys_ipo = IRODESystem(Tuple{typeof(x2_gen!)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_gen!)}; debug_config, ipo_analysis_mode=false);

let ipo_result = getfield(sys_ipo, :result), nonipo_result = getfield(sys, :result)
    @test length(ipo_result.var_to_diff) == length(nonipo_result.var_to_diff)
    @test length(ipo_result.names) == length(nonipo_result.names)
end

#=================== + derived scope =============================#
@noinline function x_derived!(scope)
    vscope = scope(:x)
    x = variable(vscope)
    observed!(x, scope(:xo))
    equation!(state_ddt(x) - x + epsilon(vscope), vscope)
end
function x2_derived!()
    x_derived!(Scope(Scope(), :x1)); x_derived!(Scope(Scope(), :x2));
    return nothing
end

sys_ipo = IRODESystem(Tuple{typeof(x2_derived!)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_derived!)}; debug_config, ipo_analysis_mode=false);

let ipo_result = getfield(sys_ipo, :result), nonipo_result = getfield(sys, :result)
    @test length(ipo_result.var_to_diff) == length(nonipo_result.var_to_diff)
    @test length(ipo_result.names) == length(nonipo_result.names)
end

#=================== + equation arg =============================#
using DAECompiler.Intrinsics: equation
@noinline function x_eqarg!(eq, scope)
    vscope = scope(:x)
    x = variable(vscope)
    observed!(x, scope(:xo))
    eq(state_ddt(x) - x + epsilon(vscope))
end
function x2_eqarg!()
    e1 = equation(Scope(Scope(), :e1)); e2 = equation(Scope(Scope(), :e2))
    x_eqarg!(e1, Scope(Scope(), :x1)); x_eqarg!(e2, Scope(Scope(), :x2));
    return nothing
end

sys_ipo = IRODESystem(Tuple{typeof(x2_eqarg!)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_eqarg!)}; debug_config, ipo_analysis_mode=false);

let ipo_result = getfield(sys_ipo, :result), nonipo_result = getfield(sys, :result)
    @test length(ipo_result.var_to_diff) == length(nonipo_result.var_to_diff)
    @test length(ipo_result.names) == length(nonipo_result.names)
end

#====================== + ScopedValue =============================#
using Base.ScopedValues
const debug_scope = ScopedValue{DAECompiler.Intrinsics.AbstractScope}()

@noinline function x_sv!()
    scope = debug_scope[]
    x = variable(scope)
    equation!(state_ddt(x) - x + epsilon(scope), scope)
end
function x2_sv!()
    with(x_sv!, debug_scope => Scope(Scope(), :x1))
    with(x_sv!, debug_scope => Scope(Scope(), :x2))
    return nothing
end

sys_ipo = IRODESystem(Tuple{typeof(x2_sv!)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_sv!)}; debug_config, ipo_analysis_mode=false);

let ipo_result = getfield(sys_ipo, :result), nonipo_result = getfield(sys, :result)
    @test length(ipo_result.var_to_diff) == length(nonipo_result.var_to_diff)
    @test length(ipo_result.names) == length(nonipo_result.names)
end

#============================ vararg ===============================#
@noinline function x_va!(args...)
    map(args) do x
        equation!(ddt(x[1]) - x[1])
    end
end

@noinline function x_va_scope!(args...)
    # Extra tuple to exercise some of the deeper nesting code paths
    x_va!(map(x->(variable(x[1]) + epsilon(x[1]), 1.0), args)...)
end

function x2_va!()
    x_va_scope!((Scope(Scope(), :x1), 2.0), (Scope(Scope(), :x2), 2.0))
end

sys_ipo = IRODESystem(Tuple{typeof(x2_va!)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_va!)}; debug_config, ipo_analysis_mode=false);

let ipo_result = getfield(sys_ipo, :result), nonipo_result = getfield(sys, :result)
    @test !any(==(Float64), ipo_result.total_incidence)
    @test length(ipo_result.var_to_diff) == length(nonipo_result.var_to_diff)
    @test length(ipo_result.names) == length(nonipo_result.names)
end

#====================== internal variable leaking ======================#
@noinline make_a_variable() = variable()

@noinline function x_internal!()
    x = make_a_variable()
    equation!(ddt(x) - x)
end
function x2_internal!()
    x_internal!(); x_internal!();
    return nothing
end

sys_ipo = IRODESystem(Tuple{typeof(x2_internal!)}; debug_config, ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_internal!)}; debug_config, ipo_analysis_mode=false);

let ipo_result = getfield(sys_ipo, :result), nonipo_result = getfield(sys, :result)
    @test length(ipo_result.var_to_diff) == length(nonipo_result.var_to_diff)
    @test length(ipo_result.names) == length(nonipo_result.names)
end

end
