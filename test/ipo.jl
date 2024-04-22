module ipo

using Test
using DAECompiler, SciMLBase, OrdinaryDiffEq, Sundials
using DAECompiler.Intrinsics
include(joinpath(Base.pkgdir(DAECompiler), "test", "testutils.jl"))

#= Test basic IPO functionality with repeated-noinlined function =#
@noinline function x!()
    x = variable()
    equation!(state_ddt(x) - x)
end
function x2!()
    x!(); x!();
    return nothing
end

sys_ipo = IRODESystem(Tuple{typeof(x2!)}; ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2!)}; ipo_analysis_mode=false);

@test length(getfield(sys_ipo, :result).var_to_diff) ==
      length(getfield(sys, :result).var_to_diff)

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

sys_ipo = IRODESystem(Tuple{typeof(x2_scope!)}; ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_scope!)}; ipo_analysis_mode=false);

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

sys_ipo = IRODESystem(Tuple{typeof(x2_gen!)}; ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_gen!)}; ipo_analysis_mode=false);

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

sys_ipo = IRODESystem(Tuple{typeof(x2_derived!)}; ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_derived!)}; ipo_analysis_mode=false);

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

sys_ipo = IRODESystem(Tuple{typeof(x2_eqarg!)}; ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_eqarg!)}; ipo_analysis_mode=false);

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

sys_ipo = IRODESystem(Tuple{typeof(x2_sv!)}; ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_sv!)}; ipo_analysis_mode=false);

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

sys_ipo = IRODESystem(Tuple{typeof(x2_va!)}; ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_va!)}; ipo_analysis_mode=false);

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

sys_ipo = IRODESystem(Tuple{typeof(x2_internal!)}; ipo_analysis_mode=true);
sys = IRODESystem(Tuple{typeof(x2_internal!)}; ipo_analysis_mode=false);

let ipo_result = getfield(sys_ipo, :result), nonipo_result = getfield(sys, :result)
    @test length(ipo_result.var_to_diff) == length(nonipo_result.var_to_diff)
    @test length(ipo_result.names) == length(nonipo_result.names)
end

end
