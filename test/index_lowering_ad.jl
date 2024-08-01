module index_lowering_ad

using Test

using DAECompiler, DAECompiler.Intrinsics, DAECompiler.ChainRulesCore
using ChainRules, SciMLBase, OrdinaryDiffEq

include(joinpath(Base.pkgdir(DAECompiler), "test/testutils.jl"))
include(joinpath(Base.pkgdir(DAECompiler), "test/irutils.jl"))

# typeinf_dae
# ===========

struct SimpleFRule end
function (::SimpleFRule)()
    a = variable(:a)
    b = a - myidentity(abs(myidentity(sin(myidentity(sim_time())))))
    equation!(b)
end
myidentity(@nospecialize x) = x

function test_frule_inlineability(ir, f, args...; frule_exists=true)
    @nospecialize f args
    @test (frule_exists ? (!==) : (===))(ChainRulesCore.frule((ChainRulesCore.NoTangent(), 1.0), f, args...), nothing)
    code = map(i->ir.stmts[i][:inst], 1:length(ir.stmts))
    if frule_exists
        @test count(isinvoke(nameof(f)), code) > 0
    else
        @test count(isinvoke(nameof(f)), code) == 0
        @test count(iscall((ir, f)), code) == 0
    end
end

let tt = Tuple{SimpleFRule}
    interp, ci = DAECompiler.typeinf_dae(tt);
    ir = ci.inferred.ir
    t = 1.0
    test_frule_inlineability(ir, sin, t; frule_exists=true)
    test_frule_inlineability(ir, abs, sin(t); frule_exists=true)
    test_frule_inlineability(ir, myidentity, t; frule_exists=false)
    test_frule_inlineability(ir, myidentity, sin(t); frule_exists=false)
    test_frule_inlineability(ir, myidentity, abs(sin(t)); frule_exists=false)
end

# forward_diff!
# =============

"""
Analytical solution:

y = -cos(t) + C₁
x = cos(t) - sin(t) + C₁
dx/dt = sin(t) - cos(t)
dz/dt = sin(t) - cos(t) - 1.
z = -cos(t) - sin(t) - t + C₂

```
julia> lineplot(sol.t, sol[1,:], canvas=DotCanvas)
     ┌────────────────────────────────────────┐
   2 │                          ..'''''''.    │
     │                        .'          '.. │
     │                      .'               '│
     │                    .''                 │
     │                   :                    │
     │                 .'                     │
     │                .'                      │
     │               .'                       │
     │              .'                        │
     │            .'                          │
     │           :                            │
     │         .'                             │
     │       .'                               │
     │     .:                                 │
   0 │....'                                   │
     └────────────────────────────────────────┘
      0                                      4


julia> lineplot(sol.t, sol[2,:], canvas=DotCanvas)
      ┌────────────────────────────────────────┐
    0 │:                                       │
      │'.                                      │
      │ :                                      │
      │  :                                     │
      │   :                                    │
      │    :                                   │
      │    '.                                  │
      │     '.                                 │
      │      '.                    .......     │
      │        :               ..''       '.   │
      │         '..         ..''            '. │
      │            '.....'''                  '│
      │                                        │
      │                                        │
   -2 │                                        │
      └────────────────────────────────────────┘
       0                                      4
```
"""
function diff_constraints()
    (; x, y, z) = variables()
    equation!(x - y + sin(sim_time()))
    equation!(state_ddt(y) - sin(sim_time()))
    equation!(state_ddt(z) - state_ddt(x) + 1.)
end

function diff_constraints2()
    (; x, y, z) = variables()
    equation!(1.0*x - 1.0*y + 1.0*sin(sim_time()))
    equation!(1.0*state_ddt(y) - 1.0*sin(sim_time()))
    equation!(1.0*state_ddt(z) - 1.0*state_ddt(x) + 1.)
end

function test_diff_constraints(entry_func)
    sys = IRODESystem(entry_func)
    for sol in solve_dae_and_ode(sys, [0.0, 0.0], [0.0, 0.0], (0.0, 4.))
        @test sol[1,:] ≈ (-cos.(sol.t) .+ 1.0)  atol=0.01
        z(t) = -cos(t) - sin(t) - t
        @test sol[2,:] ≈ (z.(sol.t) .+ 1.0) atol=0.01
    end
end
test_diff_constraints(diff_constraints)
test_diff_constraints(diff_constraints2)

@test_throws "Expected incidence of variable `x`" solve_dae(#=jac=#true, [0.], [1.], (0., 1.)) do
    (;x) = variables()
    equation!(state_ddt(2.0*x) - sin(sim_time()))
end
@test_throws "Duplicated state_ddt for variable `x`" solve_dae(#=jac=#true, [0.], [1.], (0., 1.)) do
    (;x) = variables()
    equation!(state_ddt(x) - state_ddt(x) + 1.)
end

function second_order_zero_diff()
    (; x, y, z, out) = variables()
    equation!(y - state_ddt(x))
    equation!(z - state_ddt(y))
    equation!(z - state_ddt(out))
    equation!(x - sim_time() + 1.)
end
let sys = IRODESystem(Tuple{typeof(second_order_zero_diff)})
    for sol in solve_dae_and_ode(sys, nothing, nothing, (0.0, 1.));
        @test sol[sys.out][1] ≈ sol[sys.out][end]
    end
end

function second_order()
    (; x, y, z, out) = variables()
    equation!(y - state_ddt(x))
    equation!(z - state_ddt(y))
    equation!(z - state_ddt(out))
    equation!(x - sin(sim_time()))
end
let sys = IRODESystem(Tuple{typeof(second_order)})
    for sol in solve_dae_and_ode(sys, [0.], [0.], (0.0, 1.));
        @test sol[sys.out] ≈ (x->(cos(x)-1)).(sol.t) atol=1e-4
    end
end

# Test AD involving control flow
function var_phi()
    (; x, y, z) = variables()
    if y > 0.
        a = 0.
    else
        a = z
    end
    equation!(x - 3.5 * a)
    equation!(state_ddt(z))
    equation!(state_ddt(x) - state_ddt(y))
end
function test_var_phi(var_phi)
    let sys = IRODESystem(Tuple{typeof(var_phi)})
        tsys = TransformedIRODESystem(sys)
        @test DAECompiler.num_selected_states(tsys, true) == 2
        @test DAECompiler.num_selected_states(tsys, false) == 2
        for sol in (solve_dae(sys, [3.5, 0.], [-2., 1.], (0.0, 1.), var_phi)...,
                    solve_ode(sys, [-2., 1.], (0.0, 1.), var_phi)...)
            @test sol[sys.x][1] ≈ sol[sys.x][2]
            @test sol[sys.y][1] ≈ sol[sys.y][2]
            @test sol[sys.z][1] ≈ sol[sys.z][2]
        end
    end
end
test_var_phi(var_phi)

# Add a nonlinearity into the mix
function var_phi_sin()
    (; x, y, z) = variables()
    if y > 0.
        a = 0.
    else
        a = z
    end
    equation!(x - sin(3.5 * a))
    equation!(state_ddt(z))
    equation!(state_ddt(x) - state_ddt(y))
end
test_var_phi(var_phi_sin)

# Force three states and test that having both a control flow
# and data dependency can co-exist.  X-ref issue #355
# This is always expected to have three ODE states.
function var_phi_threestate()
    (; x, y, z) = variables()
    if y > 0.
        a = sin(y)
    else
        a = z + sin(y)
    end
    equation!(x - 3.5 * a)
    equation!(state_ddt(z))
    equation!(state_ddt(x) - state_ddt(y))
end
function test_var_phi_threestate(var_phi)
    let sys = IRODESystem(Tuple{typeof(var_phi)})
        for sol in (solve_dae(sys, [3.5, 0.], [-2., 1.], (0.0, 1.), var_phi)...,
                    solve_ode(sys, [-2., 1., 0.], (0.0, 1.), var_phi)...)
            @test sol[sys.x][1] ≈ sol[sys.x][2]
            @test sol[sys.y][1] ≈ sol[sys.y][2]
            @test sol[sys.z][1] ≈ sol[sys.z][2]
        end
    end
end
test_var_phi_threestate(var_phi_threestate)

struct ArgDiff
    x::Float64
    s::Symbol
end
function (this::ArgDiff)()
    (; x, y, z) = variables()
    if y > 0.
        a = 0.
    else
        a = z
    end
    equation!(x - (this.s == :ok ? this.x * a : 2 * this.x * a))
    equation!(state_ddt(z))
    equation!(state_ddt(x) - state_ddt(y))
end
test_var_phi(ArgDiff(3.5, :ok))

# Include original test case from https://github.com/JuliaComputing/DAECompiler.jl/issues/21
const R = 1.0
const t_ramp = 10
const L = 1.0e-2

# This tests that Pantelides is able to differentiate the first equation
# to add a constraint on `di`.  We are able to observe this as `v_2` must
# equal `d/dt equation(1) * L1`, which in this case is t_ramp * L.
function serial_inductor_var_csrc()
    (;v_source, v_1, v_2, i) = variables()
    di = state_ddt(i)
    equation!(i - t_ramp*sim_time())
    equation!((v_source - v_1) / R - i)
    equation!((v_1 - v_2) / L - di)
    equation!(v_2 / L - di)
end

sys = IRODESystem(Tuple{typeof(serial_inductor_var_csrc)})
for sol in solve_dae_and_ode(sys, [0.0], [0.0], (0, 1e-5))
    @test sol[sys.i][1] ≈ 0.0
    @test sol[sys.i][end] ≈ 1e-4

    @test all(sol[sys.v_1] .≈ 0.2)
    @test all(sol[sys.v_2] .≈ t_ramp * L)
end


@testset "All paths error after index lowering" begin
    bad_sin(x) = sin(x)
    # it only fails due to the the frule being bad, so index lowering brings out the failure.
    ChainRulesCore.frule(_, ::typeof(bad_sin), x) = sin(x), error("frule broken")
    function bad_sys()
        (; a, b) = variables()
        equation!(a - state_ddt(b))
        equation!(b - bad_sin(sim_time()))
    end

    sys = IRODESystem(Tuple{typeof(bad_sys)})
    try
        tsys = TransformedIRODESystem(sys)
        @assert(false, "should have errored when making tsys")
    catch err
        @test err isa DAECompiler.UnsupportedIRException
        @test contains(string(err), "During index lowering, after diffractor, function unconditionally errors")
        @test contains(string(err), r"∂☆.*bad_sin.*Union\{\}")  # this is the key info needed
    end
end

end # module index_lowering_ad
