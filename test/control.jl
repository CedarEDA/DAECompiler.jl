module control

using Test
using DAECompiler
using DAECompiler.Intrinsics
using OrdinaryDiffEq
using SciMLBase

include("testutils.jl")

struct Control
    a::Bool
end

function (c::Control)()
    a = variable(:a)
    if c.a
        b = a - sin(sim_time())
    else
        b = a - cos(sim_time())
    end
    equation!(b)
end

let sys = IRODESystem(Tuple{Control})
    daeprob = DAEProblem(sys, [], [], (0, 1e-5), Control(true))
    odeprob = ODEProblem(sys, [], (0, 1e-5), Control(true))
    @test daeprob.u0 == odeprob.u0 == nothing
    @test init(odeprob)[sys.a] == 0.
end

function control_dependent_taint()
    vars = ntuple(_->variable(), 3)
    (a, b, c) = vars
    (da, db, dc) = map(state_ddt, vars)
    if a > 1.
        x = b
    else
        x = c
    end
    equation!(db - 15.0)
    equation!(dc - 3.0)
    equation!(da - x)
end
let sys = IRODESystem(Tuple{typeof(control_dependent_taint)})
    (; ir) = DAECompiler.StructuralAnalysisResult(sys)
    for i = 1:length(ir.stmts)
        inst = ir.stmts[i]
        if isa(inst[:inst], Core.PhiNode)
            TT = inst[:type]
            @test isa(TT, DAECompiler.Incidence)
            @test !iszero(TT.row[2])
        end
    end
end
for sol in solve_dae_and_ode(IRODESystem(Tuple{typeof(control_dependent_taint)}), [0.0, 2.0, 3.0], [0.0, 0.0, 0.0], (0, 1.))
    # Analytic solution is 3.5. How well the integrator does depends on how well it resolves the discontinuity, which it's not that
    # great at currently.
    @test sol[end][1] ≈ 3.5 atol=1e-2
    @test sol[end][2] ≈ 15.0 atol=1e-8
    @test sol[end][3] ≈ 3.0 atol=1e-8
end

"""
```julia
julia> lineplot(sol.t, sol[1,:], canvas=AsciiCanvas)
   ┌────────────────────────────────────────┐
20 │                                        │
   │                                        │
   │                                        │
   │                                   ..'''│
   │                              ..'''     │
   │                        ...'''          │
   │                       :                │
   │                      :                 │
   │                     :                  │
   │                    :                   │
   │                   :                    │
   │                  :                     │
   │                 :                      │
   │           .....'                       │
 0 │......'''''                             │
   └────────────────────────────────────────┘
    0                                      5
```
"""
function multi_control()
    vars = ntuple(_->variable(), 3)
    (a, b, c) = vars
    da = state_ddt(a)
    if b < 0.
        x = 1.
    else
        if c > 0.
            x = 2.
        else
            x = 10.
        end
    end
    equation!(da - x)
    equation!(b - (sim_time() - 2.))
    equation!(c - (sim_time() - 3.))
end
let sys = IRODESystem(Tuple{typeof(multi_control)})
    (; ir) = DAECompiler.StructuralAnalysisResult(sys)
    for i = 1:length(ir.stmts)
        inst = ir.stmts[i]
        if isa(inst[:inst], Core.PhiNode)
            TT = inst[:type]
            @test isa(TT, DAECompiler.Incidence)
            @test !iszero(TT.row[3])
            @test !iszero(TT.row[4])
        end
    end
end
for sol in solve_dae_and_ode(IRODESystem(Tuple{typeof(multi_control)}), [0.0], [0.0], (0, 5.))
    @test sol[end][1] ≈ 16 atol=1e-2
end

# Regression test for solved phi node into single bb (Issue #77)
"""
julia> lineplot(sol.t, sol[1,:], canvas=DotCanvas)
  ┌────────────────────────────────────────┐
3 │                                        │
  │                                        │
  │                               ..'''''''│
  │                            .''         │
  │                           .'           │
  │                         .'             │
  │                       .'               │
  │                      .'                │
  │                    .'                  │
  │              ....:'                    │
  │            .:'                         │
  │           .'                           │
  │         .'                             │
  │       .'                               │
0 │.....:'                                 │
  └────────────────────────────────────────┘
   0                                      5
"""
function regress77()
    a = variable()
    s1 = sin(sim_time())
    s2 = cos(sim_time())
    x = s1
    if a > 1.
        x = s2
    end
    equation!(state_ddt(a) - x^2)
end
for sol in solve_dae_and_ode(IRODESystem(Tuple{typeof(regress77)}), [0.0], [0.0], (0, 5.), regress77)
    # All we really care about here is that we don't get NaN in the initializtion,
    # which would have thrown a large error, but let's test something.
    @test sol[1,end] > 2.
end

end # module control
