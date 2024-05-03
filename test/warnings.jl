module warnings_and_errors

using Test
using DAECompiler
using DAECompiler.Intrinsics
using SciMLBase

include("testutils.jl")

struct Lorenz1{T}
    σ::T
    ρ::T
    β::T
end

function make_variable(symbol)
    return variable(symbol)
end

# x, dx/dt, y, z, a, u
function (l::Lorenz1)()
    x = make_variable(:x)
    y = make_variable(:x) # Duplicate variable assignment
    z = make_variable(:z)
    a = make_variable(:a)
    u = make_variable(:u)
    equation!.((
        u - (y - x) + 0*state_ddt(a), # test tearing and state selection
        a - (u - (y - x)), # test a == 0
        state_ddt(x) - (l.σ * u),
        state_ddt(y) - (x * (l.ρ - z) - y),
        state_ddt(z) - (x * y - l.β * z)
    ))
end

@test_logs (:warn, r"definition for scope") sys = IRODESystem(Tuple{Lorenz1{Float64}});

function genx(scope = GenScope(Scope(), :x))
    x = variable(scope)
    equation!(state_ddt(x) - x)
end

genscope() = (genx(); genx())
let sys = @test_nowarn IRODESystem(Tuple{typeof(genscope)})
    for sol in solve_dae_and_ode(sys, [0., 0.], [1., 1.], (0., 1.))
        @test all(isapprox.(sol[sys.x1], exp.(sol.t), atol=1e-2))
        @test all(isapprox.(sol[sys.x2], exp.(sol.t), atol=1e-2))
    end
end

genscope2() = genx(GenScope(Scope(), :sc)(:x))
let sys = @test_nowarn IRODESystem(Tuple{typeof(genscope2)})
    for sol in solve_dae_and_ode(sys, [0.], [1.], (0., 1.))
        @test all(isapprox.(sol[sys.sc1.x], exp.(sol.t), atol=1e-2))
    end
end

function unblanaced_too_many_eqs()
    (;x) = variables()
    equation!(x, :foo)
    qux_scope = Scope(Scope(Scope(), :Bar), :qux)
    equation!(x^2, qux_scope)
end
let sys = IRODESystem(unblanaced_too_many_eqs)
    @test_throws ["The system is unbalanced.",
        "There are 1 highest order differentiated variable(s) and 2 equation(s).",
        r"(▫.Bar.qux)|(▫.foo)",
        "was potentially redundant:"] ODEProblem(sys, nothing, (0., 1.))
end

function structurally_singular()
    (;x, u1, u2) = variables()
    equation!.((x + sin(u1 + u2),
        ddt(x) - (x + 1.0),
        cos(x) - 2.0))
end
let sys = IRODESystem(structurally_singular)
    @test_throws "The system is structurally singular.\nThis variable may be problematic:" ODEProblem(sys, nothing, (0., 1.))
end

end
