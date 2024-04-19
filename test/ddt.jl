module ddt_tests

using Test
using DAECompiler
using DAECompiler.Intrinsics
using SciMLBase
using OrdinaryDiffEq, Sundials
using ForwardDiff

include(joinpath(Base.pkgdir(DAECompiler), "test/testutils.jl"))

function ddt_trivial()
    (;x) = variables()
    equation!(ddt(x) - x)
end
test_ddx_eq_x(ddt_trivial)

function ddt_nonlinear()
    (;x,y) = variables()
    equation!(ddt(x*x)/2x - x)
    equation!(ddt(x*x)/2x - y)
end
let sys = IRODESystem(ddt_nonlinear; debug_config=(; store_ss_levels=true))
    dbg = getfield(sys, :debug_config)

    for sol in solve_ode(sys, [1., 0.], (0., 1.); reltol=1e-6)
        @test all(isapprox.(sol[sys.x], exp.(sol.t), atol=1e-4))
        @test all(isapprox.(sol[sys.y], exp.(sol.t), atol=1e-4))
    end

    for sol in solve_dae(sys, [0.], [1.], (0., 1.); reltol=1e-6)
        @test all(isapprox.(sol[sys.x], exp.(sol.t), atol=1e-4))
        @test all(isapprox.(sol[sys.y], exp.(sol.t), atol=1e-4))
    end

    for level in dbg.ss_levels
        # Verify that both equations are incident on the expected variables
        g = level.mss.structure.graph
        @test DAECompiler.nsrcs(g) == 2
        @test DAECompiler.ndsts(g) == 3
        @test DAECompiler.𝑠neighbors(g, 1) == [1,3]   # x, dx
        @test DAECompiler.𝑠neighbors(g, 2) == [1,2,3] # x, y, dx
    end
end

function ddt_nonlinear()
    (;x,y) = variables()
    equation!(ddt(x*y)/2x - x)
    equation!(ddt(x*y)/2y - y)
end
let sys = IRODESystem(ddt_nonlinear; debug_config=(; store_ss_levels=true))
    dbg = getfield(sys, :debug_config)

    for sol in (solve_dae(sys, [0., 0.], [1., 1.], (0., 1.))...,
                solve_ode(sys, [1., 1., 0., 0.], (0., 1.))..., )
        @test all(isapprox.(sol[sys.x], exp.(sol.t), atol=1e-2))
        @test all(isapprox.(sol[sys.y], exp.(sol.t), atol=1e-2))
    end

    for level in dbg.ss_levels
        # Verify that both equations are incident on the expected variables
        g = level.mss.structure.graph
        @test DAECompiler.𝑠neighbors(g, 1) == [1,2,3,4] # x, y, dx, dy
        @test DAECompiler.𝑠neighbors(g, 2) == [1,2,3,4] # x, y, dx, dy
    end
end

end
