module mtk_components

using DAECompiler
using DAECompiler.Intrinsics
using DAECompiler.MTKComponents
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Test

@mtkmodel FOL begin
    @parameters begin
        τ=1.0 # parameters
    end
    @variables begin
        x(t) # dependent variables
    end
    @equations begin
        D(x) ~ (1 - x) / τ
    end
end


const fol_mtk = FOL(; name=:fol)
const FolConnector = MTKConnector(fol_mtk, fol_mtk.x)

struct ScaledFOL{T <: MTKConnector}
    scale::Float64
    fol_conn!::T
end

function (this::ScaledFOL)()
    (; scaled_x, x_outer) = variables()
    this.fol_conn!(x_outer) 
    equation!(scaled_x - this.scale * x_outer)
end
@testset "Normal case" begin
    p = ScaledFOL(2.5, FolConnector())
    p()  # make sure no errors
    sys = IRODESystem(Tuple{typeof(p)})
    prob = ODEProblem(sys, nothing, (0.0, 10.0), p; jac=true)
    sol = solve(prob, Rodas5P())

    # connection between fol and outer variables should work
    @test sol[sys.fol.x] == sol[sys.x_outer]
    # output of fol.x is basically a log function:
    @test sol(0; idxs=sys.fol.x) ≈ 0 atol=1e-4
    @test sol(10; idxs=sys.fol.x) ≈ 1 atol=1e-4
    @test issorted(sol[sys.fol.x]; rev=false)  # monotonically increasing
    #outer variables should work
    @test sol[sys.scaled_x] ≈ 2.5*sol[sys.x_outer]
end

###
# Now let's set τ to a negative value
# This isn't realistic for a FOL (first order lag) 
# but it *is* something that we can simulate
@testset "Nonphysical case" begin
    p = ScaledFOL(2.5, FolConnector(τ=-1.0))
    p()  # make sure no errors
    sys = IRODESystem(Tuple{typeof(p)})
    prob = ODEProblem(sys, nothing, (0.0, 10.0), p; jac=true)
    sol = solve(prob, Rodas5P())

    # connection between fol and outer variables shoould work
    @test sol[sys.fol.x] == sol[sys.x_outer]
    # output of fol.x is basically a upside down expodential
    @test sol(0; idxs=sys.fol.x) ≈ 0 atol=1e-4
    @test sol(10; idxs=sys.fol.x) < -2e4
    @test issorted(sol[sys.fol.x]; rev=true)  # monotonically decreasing
    #outer variables should work
    @test  sol[sys.scaled_x] == 2.5*sol[sys.x_outer]
end

@testset "Works on things that have been structural_simplified" begin
    @mtkmodel FOL_with_aliases begin
        @parameters begin
            τ=1.0 # parameters
        end
        @variables begin
            x(t) # dependent variables
            y(t)
        end
        @equations begin
            D(x) ~ (1 - y) / τ
            y ~ x
        end
    end
    fol_wa_mtk = structural_simplify(FOL_with_aliases(; name=:fol))
    FolWAConnector = MTKConnector(fol_wa_mtk, fol_wa_mtk.x)

    p = ScaledFOL(1.0, FolWAConnector())
    p()  # make sure no errors
    sys = IRODESystem(Tuple{typeof(p)})
    prob = ODEProblem(sys, nothing, (0.0, 10.0), p; jac=true)
    sol = solve(prob, Rodas5P())

    # basic sensibile result:
    @test sol[sys.fol.x] == sol[sys.x_outer]
    @test  sol[sys.scaled_x] == sol[sys.x_outer]
    # can still access variables that structural_simplify would have removed.
    @test sol[sys.fol.x] == sol[sys.fol.y]
end

end  # module
