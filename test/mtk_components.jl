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

struct ScaledFOL{T<:FolConnector}
    scale::Float64
    fol_conn!::T
end

function (this::ScaledFOL)()
    (; scaled_x, x_outer) = variables()
    this.fol_conn!(x_outer) 
    equation!(scaled_x - this.scale * x_outer)
end

p = ScaledFOL(2.5, FolConnector())
p()  # make sure no errors
sys = IRODESystem(Tuple{typeof(p)})
prob = ODEProblem(sys, nothing, (0.0, 10.0), p; jac=true)
sol = solve(prob, Rodas5P())

# connection between fol and outer variables sould work
@test sol[sys.fol.var"x(t)"] == sol[sys.x_outer]
# output of fol.x is basically a log function:
@test sol(0; idxs=sys.fol.var"x(t)") ≈ 0 atol=1e-4
@test sol(10; idxs=sys.fol.var"x(t)") ≈ 1 atol=1e-4
@test issorted(sys.fol.var"x(t)"; rev=true)
#outer variables should work
@test 2.5 * sol[sys.scaled_x] == sol[sys.x_outer]

