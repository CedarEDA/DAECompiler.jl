using DAECompiler.Intrinsics
using DAECompiler

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D


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

struct ScaledFOL
    scale::Float64
    time_constant::Float64
end


const fol_mtk = FOL(; name=:fol)
const fol_connector! = MTKConnector(fol, fol.τ, fol.x) 

function (this::ScaledFOL)()
    (; scaled_x, x_outer) = variables()
    fol_connector!(this.time_constant, x_outer) 
    equation!(scaled_x - this.scale * x_outer)
end

sys = IRODESystem(Tuple{ScaledFOL})
prob = ODEProblem(ScaledFOL(2.5, 3.0), (0.0, 10.0); jac=true)

# connection between fol and outer variables sould work
@test sol[sys.fol.x] == sol[sys.x_outer]
# output of fol.x is basically a log function:
@test sol(sys.fol.x; t=0) ≈ 0
@test sol(sys.fol.x; t=10.0) > 0.9
@test issorted(sol.fol.x; rev=true)
#outer variables should work
@test 2.5 * sol[sys.scaled_x] == sol[sys.x_outer]

