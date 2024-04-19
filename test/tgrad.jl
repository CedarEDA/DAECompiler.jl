module tgrad

using DAECompiler, SciMLBase, Test
using DAECompiler.Intrinsics: variables, variable, equation!, state_ddt, sim_time
using FiniteDiff
using Random
Random.seed!(42)

function finitediff_tgrad(F!)
    function (dT, u, p, t)
        FiniteDiff.finite_difference_gradient!(dT, (_dT, _t)->F!(_dT, u, p, _t), t)
        return dT
    end
end

function check_tgrad(f, name=""; t=10rand())
    @testset "$name" begin
        sys = IRODESystem(Tuple{typeof(f)})
        tsys = TransformedIRODESystem(sys)
        (;state, var_eq_matching) = tsys
        (; F!, neqs) = DAECompiler.dae_finish!(state, var_eq_matching, false; allow_unassigned=true)
        fin_tgrad = finitediff_tgrad(F!)

        tgrad = DAECompiler.construct_tgrad(sys)


        dT = -zeros(neqs)
        u = randn(neqs)
        p = Base.issingletontype(typeof(f)) ? nothing : f

        @test tgrad(copy(dT), u, p, t) ≈ fin_tgrad(copy(dT), u, p, t) rtol=1e-6
    end
end

function check_tgrad_mostly_eliminated(f)
    sys = IRODESystem(Tuple{typeof(f)}, debug_config=(; store_ir_levels=true, verify_ir_levels=true))
    tgrad = DAECompiler.construct_tgrad(sys)
    src = select_ir(sys, r"^construct_tgrad.state_type=Float64$")

    @test src.stmts[end-1][:inst].args[1] == GlobalRef(Base, :memoryrefset!)
    @test src.stmts[end-1][:inst].args[2] == Core.SSAValue(length(src.stmts)-2)
    @test src.stmts[end-2][:inst].args[1] == GlobalRef(Base, :memoryref)
end


include("cases/pwl_at_time.jl")
@testset "pwl_at_time makes good code" begin
    function pwl_at_time_sys()
        (;x, y) = variables()
        equation!(y - pwl_at_time(example_wave, sim_time()))
        equation!(state_ddt(x) - state_ddt(y))
    end
    check_tgrad(pwl_at_time_sys, "pwl_at_time diff"; t=0.24)
    check_tgrad_mostly_eliminated(pwl_at_time_sys)
end



@testset "parameterized struct" begin
    struct DemoParamStruct
        α::Float64
        β::Float64
    end
    function (self::DemoParamStruct)()
        (; x, y) = variables()
        equation!(x^2 + self.α*sim_time())
        equation!(state_ddt(x) - self.β*y + 5x)
    end
    check_tgrad(DemoParamStruct(4,3))
end

@testset "big list" begin
    check_tgrad("basic") do
        x = variable(:x)
        equation!(sim_time()^2 - x^3)
    end

    check_tgrad("product") do
        x = variable(:x)
        equation!(x*sim_time())
    end

    check_tgrad("state_ddt") do
        x = variable(:x)
        equation!(sim_time() + state_ddt(x))
    end

    check_tgrad("multiple") do
        x = variable(:x)
        y = variable(:y)
        equation!(x - sin(sim_time()))
        equation!(y - cos(sim_time()))
    end

    check_tgrad("multi_arg_plus") do
        (;x, y) = variables()
        equation!(+(1.0, y, sim_time()))
        equation!(state_ddt(x) - state_ddt(y))
    end
end

end
