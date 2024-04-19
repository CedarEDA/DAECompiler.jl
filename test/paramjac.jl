module paramjac

using DAECompiler, SciMLBase, Test
using DAECompiler.Intrinsics: variables, variable, equation!, state_ddt, sim_time
using DAECompiler: determine_num_tangents
using FiniteDifferences  # Note: we use FiniteDifferences.jl rather than FiniteDiff.jl as it has better support for structs, and FiniteDiff was segfaulting for reasons we don't understand
using Random
Random.seed!(42)

const _fdm = central_fdm(5, 1; max_range=1e-2)
function finitediff_paramjac(F!)
    function (pJ, u, p, t)
        function f(x)
            out = copy(u)
            F!(out, u, x, t)
            return out
        end
        pJ .= FiniteDifferences.jacobian(_fdm, f, p)[1]
        return pJ
    end
end


function check_paramjac(f, name=""; t=10rand())
    @testset "$name" begin
        debug_config = (; store_ir_levels=true, verify_ir_levels=true)
        sys = IRODESystem(Tuple{typeof(f)}; debug_config)
        tsys = TransformedIRODESystem(sys)
        (;state, var_eq_matching) = tsys
        (; F!, neqs) = DAECompiler.dae_finish!(state, var_eq_matching, false; allow_unassigned=true)
        fin_paramjac = finitediff_paramjac(F!)

        paramjac = DAECompiler.construct_paramjac(sys)

        p = Base.issingletontype(typeof(f)) ? nothing : f
        nparams = determine_num_tangents(typeof(f))
        pJ = rand(neqs, nparams)
        u = randn(neqs)


        @test paramjac(copy(pJ), u, p, t) ≈ fin_paramjac(copy(pJ), u, p, t) rtol=1e-6
    end
end

@testset "no parameters" begin
    function a_dae_without_parameters()
        (; x, y) = variables()
        tt = sim_time()
        equation!(x^2 + tt)
        equation!(state_ddt(x) - 2y + 3x)
    end

    debug_config = (; store_ir_levels=true, verify_ir_levels=true)
    sys = IRODESystem(Tuple{typeof(a_dae_without_parameters)}; debug_config)
    tsys = TransformedIRODESystem(sys)

    paramjac = DAECompiler.construct_paramjac(sys)
    @test paramjac(Float64[;;], [1.0, 2.0], nothing, 1.0) == Float64[;;]
end


struct Demo3{T}
    x1::T
    x2::T
    x3::T
end

function (self::Demo3)()
    (; x, y) = variables()
    tt = sim_time()
    equation!(x^2 + self.x1*tt)
    equation!(state_ddt(x) - self.x2*y + self.x3*x)
end
check_paramjac(Demo3(1.1, 2.1, 3.1), "flat float")


struct Demo1Nest3{T}
    x1::T
    x2::Demo3{T}
end
function (self::Demo1Nest3)()
    (; x, y) = variables()
    tt = sim_time()
    equation!(x^2 + self.x1*self.x2.x1*tt)
    equation!(state_ddt(x) - self.x2.x2*y + self.x2.x3*x)
end
check_paramjac(Demo1Nest3(-0.5, Demo3(1.1, 2.1, 3.1)), "Nested")


struct Demo1
    x::Float64
end

function (self::Demo1)()
    a = self.x*2.0  # in this test we need accessing that parameter to be the first thing we do, to check we insert param bob at start, not at position 2.
    (; x, y) = variables()
    tt = sim_time()
    equation!(x^2 + a*tt)
    equation!(y^2 + x^2)
end
check_paramjac(Demo1(1.1), "access param in first stmt")

end  # module