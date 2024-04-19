module statemapping_derivatives
using DAECompiler, Test
using DAECompiler.Intrinsics: variable, variables, equation, equation!, state_ddt, sim_time, observed!
using DAECompiler: compile_batched_reconstruct_derivatives, compile_batched_reconstruct_func
using FiniteDifferences

const _fdm = central_fdm(5, 1; max_range=1e-2)

function check_der_reconstruct(f, name=""; dae_var_broken=false)
    f()  # make sure it doesn't error

    sys = IRODESystem(Tuple{typeof(f)}, debug_config=Dict(:store_ir_levels=>true, :verify_ir_levels=>true))
    tsys = (; state, var_eq_matching) = TransformedIRODESystem(sys)

    t = 10rand()
    p = Base.issingletontype(typeof(f)) ? nothing : f

    @testset "$(name)" begin
    @testset "ODE" begin
        isdae=false
        reconstruct! = compile_batched_reconstruct_func(tsys, 1:length(var_eq_matching), 1:state.nobserved, isdae)
        der_reconstruct! = compile_batched_reconstruct_derivatives(tsys,  1:length(var_eq_matching), 1:state.nobserved, false, isdae)
        (; neqs) = DAECompiler.assign_vars_and_eqs(tsys, isdae)
        
        u = randn(neqs)
        nparams = DAECompiler.determine_num_tangents(DAECompiler.parameter_type(get_sys(tsys)))
        advar_du = similar(u, length(var_eq_matching), neqs)
        advar_dp = similar(u, length(var_eq_matching), nparams)
        adobs_du = similar(u, state.nobserved, neqs)
        adobs_dp = similar(u, state.nobserved, nparams)
        der_reconstruct!(advar_du, advar_dp, adobs_du, adobs_dp, u, p, t)

        function reconstruct_wrap(u, p)
            var_out = similar(u, length(var_eq_matching))
            obs_out = similar(u, state.nobserved)
            reconstruct!(var_out, obs_out, u, p, t)
            return var_out, obs_out
        end

        fdvar_du = FiniteDifferences.jacobian(_fdm, u -> reconstruct_wrap(u, p)[1], u)[1]
        @test advar_du ≈ fdvar_du

        if nparams > 0
            fdvar_dp = FiniteDifferences.jacobian(_fdm, p -> reconstruct_wrap(u, p)[1], p)[1]
            @test advar_dp ≈ fdvar_dp
        else
            @test isempty(advar_dp)
        end

        if state.nobserved > 0 
            fdobs_du = FiniteDifferences.jacobian(_fdm, u -> reconstruct_wrap(u, p)[2], u)[1]
            @test adobs_du ≈ fdobs_du

            if nparams > 0
                fdobs_dp = FiniteDifferences.jacobian(_fdm, p -> reconstruct_wrap(u, p)[2], p)[1]
                @test adobs_dp ≈ fdobs_dp
            else
                @test isempty(adobs_dp)
            end
        else
            @test isempty(adobs_du)
            @test isempty(adobs_dp)
        end
    end
    @testset "DAE" begin
        isdae=true
        reconstruct! = compile_batched_reconstruct_func(tsys, 1:length(var_eq_matching), 1:state.nobserved, isdae)
        der_reconstruct! = compile_batched_reconstruct_derivatives(tsys,  1:length(var_eq_matching), 1:state.nobserved, false, isdae)
        (; neqs) = DAECompiler.assign_vars_and_eqs(tsys, isdae)
        
        u = randn(neqs)
        du = randn(neqs)

        nparams = DAECompiler.determine_num_tangents(DAECompiler.parameter_type(get_sys(tsys)))
        advar_du = similar(u, length(var_eq_matching), neqs)
        advar_dp = similar(u, length(var_eq_matching), nparams)
        adobs_du = similar(u, state.nobserved, neqs)
        adobs_dp = similar(u, state.nobserved, nparams)
        der_reconstruct!(advar_du, advar_dp, adobs_du, adobs_dp, du, u, p, t)

        function reconstruct_wrap(u, p)
            var_out = similar(u, length(var_eq_matching))
            obs_out = similar(u, state.nobserved)
            reconstruct!(var_out, obs_out, du, u, p, t)
            return var_out, obs_out
        end

        fdvar_du = FiniteDifferences.jacobian(_fdm, u -> reconstruct_wrap(u, p)[1], u)[1]
        @test advar_du ≈ fdvar_du broken=dae_var_broken

        if nparams > 0
            fdvar_dp = FiniteDifferences.jacobian(_fdm, p -> reconstruct_wrap(u, p)[1], p)[1]
            @test advar_dp ≈ fdvar_dp broken=dae_var_broken
        else
            @test isempty(advar_dp)
        end

        if state.nobserved > 0
            fdobs_du = FiniteDifferences.jacobian(_fdm, u -> reconstruct_wrap(u, p)[2], u)[1]
            @test adobs_du ≈ fdobs_du

            if nparams > 0
                fdobs_dp = FiniteDifferences.jacobian(_fdm, p -> reconstruct_wrap(u, p)[2], p)[1]
                @test adobs_dp ≈ fdobs_dp
            else
                @test isempty(adobs_dp)
            end
        else
            @test isempty(adobs_du)
            @test isempty(adobs_dp)
        end
    end
    end
end

@testset "observed" begin    
    check_der_reconstruct("1 state, linear obs") do
        (;x) = variables()
        observed!(2x, :two_x)
        observed!(3x, :three_x)
        equation!(x-state_ddt(x))
    end

    check_der_reconstruct("1 state, non-linear obs") do
        (;x) = variables()
        observed!(sin(x), :sin_x)
        observed!(x^2, :xsquared)
        observed!(exp(x), :exp_x)
        equation!(x-state_ddt(x))
    end
end

@testset "parameterized struct" begin
    struct DemoParamStruct
        α::Float64
        β::Float64
    end
    function (self::DemoParamStruct)()
        (; x, y) = variables()
        observed!(self.α * self.β^2 * y^2 + self.β * x)
        observed!(self.α * self.β^2)
        observed!(x * y^2)
        equation!(self.α*x + y^3)
        equation!(state_ddt(x) - self.β*y + 5x)
    end
    check_der_reconstruct(DemoParamStruct(4,3), "parameterized struct")
end

# Copied from Jacobian tests, has lots of tests that have solved_variables
@testset "big list" begin
    # basic test fully linear
    check_der_reconstruct("basic linear") do
        (; x, y) = variables()
        equation!(4x+y^3)
        equation!(state_ddt(x) - 3y + 5x)
    end

    # basic test with some nonlinear components
    check_der_reconstruct("basic nonlinear") do
        (; x, y) = variables()
        equation!(state_ddt(x) - y + x)
        equation!(x+y^2)
    end

    # Has no is_solved_variables (as all things are prime powers so can't be expressed in terms of each other)
    check_der_reconstruct("prime powers") do
        (; x, y, z) = variables()
        equation!(4x^2+y^3)
        equation!(2y^5 + x^7)
        equation!(z^9 - x^13)
    end

    check_der_reconstruct("cross-state multiplication") do
        (; x, y) = variables()
        a = 2x
        equation!(state_ddt(x) - y + x)
        equation!(a*y - x)
    end

    check_der_reconstruct("zero all but one const") do
        (; x, y, z) = variables()
        equation!(y-z)
        equation!(state_ddt(x) - z)
        equation!(y^3/x)
    end

    check_der_reconstruct("Massive simplify"; dae_var_broken=true) do
        (; x, y, z) = variables()
        equation!(state_ddt(y) - z + x)
        equation!(state_ddt(x) - z)
        equation!(y + x)
    end

    # output is unteathered from base variables due to only having equations in terms of derivatives
    check_der_reconstruct("untethered"; dae_var_broken=true) do
        (;x, y) = variables()
        dx = state_ddt(x)
        dy = state_ddt(y)
        equation!(dx^2+dy^3)
        equation!(3dy + 7dx)
    end

    # Has implicit equations due to both variable and its derivative being selected
    check_der_reconstruct("imp untethered O(1)+O(2)") do
        (;x, y) = variables()
        dx = state_ddt(x)
        dy = state_ddt(y)
        ddy = state_ddt(dy)
        equation!(4dx+dy^3)
        equation!(ddy - 3dy + 5dx)
    end

    # higher order with levels skipped
    check_der_reconstruct("imp O(1)+O(3)"; dae_var_broken=true) do
        (;x, y) = variables()
        dx = state_ddt(x)
        dy = state_ddt(y)
        ddy = state_ddt(dy)
        ddx = state_ddt(dx)
        dddx = state_ddt(ddx)
        equation!(dx^2+dy^3)
        equation!(3dy + 9dddx + 5dx)
    end
end

end  # module
