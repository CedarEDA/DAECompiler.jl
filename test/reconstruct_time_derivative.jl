module reconstruct_time_derivative
using DAECompiler
using Test
using OrdinaryDiffEq
using DAECompiler.Intrinsics
using DAECompiler.Intrinsics: state_ddt
using DAECompiler: reconstruct_time_deriv

@testset "time derivative" begin
    @testset "just state" begin
        function foo()
            y = variable(:y)
            equation!(state_ddt(y) - y - 10.0)  # y is connected to sim_time via the selected state
        end
        foo()

        sys = IRODESystem(Tuple{typeof(foo)}; debug_config=(; store_ir_levels=true));
        tsys = TransformedIRODESystem(sys)
        prob = ODEProblem(tsys, nothing, (1.0, 10.0), [1.0]; jac=true);
        sol = solve(prob, Rodas5P())

        times = [2.0, 4.0, 8.0]
        dy = reconstruct_time_deriv(sol, [sys.y], times)
        @test dy .- 10.0 ≈ hcat(sol(times).u...)  rtol=1e-3  # derviative of exp is exp
    end

    @testset "uses epsilon" begin
        # This is as per the above "just state" but it uses epsilon, which should have no effect
        # This used to segfault before we fixed it
        function foo()
            y = variable(:y)
            ε = epsilon(:ε)
            equation!(state_ddt(y) - y - 10.0 + ε)  # y is connected to sim_time via the selected state
        end
        foo()

        sys = IRODESystem(Tuple{typeof(foo)}; debug_config=(; store_ir_levels=true));
        tsys = TransformedIRODESystem(sys)
        prob = ODEProblem(tsys, nothing, (1.0, 10.0), [1.0]; jac=true);
        sol = solve(prob, Rodas5P())

        times = [2.0, 4.0, 8.0]
        dy = reconstruct_time_deriv(sol, [sys.y], times)
        @test dy .- 10.0 ≈ hcat(sol(times).u...)  rtol=1e-3  # derviative of exp is exp
    end

    @testset "big test" begin
        function foo()
            x = variable(:x)
            y = variable(:y)
            observed!(2*x, :two_x) # two_x is connected to sim_time via the nonselected variable x
            observed!(2*y, :two_y)  # two_y is connected to sim_time via the selected state y
            observed!(1 + 2*2.0, :five) # five no connection to time
            equation!(state_ddt(y) - y + 10.0)  # y is connected to sim_time via the selected state
            equation!(x - sin(sim_time()))  # x is connected directly to sim_time as a nonselected variable
            observed!(y + sim_time(), :y_plus_time)  # y plus time is connected both directly to sim_time and also via the selected state y

            x2 = variable(:x2)
            equation!(x2 - x)  # alias for nonselected state x
            y2 = variable(:y2)
            equation!(y2 - y)  # alias for selected state y
        end
        foo()

        sys = IRODESystem(Tuple{typeof(foo)}; debug_config=(; store_ir_levels=true));
        tsys = TransformedIRODESystem(sys)
        prob = ODEProblem(tsys, nothing, (0.0, 10.0), Float64[1.0]; jac=true);
        sol = solve(prob, Rodas5P())

        times = [0, π, 2π, 8.0]
        dy, dtwo_y, dy2 = eachrow(reconstruct_time_deriv(sol, [sys.y, sys.two_y, sys.y2], times))
        @test length(dy) == 4
        @test 2dy == dtwo_y
        @test all(<(0), dy) || all(>(0), dy)
        @test issorted(dy) || issorted(dy; rev=true)
        @test dy == dy2

        dy_plus_time = reconstruct_time_deriv(sol, [sys.y_plus_time], times)
        @test dy_plus_time[:] ≈ dy .+ 1.0

        dx, dtwo_x, dx2, dfive = eachrow(reconstruct_time_deriv(sol, [sys.x, sys.two_x, sys.x2, sys.five], times))
        @test dx == cos.(times)
        @test 2dx == dtwo_x
        @test dfive == [0.0, 0.0, 0.0, 0.0]
        @test dx == dx2
    end
end

end  # module
