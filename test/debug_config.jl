module debug_config

using Test
using DAECompiler, SciMLBase, OrdinaryDiffEq, Sundials

module non_debug
    using DAECompiler, SciMLBase, OrdinaryDiffEq, Sundials
    include(joinpath(Base.pkgdir(DAECompiler), "test/lorenz.jl"))

    u0 = [1.0, 0.0, 0.0]
    tspan = (0.0, 100.0)
    x = Lorenz1ts(10.0, 28.0, 8.0/3.0)

    sys = IRODESystem(Tuple{typeof(x)});
    daeprob = DAEProblem(sys, zero(u0), u0, tspan, x);
end

module debug
    using DAECompiler, SciMLBase, OrdinaryDiffEq, Sundials
    include(joinpath(Base.pkgdir(DAECompiler), "test/lorenz.jl"))

    u0 = [1.0, 0.0, 0.0]
    tspan = (0.0, 100.0)
    x = Lorenz1ts(10.0, 28.0, 8.0/3.0)

    debug_config = (;
        store_ir_levels = true,
        ir_log = joinpath(mktempdir(), "test"),
        replay_log = true,
    )

    sys = IRODESystem(Tuple{typeof(x)}; debug_config);
    daeprob = DAEProblem(sys, zero(u0), u0, tspan, x);
    odeprob = ODEProblem(sys, u0, tspan, x; jac=true);
end

let tsys = non_debug.daeprob.f.sys
    non_debug_ir_levels = DAECompiler.DebugConfig(tsys).ir_levels
    @test isnothing(non_debug_ir_levels)
end

let tsys = debug.daeprob.f.sys
    debug_ir_levels = DAECompiler.DebugConfig(tsys).ir_levels
    debug_ir_log = DAECompiler.DebugConfig(tsys).ir_log

    # When store_ir_levels is enabled, we should save more ir_levels
    # than without
    @test !isnothing(debug_ir_levels) && length(debug_ir_levels) > 0

    # The debug run should have saved IR in the requested directory,
    # including the unoptimized IR from pipeline entry.
    @test isfile(joinpath(debug_ir_log, "01.Lorenz1ts.compute_structure.unoptimized.ir"))

    # Solve the ODE system, ensure that it logs everything out to `replay_log`
    prob = debug.odeprob
    sol = solve(prob, Rodas5P())
    replay_log = getfield(get_sys(prob), :debug_config).replay_log
    @test length(sol[get_sys(prob).obby]) == length(sol.t)

    # Ensure that our replay log gets filled
    @test !isempty(replay_log["RHS"])
    @test !isempty(replay_log["jacobian"])
    @test !isempty(replay_log["tgrad"])
    @test !isempty(replay_log["reconstruct"])

    # Ensure that the replays are faithful
    out = zeros(length(prob.u0))
    for args in copy(replay_log["RHS"])
        prob.f.f(out, args[2:end]...)
        @test out == args[1]
    end
    J = zeros(length(prob.u0), length(prob.u0))
    for args in copy(replay_log["jacobian"])
        prob.f.jac(J, args[2:end]...)
        @test J == args[1]
    end
    dT = zeros(length(prob.u0))
    for args in copy(replay_log["tgrad"])
        prob.f.tgrad(dT, args[2:end]...)
        @test dT == args[1]
    end
end

end # module debug_config
