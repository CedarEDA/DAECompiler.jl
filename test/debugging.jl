module Debugging

using Test
using DAECompiler
using DAECompiler.Intrinsics
using InteractiveUtils: @code_typed
using Sundials
using SciMLBase
using OrdinaryDiffEq

const -ᵢ = Core.Intrinsics.sub_float

@noinline function twoeq!()
    x = continuous()
    y = continuous()
    always!(ddt(x) -ᵢ x)
    always!(ddt(y) -ᵢ y)
end

@testset "IR statements in stacktraces" begin
    function test_stmt_debuginfo(f)
        exc = bt = nothing
        try
            f()
        catch e
            exc = e
            bt = catch_backtrace()
        end
        @test isa(exc, BoundsError)
        output = sprint(Base.show_backtrace, bt)
        @test contains(output, "inferred type: SubArray{Float64") # insert_ssa_debuginfo
    end

    # use a short `u0` to trigger an error and get a stacktrace
    u0 = Float64[0.0]

    settings = DAECompiler.Settings(; mode = DAECompiler.ODENoInit, insert_ssa_debuginfo = true)
    odef, _ = DAECompiler.factory(Val(settings), twoeq!)
    prob = ODEProblem(odef, u0, (0.0, 1.0))
    test_stmt_debuginfo(() -> solve(prob, Rodas5()))

    settings = DAECompiler.Settings(; mode = DAECompiler.DAENoInit, insert_ssa_debuginfo = true)
    daef, differential_vars = DAECompiler.factory(Val(settings), twoeq!)
    prob = DAEProblem(daef, u0, u0, (0.0, 1.0))
    test_stmt_debuginfo(() -> solve(prob, IDA()))
end;

@testset "`DebugInfo`" begin
    settings = DAECompiler.Settings(; mode = DAECompiler.ODENoInit, insert_ssa_debuginfo = true)
    odef, _ = DAECompiler.factory(Val(settings), twoeq!)
    src = first(@code_typed debuginfo=:source odef.f(Float64[], Float64[], SciMLBase.NullParameters(), 1.0))
    output = sprint(show, src)
    @test contains(output, "inferred type:")

    settings = DAECompiler.Settings(; mode = DAECompiler.DAENoInit, insert_ssa_debuginfo = true)
    daef, _ = DAECompiler.factory(Val(settings), twoeq!)
    src = first(@code_typed debuginfo=:source daef.f(Float64[], Float64[], Float64[], SciMLBase.NullParameters(), 1.0))
    output = sprint(show, src)
    @test contains(output, "inferred type:")

    settings = DAECompiler.Settings(; mode = DAECompiler.ODENoInit, insert_stmt_debuginfo = true)
    odef, _ = DAECompiler.factory(Val(settings), twoeq!)
    src = first(@code_typed debuginfo=:source odef.f(Float64[], Float64[], SciMLBase.NullParameters(), 1.0))
    output = sprint(show, src)
    @test contains(output, "test/debugging.jl") && contains(output, "twoeq!")
    @test contains(output, "ode_factory.jl")
    @test contains(output, "intrinsics.jl") && contains(output, "continuous")

    settings = DAECompiler.Settings(; mode = DAECompiler.DAENoInit, insert_stmt_debuginfo = true)
    daef, _ = DAECompiler.factory(Val(settings), twoeq!)
    src = first(@code_typed debuginfo=:source daef.f(Float64[], Float64[], Float64[], SciMLBase.NullParameters(), 1.0))
    output = sprint(show, src)
    @test contains(output, "test/debugging.jl") && contains(output, "twoeq!")
    @test contains(output, "dae_factory.jl")
    @test contains(output, "intrinsics.jl") && contains(output, "continuous")
end;

end
