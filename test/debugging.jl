module Debugging

using Test
using DAECompiler
using DAECompiler.Intrinsics
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
        buffer = IOBuffer()
        Base.show_backtrace(buffer, bt)
        output = String(take!(seekstart(buffer)))
        @test contains(output, "inferred type: SubArray{Float64")
    end

    # use an empty `u0` to trigger a stacktrace
    u0 = Float64[0.0]

    settings = DAECompiler.Settings(; mode = DAECompiler.ODENoInit, insert_stmt_debuginfo = true)
    odef, _ = DAECompiler.factory(Val(settings), twoeq!)
    prob = ODEProblem(odef, u0, (0.0, 1.0))
    test_stmt_debuginfo(() -> solve(prob, Rodas5()))

    settings = DAECompiler.Settings(; mode = DAECompiler.DAENoInit, insert_stmt_debuginfo = true)
    daef, differential_vars = DAECompiler.factory(Val(settings), twoeq!)
    prob = DAEProblem(daef, u0, u0, (0.0, 1.0))
    test_stmt_debuginfo(() -> solve(prob, IDA()))
end;

end
