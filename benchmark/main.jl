using Test
using SciMLBase, Sundials

include("thermalfluid.jl")

Benchmark{3}()()
@test isa(code_lowered(DAECompiler.factory, Tuple{Val{DAECompiler.Settings(mode=DAECompiler.DAENoInit)}, Benchmark{3}})[1], Core.CodeInfo)
let sol = solve(DAECProblem(Benchmark{3}(), [1:9;] .=> 0.), IDA())
    @test_broken sol.retcode == ReturnCode.Success
end
