module transform_common
using DAECompiler
using DAECompiler: IRODESystem, TransformedIRODESystem
using DAECompiler.Intrinsics
using DAECompiler.Intrinsics: state_ddt
using Test

@testset "checked_intrinstics_removed + replace_if_intrinsic" begin
    function foo()
        y = variable(:y)
        x = variable(:x)
        ε = epsilon(:ε)
        equation!(state_ddt(y) - y - 10.0 + ε)
        equation!(x - sim_time())
    end
    foo()
    sys = IRODESystem(Tuple{typeof(foo)})
    tsys = TransformedIRODESystem(sys)

    ir = tsys.state.ir
    @test_throws DAECompiler.UnexpectedIntrinsicException DAECompiler.check_for_daecompiler_intrinstics(ir)

    # use replace_if_intrinsic to remove all intrinstic
    du = [1.0, 2.0]
    u = [10.0, 20.0]
    p = nothing
    t = 0.5
    var_assignment = Dict(1=>(1,false), 2=>(2,false))
    compact = Core.Compiler.IncrementalCompact(ir)
    for ((_, idx), stmt) in compact
        ssa = Core.SSAValue(idx)
        DAECompiler.replace_if_intrinsic!(compact, ssa, du, u, p, t, var_assignment)
    end
    ir = Core.Compiler.finish(compact)

    # the following should not throw anymore
    @test DAECompiler.check_for_daecompiler_intrinstics(ir) isa Any
end


end  # module
