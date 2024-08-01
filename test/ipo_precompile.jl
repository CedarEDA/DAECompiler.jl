module ipo_precompile

using Test
using DAECompiler
using DAECompiler.Intrinsics
using SciMLBase, OrdinaryDiffEq, Sundials

include("utils/precompile_utils.jl")

function test_no_dae_compile(f)
    (old_nsicmcompiles, old_nrhscompiles, old_nfactorycompiles) = (DAECompiler.nsicmcompiles, DAECompiler.nrhscompiles, DAECompiler.nfactorycompiles)
    f()
    @test old_nsicmcompiles    == DAECompiler.nsicmcompiles
    @test old_nrhscompiles     == DAECompiler.nrhscompiles
    @test old_nfactorycompiles == DAECompiler.nfactorycompiles
end

precompile_test_harness("Basic IPO Precompile") do load_path
    write(joinpath(load_path, "BasicIPO.jl"),
        """
        module BasicIPO

        using Test
        using DAECompiler
        using DAECompiler.Intrinsics
        import DAECompiler.Intrinsics: state_ddt
        using SciMLBase, OrdinaryDiffEq, Sundials

        @noinline function x!()
            x = variable()
            equation!(state_ddt(x) - x)
        end
        function x2!()
            x!(); x!();
            return nothing
        end

        solve(DAECProblem(x2!, (1, 2) .=> 1.), IDA())
        end
        """)
    Base.compilecache(Base.PkgId("BasicIPO"))
    BasicIPO = (@eval (using BasicIPO; BasicIPO))
    Core.eval(Main, :(BasicIPO = $(BasicIPO)))

    println("=================== Loaded Precompile ============================")

    # This should not require recompilation
    invokelatest() do
        test_no_dae_compile() do
            solve(DAECProblem(BasicIPO.x2!, (1, 2) .=> 1.), IDA())
        end
    end
end

finish_precompile_test!()

end
