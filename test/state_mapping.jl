module state_mapping
using SciMLBase
using Random
using DAECompiler: DAECompiler
using Test
using DAECompiler
using DAECompiler.Intrinsics
using SymbolicIndexingInterface

@testset "spit and join syms" begin
    split_and_sort_syms = DAECompiler.split_and_sort_syms
    join_syms = DAECompiler.join_syms

    "Dummy stand-in for `ScopeRef` for testing purposes"
    struct DummyRef
        idx::Int
        is_obs::Bool
    end
    function DAECompiler._sym_to_index(_, ref::DummyRef)
        ref.is_obs ? DAECompiler.NameLevel(nothing, ref.idx, nothing, nothing, nothing) :
            DAECompiler.NameLevel(ref.idx, nothing, nothing, nothing, nothing)
    end

    indexes = shuffle(10:99)[1:50]
    obs_mask = rand(Bool, 50)
    syms = DummyRef.(indexes, obs_mask)

    (var_inds, obs_inds)=split_and_sort_syms(nothing, syms)

    var_data = zeros(Int, length(var_inds), 3)
    var_data[:, 1] .= var_inds

    obs_data = zeros(Int, length(obs_inds), 3)
    obs_data[:, 1] .= obs_inds


    merged_data = join_syms(nothing, syms, var_data, obs_data)
    @test merged_data[:, 1] == indexes

    # check passing in inds from earlier also works the same:
    @test merged_data == join_syms(nothing, syms, var_data, obs_data, (var_inds, obs_inds))
end

@testset "variable_symbols" begin
    function foo()
        x = variable(:x)
        A = Scope(Scope(), :A)
        Ax = variable(Scope(A, :x))
        AA =Scope(A, :AA)
        AAx = variable(Scope(AA, :x))

        equation!(x - ddt(x))
        equation!(Ax - ddt(Ax))
        equation!(AAx - ddt(AAx))
    end
    sys = IRODESystem(Tuple{typeof(foo)})
    tsys = TransformedIRODESystem(sys)
    @test SymbolicIndexingInterface.variable_symbols(tsys) == [Symbol("▫.x"), Symbol("▫.A.x"), Symbol("▫.A.AA.x")]
end

end  # module
