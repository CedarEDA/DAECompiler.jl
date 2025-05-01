module DAECompiler

    using Graphs
    using StateSelection
    using StateSelection: DiffGraph
    using Diffractor
    using OrderedCollections
    using Compiler
    using Compiler: IRCode, IncrementalCompact, argextype, singleton_type, isexpr, widenconst
    using Core.IR
    using SciMLBase
    using AutoHashEquals

    include("utils.jl")
    include("intrinsics.jl")
    include("analysis/utils.jl")
    include("analysis/lattice.jl")
    include("analysis/ADAnalyzer.jl")
    include("analysis/scopes.jl")
    include("analysis/cache.jl")
    include("analysis/refiner.jl")
    include("analysis/ipoincidence.jl")
    include("analysis/structural.jl")
    include("transform/state_selection.jl")
    include("transform/common.jl")
    include("transform/runtime.jl")
    include("transform/tearing/schedule.jl")
    include("transform/codegen/dae_factory.jl")
    include("transform/codegen/init_factory.jl")
    include("transform/codegen/rhs.jl")
    include("transform/codegen/init_uncompress.jl")
    include("transform/autodiff/ad_common.jl")
    include("transform/autodiff/ad_runtime.jl")
    include("transform/autodiff/index_lowering.jl")
    include("interface.jl")
    include("problem_interface.jl")
end
