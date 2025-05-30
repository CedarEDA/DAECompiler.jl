include("basic.jl")
include("debugging.jl")
include("reflection.jl")
include("incidence.jl")
include("ipo.jl")
include("ssrm.jl")
include("regression.jl")
include("errors.jl")
include("invalidation.jl")

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), "benchmark")) do
  include("../benchmark/thermalfluid.jl")
end
