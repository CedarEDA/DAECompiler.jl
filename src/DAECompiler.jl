module DAECompiler

using SciMLBase
using StateSelection
using StateSelection: SelectedState, complete
using Diffractor
using Base.Experimental: @opaque
using Base.Meta: isexpr
using ExprTools: splitdef
using CentralizedCaches: @new_cache
using Accessors
using OrderedCollections

export IRTransformationState, IRODESystem, TransformedIRODESystem
export get_epsjac, reconstruct_sensitivities, get_transformed_sys, get_sys, select_ir

function reconstruct_sensitivities(args...)
    error("This method requires SciMLSensitivity")
end

const CC = Core.Compiler
import .CC: get_inference_world
using Base: get_world_counter

const INIT_HOOKS = Function[]
push_inithook!(f) = push!(INIT_HOOKS, f)
__init__() = foreach(@nospecialize(f)->f(), INIT_HOOKS)

include("utils.jl")

include("breadcrumbs.jl")
include("JITOpaqueClosures.jl")
include("runtime.jl")
import .Intrinsics: state_ddt
include("analysis/compiler_reexports.jl")
include("analysis/lattice.jl")
include("cache.jl")
include("analysis/interpreter.jl")
include("irodesystem.jl")
include("analysis/compiler.jl")

include("transform/common.jl")
include("transform/ad_common.jl")
include("transform/mtk_passes.jl")
include("transform/tearing_schedule.jl")
include("transform/dae_transform.jl")
include("transform/dae_finish.jl")

include("transform/jacobian_batching_utils.jl")
include("transform/classic_derivatives/common.jl")
include("transform/classic_derivatives/jacobian.jl")
include("transform/classic_derivatives/tgrad.jl")
include("transform/classic_derivatives/paramjac.jl")
include("transform/classic_derivatives/epsjac.jl")
include("transform/state_reconstruct/common.jl")
include("transform/state_reconstruct/compress.jl")
include("transform/state_reconstruct/reconstruct.jl")
include("transform/state_reconstruct/derivative.jl")
include("transform/state_reconstruct/time_derivative.jl")
include("transform/callbacks/iterative.jl")
include("transform/callbacks/vector_continous.jl")

include("analysis/index_lowering_ad.jl")
include("analysis/debugging.jl")
include("analysis/extra_rules.jl")
include("state_mapping.jl")
include("custom_init.jl")

include("../ext/DAECompilerSciMLSensitivityExt.jl")

using PrecompileTools
# enable this once we establish a way to precompile external abstract interpreter
# after https://github.com/JuliaLang/julia/pull/52233
@static v"1.11.0-DEV.1552" > VERSION ≥ v"1.11.0-DEV.1287" && @setup_workload let
    using SciMLBase
    include("../test/lorenz.jl")

    u0 = [1.0, 0.0, 0.0]
    tspan = (0.0, 100.0)
    x = Lorenz1(10.0, 28.0, 8.0/3.0)

    @info "Generating DAECompiler precompilation cache..."

    @compile_workload let
        sys = IRODESystem(Tuple{typeof(x)});
        daeprob = DAEProblem(sys, zero(u0), u0, tspan, x);
        prob = ODEProblem(sys, u0, tspan, x);
    end

    # HACK to register an initialization callback that fixes up `max_world` which is
    # overridden to `one(UInt) == WORLD_AGE_REVALIDATION_SENTINEL` by staticdata.c:
    # otherwise using cached analysis results would result in world age assertion error.
    function override_precompiled_cache()
        for (_, cache) in GLOBAL_CODE_CACHE
            for (_, codeinst) in cache.cache
                if ((@atomic :monotonic codeinst.min_world) > zero(UInt) &&
                    (@atomic :monotonic codeinst.max_world) == one(UInt)) # == WORLD_AGE_REVALIDATION_SENTINEL

                    @atomic :monotonic codeinst.max_world = typemax(UInt)
                end
            end
            Base.rehash!(cache.cache) # another HACK to avoid JuliaLang/julia#52915
        end
    end
    override_precompiled_cache() # to precompile this callback itself
    push_inithook!(override_precompiled_cache)
end

# Based on https://github.com/JuliaLang/julia/blob/cfcf8a026276f31eff170fac6ede9d07297d56cf/stdlib/Sockets/test/runtests.jl#L658-L674
mutable struct RLimit
    cur::Int64
    max::Int64
end
const RLIMIT_STACK = 3 # from /usr/include/sys/resource.h
function get_max_stacksize()
    rlim = Ref(RLimit(0, 0))
    # Get the current maximum stack size in bytes
    rc = ccall(:getrlimit, Cint, (Cint, Ref{RLimit}), RLIMIT_STACK, rlim)
    @assert rc == 0
    return rlim[].max
end

function set_stacksize!(stacksize::Int)
    rlim = Ref(RLimit(0, 0))
    # Get the current maximum stack size in bytes
    rc = ccall(:getrlimit, Cint, (Cint, Ref{RLimit}), RLIMIT_STACK, rlim)
    @assert rc == 0
    current = rlim[].cur
    rlim[].cur = stacksize
    rc = ccall(:setrlimit, Cint, (Cint, Ref{RLimit}), RLIMIT_STACK, rlim)
    if rc != 0
        rc = ccall(:getrlimit, Cint, (Cint, Ref{RLimit}), RLIMIT_STACK, rlim)
        @assert rc == 0
        @warn "Could not raise stacksize" target=stacksize actual=rlim[].cur errno=Base.Libc.errno()
    end
    return nothing
end
function default_stacksize!()
    # If we're on anything other than windows, set the stack size as cloe to 64MB as we can
    if !Sys.iswindows()
        set_stacksize!(min(64*1024*1024, get_max_stacksize()))
    end
end
push_inithook!(default_stacksize!) # ensure we do not run out of stack.

end # module
