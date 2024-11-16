using Test
using Compiler
@eval Compiler begin
function compute_ir_rettype(ir::IRCode)
    rt = Union{}
    for i = 1:length(ir.stmts)
        stmt = ir[SSAValue(i)][:stmt]
        if isa(stmt, Core.ReturnNode) && isdefined(stmt, :val)
            rt = Compiler.tmerge(Compiler.argextype(stmt.val, ir), rt)
        end
    end
    return Compiler.widenconst(rt)
end

function compute_oc_signature(ir::IRCode, nargs::Int, isva::Bool)
    argtypes = Vector{Any}(undef, nargs)
    for i = 1:nargs
        argtypes[i] = Compiler.widenconst(ir.argtypes[i+1])
    end
    if isva
        lastarg = pop!(argtypes)
        if lastarg <: Tuple
            append!(argtypes, lastarg.parameters)
        else
            push!(argtypes, Vararg{Any})
        end
    end
    return Tuple{argtypes...}
end

function Core.OpaqueClosure(ir::IRCode, @nospecialize env...;
                           isva::Bool = false,
                           slotnames::Union{Nothing,Vector{Symbol}}=nothing,
                           kwargs...)
    # NOTE: we need ir.argtypes[1] == typeof(env)
    ir = Core.Compiler.copy(ir)
    # if the user didn't specify a definition MethodInstance or filename Symbol to use for the debuginfo, set a filename now
    ir.debuginfo.def === nothing && (ir.debuginfo.def = :var"generated IR for OpaqueClosure")
    nargtypes = length(ir.argtypes)
    nargs = nargtypes-1
    sig = compute_oc_signature(ir, nargs, isva)
    rt = compute_ir_rettype(ir)
    src = ccall(:jl_new_code_info_uninit, Ref{CodeInfo}, ())
    if slotnames === nothing
        src.slotnames = fill(:none, nargtypes)
    else
        length(slotnames) == nargtypes || error("mismatched `argtypes` and `slotnames`")
        src.slotnames = slotnames
    end
    src.slotflags = fill(zero(UInt8), nargtypes)
    src.slottypes = copy(ir.argtypes)
    src.isva = isva
    src.nargs = UInt64(nargtypes)
    src = ir_to_codeinf!(src, ir)
    src.rettype = rt
    return Base.Experimental.generate_opaque_closure(sig, Union{}, rt, src, nargs, isva, env...; kwargs...)
end
end
    
@testset "state_mapping.jl" include("state_mapping.jl")
@testset "interpreter.jl" include("interpreter.jl")
@testset "compiler_and_lattice.jl" include("compiler_and_lattice.jl")
@testset "JITOpaqueClosures.jl" include("JITOpaqueClosures.jl")
@testset "robertson.jl" include("robertson.jl")
@testset "ipo.jl" include("ipo.jl")
@testset "lorenz.jl" include("lorenz_tests.jl")
@testset "pendulum.jl" include("pendulum_tests.jl")
@testset "mass_matrix.jl" include("mass_matrix.jl")
@testset "custom_init.jl" include("custom_init.jl")
@testset "custom_inline.jl" include("custom_inline.jl")
@testset "control.jl" include("control.jl")
#@testset "dynamic_state_error.jl" include("dynamic_state_error.jl")
@testset "index_lowering_ad.jl" include("index_lowering_ad.jl")
@testset "ddt.jl" include("ddt.jl")
@testset "implied_alias.jl" include("implied_alias.jl")
@testset "regression.jl" include("regression.jl")
@testset "transform_common.jl" include("transform_common.jl")
@testset "reconstruct.jl" include("reconstruct.jl")
@testset "reconstruct_time_derivative.jl" include("reconstruct_time_derivative.jl")
@testset "tearing_schedule.jl" include("tearing_schedule.jl")
@testset "jacobian_batching_utils.jl" include("jacobian_batching_utils.jl")
@testset "jacobian.jl" include("jacobian.jl")
@testset "tgrad.jl" include("tgrad.jl")
@testset "paramjac.jl" include("paramjac.jl")
@testset "periodic_callback.jl" include("periodic_callback.jl")
@testset "statemapping_derivatives.jl" include("statemapping_derivatives.jl")
@testset "invalidation.jl" include("invalidation.jl")
@testset "frule_invalidation.jl" include("frule_invalidation.jl")
@testset "debug_config.jl" include("debug_config.jl")
@testset "warnings.jl" include("warnings.jl")
@testset "sensitivity.jl" include("sensitivity.jl")
@testset "sensitivity_rccircuit.jl" include("sensitivity_rccircuit.jl")
@testset "epsilon.jl" include("epsilon.jl")
@testset "cthulhu.jl" include("cthulhu.jl")
@testset "mtk_components.jl" include("mtk_components.jl")

# must be last to minimize risks from monkeypatching
@testset "MSL" include("MSL/run_msl_tests.jl")
