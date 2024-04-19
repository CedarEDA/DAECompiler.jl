module interpreter

using Test
using DAECompiler
using DAECompiler: Incidence, DAEInterpreter, get_toplevel_mi_from_ir
using Core.Compiler: IRCode, IncrementalCompact, NewInstruction, Argument, insert_node_here!
include(joinpath(Base.pkgdir(DAECompiler), "test", "lorenz.jl"))

function has_dae_intrinsics(interp::DAEInterpreter, @nospecialize(tt))
    match = Base._which(tt)
    mi = Core.Compiler.specialize_method(match)
    return has_dae_intrinsics(interp, mi)
end
function has_dae_intrinsics(interp::DAEInterpreter, mi::Core.MethodInstance)
    codeinst = Core.Compiler.getindex(Core.Compiler.code_cache(interp), mi)
    inferred = (@atomic :monotonic codeinst.inferred)::DAECompiler.DAECache
    return inferred.info.has_dae_intrinsics
end

let x = Lorenz1(10.0, 28.0, 8.0/3.0)
    x() # Sanity check
    interp, frame = DAECompiler.typeinf_dae(Tuple{typeof(x)})
    @test  has_dae_intrinsics(interp, frame.linfo)
    @test !has_dae_intrinsics(interp, Tuple{typeof(-),Float64,Float64})
    @test !has_dae_intrinsics(interp, Tuple{typeof(*),Float64,Float64})
end

# tfunc tests

const CC = Core.Compiler

"""
Returns a MethodInstance and IR corresponding to an abstract call to
`f(::types[1], ::types[2], ::types[3], ...)` where types can include
Const and Incidence types, if desired.
"""
function create_tfunc_ir(f::Function, types)
    ir = IRCode()
    ir[Core.SSAValue(1)] = nothing

    empty!(ir.argtypes)
    push!(ir.argtypes, Tuple{})
    append!(ir.argtypes, types)
    compact = IncrementalCompact(ir)

    # call `f(args...)`
    ret = insert_node_here!(
        compact,
        NewInstruction(Expr(:call, f), Any, Int32(0)),
    )
    append!(compact[ret][:stmt].args, [Argument(1 + i) for i=1:length(types)])
    insert_node_here!(
        compact,
        NewInstruction(Core.ReturnNode(ret), Nothing, Int32(0)),
    )
    ir = CC.finish(compact)
    cfg = CC.compute_basic_blocks(ir.stmts.stmt)
    append!(ir.cfg.blocks, cfg.blocks)
    append!(ir.cfg.index, cfg.index)

    # create MethodInstance
    return ir, get_toplevel_mi_from_ir(ir, Module())
end
function tfunc_verify_rt(f::Function, types, expected_return_type)
    ir, mi = create_tfunc_ir(Base.muladd, types)
    return DAECompiler.infer_ir!(ir, DAEInterpreter(), mi) === expected_return_type
end

# Standard Const propagation should work as usual
@test tfunc_verify_rt(Base.muladd,
                      (Core.Const(2.5), Core.Const(1.0), Core.Const(10.0)),
                      Core.Const(Base.muladd(2.5, 1.0, 10.0)))

# DAECompiler custom tfunc should propagate the 0.0, so that this is Const
# (i.e. it will assume the other argument is finite and non-NaN)
@test tfunc_verify_rt(Base.muladd,
                      (Core.Const(0.0), Incidence(Float64), Core.Const(10.0)),
                      Core.Const(10.0))

@test tfunc_verify_rt(Base.muladd,
                      (Core.Const(0.0), Float64, Core.Const(10.0)),
                      Float64)

end # module interpreter
