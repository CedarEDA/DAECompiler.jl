using Base.Meta
using Base: quoted

"Set inst to be a differentiable nothing of given order"
function dnullout_inst!(inst, order=1)
    inst[:inst] = quoted(Diffractor.DNEBundle{order}(nothing))
    inst[:type] = typeof(Diffractor.DNEBundle{order}(nothing))
    inst[:flag] = Compiler.IR_FLAG_EFFECT_FREE | Compiler.IR_FLAG_NOTHROW | Compiler.IR_FLAG_CONSISTENT
    return nothing
end