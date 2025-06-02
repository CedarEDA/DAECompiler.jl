module Reflection

using Test
using DAECompiler
using DAECompiler.Intrinsics
using Compiler: IRCode

@noinline callee!(x, y) = always!(x + y)

function ssrm()
    x₁ = continuous()
    x₂ = continuous()
    x₃ = continuous()
    x₄ = continuous()
    ẋ₁ = ddt(x₁)
    ẋ₂ = ddt(x₂)
    ẋ₃ = ddt(x₃)
    ẋ₄ = ddt(x₄)
    ẍ₁ = ddt(ẋ₁)
    ẍ₂ = ddt(ẋ₂)
    ẍ₃ = ddt(ẋ₃)
    always!((ẍ₁+ẍ₂)-(ẋ₁+ẋ₂)+x₄)
    always!((ẍ₁+ẍ₂)+x₃)
    always!(x₂+ẍ₃+ẋ₄)
    callee!(x₃,ẋ₄)
end

ir = code_structure_by_type(Tuple{typeof(ssrm)})
@test isa(ir, IRCode)
ir = code_structure(ssrm)
@test isa(ir, IRCode)
ir = @code_structure ssrm()
@test isa(ir, IRCode)
ir = @code_structure world = Base.get_world_counter() ssrm()
@test isa(ir, IRCode)
result = @code_structure result = true ssrm()
@test isa(result, DAECompiler.DAEIPOResult)
result = @code_structure matched = true ssrm()
@test isa(result, DAECompiler.MatchedSystemStructure)
@test contains(sprint(show, result), r"%\d+")

end # module
