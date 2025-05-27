module _Incidence

using Test
using DAECompiler
using DAECompiler: Incidence, IncidenceValue, Linearity, refresh, linear, linear_time_dependent, linear_state_dependent, linear_time_and_state_dependent, nonlinear
using DAECompiler.Intrinsics
using Compiler: SSAValue, Const
using SparseArrays: rowvals, nonzeros

const *ᵢ = Core.Intrinsics.mul_float
const +ᵢ = Core.Intrinsics.add_float
const -ᵢ = Core.Intrinsics.sub_float

walk(ex::Expr, inner, outer) = outer(Expr(ex.head, map(inner, ex.args)...))
walk(ex, inner, outer) = outer(ex)

postwalk(f, ex) = walk(ex, x -> postwalk(f, x), f)

"""
    @infer_incidence u₁ + u₂ + t
    @infer_incidence begin
      x = u₁ + u₂
      x + u₃
    end
    @infer_incidence u₁ + u₂ true    # a second argument of `true` makes it return the IR, not the Incidence
    @infer_incidence exp(u₂) + u₁    # will create u₂ as the second continuous variable

Return the `Incidence` object after inferring the structure of the provided code,
substituting any variables starting by 'u'. Variables are created as `continuous()`
in lexicographic order, so u₂ will appear before u₁. A special `t` variable may be
used, which will be created as `sim_time()`.

Note that if variable indices are not contiguous starting from 1, incidence annotations
will not correspond to the input symbols. `@infer_incidence 1 + u₂` for example will be inferred
to `Incidence(1.0 + u₁)`.
"""
macro infer_incidence(ex, return_ir = false)
  variables = Symbol[]
  uses_time = false
  ex = postwalk(ex) do subex
    # Force all observed function calls to be `@noinline`,
    # the caller probably wants to test IPO.
    Meta.isexpr(subex, :call) && return :(@noinline $subex)
    isa(subex, Symbol) || return subex
    uses_time |= subex === :t
    startswith(string(subex), "u") || return subex
    !in(subex, variables) && push!(variables, subex)
    return subex
  end
  sort!(variables, by = string)
  prelude = Expr(:block, [:($(esc(var)) = continuous()) for var in variables]...)
  uses_time && pushfirst!(prelude.args, :($(esc(:t)) = sim_time()))
  quote
    function incidence()
      $prelude
      $(esc(ex))
    end
    ir = code_structure_by_type(Tuple{typeof(incidence)})
    $return_ir ? ir : begin
      ssa = SSAValue(length(ir.stmts) - 1)
      ir[ssa][:type]::Incidence
    end
  end
end

dependencies(row) = sort(rowvals(row) .=> nonzeros(row), by = first)

@testset "Incidence" begin
  err = "state or time independence is not supported for nonlinearities"
  @test_throws err Linearity(state_dependent = false, nonlinear = true)
  @test_throws err Linearity(time_dependent = false, nonlinear = true)

  @testset "Construction & printing" begin
    incidence = Incidence(Const(1.0))
    @test incidence.typ === Const(1.0)
    @test dependencies(incidence.row) == []
    @test repr(incidence) == "Incidence(1.0)"
    @test incidence == incidence"1.0"

    incidence = Incidence(Float64)
    @test repr(incidence) == "Incidence(a)"
    @test incidence == incidence"a"

    incidence = Incidence(Float64, IncidenceValue[1.0])
    @test dependencies(incidence.row) == [1 => 1]
    @test repr(incidence) == "Incidence(a + t)"
    @test incidence == incidence"a + t"

    incidence = Incidence(String, IncidenceValue[1.0])
    @test repr(incidence) == "Incidence(String, t)"
    @test incidence == incidence"String, t"

    incidence = Incidence(1)
    @test incidence.typ === Const(0.0)
    @test dependencies(incidence.row) == [2 => 1]
    @test repr(incidence) == "Incidence(u₁)"
    @test incidence == incidence"u₁"

    incidence = Incidence(3)
    @test dependencies(incidence.row) == [4 => 1]
    @test repr(incidence) == "Incidence(u₃)"
    @test incidence == incidence"u₃"

    incidence = Incidence(Const(3.0), IncidenceValue[0.0, 0.0, 2.0, 1.0])
    @test repr(incidence) == "Incidence(3.0 + 2.0u₂ + u₃)"
    @test incidence == incidence"3.0 + 2.0u₂ + u₃"

    incidence = Incidence(Const(0.0), IncidenceValue[4.0, 0.0, 2.0])
    @test repr(incidence) == "Incidence(4.0t + 2.0u₂)"
    @test incidence == incidence"4.0t + 2.0u₂"

    incidence = Incidence(Const(0.0), IncidenceValue[nonlinear])
    @test repr(incidence) == "Incidence(f(t))"
    @test incidence == incidence"f(t)"

    incidence = Incidence(Const(0.0), IncidenceValue[linear])
    @test repr(incidence) == "Incidence(∝t)"
    @test incidence == incidence"∝t"

    incidence = Incidence(Const(0.0), IncidenceValue[1.0, nonlinear])
    @test repr(incidence) == "Incidence(t + f(u₁))"
    @test incidence == incidence"t + f(u₁)"

    incidence = Incidence(Const(0.0), IncidenceValue[1.0, linear])
    @test repr(incidence) == "Incidence(t + ∝u₁)"
    @test incidence == incidence"t + ∝u₁"

    incidence = Incidence(Const(0.0), IncidenceValue[linear, linear, linear])
    @test repr(incidence) == "Incidence(∝t + ∝u₁ + ∝u₂)"
    @test incidence == incidence"∝t + ∝u₁ + ∝u₂"

    incidence = Incidence(Const(0.0), IncidenceValue[linear_state_dependent, linear_time_dependent, linear])
    @test repr(incidence) == "Incidence(∝u₂ + f(∝ₛt, ∝ₜu₁))"
    @test incidence == incidence"∝u₂ + f(∝ₛt, ∝ₜu₁)"
    @test incidence == incidence"Incidence(∝u₂ + f(∝ₛt, ∝ₜu₁))"

    incidence = Incidence(Const(0.0), IncidenceValue[linear_state_dependent, linear_time_dependent, nonlinear])
    @test repr(incidence) == "Incidence(f(∝ₛt, ∝ₜu₁, u₂))"
    @test incidence == incidence"f(∝ₛt, ∝ₜu₁, u₂)"

    incidence = Incidence(Const(0.0), IncidenceValue[nonlinear, linear_time_dependent, nonlinear])
    @test repr(incidence) == "Incidence(f(t, ∝ₜu₁, u₂))"
    @test incidence == incidence"f(t, ∝ₜu₁, u₂)"

    @test_throws "inconsistent with an absence of time incidence" Incidence(Const(0.0), IncidenceValue[0.0, linear_time_dependent])
    @test_throws "inconsistent with an absence of state incidence" Incidence(Const(0.0), IncidenceValue[linear_state_dependent])
    @test_throws "absence of state dependence for time" Incidence(Const(0.0), IncidenceValue[1.0, linear_time_dependent])
    @test_throws "absence of state dependence for time" Incidence(Const(0.0), IncidenceValue[linear, linear_time_dependent])
  end

  @testset "Basic lattice operations with intrinsic tfuncs" begin
    incidence = @infer_incidence t
    @test incidence.typ === Const(0.0)
    @test dependencies(incidence.row) == [1 => 1]
    @test incidence == incidence"t"

    incidence = @infer_incidence 5. +ᵢ u₁
    @test incidence.typ === Const(5.0)
    @test dependencies(incidence.row) == [2 => 1]
    @test incidence == incidence"5.0 + u₁"

    incidence = @infer_incidence 5.0 +ᵢ t +ᵢ u₁
    @test incidence.typ === Const(5.0)
    @test dependencies(incidence.row) == [1 => 1, 2 => 1]
    @test incidence == incidence"5.0 + t + u₁"

    incidence = @infer_incidence t *ᵢ u₁
    @test incidence.typ === Const(0.0)
    @test dependencies(incidence.row) == [1 => linear_state_dependent, 2 => linear_time_dependent]
    @test incidence == incidence"f(∝ₛt, ∝ₜu₁)"

    incidence = @infer_incidence u₁
    @test incidence.typ === Const(0.0)
    @test dependencies(incidence.row) == [2 => 1]
    @test incidence == incidence"u₁"

    incidence = @infer_incidence u₁ +ᵢ u₂
    @test incidence.typ === Const(0.0)
    @test dependencies(incidence.row) == [2 => 1, 3 => 1]
    @test incidence == incidence"u₁ + u₂"

    incidence = @infer_incidence u₁ *ᵢ u₂
    @test incidence.typ === Const(0.0)
    @test dependencies(incidence.row) == [2 => linear_state_dependent, 3 => linear_state_dependent]
    @test incidence == incidence"f(∝ₛu₁, ∝ₛu₂)"

    incidence = @infer_incidence (2.0 +ᵢ u₁) *ᵢ (3.0 +ᵢ u₂)
    @test incidence.typ === Const(6.0)
    @test dependencies(incidence.row) == [2 => linear_state_dependent, 3 => linear_state_dependent]
    @test incidence == incidence"6.0 + f(∝ₛu₁, ∝ₛu₂)"

    incidence = @infer_incidence (2.0 +ᵢ u₁) *ᵢ (3.0 +ᵢ u₁ *ᵢ u₂)
    @test incidence.typ === Const(6.0)
    @test dependencies(incidence.row) == [2 => nonlinear, 3 => linear_state_dependent]
    @test incidence == incidence"6.0 + f(u₁, ∝ₛu₂)"

    incidence = @infer_incidence (2.0 +ᵢ u₁) *ᵢ (3.0 +ᵢ u₁ *ᵢ u₂) +ᵢ u₃
    @test incidence.typ === Const(6.0)
    @test dependencies(incidence.row) == [2 => nonlinear, 3 => linear_state_dependent, 4 => 1.0]
    @test incidence == incidence"6.0 + u₃ + f(u₁, ∝ₛu₂)"

    incidence = @infer_incidence (u₁ +ᵢ u₂) *ᵢ t
    @test incidence.typ === Const(0.0)
    @test dependencies(incidence.row) == [1 => linear_state_dependent, (2:3 .=> linear_time_dependent)...]
    @test incidence == incidence"f(∝ₛt, ∝ₜu₁, ∝ₜu₂)"

    incidence = @infer_incidence u₁ *ᵢ u₂ +ᵢ (u₁ +ᵢ t) *ᵢ u₃
    @test dependencies(incidence.row) == [(1:3 .=> linear_state_dependent)..., 4 => linear_time_and_state_dependent]
    @test incidence == incidence"f(∝ₛt, ∝ₛu₁, ∝ₛu₂, ∝ₜₛu₃)"
  end

  @testset "IPO" begin
    # NOTE: Most of additive terms (`.typ`) can't be precise given the current IPO representation.
    # We expect `Const(0.0)` in most cases, but widen to `Float64`.

    incidence = @infer_incidence 1.0t * 3.0
    @test dependencies(incidence.row) == [1 => linear]
    @test incidence == incidence"a + ∝t"

    incidence = @infer_incidence (1.0 + u₁ +ᵢ u₂) * 1.0
    @test incidence.typ === Float64
    @test dependencies(incidence.row) == (2:3 .=> linear_state_dependent)
    @test incidence == incidence"a + f(∝ₛu₁, ∝ₛu₂)"

    incidence = @infer_incidence (2.0 + u₁) * (3.0 + u₂)
    @test dependencies(incidence.row) == [2 => linear_state_dependent, 3 => linear_state_dependent]
    @test incidence == incidence"a + f(∝ₛu₁, ∝ₛu₂)"

    incidence = @infer_incidence 5.0 + u₁
    @test incidence.typ === Const(5.0)
    @test dependencies(incidence.row) == [2 => 1]
    @test incidence == incidence"5.0 + u₁"

    incidence = @infer_incidence u₁ * u₁
    @test dependencies(incidence.row) == [2 => nonlinear]
    @test incidence == incidence"f(u₁)"

    incidence = @infer_incidence t * t
    @test dependencies(incidence.row) == [1 => nonlinear]
    @test incidence == incidence"f(t)"

    mul3(a, b, c) = a *ᵢ (b *ᵢ c)
    incidence = @infer_incidence mul3(t, u₁, u₂)
    @test dependencies(incidence.row) == [1 => linear_state_dependent, (2:3 .=> linear_time_and_state_dependent)...]
    @test incidence == incidence"f(∝ₛt, ∝ₜₛu₁, ∝ₜₛu₂)"

    incidence = @infer_incidence mul3(t, u₁, u₁)
    @test dependencies(incidence.row) == [1 => linear_state_dependent, 2 => nonlinear]
    @test incidence == incidence"f(∝ₛt, u₁)"

    incidence = @infer_incidence mul3(t, u₁, t)
    # If we knew which state is used for state dependence,
    # state should be inferred as linear_time_dependent.
    @test dependencies(incidence.row) == [1 => nonlinear, 2 => linear_time_and_state_dependent]
    @test incidence == incidence"f(t, ∝ₜₛu₁)"

    incidence = @infer_incidence mul3(u₂, u₁, u₂)
    @test dependencies(incidence.row) == [2 => linear_state_dependent, 3 => nonlinear]
    @test incidence == incidence"f(∝ₛu₁, u₂)"

    _muladd(a, b, c) = a +ᵢ b *ᵢ c
    incidence = @infer_incidence _muladd(u₁, u₁, u₂)
    # We widen to `nonlinear` because we can't yet infer that `b := u₁` is
    # not multiplied by `a := u₁`. The solution would be to see that `a`
    # is linear but state-independent and therefore can't be a factor of `b`.
    @test dependencies(incidence.row) == [2 => nonlinear, 3 => linear_state_dependent]
    @test incidence == incidence"f(u₁, ∝ₛu₂)"

    # Here we still wouldn't be able to use the above solution because `a := u₁` is state-dependent.
    # So `c := u₁` having a state-dependent coefficient might be multiplied by `a` a.k.a itself
    # which would make it nonlinear, so IPO can only infer `u₁` as nonlinear.
    _muladd2(a, b, c, d) = d *ᵢ a +ᵢ b *ᵢ c
    incidence = @infer_incidence _muladd2(u₁, u₂, u₁, u₃)
    @test dependencies(incidence.row) == [2 => nonlinear, 3 => linear_state_dependent, 4 => linear_state_dependent]
    @test incidence == incidence"f(u₁, ∝ₛu₂, ∝ₛu₃)"

    incidence = @infer_incidence exp(u₁)
    @test dependencies(incidence.row) == [2 => nonlinear]
    @test incidence == incidence"a + f(u₁)"

    incidence = @infer_incidence t * exp(u₁)
    @test dependencies(incidence.row) == [1 => linear_state_dependent, 2 => nonlinear]
    @test incidence == incidence"a + f(∝ₛt, u₁)"

    incidence = @infer_incidence u₁ * exp(t)
    @test dependencies(incidence.row) == [1 => nonlinear, 2 => linear_time_dependent]
    @test incidence == incidence"a + f(t, ∝ₜu₁)"

    incidence = @infer_incidence u₁ * exp(t + u₂)
    @test dependencies(incidence.row) == [1 => nonlinear, 2 => linear_time_and_state_dependent, 3 => nonlinear]
    @test incidence == incidence"a + f(t, ∝ₜₛu₁, u₂)"

    incidence = @infer_incidence atan(u₁, u₂)
    @test dependencies(incidence.row) == [2 => nonlinear, 3 => nonlinear]
    @test incidence == incidence"a + f(u₁, u₂)"
  end
end;

end
