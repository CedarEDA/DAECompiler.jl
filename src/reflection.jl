export code_structure_by_type, code_structure, @code_structure,
       code_ad_by_type, code_ad, @code_ad, @incidence_str

function code_structure(@nospecialize(f), @nospecialize(types = Base.default_tt(f)); kwargs...)
  tt = Base.signature_type(f, types)
  return code_structure_by_type(tt; kwargs...)
end

function code_ad(@nospecialize(f), @nospecialize(types = Base.default_tt(f)); kwargs...)
  tt = Base.signature_type(f, types)
  return code_ad_by_type(tt; kwargs...)
end

function _code_ad_by_type(@nospecialize(tt::Type);
                                world::UInt = Base.tls_world_age(),
                                force_inline_all::Bool = false)

  fT = fieldtype(tt, 1)
  # We could pick any other mode, this is for invalidation purposes only.
  settings = Settings(; mode = DAE, force_inline_all)
  factory_mi = get_method_instance(Tuple{typeof(factory),Val{settings},typeof(fT)}, world)
  # First, perform ordinary type inference, under the assumption that we may need to AD
  # parts of the function later.
  ci = ad_typeinf(world, tt; force_inline_all, edges=Core.svec(factory_mi))
end
code_ad_by_type(@nospecialize(tt::Type); kwargs...) =
  _code_ad_by_type(tt; kwargs...).inferred.ir

function code_structure_by_type(@nospecialize(tt::Type); world::UInt = Base.tls_world_age(), result = false, matched = false, mode = DAE, kwargs...)
  ci = _code_ad_by_type(tt; world, kwargs...)
  _result = structural_analysis!(ci, world)
  isa(_result, UncompilableIPOResult) && throw(_result.error)
  !matched && return result ? _result : _result.ir
  result = _result

  structure = make_structure_from_ipo(result)

  tstate = TransformationState(result, structure, copy(result.total_incidence))
  err = StateSelection.check_consistency(tstate, nothing)
  err !== nothing && throw(err)

  ret = top_level_state_selection!(tstate)
  isa(ret, UncompilableIPOResult) && throw(ret.error)

  (diff_key, init_key) = ret
  key = in(mode, (DAE, DAENoInit, ODE, ODENoInit)) ? diff_key : init_key

  # Removing `@invokelatest` segfaults in LLVM with "Unexpected instruction".
  var_eq_matching = @invokelatest matching_for_key(tstate, key)
  return StateSelection.MatchedSystemStructure(result, structure, var_eq_matching)
end

"""
    @code_structure ssrm()
    @code_structure world = UInt(1) [other_parameters...] ssrm()
    @code_structure result = true ssrm() # returns DAEIPOResult
    @code_structure matched = true ssrm() # returns MatchedSystemStructure

Return the IR after structural analysis of the passed function call.

A method instance corresponding to `ssrm()` is first extracted, then used
for type inference with the [`ADAnalyzer`](@ref). The inferred `CodeInstance`
then goes through structural analysis, and the resulting IR is returned.

Parameters:
- `world::UInt = Base.get_world_counter()`: the world in which to operate.
- `force_inline_all::Bool = false`: if `true`, make inlining heuristics choose to always inline where possible.
- `result::Bool = false`: if `true`, return the full [`DAEIPOResult`](@ref) instead of just the `IRCode`.
- `matched::Bool = false`: if `true`, return the [`MatchedSystemStructure`](@ref) after top-level state selection
  for visualization purposes.

!!! warning
    This will cache analysis results. You might want to invalidate with `DAECompiler.refresh()` between calls to `@code_structure`.
"""
macro code_structure(exs...)
  gen_call_with_extracted_types_and_kwargs(__module__, :code_structure, exs)
end

macro code_ad(exs...)
  gen_call_with_extracted_types_and_kwargs(__module__, :code_ad, exs)
end
