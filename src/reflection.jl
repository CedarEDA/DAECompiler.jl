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

function code_structure_by_type(@nospecialize(tt::Type); world::UInt = Base.tls_world_age(), result = false, kwargs...)
  ci = _code_ad_by_type(tt; world, kwargs...)
  # Perform or lookup DAECompiler specific analysis for this system.
  _result = structural_analysis!(ci, world)
  isa(_result, UncompilableIPOResult) && throw(_result.error)
  return result ? _result : _result.ir
end


"""
    @code_structure ssrm()
    @code_structure world = UInt(1) [other_parameters...] ssrm()

Return the IR after structural analysis of the passed function call.

A method instance corresponding to `ssrm()` is first extracted, then used
for type inference with the [`ADAnalyzer`](@ref). The inferred `CodeInstance`
then goes through structural analysis, and the resulting IR is returned.

Parameters:
- `world::UInt = Base.get_world_counter()`: the world in which to operate.
- `force_inline_all::Bool = false`: if `true`, make inlining heuristics choose to always inline where possible.

!!! warning
    This will cache analysis results. You might want to invalidate with `DAECompiler.refresh()` between calls to `@code_structure`.
"""
macro code_structure(exs...)
  gen_call_with_extracted_types_and_kwargs(__module__, :code_structure, exs)
end

macro code_ad(exs...)
  gen_call_with_extracted_types_and_kwargs(__module__, :code_ad, exs)
end
