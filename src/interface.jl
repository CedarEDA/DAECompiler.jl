struct Settings
    dae::Bool
    force_inline_all::Bool
end
Settings(; dae::Bool=true, force_inline_all::Bool=false) = Settings(dae, force_inline_all)

using StateSelection: Unassigned, SelectedState, unassigned
function top_level_state_selection(result)
    @show result.nexternalvars

    # For the top-level problem, all external vars are state-invariant, and we do no other fissioning
    param_vars = BitSet(1:result.nexternalvars)

    structure = make_structure_from_ipo(result)
    StateSelection.complete!(structure)

    varwhitelist = StateSelection.computed_highest_diff_variables(structure)

    for param in param_vars
        varwhitelist[param] = false
    end

    # Max match is the (unique) tearing result given the choice of states
    var_eq_matching = StateSelection.complete(StateSelection.maximal_matching(structure.graph, Union{Unassigned, SelectedState};
        dstfilter = var->varwhitelist[var]))

    var_eq_matching = StateSelection.partial_state_selection_graph!(structure, var_eq_matching)

    diff_vars = BitSet()
    alg_vars = BitSet()

    for (v, match) in enumerate(var_eq_matching)
        v in param_vars && continue
        if match === SelectedState()
            push!(diff_vars, v)
        elseif match === unassigned
            push!(alg_vars, v)
        end
    end

    key = TornCacheKey(diff_vars, alg_vars, param_vars, Vector{Pair{BitSet, BitSet}}())
end

"""
    factory_gen(world, source, _gen, settings, f)

This is the compile-time entry point for DAECompiler code generation. It drives all other pieces.
"""
function factory_gen(world::UInt, source::Method, @nospecialize(_gen), settings, @nospecialize(fT))
    settings = settings.parameters[1]

    # First, perform ordinary type inference, under the assumption that we may need to AD
    # parts of the function later.
    ci = ad_typeinf(world, Tuple{fT}; force_inline_all=settings.force_inline_all)

    # Perform or lookup DAECompiler specific analysis for this system.
    result = structural_analysis!(ci, world)

    if isa(result, UncompilableIPOResult)
        return Base.generated_body_to_codeinfo(
            Expr(:lambda, Any[:var"#self", :settings, :f], Expr(:block, Expr(:return, Expr(:call, throw, result.error)))),
            @__MODULE__, false)
    end

    # TODO: Pantelides here

    # Select differential and algebraic states
    key = top_level_state_selection(result)

    tearing_schedule!(result, ci, key, world)

    # Generate the IR implementation of `factory`, returning the DAEFunction
    @assert settings.dae
    ir_factory = dae_factory_gen(result, ci, key, world)
    src = ir_to_src(ir_factory)

    src.ssavaluetypes = length(src.code)
    src.min_world = @atomic ci.min_world
    src.max_world = @atomic ci.max_world
    src.edges = Core.svec(ci.def)

    return src
end

"""
    factory(::Val{::Settings}, f)

Given Julia function `f` compatible with DAECompiler's model representation, return a representation
suitable for solving using numerical integrators. The particular representation depends on the passed
settings.

For `dae`, return a `DAEFunction` suitable for use with DAEProblem.
The DAEFunction will be specific to the parameterization of `f`.

To obtain a new parameterization, re-run this function. The runtime complexity of this function is
at most equivalent to one ordinary evaluation of `f`, but this function may have significant
compile-time cost (cached as usual for Julia code).
"""
function factory end
factory(f) = factory(Val(Settings()), f)

function refresh()
    @eval function factory(settings::Val #= {::Settings} =#, f)
        $(Expr(:meta, :generated_only))
        $(Expr(:meta, :generated, factory_gen))
    end
end
refresh()
