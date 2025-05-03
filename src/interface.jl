@enum GenerationMode begin
    DAE
    DAENoInit

    ODE
    ODENoInit

    # These are primarily for debug
    InitUncompress
end

struct Settings
    mode::GenerationMode
    force_inline_all::Bool
end
Settings(; mode::GenerationMode=DAE, force_inline_all::Bool=false) = Settings(mode, force_inline_all)

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

    # Select differential and algebraic states
    structure = make_structure_from_ipo(result)
    tstate = TransformationState(result, structure, copy(result.total_incidence))
    (diff_key, init_key) = top_level_state_selection!(tstate)

    if settings.mode in (DAE, DAENoInit, ODE, ODENoInit)
        tearing_schedule!(tstate, ci, diff_key, world)
    end
    if settings.mode in (InitUncompress, DAE, ODE)
        tearing_schedule!(tstate, ci, init_key, world)
    end

    # Generate the IR implementation of `factory`, returning the DAEFunction/ODEFunction
    if settings.mode in (DAE, DAENoInit)
        ir_factory = dae_factory_gen(tstate, ci, diff_key, world, settings.mode == DAE ? init_key : nothing)
    elseif settings.mode in (ODE, ODENoInit)
        ir_factory = ode_factory_gen(tstate, ci, diff_key, world, settings.mode == ODE ? init_key : nothing)
    elseif settings.mode == InitUncompress
        ir_factory = init_uncompress_gen(result, ci, init_key, diff_key, world)
    else
        return :(error("Unknown generation mode: $(settings.mode)"))
    end

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

For `ode`, return an `ODEFunction` suitable for use with ODEProblem.
The ODEFunction will be specific to the parameterization of `f`.

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
