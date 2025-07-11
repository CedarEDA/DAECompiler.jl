using SciMLBase, DiffEqBase

export DAECProblem, ODECProblem
export ▫

const ▫ = Scope()

struct DAECProblem{F, I, G, T, K} <: SciMLBase.AbstractDAEProblem{Nothing, Nothing, T, true}
    f::F
    init::I
    guesses::G
    tspan::T
    kwargs::K
    settings::Settings
    # TODO: `f` and parameters are the same thing, and we derive du0 and u0 from
    # `init`, but DiffEqBase accesses this before hitting get_concrete_problem.
    # Can we adjust upstream do make this nicer?
    p::Missing
    u0::Nothing
    du0::Nothing
end

function DAECProblem(f, init::Union{Vector, Tuple{Vararg{Pair}}}, tspan::Tuple{Real, Real} = (0., 1.);
                     guesses = nothing,
                     force_inline_all=false,
                     insert_stmt_debuginfo=false,
                     insert_ssa_debuginfo=false,
                     skip_optimizations=false,
                     kwargs...)
    settings = Settings(; mode = DAENoInit, force_inline_all, insert_stmt_debuginfo, insert_ssa_debuginfo, skip_optimizations)
    DAECProblem(f, init, guesses, tspan, kwargs, settings, missing, nothing, nothing)
end

function DAECProblem(f, tspan::Tuple{Real, Real} = (0., 1.);
                     guesses = nothing,
                     force_inline_all=false,
                     insert_stmt_debuginfo=false,
                     insert_ssa_debuginfo=false,
                     skip_optimizations=false,
                     kwargs...)
    settings = Settings(; mode = DAE, force_inline_all, insert_stmt_debuginfo, insert_ssa_debuginfo, skip_optimizations)
    DAECProblem(f, nothing, guesses, tspan, kwargs, settings, missing, nothing, nothing)
end

function DiffEqBase.get_concrete_problem(prob::DAECProblem, isadaptive; kwargs...)
    (daef, differential_vars) = factory(Val(prob.settings), prob.f)

    u0 = zeros(length(differential_vars))
    du0 = zeros(length(differential_vars))

    if prob.init !== nothing
        for (which, val) in prob.init
            u0[which] = val
        end
    end

    return DiffEqBase.get_concrete_problem(
        DAEProblem(daef, du0, u0, prob.tspan; differential_vars, prob.kwargs...),
        isadaptive)
end

struct ODECProblem{F, I, G, T, K} <: SciMLBase.AbstractODEProblem{Nothing, T, true}
    f::F
    init::I
    guesses::G
    tspan::T
    kwargs::K
    settings::Settings
    # TODO: `f` and parameters are the same thing, and we derive u0 from
    # `init`, but DiffEqBase accesses this before hitting get_concrete_problem.
    # Can we adjust upstream do make this nicer?
    p::Missing
    u0::Nothing
end

function ODECProblem(f, init::Union{Vector, Tuple{Vararg{Pair}}}, tspan::Tuple{Real, Real} = (0., 1.);
                     guesses = nothing,
                     force_inline_all=false,
                     insert_stmt_debuginfo=false,
                     insert_ssa_debuginfo=false,
                     skip_optimizations=false,
                     kwargs...)
    settings = Settings(; mode = ODENoInit, force_inline_all, insert_stmt_debuginfo, insert_ssa_debuginfo, skip_optimizations)
    ODECProblem(f, init, guesses, tspan, kwargs, settings, missing, nothing)
end

function ODECProblem(f, tspan::Tuple{Real, Real} = (0., 1.);
                     guesses = nothing,
                     force_inline_all=false,
                     insert_stmt_debuginfo=false,
                     insert_ssa_debuginfo=false,
                     skip_optimizations=false,
                     kwargs...)
    settings = Settings(; mode = ODE, force_inline_all, insert_stmt_debuginfo, insert_ssa_debuginfo, skip_optimizations)
    ODECProblem(f, nothing, guesses, tspan, kwargs, settings, missing, nothing)
end

function DiffEqBase.get_concrete_problem(prob::ODECProblem, isadaptive; kwargs...)
    (odef, n) = factory(Val(prob.settings), prob.f)

    u0 = zeros(n)

    if prob.init !== nothing
        for (which, val) in prob.init
            u0[which] = val
        end
    end

    return DiffEqBase.get_concrete_problem(
        ODEProblem(odef, u0, prob.tspan; prob.kwargs...),
        isadaptive)
end
