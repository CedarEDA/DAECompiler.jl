using SciMLBase, DiffEqBase

export DAECProblem
export ▫

const ▫ = Scope()

struct DAECProblem{F, I, G, T, K} <: SciMLBase.AbstractDAEProblem{Nothing, Nothing, T, true}
    f::F
    init::I
    guesses::G
    tspan::T
    kwargs::K
    # TODO: `f` and parameters are the same thing, and we derive du0 and u0 from
    # `init`, but DiffEqBase accesses this before hitting get_concrete_problem.
    # Can we adjust upstream do make this nicer?
    p::Missing
    u0::Nothing
    du0::Nothing
end

function DAECProblem(f, init, tspan = (0., 1.); guesses = nothing, kwargs...)
    DAECProblem(f, init, guesses, tspan, kwargs, missing, nothing, nothing)
end

function DiffEqBase.get_concrete_problem(prob::DAECProblem, isadaptive; kwargs...)
    (daef, differential_vars) = factory(Val(Settings(dae=true)), prob.f)

    u0 = zeros(length(differential_vars))
    du0 = zeros(length(differential_vars))

    for (which, val) in prob.init
        u0[which] = val
    end

    return DiffEqBase.get_concrete_problem(
        DAEProblem(daef, du0, u0, prob.tspan; differential_vars, prob.kwargs...),
        isadaptive)
end
