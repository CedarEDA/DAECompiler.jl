using DAECompiler: equation!, state_ddt, variables

struct Pendulum{T}
    length::T
    mass::T
end

# x, y, T, θ
function (p::Pendulum)()
    (; x, y, T, θ) = variables()
    λ = T / (p.length * p.mass)
    g = 10.
    equation!.((
        state_ddt(state_ddt(x)) + λ * x,
        state_ddt(state_ddt(y)) + λ * y + g,
        x - p.length * sin(θ),
        y - p.length * cos(θ)
    ))
end

struct FirstOrderPendulum{T}
    length::T
    mass::T
end

# x, dx/dt, y, dy/dt, T, θ
function (p::FirstOrderPendulum)()
    (; x, ẋ, y, ẏ, T, θ) = variables()
    λ = T / (p.length * p.mass)
    g = 10.
    equation!.((
        state_ddt(x) - ẋ,
        state_ddt(y) - ẏ,
        state_ddt(ẋ) + λ * x,
        state_ddt(ẏ) + λ * y + g,
        x - p.length * sin(θ),
        y - p.length * cos(θ)
    ))
end

# Another variant of the problem. DAECompiler encounters a singular
# Jacobian at x=0 or y=0 when solving this, so it doesn't yet work.
# Dynamic pivoting will be required to support this example correctly.

struct SingularJacobianPendulum{T}
    length::T
    mass::T
end

# x, y, T
function (p::SingularJacobianPendulum)()
    (; x, y, T) = variables()
    λ = T / (p.length * p.mass)
    g = 10.
    equation!.((
        state_ddt(state_ddt(x)) + λ * x,
        state_ddt(state_ddt(y)) + λ * y + g,
        x^2 + y^2 - p.length^2
    ))
end
