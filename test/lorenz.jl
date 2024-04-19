using DAECompiler: equation, equation!, state_ddt, variables, singularity_root!, time_periodic_singularity!, sim_time, observed!

struct Lorenz1{T}
    σ::T
    ρ::T
    β::T
end

# x, dx/dt, y, z, a, u
function (l::Lorenz1)()
    (; x, y, z, a, u) = variables()
    observed!(x + y, :obby)
    equation!.((
        u - (y - x) + 0*state_ddt(a), # test tearing and state selection
        a - (u - (y - x)), # test a == 0
        state_ddt(x) - (l.σ * u),
        state_ddt(y) - (x * (l.ρ - z) - y),
        state_ddt(z) - (x * y - l.β * z)
    ))
end

struct Lorenz1ts{T}
    σ::T
    ρ::T
    β::T
end

# x, dx/dt, y, z, a, u
@inline function (l::Lorenz1ts)()
    (; x, y, z, a, u) = variables()
    observed!(x + y, :obby)
    equation!.((
        u - (y - x) + 0*state_ddt(a), # test tearing and state selection
        a - (u - (y - x)), # test a == 0
        state_ddt(x) - (l.σ * u),
        state_ddt(y) - (x * (l.ρ - z) - y),
        state_ddt(z) - (x * y - l.β * z)
    ))
    # test discontinuities
    singularity_root!(50.0-sim_time())
end

struct Lorenz1cb{T}
    σ::T
    ρ::T
    β::T
end

# x, dx/dt, y, z, a, u
@inline function (l::Lorenz1cb)()
    (; x, y, z, a, u) = variables()
    observed!(x + y, :obby)
    equation!.((
        u - (y - x) + 0*state_ddt(a), # test tearing and state selection
        a - (u - (y - x)), # test a == 0
        state_ddt(x) - (l.σ * u),
        state_ddt(y) - (x * (l.ρ - z) - y),
        state_ddt(z) - (x * y - l.β * z)
    ))
    singularity_root!(x) # test callback
    singularity_root!(50.0-3*sim_time()) # test discontinuities

    # Test periodic callbacks
    time_periodic_singularity!(5.0, 10.0, 4)

    # Test infinite periods
    time_periodic_singularity!(13.37, Inf, -1)

    # Test that a count of `0` adds no tstops
    time_periodic_singularity!(2*13.37, 50.0, 0)
end

struct Lorenz1split{T}
    σ::T
    ρ::T
    β::T
end

# x, dx/dt, y, z, a, u
function (l::Lorenz1split)()
    (; x, y, z, a, u) = variables()
    observed!(x + y, :obby)
    let e1 = equation(),
        e2 = equation(),
        e3 = equation(),
        e4 = equation(),
        e5 = equation()

        xy = x-y
        e1(u); e1(xy);
        e2(a); e2(-u); e2(-xy);
        e3(state_ddt(x)); e3(-l.σ*u);
        e4(state_ddt(y)); e4(-x*(l.ρ-z)); e4(y);
        e5(state_ddt(z)); e5(-x * y); e5(l.β*z)
    end
end
