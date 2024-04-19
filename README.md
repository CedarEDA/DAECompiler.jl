# DAECompiler

<a href="https://help.juliahub.com/daecompiler/dev/"><img src='https://img.shields.io/badge/docs-dev-blue.svg'/></a>

> [!WARNING]
> The public release of Cedar is ongoing. You are welcome to look around, but things will be unstable and various pieces may be missing. If you are not feeling adventurous, please check back in a few weeks.

DAECompiler is the core compiler engine of the Cedar EDA platform.
It provides a domain-agnostic framework for high-performance simulation
of Differential Algebraic Equations (DAEs).

DAECompiler takes a description of a system of DAEs written in a restricted subset of Julia and performs suitable transformations,
including state selection and generation of jacobian evaluation, discontinuity callbacks, etc. to enable high-performance simulation
using state-of-the art ODE and DAE solvers.

This package is domain agnostic, any code related to analog circuit simulation can be found in [CedarSim](https://github.com/CedarEDA/CedarSim.jl) and related packages. It also does not contain any differential
equation solvers - these are provided by [SciML](https://sciml.ai/).

### DAECompiler vs ModelingToolkit, etc.

There's a few ways to think of DAECompiler:

- As an optimizing compiler for DAE simulation (as described above)
- As ModelingToolkit, but using an embedded Julia DSL rather than tracing
- As JuliaSimCompiler, but using the Core.Compiler IR rather than a custom IR

DAECompiler shares some code with ModelingToolkit and JuliaSimCompiler. In particular, MTK's structural analysis passes are shared by all three packages.

It is possible to use DAECompiler (currently via some type-piracy, see [`test/MSL/modeling_toolkit_helper.jl`](test/MSL/modeling_toolkit_helper.jl)) you can even use DAECompiler as the backend for the modelling toolkit DSL.

The primary design objective of this package is to handle large models (primarily of MOSFET) found in the EDA space.

## License / Contributing

The Cedar EDA platform is dual-licensed under a commercial license and CERN-OHL-S v2. Please see the LICENSE file for more
information and the LICENSE.FAQ.md file for more information on how to
use Cedar under the CERN-OHL-S v2 license.

We are accepting PRs on all Cedar repositories, although you must sign the Cedar Contributor License Agreement (CLA) for us to be able to incorporate your changes into the upstream repository. Additionally, if you would like to make major or architectural changes, please discuss this with us *before* doing the work. Cedar is a complicated piece of software, with many moving pieces, not all of which are publicly available. Because of this, we may not be able to take your changes, even if they are correct and useful (so again, please talk to us first).

# Getting started

DAECompiler is an internal dependency of CedarEDA/CedarSim. You should in general not need to use it directly. For development, you can set up the [Cedar public registry](https://github.com/CedarEDA/PublicRegistry/) and use the regular julia package development workflow.

## Simple example

DAECompiler's input is a single julia function (which may itself call a restricted subset of other julia functions) that evenutally calls
any of a small number of DAECompiler intrinsics to define a DAE system.
These intrinsics are defined in `DAECompiler.Intrinsics`. The most relevant
intrinsics are:

- *variable()*: Introduces a new Float64 variable and return it
- *ddt(x)*: Take the derivative of the expression `x` with respect to time
- *equation!(x)*: Declare the expression `x` as an equation of the DAE system. The ODE solver will attempt to find values for all `variable`s that (at every time point) ensure that the expression `x` becomes 0.

Please see the documentation for a full list of intrinsics (e.g. to declare discontinuities) and helper functions (e.g. the `variables()` helper, which introduces several variables) at once.

The function provided to DAECompiler may also be the call overload of a julia struct (passing a function is the degenerate case of passing a struct with no fields, whose call over load is the body of the function), in which case the fields of the struct are taken as the parameterization of the system. This parameterization can then be changed without requiring re-compilation.

> [!WARNING]
> The subset of julia that is supported by DAECompiler is currently not specified, but is implementation defined by what CedarSim requires. Generally, you can expect arithmetic, small struct manipulation, etc to work. Do not use mutation and do not attempt side effects inside the DAECompiler system specification.

Here is a simple getting started example. The system being simulated is the well-known [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system).

```julia
using DAECompiler, SciMLBase
using DAECompiler.Intrinsics

struct Lorenz{T}
    σ::T
    ρ::T
    β::T
end

# x, dx/dt, y, z
function (l::Lorenz)()
    (; x, y, z) = variables()
    equation!.((
        ddt(x) - (l.σ * u),
        ddt(y) - (x * (l.ρ - z) - y),
        ddt(z) - (x * y - l.β * z)
    ))
end

u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 100.0)
x = Lorenz(10.0, 28.0, 8.0/3.0)
sys = IRODESystem(Tuple{typeof(x)});
daeprob = DAEProblem(sys, zero(u0), u0, tspan, x);

using Test, OrdinaryDiffEq
daesol = solve(daeprob, DFBDF(autodiff=false));
@test all(x->abs(x) < 100, daesol)
```

## Dev Docs

> [!NOTICE]
> These will be moved to the proper place shortly.

### Compilation pipeline

This diagram showcases a rough dataflow diagram of DAECompiler's compilation pipeline, in particular it shows the flow of the IR as it is processed by the various subsystems within DAECompiler, and their eventual use in the solving of the numerical systems.
```mermaid
flowchart TD
    input>"DAE Definition function \n(uses DAECompiler.Intrinsics)"]
    struct["Structural Transforms"]
    tearing
    prepare["prepare for differentiation \n(run optimizer)"]
    F!
    paramjac("paramjac \n(ODE only)")
    jac(jac)
    tgrad("tgrad \n(ODE only)")
    reconstruct("reconstruct \n(observed! and variables)")
    reconstruct_der("reconstruct der \n(derivatives of reconstruct wrt states)")
    reconstruct_time_der("reconstruct time der \n(derivatives of reconstruct wrt time)")
    callbacks("callbacks \n(singularity_root!)")
    jac
    jac_proto("jacobian prototype \n(determing a sparse matrix\n with jac structure)")

    input--> struct --> tearing
    tearing ---> F!
    tearing ---> callbacks
    tearing ---> reconstruct
    tearing ---> reconstruct_der
    tearing ---> reconstruct_time_der
    tearing ---> jac_proto
    tearing --> prepare
    prepare --> jac
    prepare --> paramjac
    prepare --> tgrad
```


### Derivatives

DAECompiler includes code for finding the derivatives of many parts DAEs/ODEs defined using it.
Here for reference is a table showing what functions define the derivatives of various parts with respect to other parts

|                              | **f (RHS)** | **sim_time** | **selected states** | **variables & observations** | **parameters**                               |
|------------------------------|-------------|--------------|---------------------|------------------------------|----------------------------------------------|
| **f (RHS)**                  | 1           | tgrad        | jac                 |                              | paramjac                                     |
| **sim_time**                 |             | 1            | [`sol(t, Val{1})`](https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/#Interpolations-and-Calculating-Derivatives) | reconstruct\_time\_deriv  | 0.0                                          |
| **selected states**          |             |              | 1                   | reconstruct\_der             | SciMLSensitivity extract\_local\_sensitivity |
| **variables & observations** |             |              |                     | 1                            | reconstruct\_der reconstruct\_sensitivities  |
| **parameters**               |             |              |                     |                              | 1                                            |

### Caches
We cache various things at various times to avoid repeating work.
A large portion of these are various generated code such as different specializions of the derivative functions.
This means that sometimes after modifying code rerunning it will still return the old result.
To help manage this we use [CentralizedCaches.jl](https://github.com/JuliaComputing/CentralizedCaches.jl).
They are declared with `@new_cache` and can all be cleared by using `clear_all_caches!(DAECompiler)`.
