# DAECompiler

<a href="https://help.juliahub.com/daecompiler/dev/"><img src='https://img.shields.io/badge/docs-dev-blue.svg'/></a>

> [!WARNING]
> This package is not currently intended for (standalone) public consumption.

DAECompiler is a compiler library for the (application domain-agnostic) transformation, optimization and
high-performance simulation of (potentially large) Differential Algebraic Equations (DAEs).

DAECompiler is the core compiler engine of the Cedar EDA platform, as well as JuliaHub'd Dyad multi-domain modeling and simulation tool.

DAECompiler takes a description of a system of DAEs written in a restricted subset of Julia and performs suitable transformations,
including state selection and generation of jacobian evaluation, discontinuity callbacks, etc. to enable high-performance simulation
using state-of-the art ODE and DAE solvers.

This package is domain agnostic and does not provide a high level modeling interface.
Rather, it is intended as the execution backend for such a modeling package.
It also does not contain any differential equation solvers - these are provided by [SciML](https://sciml.ai/).

## License / Contributing

DAECompiler is triple licensed under commercial licenses, CERN OHL-S v2 and the Dyad source available license. See LICENSE for further information.

We are currently not accepting external pull requests on this repository.

# Getting started

DAECompiler is an internal dependency of CedarEDA/CedarSim. You should in general not need to use it directly. For development, you can set up the [Cedar public registry](https://github.com/CedarEDA/PublicRegistry/) and use the regular julia package development workflow.

## Simple example

DAECompiler's input is a single julia function (which may itself call a restricted subset of other julia functions) that evenutally calls
any of a small number of DAECompiler intrinsics to define a DAE system.
These intrinsics are defined in `DAECompiler.Intrinsics`. The most relevant
intrinsics are:

- *continuous()*: Introduces a new Float64 variable and return it
- *ddt(x)*: Take the derivative of the expression `x` with respect to time
- *always!(x)*: Declare the expression `x` as an equation of the DAE system. The ODE solver will attempt to find values for all `continuous`s that (at every time point) ensure that the expression `x` becomes 0.

Please see the documentation for a full list of intrinsics (e.g. to declare discontinuities) and helper functions (e.g. the `variables()` helper, which introduces several variables) at once.

The function provided to DAECompiler may also be the call overload of a julia struct (passing a function is the degenerate case of passing a struct with no fields, whose call over load is the body of the function), in which case the fields of the struct are taken as the parameterization of the system. This parameterization can then be changed without requiring re-compilation.

> [!WARNING]
> The subset of julia that is supported by DAECompiler is currently not specified, but is implementation defined by what CedarSim requires. Generally, you can expect arithmetic, small struct manipulation, etc to work. Do not use mutation and do not attempt side effects inside the DAECompiler system specification.
