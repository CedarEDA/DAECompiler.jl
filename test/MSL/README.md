# MSL

These tests are ported from the [Modelling Toolkit Standard Library](https://github.com/SciML/ModelingToolkitStandardLibrary.jl/) (MSL), which are in turn ported from the Modelica Standard Library.

- `modelling_toolkit_helper.jl` adds the required overloads to take a Modelling Toolkit model and feed it to a DAECompiler system. Some of it reasonable, other parts, truely evil type-piracy and monkey patching.
 - `run_msl_tests.jl` loads that, then includes each file from the MSL test suite.

Note: we use a custom branch of MSL.
```
Remote (git): git@github.com:staticfloat/ModelingToolkitStandardLibrary.jl.git
Remote (https): https://github.com/staticfloat/ModelingToolkitStandardLibrary.jl.git
Branch: sf/daecompiler_compatible
```

It has only [a few minimal changes](https://github.com/SciML/ModelingToolkitStandardLibrary.jl/compare/main...staticfloat:ModelingToolkitStandardLibrary.jl:sf/daecompiler_compatible).
When modifying it please keep the changes well isolated per commit, we will need to rebase it occationally to get new tests.