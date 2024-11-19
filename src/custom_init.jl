using NonlinearSolve
import OrdinaryDiffEqCore: BrownFullBasicInit
export CustomBrownFullBasicInit

function CustomBrownFullBasicInit(;abstol=1e-10, nlsolve=RobustMultiNewton(autodiff=AutoFiniteDiff()))
    BrownFullBasicInit(abstol, nlsolve)
end
