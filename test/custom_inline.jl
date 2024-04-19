module custom_inline

using Test
using DAECompiler
using DAECompiler.Intrinsics
using SciMLBase

function foo_variables()
    a = ntuple(x->variable(), 10)
    ntuple(i->equation!(a[i] - i), 10)
    sum(a)
end

bar1() = equation!(variable() - foo_variables()*sim_time())
bar2() = equation!(variable() - sin(sim_time()))

daeprob1 = DAEProblem(IRODESystem(Tuple{typeof(bar1)}), [], [], bar1, (0, 1e-5))
odeprob1 = ODEProblem(IRODESystem(Tuple{typeof(bar1)}), [], bar1, (0, 1e-5))
daeprob2 = DAEProblem(IRODESystem(Tuple{typeof(bar2)}), [], [], bar2, (0, 1e-5))
odeprob2 = ODEProblem(IRODESystem(Tuple{typeof(bar2)}), [], bar2, (0, 1e-5))

@test daeprob1.u0 == daeprob2.u0 == odeprob1.u0 == odeprob2.u0 == nothing

end # custom_inline
