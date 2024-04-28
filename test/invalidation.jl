module invalidation

using Test

using DAECompiler, StateSelection, SciMLBase, OrdinaryDiffEq, Sundials
using DAECompiler: equation!, state_ddt, variables

const results1 = Number[]
const results2 = Number[]

struct Lorenz1
    σ::Float64
    ρ::Float64
    β::Float64
end

x::Number ⊖ y::Number = x ⊟ y
x::Number ⊟ y::Number = begin # the target of invalidation
    result = x - y
    push!(results1, result)
    return result
end

# x, dx/dt, y, z, a, u
function (l::Lorenz1)()
    (; x, y, z, a, u) = variables()
    equation!.((
        u ⊖ (y ⊖ x),       # test tearing
        a ⊖ (u ⊖ (y ⊖ x)), # test a == 0
        state_ddt(x) - (l.σ * u),
        state_ddt(y) - (x * (l.ρ - z) - y),
        state_ddt(z) - (x * y - l.β * z)
    ))
end

function solve_lorenz1()
    x = Lorenz1(10.0, 28.0, 8.0/3.0)
    u0 = [1.0, 0.0, 0.0]
    tspan = (0.0, 100.0)
    sys = IRODESystem(Tuple{typeof(x)});
    interp = getfield(sys, :interp)
    daeprob = DAEProblem(sys, zero(u0), u0, tspan, x);
    daesol = solve(daeprob, IDA())
    return daesol, interp
end

empty!(results1); empty!(results2);
res1, interp = solve_lorenz1();
@test !isempty(results1) && isempty(results2)

function isfunc(name::Symbol)
    return function (mi::Core.MethodInstance)
        def = mi.def
        isa(def, Method) || return false
        return def.module === (@__MODULE__) && def.name === name
    end
end

# check if there are cached analysis results
let cache = Core.Compiler.code_cache(interp)
    mi1 = Core.Compiler.specialize_method(
        only(methods(⊖)), Tuple{typeof(⊖),Float64,Float64}, Core.svec())
    @test Core.Compiler.haskey(cache, mi1)
    @test Core.Compiler.getindex(cache, mi1).max_world == typemax(UInt)
    mi2 = Core.Compiler.specialize_method(
        only(methods(⊟)), Tuple{typeof(⊟),Float64,Float64}, Core.svec())
    @test Core.Compiler.haskey(cache, mi2)
    @test Core.Compiler.getindex(cache, mi2).max_world == typemax(UInt)
end

x::Number ⊟ y::Number = begin # now redefine it
    result = x + y
    push!(results2, result)
    return result
end

# check if the cached analysis results are invalidated
let cache = Core.Compiler.code_cache(interp)
    mi1 = Core.Compiler.specialize_method(
        only(methods(⊖)), Tuple{typeof(⊖),Float64,Float64}, Core.svec())
    @test Core.Compiler.haskey(cache, mi1)
    @test Core.Compiler.getindex(cache, mi1).max_world != typemax(UInt)
    mi2 = Core.Compiler.specialize_method(
        last(sort(methods(⊟); by=x->x.primary_world)), Tuple{typeof(⊟),Float64,Float64}, Core.svec())
    @test !Core.Compiler.haskey(cache, mi2)
end

empty!(results1); empty!(results2);
res2, = solve_lorenz1();
@test isempty(results1) && !isempty(results2)

@test res1.u ≠ res2.u

end # module invalidation
