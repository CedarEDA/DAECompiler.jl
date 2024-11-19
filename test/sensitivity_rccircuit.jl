module sensitivity_rccircuit
using DAECompiler
using DAECompiler.Intrinsics;
using DAECompiler.Intrinsics: state_ddt
using OrdinaryDiffEqRosenbrock
using SciMLSensitivity
using Test
using Roots
using ChainRulesCore
using Diffractor
using FiniteDifferences

struct RCCircuit <: AbstractVector{Float64}
    r::Float64
    c::Float64
end
Base.size(::RCCircuit) = (2,)
Base.getindex(rc::RCCircuit, ii::Int) = getfield(rc, ii)

"""
    RCCircuit()

Basic RC Circuit with r=0.2 and C=0.5

```
  ┌────────────────────────────────────────┐
1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⢀⣀⣠⠤⠤⠤⠖⠒⠒⠒⠒⠊⠉⠉⠉⠉⠉⠉⠉│ u1(t): voltage across the capacitor
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⠀⠀⣀⡤⠖⠚⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣗⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀⠀⢱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⠀⠀⠀⠀⠀⠀⠀⣠⠏⠀⠀⠀⠀⠀⠀⠀⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⠀⠀⠀⠀⠀⠀⡴⠁⠀⠀⠀⠀⠀⠀⠀⠀⠈⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⠀⠀⠀⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⠀⠀⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢣⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⠀⠀⠀⡼⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⠀⠀⣰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⠀⢠⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⠀⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⢦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  │⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠒⠦⢄⣀⡀⠀⠀⠀⠀⠀│
0 │⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠑⠒⠒⠲│ u2(t): current through circuit
  └────────────────────────────────────────┘
  ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀t⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀0.5⠀
```
"""
RCCircuit() = RCCircuit(0.2, 0.5)
# Rises from 0 toi 0.8, in about 0.16 seconds, in about 0.6 seconds almost reaches 1.0

function (p::RCCircuit)()
    resistor_v = variable(:resistor_v)
    resistor_i = variable(:resistor_i)
    resistor_p_v = variable(:resistor_p_v)
    resistor_p_i = variable(:resistor_p_i)
    resistor_n_v = variable(:resistor_n_v)
    resistor_n_i = variable(:resistor_n_i)
    capacitor_v = variable(:capacitor_v)
    capacitor_i = variable(:capacitor_i)
    capacitor_p_v = variable(:capacitor_p_v)
    capacitor_p_i = variable(:capacitor_p_i)
    capacitor_n_v = variable(:capacitor_n_v)
    capacitor_n_i = variable(:capacitor_n_i)
    source_output_u = variable(:source_output_u)
    voltage_v = variable(:voltage_v)
    voltage_i = variable(:voltage_i)
    voltage_p_v = variable(:voltage_p_v)
    voltage_p_i = variable(:voltage_p_i)
    voltage_n_v = variable(:voltage_n_v)
    voltage_n_i = variable(:voltage_n_i)
    voltage_V_u = variable(:voltage_V_u)
    ground_g_v = variable(:ground_g_v)
    ground_g_i = variable(:ground_g_i)


    equation!(resistor_p_i + resistor_n_i)
    equation!(resistor_p_v - resistor_n_v - resistor_v)
    equation!(resistor_p_i - resistor_i)
    equation!(p.r*resistor_i - resistor_v)  # Ohms law on the Resistor
    equation!((capacitor_p_v - capacitor_n_v) - capacitor_v)
    equation!(capacitor_p_i + capacitor_n_i)
    equation!(capacitor_p_i - capacitor_i)
    equation!(capacitor_i/p.c - state_ddt(capacitor_v))  # Capacitor IV equatiom
    equation!(1 - source_output_u)  # Voltage is 1
    equation!((voltage_p_v - voltage_n_v) - voltage_v)
    equation!(voltage_p_i + voltage_n_i)
    equation!(voltage_p_i - voltage_i)
    equation!(voltage_V_u - voltage_v)
    equation!(ground_g_v)     # Ground is 0
    equation!(voltage_V_u - source_output_u)
    equation!(resistor_p_v - voltage_p_v)
    equation!(resistor_p_i + voltage_p_i)
    equation!(capacitor_p_v - resistor_n_v)
    equation!(capacitor_p_i + resistor_n_i)
    equation!(voltage_n_v - capacitor_n_v)
    equation!(ground_g_v - capacitor_n_v)
    equation!(ground_g_i + voltage_n_i + capacitor_n_i)  # KCL at ground
end

const tspan = (0.0, 5.0)

@testset "sensibility tests" begin
    sys = IRODESystem(Tuple{RCCircuit}, debug_config=Dict(:store_ir_levels=>true, :verify_ir_levels=>true))
    prob = ODEProblem(sys, nothing, tspan, RCCircuit())
    sol = solve(prob, Rodas5P(autodiff=false),  initializealg=CustomBrownFullBasicInit())

    # make sure the basic simulation worked
    v = sol[sys.capacitor_v]
    @test length(v)>=2
    @test -0.01<=v[1]<=0.01
    @test 0.99<=v[end]<=1.01
    @test all(>(-1e-10), diff(v))  # must be monotonically increasing (modulo noise)
end

@testset "batch_reconstruct frule" begin
    # This is just a unit test for batch_reconstruct frule
    r = 0.2; c = 0.5
    batch_reconstruct = DAECompiler.batch_reconstruct
    sys = IRODESystem(Tuple{RCCircuit}, debug_config=Dict(:store_ir_levels=>true, :verify_ir_levels=>true))
    sprob = ODEForwardSensitivityProblem(sys, nothing, tspan, RCCircuit(r, c))
    ssol = solve(sprob, Rodas5P(autodiff=false),  initializealg=CustomBrownFullBasicInit())

    function test_batch_reconstruct_frule(refs, ṙ, ċ, ts, ṫs)
        fdm = central_fdm(5, 1; max_range=1e-2)
        primal_expected = batch_reconstruct(ssol, refs, ts)
        ṡsol = Tangent{typeof(ssol)}(prob = Tangent{typeof(ssol.prob)}(p = [ṙ, ċ]));
        primal_actual, tangent = frule((NoTangent(), ṡsol, NoTangent(), ṫs), batch_reconstruct, ssol, refs, ts)

        @test primal_actual == primal_expected
        @test size(tangent) == size(primal_actual)

        fd_tangent = if iszero(ṫs)
            # varying ṙ and ċ
            function recon_at_rc(r, c)
                sprob_d = ODEForwardSensitivityProblem(sys, nothing, tspan, RCCircuit(r, c))
                ssol_d = solve(sprob_d, Rodas5P(autodiff=false),  initializealg=CustomBrownFullBasicInit())
                return batch_reconstruct(ssol_d, refs, ts)
            end
            jvp(fdm, recon_at_rc, (r, ṙ), (c, ċ))
        else
            @assert iszero(ṙ)
            @assert iszero(ċ)
            # not varying ṙ and ċ
            recon_at_t(t) = batch_reconstruct(ssol, refs, t)
            jvp(fdm, recon_at_t, (ts, ṫs))
        end

        @test tangent ≈ fd_tangent rtol=0.05
    end

    for refs in ([sys.capacitor_v], [sys.capacitor_v, sys.capacitor_i])
        test_batch_reconstruct_frule(refs, 0.0, 0.5, [0.16], ZeroTangent())
        test_batch_reconstruct_frule(refs, 1.5, 0.5, [0.16], ZeroTangent())
        test_batch_reconstruct_frule(refs, 0.0, 0.0, [0.16], [1.0])
        test_batch_reconstruct_frule(refs, 0.0, 0.0, [0.05, 0.10, 0.16], [1.0, 2.0, 3.0])
    end

    test_batch_reconstruct_frule([sys.capacitor_v], 0.1, 0.5, [0.16], ZeroTangent())
end

# Hack: Right now the Roots.Callable_Function this makes isn't supported by zero_tangent as it is recursive
# but that doesn't matter as actually it isn't allowed to carry any deriviative information
# due to https://github.com/JuliaMath/Roots.jl/issues/408
ChainRulesCore.zero_tangent(::Roots.Callable_Function) = ChainRulesCore.NoTangent()
@testset "Rise time derviative" begin
    # Main integration test, this is the purpose of this file
    # Roots.jl based solution, but this could be substituted out for anything else using implicit function theorem
    # Quoting Roots.jl: https://github.com/JuliaMath/Roots.jl/blob/ac16874b512a06ec4deed416c0911eef3fdf77c2/src/chain_rules.jl
    # > View find_zero as solving `f(x, p) = 0` for `xᵅ(p)`.
    # > This is implicitly defined. By the implicit function theorem, we have:
    # > ∇f = 0 => ∂/∂ₓ f(xᵅ, p) ⋅ ∂xᵅ/∂ₚ + ∂/∂ₚf(x\^α, p) ⋅ I = 0
    # > or ∂xᵅ/∂ₚ = - ∂/∂ₚ f(xᵅ, p)  / ∂/∂ₓ f(xᵅ, p)
    #
    # Eventually, once we're using CedarWaves, we will do this same implicit function theorem transformation on
    # functions such as `crosses()`, eliminating the need to AD our thresholding/search code, instead boiling
    # everything down to just:
    # -  ∂/∂ₚ f(xᵅ, p) (which is reconstruct_sensitivities)
    # - `∂/∂ₓ f(xᵅ, p)` (which is reconstruct_time_deriv)

    sys = IRODESystem(Tuple{RCCircuit}, debug_config=Dict(:store_ir_levels=>true, :verify_ir_levels=>true))
    sprob = ODEForwardSensitivityProblem(sys, nothing, tspan, RCCircuit())
    ssol = solve(sprob, Rodas5P(autodiff=false),  initializealg=CustomBrownFullBasicInit())
    ref = sys.capacitor_v

    function v_minus_v_hi(t, (ssol, ref))
        refs = @ignore_derivatives [ref]  #XXX: Diffractor shouldn't be trying to AD this anyway, but it is

        v = only(DAECompiler.batch_reconstruct(ssol, refs, [t]))
        return v-0.8  # we arbitarily select 0.8 as out v_hi threshold
    end;
    rise_time_primal = find_zero(v_minus_v_hi, tspan, (ssol, ref))

    # Set the signal since we want senstitivity to the  capacitance we do:
    ṙ = 0.0; ċ = 1.0;
    # if we wanted sensitivity to resistance  would do:  ṙ = 1.0; ċ = 0.0;
    # and we can loop over each in turn to get gradient
    ṡsol = Tangent{typeof(ssol)}(prob = Tangent{typeof(ssol.prob)}(p = [ṙ, ċ]));
    ṡsol_and_ṙef = Tangent{typeof((ssol, ref))}(ṡsol, NoTangent())
    # compute the rise time and the drise_time/dcapacitance sensitivity
    rise_time, sensitivity = frule(
        Diffractor.DiffractorRuleConfig(),
        (ZeroTangent(), ZeroTangent(), NoTangent(), ṡsol_and_ṙef),
        Roots.solve, ZeroProblem(v_minus_v_hi, tspan), Bisection(), (ssol, ref)
    )

    @test rise_time == rise_time_primal
    @test 0 < rise_time < 1
    @test sensitivity ≈ 2*rise_time rtol=1e-4  # computed by hand, property of the RC circuit
end

end  # module
