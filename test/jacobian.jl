module Jacobian

using DAECompiler, SciMLBase, Test
using DAECompiler.Intrinsics: variables, variable, equation!, state_ddt, sim_time
using FiniteDiff
using Random
using SparseArrays
Random.seed!(42)

"""
    finitediff_jacobian(f!; isdae)

Uses finite differencing to generate an appropriate jacobian generating function.
For `isdae` this is `jac!(J, du, u, p, γ, t)`
For ODE (!isdae) this is `jac!(J, u, p, t)`
`finish_result` should be the output of `dae_finish`
"""
function finitediff_jacobian(f!; isdae)
    if isdae
        function (J, du, u, p, γ, t)
            J2 = copy(J)
            #=dG/du =# FiniteDiff.finite_difference_jacobian!(J, (_out, _u)->f!(_out, du, _u, p, t), u)
            #=dG/ddu=# FiniteDiff.finite_difference_jacobian!(J2, (_out, _du)->f!(_out, _du, u, p, t), du)
            J .+= γ*J2
            return J
        end
    else  # ODE
        function (J, u, p, t)
            FiniteDiff.finite_difference_jacobian!(J, (_du, _u)->f!(_du, _u, p, t), u)
            return J
        end
    end
end

function jac_testing_helpers(f; isdae)
    debug_config = (; store_ir_levels=true, verify_ir_levels=true)
    sys = IRODESystem(Tuple{typeof(f)}; debug_config);
    tsys = TransformedIRODESystem(sys)
    (;state, var_eq_matching) = tsys
    (;F!, neqs) = DAECompiler.dae_finish!(state, var_eq_matching, isdae; allow_unassigned=true)
    return (;
        jac! = DAECompiler.construct_jacobian(sys; isdae),
        fin_jac! = finitediff_jacobian(F!; isdae),
        neqs
    )
end

"""
function to check jacobians from `construct_jacobian` against finite differencing.
"""
function check_jac(f, name="")
    @testset "$name" begin
        p = Base.issingletontype(typeof(f)) ? nothing : f
        @testset "ODE" begin
            (;jac!, fin_jac!, neqs) = jac_testing_helpers(f; isdae=false)
            J = -zeros(neqs,neqs)
            u = randn(neqs)
            t = 10rand()
            J = jac!(J, u, p, t)
            J_fin = fin_jac!(-0.0 .* J, u, p, t)
            @test J ≈ J_fin rtol=1e-6
        end
        # DAE
        @testset "DAE" begin
            (;jac!, fin_jac!, neqs) = jac_testing_helpers(f; isdae=true)
            J = -zeros(neqs,neqs)
            u = randn(neqs)
            du = randn(neqs)
            γ = 3.141592  #randn()
            t = 10rand()
            J = jac!(J, u, du, p, γ, t)
            J_fin = fin_jac!(-0.0 .* J, u, du, p, γ, t)
            @test J ≈ J_fin rtol=1e-6
        end
    end
end


@testset "basic functionality" begin
    # make sure it doesn't break if it doesn't `return nothing`
    # Earlier versions broke depending on if with `nothing`ed out last expression or not
    # Unrealistic Artificial case that always hits this, explictly returning a integer
    function qux()
        (; x, y) = variables()
        equation!(state_ddt(x) - y + x)
        equation!(x+y^2)
        return 5
    end
    @testset "ODE" begin
        (;jac!, fin_jac!, neqs) = jac_testing_helpers(qux; isdae=false)
        @assert neqs==2

        J = fill(NaN, 2,2);
        jac!(J, [10.0,100.0], nothing, 0.0);  # mutates J
        @test !all(isnan, J)  # just make sure it did something rather than erroring.

        J_fin = fin_jac!(NaN .* J,  [10.0,100.0], nothing, 0.0)
        @test J_fin ≈ J rtol=1e-6
    end

    @testset "DAE" begin
        (;jac!, fin_jac!, neqs) = jac_testing_helpers(qux; isdae=true)
        @assert neqs==2

        J = fill(NaN, 2,2);
        @test J==jac!(J, [10.0, 100.0],[10.0, 100.0], nothing, 0.0, 0.0)

        # test a range of γ
        @test isapprox(
            jac!(copy(J), [10.0, 100.0],[10.0, 100.0], nothing, 0.0, 0.0),
            fin_jac!(J, [10.0, 100.0],[10.0, 100.0], nothing, 0.0, 0.0),
        )
        @test isapprox(
            jac!(copy(J), [10.0, 100.0],[10.0, 100.0], nothing, 0.5, 0.0),
            fin_jac!(J, [10.0, 100.0],[10.0, 100.0], nothing, 0.5, 0.0),
        )
        @test isapprox(
            jac!(copy(J), [10.0, 100.0],[10.0, 100.0], nothing, 1.0, 0.0),
            fin_jac!(J, [10.0, 100.0],[10.0, 100.0], nothing, 1.0, 0.0),
        )
        @test isapprox(
            jac!(copy(J), [10.0, 100.0],[10.0, 100.0], nothing, -8.5, 0.0),
            fin_jac!(J, [10.0, 100.0],[10.0, 100.0], nothing, -8.5, 0.0),
        )
    end
end

@testset "Sparse Jacobian type" begin
    function bar()
        (; x, y, z) = variables()
        equation!(y^2 - sim_time())
        equation!(x+y^2)
        equation!(x+z^2)
    end

    sys = IRODESystem(Tuple{typeof(bar)});
    @testset "ODE" begin
        J = spzeros(3,3)
        jac! = DAECompiler.construct_jacobian(sys; isdae=false)
        J_out = jac!(J, [1.0, 2.0, 3.0], nothing, 1.5)
        @test J_out === J
        # This DAE has some sparsity so should have structural zeros in output
        @test nnz(J) < length(J)
    end

    @testset "DAE" begin
        J = spzeros(3,3)
        jac! = DAECompiler.construct_jacobian(sys; isdae=true)
        J_out = jac!(J, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], nothing, 3.1415, 1.5)
        @test J_out === J
        # This DAE has some sparsity so should have structural zeros in output
        @test nnz(J) < length(J)
    end
end



@testset "parameterized struct" begin
    struct DemoParamStruct
        α::Float64
        β::Float64
    end
    function (self::DemoParamStruct)()
        (; x, y) = variables()
        equation!(self.α*x + y^3)
        equation!(state_ddt(x) - self.β*y + 5x)
    end
    check_jac(DemoParamStruct(4,3))
end

@testset "big list" begin
    # basic test fully linear
    check_jac("basic linear") do
        (; x, y) = variables()
        equation!(4x+y^3)
        equation!(state_ddt(x) - 3y + 5x)
    end

    # basic test with some nonlinear components
    check_jac("basic nonlinear") do
        (; x, y) = variables()
        equation!(state_ddt(x) - y + x)
        equation!(x+y^2)
    end

    # Has no is_solved_variables (as all things are prime powers so can't be expressed in terms of each other)
    check_jac("prime powers") do
        (; x, y, z) = variables()
        equation!(4x^2+y^3)
        equation!(2y^5 + x^7)
        equation!(z^9 - x^13)
    end

    check_jac("cross-state multiplication") do
        (; x, y) = variables()
        a = 2x
        equation!(state_ddt(x) - y + x)
        equation!(a*y - x)
    end

    # Used to trigger https://github.com/JuliaComputing/DAECompiler.jl/issues/278
    # Has the property that z=y=dx/dt=0 and x can be anything nonzero and is constant
    # In DAE form this has a state_ddt that is in u (due to aliasing)
    check_jac("zero all but one const") do
        (; x, y, z) = variables()
        equation!(y-z)
        equation!(state_ddt(x) - z)
        equation!(y^3/x)
    end

    # Simplifies down to just 1 state variable
    # triggers both "failed to map from derivative to variable", and
    # still got right answer though.
    # hits !SelectedState check
    check_jac("Massive simplify") do
        (; x, y, z) = variables()
        equation!(state_ddt(y) - z + x)
        equation!(state_ddt(x) - z)
        equation!(y + x)
    end

    # One that Pantelides has done nontrivial things to:
    # for DAE has an implicit equation
    check_jac("Pantelides + imp eq") do
        (; x, y, z) = variables()
        equation!(y^2-z)
        equation!(state_ddt(y) - state_ddt(x))
        equation!(y^3-3z-x)
    end

    # hits Pantelides
    # https://github.com/JuliaComputing/DAECompiler.jl/issues/273
    check_jac("Pantelides") do
        (; x, y, z) = variables()
        equation!(y-z)
        equation!(z + state_ddt(y))
        equation!(y^3-3z-x)
    end

    # Engages Pantelides and can be solved normally if we don't use `construct_jacobian`
    # used to breaks with construct_jacobian because nesting diffractor seems to screw up some things.
    # for DAE has an implicit equation
    # Plus it needs to fix from https://github.com/JuliaDiff/Diffractor.jl/pull/162 or it will segfault
    check_jac("Pantelides + imp eq + branch") do
        (; x, y, z) = variables()
        if y > 0.
            a = 0.
        else
            a = z
        end
        equation!(x - 3.5 * a)
        equation!(state_ddt(z))
        equation!(state_ddt(x) - state_ddt(y))
    end

    # output is unteathered from base variables due to only having equations in terms of derivatives
    check_jac("untethered") do
        (;x, y) = variables()
        dx = state_ddt(x)
        dy = state_ddt(y)
        equation!(dx^2+dy^3)
        equation!(3dy + 7dx)
    end
    # Has implicit equations due to both variable and its derivative being selected
    #
    check_jac("imp untethered O(1)+O(2)") do
        (;x, y) = variables()
        dx = state_ddt(x)
        dy = state_ddt(y)
        ddy = state_ddt(dy)
        equation!(4dx+dy^3)
        equation!(ddy - 3dy + 5dx)
    end

    # higher order with levels skipped
    check_jac("imp O(1)+O(3)") do
        (;x, y) = variables()
        dx = state_ddt(x)
        dy = state_ddt(y)
        ddy = state_ddt(dy)
        ddx = state_ddt(dx)
        dddx = state_ddt(ddx)
        equation!(dx^2+dy^3)
        equation!(3dy + 9dddx + 5dx)
    end


    # This case has intrinstics that are generated during tearing. Which runs into:
    # https://github.com/JuliaComputing/DAECompiler.jl/issues/401
    check_jac("Generated intrinstics during tearing") do
        (; x, T, θ) = variables()

        equation!(state_ddt(state_ddt(x)) + T * x)
        equation!(x - sin(θ))
        equation!(cos(θ))
    end
end

@testset "zero! helper" begin
    @testset "Array" begin
        x = [1.0 2.0 3.0; 4.0 5.0 6.0; 0.1 0.0 -0.1]
        @test DAECompiler.zero!(x) === x
        @test all(iszero, x)
    end

    @testset "SparseArray" begin
        # must preserve sparsity pattern
        sp_x = sparse([1, 2, 3, 1], [2, 2, 2, 3], [0.4, 0.5, 0.1, 0.7], 3, 3)
        @test DAECompiler.zero!(sp_x) === sp_x
        @test findnz(sp_x) == ([1, 2, 3, 1], [2, 2, 2, 3], [0.0, 0.0, 0.0, 0.0])
    end
end

end # module Jacobian
