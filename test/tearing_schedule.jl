module TearingSchedule

using DAECompiler, SciMLBase, Test
using DAECompiler.Intrinsics: variable, equation!, state_ddt
using Base.Meta

"extracts the tearing schedule ir from a dae defined by the function `f`"
function tearing_ir(f)
    debug_config=(; store_ir_levels=true, verify_ir_levels=true)
    sys = IRODESystem(Tuple{typeof(f)}; debug_config);
    tsys = TransformedIRODESystem(sys)

    (;state, var_eq_matching) = tsys
    ret = copy(state.ir)

    DAECompiler.dae_finish!(state, var_eq_matching, false; allow_unassigned=true)

    return ret
end

@testset "incidence on inserted *" begin
    ir = tearing_ir() do
        x = variable(:x)
        y = variable(:y)
        equation!(x^2-2y^2)
        equation!(x - 4y)
    end

    # Hunt down the mul_float and check it is acting right
    found=false
    for ii in 1:length(ir.stmts)
        stmt = ir[Core.SSAValue(ii)]
        inst = stmt[:inst]
        isexpr(inst, :call, 3) || continue
        inst.args[1] === (*) || continue
        inst.args[2] isa Float64 || continue
        inst.args[3] isa Core.SSAValue || continue
        found = true
        break  # found it, we are done
    end
    @test found
end

@testset "incidence on inserted * for alias with non-unitary coeff" begin
    ir = tearing_ir() do
        x = variable(:x)
        y = variable(:y)
        equation!(x-3state_ddt(y)^2)
        equation!(x + y)
    end

    # Hunt down the mul_float and check it is acting right
    found=false
    for ii in 1:length(ir.stmts)
        stmt = ir[Core.SSAValue(ii)]
        inst = stmt[:inst]
        isexpr(inst, :call, 3) || continue
        inst.args[1] === (*) || continue
        inst.args[3] isa Core.SSAValue || continue
        inst.args[2] isa Float64 || continue
        found = true

        coefficient = inst.args[2]
        var_incidence = ir[inst.args[3]][:type]
        break  # found it, we are done
    end
    @test found
end

@testset "incidence on inserted +" begin
    ir = tearing_ir() do
        x = variable(:x)
        y = variable(:y)
        z = variable(:z)
        equation!(x^2-2y^2)
        equation!(x - 4y-4z)
        equation!(x - 4z+1)
    end

    # Hunt down the add_float and check it is acting right
    found=false
    for ii in 1:length(ir.stmts)
        stmt = ir[Core.SSAValue(ii)]
        inst = stmt[:inst]
        isexpr(inst, :call, 3) || continue
        inst.args[1] === (+) || continue
        inst.args[2] isa Core.SSAValue || continue
        inst.args[3] isa Core.SSAValue || continue
        found = true
        break  # found it, we are done
    end
    @test found
end


@testset "incidence on inserted /" begin
    ir = tearing_ir() do
        x = variable(:x)
        y = variable(:y)
        z = variable(:z)
        equation!(x-2y+z)
        equation!(x - 4y)
        equation!(z - state_ddt(z))
    end

    # Hunt down the div_float and check it is acting right
    found=false
    for ii in 1:length(ir.stmts)
        stmt = ir[Core.SSAValue(ii)]
        inst = stmt[:inst]
        isexpr(inst, :call, 3) || continue
        inst.args[1] === (/) || continue
        inst.args[2] isa Core.SSAValue || continue
        inst.args[3] isa Float64 || continue
        found = true
        break  # found it, we are done
    end
    @test found
end

@testset "incidence on other inserted /" begin
    ir = tearing_ir() do
        x = variable(:x)
        y = variable(:y)
        equation!(x - sin(y))
        equation!(sin(x) - state_ddt(y))
    end

    # Hunt down the div_float and check it is acting right
    found=false
    for ii in 1:length(ir.stmts)
        stmt = ir[Core.SSAValue(ii)]
        inst = stmt[:inst]
        isexpr(inst, :call, 3) || continue
        inst.args[1] === (/) || continue
        inst.args[2] isa Core.SSAValue || continue
        inst.args[3] isa Float64 || continue
        found = true
        break  # found it, we are done
    end
    @test found
end

end # module TearingSchedule
