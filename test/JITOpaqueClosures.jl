module JITOpaqueClosures
using DAECompiler, Test
using DAECompiler: JITOpaqueClosure

@testset "basic use" begin
    my_typeof(x) = string(typeof(x))

    my_typeof2 = JITOpaqueClosure{:my_typeof2}() do arg_types...
        input_ir = first(only(Base.code_ircode(my_typeof, Tuple{Any})))
        ir = Core.Compiler.copy(input_ir)
        
        compact = Core.Compiler.IncrementalCompact(ir)
        # Force everything to be reinferred
        for ((_, idx), inst) in compact
            ssa = Core.SSAValue(idx)
            if Meta.isexpr(inst, :invoke)
                compact[ssa][:inst] = Expr(:call, inst.args[2:end]...)        
            end
            compact[ssa][:type] = Any
            compact[ssa][:flag] |= Core.Compiler.IR_FLAG_REFINED
        end

        ir = Core.Compiler.finish(compact)
        ir = Core.Compiler.compact!(ir)
        empty!(ir.argtypes)
        push!(ir.argtypes, Tuple{})
        append!(ir.argtypes, arg_types)
        
        interp = Core.Compiler.NativeInterpreter()
        mi = DAECompiler.get_toplevel_mi_from_ir(ir, @__MODULE__);
        DAECompiler.infer_ir!(ir, interp, mi)
        return Core.OpaqueClosure(ir; do_compile=true)
    end

    @test my_typeof2(Int64(1)) == "Int64"
    @test my_typeof2(Int64(1)) == "Int64"

    @test my_typeof2(1.5) == "Float64"
    @test my_typeof2(2.3) == "Float64"
end 

@testset "threading" begin
    # Note this is in no way a comprehensive test of thread safety

    builder_called = 0
    foo = JITOpaqueClosure{:threading_test}() do arg_types...
        builder_called+=1
        sleep(0.5)
        return identity  # Actually just returning a normal function for testing purposes
    end
    
    @sync for ii in 1:10
        Threads.@spawn foo(42)
    end
    @test builder_called == 1  # make sure only ran the builder function once 
end

@testset "goldclass" begin
    plus = JITOpaqueClosure{:plus, Tuple{Int,Int}}() do arg_types...
        return +  # just return a normal function for testing purposes
    end

    # Testing the internals to prove we did indeed goldclass this
    @test (@inferred plus(1,1)) == 2
    @test isempty(plus.cache)

    @test plus(1.0, 2.0) == 3.0
    @test length(plus.cache) == 1
end


@testset "right number of args checking" begin
    plus = JITOpaqueClosure{:plus, Tuple{Int,Int}}() do arg_types...
        return +  # just return a normal function for testing purposes
    end
    @test_throws MethodError plus(1)


    no_check_plus  = JITOpaqueClosure{:no_check_plus}() do arg_types...
        return +  # just return a normal function for testing purposes
    end
    @test no_check_plus(1) == 1
end

end
