"""
    compile_state_compression_func(tsys::TransformedIRODESystem, isdae::Bool)

Compiles a state compression function that maps from an uncompressed state vector
(e.g. one with every state) to a compressed state vector (e.g. only selected states).
At the moment, this is quite easy, we just throw away unselected states, and copy
selected states over.  In the future, there may be a more complex relationship
between our compressed states and the uncompressed states, so we retain the full
generality of a compression function with the signature:

    compress([du_compressed], u_compressed, u_uncompressed)

Where `u_uncompressed` is a vector of length equal to the total number of variables.
If this is for an ODE problem, `du_compressed` is not returned, only `u_compressed`.

The inverse of the compression function is the reconstruction function, built through
the usage of `compile_batched_reconstruct_func()`
"""
@breadcrumb "ir_levels" function compile_state_compression_func(tsys::TransformedIRODESystem, isdae::Bool)
    # Extract some information from our arguments
    (; var_assignment, var_num) = assign_vars_and_eqs(tsys, isdae)

    # We build our IR through the use of the `IncrementalCompact` API.
    ir_compress = IRCode()
    ir_compress[SSAValue(1)] = nothing

    # Input arguments are:
    #  - Output compressed `du` states (if a DAE)
    #  - Output compressed `u` states
    #  - Input uncompressed states (vector of length `num_variables`)
    push!(ir_compress.argtypes, Tuple{})
    if isdae
        push!(ir_compress.argtypes, AbstractVector{<:})
    end
    push!(ir_compress.argtypes, AbstractVector{<:})
    push!(ir_compress.argtypes, AbstractVector{<:})
    compact = IncrementalCompact(ir_compress)

    if isdae
        du_compressed, u_compressed, u_uncompressed = Argument.(2:4)
    else
        u_compressed, u_uncompressed = Argument(2:3)
    end

    # Initialize all our states with `NaN`
    if isdae
        insert_node_here!(
            compact,
            NewInstruction(Expr(:call, fill!, du_compressed, NaN), Any, Int32(1)),
        )
    end
    insert_node_here!(
        compact,
        NewInstruction(Expr(:call, fill!, u_compressed, NaN), Any, Int32(1)),
    )

    # For every variable, if it is a selected state (e.g. not `slot == 0`), insert the equivalent
    # of `u_compressed[slot] = u_uncompressed[v]`
    for (v, (slot, in_du)) in enumerate(var_assignment)
        slot == 0 && continue
        ref = insert_node_here!(
            compact,
            NewInstruction(Expr(:call, Base.getindex, u_uncompressed, v), Any, Int32(1)),
        )
        compressed_array = in_du ? du_compressed : u_compressed
        insert_node_here!(
            compact,
            NewInstruction(Expr(:call, Base.setindex!, compressed_array, ref, slot), Any, Int32(1)),
        )
    end

    # If we're a DAE, we return a tuple of `(du_compressed, u_compressed)`
    if isdae
        ret = insert_node_here!(
            compact,
            NewInstruction(Expr(:call, Core.tuple, du_compressed, u_compressed), Any, Int32(1)),
        )
    else
        ret = u_compressed
    end
    insert_node_here!(
        compact,
        NewInstruction(ReturnNode(ret), Nothing, Int32(1)),
    )

    ir_compress = finish(compact)
    # Give ourselves a proper control flow graph so optimizations work
    cfg = CC.compute_basic_blocks(ir_compress.stmts.stmt)
    append!(ir_compress.cfg.blocks, cfg.blocks)
    append!(ir_compress.cfg.index, cfg.index)

    DebugConfig(tsys).verify_ir_levels && check_for_daecompiler_intrinstics(ir_compress)

    goldclass_sig = if isdae
        Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    else
        Tuple{Vector{Float64}, Vector{Float64}}
    end
    return JITOpaqueClosure{:compress, goldclass_sig}() do arg_types...
        ir = copy(ir_compress)
        empty!(ir.argtypes)
        push!(ir.argtypes, Tuple{})
        append!(ir.argtypes, arg_types)

        # Just do a little bit of optimization so that it's properly inferred, etc...
        mi = get_toplevel_mi_from_ir(ir, get_sys(tsys))
        fallback_interp = getfield(get_sys(tsys), :fallback_interp)
        NewInterp = typeof(fallback_interp)
        opt_params = OptimizationParams(; compilesig_invokes=false,  preserve_local_sources=true)
        newinterp = NewInterp(fallback_interp; opt_params)

        infer_ir!(ir, newinterp, mi)
        return Core.OpaqueClosure(ir; do_compile=true)
    end
end