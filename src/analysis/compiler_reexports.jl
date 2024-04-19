using Core.IR
using Core.Compiler: IRCode, Instruction, InstructionStream, IncrementalCompact,
    NewInstruction, DomTree, BBIdxIter, AnySSAValue, UseRef, UseRefIterator,
    block_for_inst, cfg_simplify!, is_known_call, argextype, getfield_tfunc, finish,
    singleton_type, widenconst, dominates_ssa, ⊑, userefs

# TODO: This really needs to go into a uniform compiler stdlib.
Base.iterate(compact::IncrementalCompact, state) = Core.Compiler.iterate(compact, state)
Base.iterate(compact::IncrementalCompact) = Core.Compiler.iterate(compact)
Base.iterate(abu::CC.AbsIntStackUnwind, state...) = CC.iterate(abu, state...)

Base.setindex!(compact::IncrementalCompact, @nospecialize(v), idx::SSAValue) = Core.Compiler.setindex!(compact,v,idx)
Base.setindex!(ir::IRCode, @nospecialize(v), idx::SSAValue) = Core.Compiler.setindex!(ir,v,idx)
Base.setindex!(inst::Instruction, @nospecialize(v), sym::Symbol) = Core.Compiler.setindex!(inst,v,sym)
Base.getindex(compact::IncrementalCompact, idx::AnySSAValue) = Core.Compiler.getindex(compact,idx)

Base.setindex!(urs::InstructionStream, @nospecialize args...) = Core.Compiler.setindex!(urs, args...)
Base.setindex!(ir::IRCode, @nospecialize args...) = Core.Compiler.setindex!(ir, args...)
Base.getindex(ir::IRCode, @nospecialize args...) = Core.Compiler.getindex(ir, args...)

Base.IteratorSize(::Type{CC.AbsIntStackUnwind}) = Base.SizeUnknown()

# TODO: Move this to Core.Compiler
CC.block_for_inst(ir::IRCode, s::SSAValue) = block_for_inst(ir, s.id)
function CC.dominates_ssa(ir::IRCode, domtree::DomTree, x::SSAValue, y::SSAValue; dominates_after=false)
    xb = block_for_inst(ir, x)
    yb = block_for_inst(ir, y)
    if xb == yb
        nstmts = length(ir.stmts)
        xid = x.id
        yid = y.id
        if xid > length(ir.stmts)
            xinfo = ir.new_nodes.info[xid - nstmts]
            xid = xinfo.pos
        end
        if yid > length(ir.stmts)
            yinfo = ir.new_nodes.info[yid - nstmts]
            yid = yinfo.pos
        end
        if xid == yid && (xinfo !== nothing || yinfo !== nothing)
            if xinfo !== nothing && yinfo !== nothing
                if xinfo.attach_after == yinfo.attach_after
                    return x.id < y.id
                end
                return yinfo.attach_after
            elseif xinfo !== nothing
                return !xinfo.attach_after
            else
                return (yinfo::NewNodeInfo).attach_after
            end
        end
        return dominates_after ? xid <= yid : xid < yid
    end
    return dominates(domtree, xb, yb)
end

function replace_argument!(ir::IRCode, idx::Int, argn::Argument, @nospecialize(x))
    urs = userefs(ir.stmts[idx][:inst])
    found = false
    for ur in urs
        if ur[] == argn
            ur[] = x
            found = true
        end
    end
    found || return
    ir.stmts[idx][:inst] = urs[]
end

function replace_argument!(compact::IncrementalCompact, idx::Int, argn::Argument, @nospecialize(x))
    ssa = SSAValue(idx)
    urs = userefs(compact[ssa][:inst])
    # Dumb use count hack
    compact[ssa] = nothing
    found = false
    for ur in urs
        if ur[] == argn
            ur[] = x
            found = true
        end
    end
    compact[ssa] = urs[]
end

Base.copy(phi::PhiNode) = Core.PhiNode(copy(phi.edges), copy(phi.values))
Base.push!(bs::CC.BitSet, i::Int) = CC.push!(bs, i)
Base.push!(bs::CC.BitSetBoundedMinPrioritySet, i::Int) = CC.push!(bs, i)