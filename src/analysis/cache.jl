@enum VarEqClassification begin
    Owned
    External
    CalleeInternal
end

struct TornIR
    # Ends with a return of a tuple that is assumed to be
    # first element of the closure environment for all subsequent IRs
    ir_sicm::Union{Nothing, IRCode}

    ir_seq::Vector{IRCode}
end

@auto_hash_equals struct TornCacheKey
    # If nothing, includes init equations
    diff_states::Union{BitSet, Nothing}
    alg_states::BitSet
    param_vars::BitSet
    explicit_eqs::BitSet
    var_schedule::Vector{Pair{BitSet, BitSet}}
end

"""
    StructuralSSARef

Represents an SSA reference to the IR after structural analysis. Used as keys for callees, etc.
"""
struct StructuralSSARef
    id::Int
end

struct DAEIPOResult
    ir::IRCode
    opaque_eligible::Bool
    extended_rt::Any
    argtypes
    nexternalargvars::Int # total vars is length(var_to_diff)
    nsysmscopes::Int
    nexternaleqs::Int
    ncallees::Int
    var_to_diff::DiffGraph
    varclassification::Vector{VarEqClassification}
    total_incidence::Vector{Any}
    eqclassification::Vector{VarEqClassification}
    eq_callee_mapping::Vector{Union{Nothing, Vector{Pair{StructuralSSARef, Int}}}}
    names::OrderedDict{Any, ScopeDictEntry} # TODO: OrderedIdDict
    varkinds::Vector{Union{Intrinsics.VarKind, Nothing}}
    eqkinds::Vector{Union{Intrinsics.EqKind, Nothing}}
    # TODO: Chain these rather than copying them
    warnings::Vector{UnsupportedIRException}
end

struct UncompilableIPOResult
    warnings::Vector{UnsupportedIRException}
    error::UnsupportedIRException
end

function add_equation_row!(graph, solvable_graph, ieq::Int, inc::Incidence)
    for (v, coeff) in zip(rowvals(inc.row), nonzeros(inc.row))
        v == 1 && continue
        add_edge!(graph, ieq, v-1)
        isa(coeff, Float64) && add_edge!(solvable_graph, ieq, v-1)
    end
end
add_equation_row!(graph, solvable_graph, ieq::Int, c::Const) = nothing

using StateSelection: DiffGraph, BipartiteGraph

Base.@kwdef mutable struct DAESystemStructure <: StateSelection.SystemStructure
    # Maps the (index of) a variable to the (index of) the variable describing
    # its derivative.
    var_to_diff::DiffGraph
    eq_to_diff::DiffGraph
    graph::BipartiteGraph{Int, Nothing}
    solvable_graph::Union{BipartiteGraph{Int, Nothing}, Nothing}
end
StateSelection.n_concrete_eqs(structure::DAESystemStructure) = StateSelection.n_concrete_eqs(structure.graph)

function Base.copy(structure::DAESystemStructure)
    var_types = structure.var_types === nothing ? nothing : copy(structure.var_types)
    DAESystemStructure(copy(structure.var_to_diff), copy(structure.eq_to_diff),
        copy(structure.graph), copy(structure.solvable_graph),
        var_types, structure.only_discrete)
end

function make_structure_from_ipo(ipo::DAEIPOResult)
    neqs = length(ipo.total_incidence)
    nvars = length(ipo.var_to_diff)
    var_to_diff = copy(ipo.var_to_diff)
    eq_to_diff = DiffGraph(neqs)
    graph = BipartiteGraph(neqs, nvars)
    solvable_graph = BipartiteGraph(neqs, nvars)

    for (ieq, inc) in enumerate(ipo.total_incidence)
        add_equation_row!(graph, solvable_graph, ieq, inc)
    end

    structure = DAESystemStructure(StateSelection.complete(var_to_diff), StateSelection.complete(eq_to_diff), graph, solvable_graph)
end
