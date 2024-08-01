using StateSelection: DiffGraph

struct Gen
    id::Intrinsics.ScopeIdentity
    name::Symbol
end

const LevelKey = Union{Symbol, Gen, PartialScope}

"""
    struct NameLevel

Represents one level in the unified scope hierachy. DAECompiler tracks separate
name hierachies for each kind of primitive (variables, observed, equations,
epsilons), but we often want to operate on them jointly (e.g. for GenScope), so
it makes sense to represent them in a unified tree.

This struct represents one level in the tree. For each primitive kind
(var, obs, eq, eps), it stores the current system index for said primitive at
the current leaf. It also stores the children of this node.
"""
struct NameLevel
    var::Union{Nothing, Int}
    obs::Union{Nothing, Int}
    eq::Union{Nothing, Int}
    eps::Union{Nothing, Int}
    children::Union{Nothing, OrderedDict{LevelKey, NameLevel}}
end
NameLevel() =
    NameLevel(nothing, nothing, nothing, nothing, nothing)
NameLevel(children::OrderedDict{LevelKey, NameLevel}) =
    NameLevel(nothing, nothing, nothing, nothing, children)

struct UnsupportedIRException <: Exception
    msg::String
    ir::IRCode
end

struct TornCacheKey
    diff_states::BitSet
    alg_states::BitSet
    param_vars::BitSet
    var_schedule::Vector{Pair{BitSet, BitSet}}
end

struct TornIR
    # Ends with a return of a tuple that is assumed to be
    # first element of the closure environment for all subsequent IRs
    ir_sicm::Union{Nothing, IRCode}

    ir_seq::Vector{IRCode}
end

@enum VarEqKind begin
    Owned
    External
    CalleeInternal
end

@static if !Base.__has_internal_change(v"1.12-alpha", :methodspecialization)
    const MethodSpecialization = Core.MethodInstance
else
    import Core: MethodSpecialization
end

struct DAEIPOResult
    ir::IRCode
    extended_rt::Any
    argtypes
    nexternalvars::Int # total vars is length(var_to_diff)
    nsysmscopes::Int
    nexternaleqs::Int
    ncallees::Int
    nimplicitoutpairs::Int
    var_to_diff::DiffGraph
    var_kind::Vector{VarEqKind}
    total_incidence::Vector{Any}
    eq_kind::Vector{VarEqKind}
    eq_callee_mapping::Vector{Union{Nothing, Vector{Pair{SSAValue, Int}}}}
    names::OrderedDict{LevelKey, NameLevel}
    nobserved::Int
    neps::Int
    ic_nzc::Int
    vcc_nzc::Int
    # TODO: Chain these rather than copying them
    warnings::Vector{UnsupportedIRException}

    # TODO: Should this by a code instance
    tearing_cache::Dict{TornCacheKey, TornIR}

    # TODO: Should this be looked up via the regular code instance cache instead?
    sicm_cache::Dict{TornCacheKey, MethodSpecialization}
    dae_finish_cache::Dict{TornCacheKey, MethodSpecialization}
end

struct UncompilableIPOResult
    warnings::Vector{UnsupportedIRException}
    error::UnsupportedIRException
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

    structure = SystemStructure(complete(var_to_diff), complete(eq_to_diff), graph, solvable_graph)
end
