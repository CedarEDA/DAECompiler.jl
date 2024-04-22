
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

struct DAEIPOResult
    ir::IRCode
    extended_rt::Any
    argtypes
    nexternalvars::Int # total vars is length(var_to_diff)
    nsysmscopes::Int
    nexternaleqs::Int
    var_to_diff::DiffGraph
    total_incidence::Vector{Any}
    eq_callee_mapping::Vector{Union{Nothing, Pair{SSAValue, Int}}}
    names::OrderedDict{LevelKey, NameLevel}
    nobserved::Int
    neps::Int
    ic_nzc::Int
    vcc_nzc::Int
    # TODO: Chain these rather than copying them
    warnings::Vector{UnsupportedIRException}
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

    structure = SystemStructure(complete(var_to_diff), complete(eq_to_diff), graph, solvable_graph, nothing, false)
end
