"""
    VarObsNames

Names for variables and observables. Leaves are `Pair{isob::Bool, var_num::Int}`,
with the `isob` indicating observables.
"""
const VarObsNames = Dict{Union{Symbol, PartialScope},Any}

"""
    EqNames

Names for equations.
"""
const EqNames = Dict{Union{Symbol, PartialScope}, Any}

"""
    EqNames

Names for epsilons.
"""
const EpsNames = Dict{Union{Symbol, PartialScope}, Any}

struct UnsupportedIRException <: Exception
    msg::String
    ir::IRCode
end

struct DAEIPOResult
    ir::IRCode
    argtypes
    nexternalvars::Int
    ntotalvars::Int
    nsysmscopes::Int
    var_to_diff::DiffGraph
    ret
    total_incidence::Vector{Any}
    eq_callee_mapping::Vector{Union{Nothing, Pair{SSAValue, Int}}}
    var_callee_mapping::Vector{Union{Nothing, Pair{SSAValue, Int}}}
    var_obs_names::VarObsNames
    eq_names::EqNames
    eps_names::EpsNames
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
