using ModelingToolkit
using ModelingToolkit: DiffGraph, BipartiteGraph
using ModelingToolkit: SystemStructure, MatchedSystemStructure, TransformationState
using ModelingToolkit.BipartiteGraphs
using ModelingToolkit: SparseMatrixCLIL
using ModelingToolkit: StructuralTransformations
using Core.Compiler: IRCode, SSAValue, NewSSAValue, MethodInstance
using TimerOutputs: @timeit, TimerOutput
using Tracy: @tracepoint
using Preferences: @load_preference

struct IRCodeRecord
    ir::IRCode
    name::String
end

struct SystemStructureRecord
    mss::MatchedSystemStructure
    name::String
end

# Vector containing the IR code each compiler pass generates as we make our way through DAECompiler.
# If `:store_ir_levels` is not in `debug_config, this only ever contains a single IRCodeRecord
# NOTE: Because we consider the IR transformations done in `dae_finish!` to be a separate step
# (because you can take a single `IRODESystem` and generate both ODE and DAE forms of that system)
# we do not save their IR code here, but in a separate `ir_levels` within the `TransformedIRODESystem`
const IRCodeRecords = Vector{IRCodeRecord}
# struct IRCodeRecords <: AbstractVector{IRCodeRecord}
#     records::Vector{IRCodeRecord}
# end
# IRCodeRecords() = IRCodeRecords(IRCodeRecord[])
# Base.push!(ir_levels::IRCodeRecords, record::IRCodeRecord) = push!(ir_levels.records, record)

# Vector containing the MatchedSystemStructure each MTK pass generates.
# If `:store_ss_levels` is not in `debug_config, this is empty.
const SystemStructureRecords = Vector{SystemStructureRecord}
# struct SystemStructureRecords <: AbstractVector{SystemStructureRecord}
#     records::Vector{SystemStructureRecord}
# end
# SystemStructureRecords() = SystemStructureRecords(SystemStructureRecord[])
# Base.push!(ss_levels::SystemStructureRecords, record::SystemStructureRecord) = push!(ss_levels.records, record)

if @load_preference("enable_timer", false)::Bool
    const Timings = TimerOutput
    macro may_timeit(debug_config, name, body)
        body_with_tracy = Expr(:macrocall, Symbol("@tracepoint"), __source__, name, esc(body))
        Expr(:macrocall, Symbol("@timeit"), __source__, esc(:($debug_config.timings)), name, body_with_tracy)
    end
else
    const Timings = Nothing
    macro may_timeit(debug_config, name, body)
        esc(body)
    end
end

"""
    DebugConfig

A struct to configure various debugging behaviors for the compilation process.
Current flags include:
- `ir_log::String`: a directory to save stored IR to (in plain text format)
- `store_ir_levels::Bool`: whether to store the IR at every step of the compilation pipeline.
- `verify_ir_levels::Bool`: whether to verify the IR at every step of the compilation pipelin
"""
struct DebugConfig
    system_name::String
    timings::Timings
    ir_log::Union{Nothing,String}
    ir_levels::Union{Nothing,IRCodeRecords}
    ss_levels::Union{Nothing,SystemStructureRecords}
    replay_log::Union{Nothing,Dict{String,Vector{Tuple}}}
    verify_ir_levels::Bool
end

const NO_DEBUG = DebugConfig("", Timings(), nothing, nothing, nothing, nothing, false)

function DebugConfig(debug_config::DebugConfig=NO_DEBUG;
    system_name::String=debug_config.system_name,
    timings::Timings=debug_config.timings,
    ir_log::Union{Nothing,String}=debug_config.ir_log,
    ir_levels::Union{Nothing,IRCodeRecords}=debug_config.ir_levels,
    ss_levels::Union{Nothing,SystemStructureRecords}=debug_config.ss_levels,
    replay_log::Union{Nothing,Dict{String,Vector{Tuple}}}=debug_config.replay_log,
    verify_ir_levels::Bool=debug_config.verify_ir_levels)
    return DebugConfig(
        system_name,
        timings,
        ir_log,
        ir_levels,
        ss_levels,
        replay_log,
        verify_ir_levels)
end

# the general main constructor
function DebugConfig(debug_config, tt::DataType)
    debug_config isa DebugConfig && return debug_config
    system_name = get(debug_config, :system_name) do
        string(only(tt.parameters).name.name)
    end
    timings = get(Timings, debug_config, :timings)
    ir_log = get(debug_config, :ir_log, nothing)
    if get(debug_config, :store_ir_levels, false)
        ir_levels = IRCodeRecords()
    else
        ir_levels = get(debug_config, :ir_levels, nothing)
    end
    if get(debug_config, :store_ss_levels, false)
        ss_levels = SystemStructureRecords()
    else
        ss_levels = get(debug_config, :ss_levels, nothing)
    end
    if get(debug_config, :replay_log, false)
        replay_log = Dict{String,Vector{Tuple}}()
    else
        replay_log = get(debug_config, :replay_log, nothing)
    end
    verify_ir_levels = get(debug_config, :verify_ir_levels, false)
    return DebugConfig(system_name, timings, ir_log, ir_levels, ss_levels, replay_log, verify_ir_levels)
end

"""
    sys::IRODESystem
    IRODESystem(tt::Type{<:Tuple{Any}}; method_table, debug_config) -> IRODESystem
    IRODESystem(entry_func; kwargs...) -> IRODESystem

The DAE system.
The `IRODESystem(tt::Type{<:Tuple{Base.Callable}})` constructor first carries out (ordinary)
type inference using `DAEInterpreter` on the call graph that can be reached from the entry
nullary call represented by `tt`. Following this, it performs structural analysis on the
optimized IR code of the entry call. The result of the structural analysis are stored into
`IRODESystem` and is used when the system is transformed into different ODE/DAE problems.
"""
struct IRODESystem <: ModelingToolkit.AbstractODESystem
    interp::DAEInterpreter
    mi::MethodInstance
    fallback_interp::AbstractInterpreter
    result::DAEIPOResult
    debug_config::DebugConfig

    function IRODESystem(tt::Type{<:Tuple{Any}};
                         fallback_interp::AbstractInterpreter = Core.Compiler.NativeInterpreter(),
                         debug_config = (;),
                         ipo_analysis_mode = false)
        debug_config = DebugConfig(debug_config, tt)
        @may_timeit debug_config "typeinf_dae" interp, frame = typeinf_dae(tt; ipo_analysis_mode)
        mi = frame.linfo
        @may_timeit debug_config "compute_structure" result = compute_structure(interp, mi, debug_config)
        for warning in result.warnings
            @warn warning.msg ir = warning.ir
        end
        if isa(result, UncompilableIPOResult)
            throw(result.error)
        end
        return new(interp, mi, fallback_interp, result, debug_config)
    end
end
IRODESystem(entry_func; kwargs...) = IRODESystem(Tuple{Core.Typeof(entry_func)}; kwargs...)

"""
    state::IRTransformationState

The intermediate state for the MTK transformations for [`IRODESystem`](@ref).
"""
mutable struct IRTransformationState <: TransformationState{IRODESystem}
    ir::IRCode
    callback_func::Function
    structure::SystemStructure
    const var_obs_names::VarObsNames
    const eq_names::EqNames
    const eps_names::EpsNames
    const nobserved::Int
    const neps::Int
    const ic_nzc::Int
    const vcc_nzc::Int
    const sys::IRODESystem

    function IRTransformationState(sys::IRODESystem)
        structure = StructuralAnalysisResult(sys)
        return new(
            copy(structure.ir),
            ()->CallbackSet(),
            make_structure_from_ipo(structure),
            copy(structure.var_obs_names),
            copy(structure.eq_names),
            copy(structure.eps_names),
            structure.nobserved,
            structure.neps,
            structure.ic_nzc,
            structure.vcc_nzc,
            sys)
    end
end

StructuralAnalysisResult(sys::IRODESystem) = getfield(sys, :result)
DebugConfig(sys::IRODESystem) = getfield(sys, :debug_config)

function add_equation_row!(graph, solvable_graph, ieq::Int, inc::Incidence)
    for (v, coeff) in zip(rowvals(inc.row), nonzeros(inc.row))
        v == 1 && continue
        add_edge!(graph, ieq, v-1)
        coeff !== nonlinear && add_edge!(solvable_graph, ieq, v-1)
    end
end

DebugConfig(obj) = DebugConfig(get_sys(obj))


"""
    record_ir_leaf_name(name)

Get the "leaf name" for an `ir_level` breadcrumb trail.  This has two sources
of complexity:

* First, the trail could be empty, as is the case in the "unoptimized" IR level.
  In this case, we want the leaf name to just be the given `name`.
* Second, the name could be empty, as is the case for our "final" IR levels for
  a function.  In this case, we want to omit the final `.`, returning only the
  joined breadcrumb trail.
"""
function get_record_ir_leaf_name(name::String)
    name_stack = [string(bc.name) for bc in breadcrumb_trail("ir_levels")]
    if isempty(name_stack)
        return name
    else
        if !isempty(name)
            push!(name_stack, name)
        end
        return join(name_stack, ".")
    end
end

"""
    record_ir!(debug_config::DebugConfig, name::String, ir::IRCode, verification_lattice::AbstractLattice)

Given a chunk of IR in `ir`, record it into the vector `ir_levels` under
the given `name`, but only if `store_ir_levels` is set to `true`.  If
`verify_ir_levels` is `true`, use `verification_lattice` to verify the given
IR, potentially throwing an exception if the verification fails.
"""
function record_ir!(debug_config::DebugConfig,
                    name::String,
                    ir::IRCode,
                    verification_lattice::Core.Compiler.AbstractLattice =
                        CC.PartialsLattice(DAELattice()))
    # If the user provides `""` as the name, we just use the stack as the name.
    fulllevelname = get_record_ir_leaf_name(name)
    if isempty(fulllevelname)
        @warn("Invalid empty ir_levels name!  Did you forget a @breadcrumb macro?")
    end

    (; ir_levels, ir_log) = debug_config
    if ir_levels !== nothing
        record = IRCodeRecord(copy(ir), fulllevelname)
        push!(ir_levels, record)
    end

    # Save IR to file, if requested
    if ir_log !== nothing
        mkpath(ir_log)
        levelname = string(length(ir_levels); pad=2)
        systemname = debug_config.system_name
        filename = "$(levelname).$(systemname).$(fulllevelname).ir"
        open(joinpath(ir_log, filename), "w") do io
            show(io, ir)
        end
    end

    # Always verify the IR if we've been asked to.  This occurs even if
    # `store_ir_levels` is set to `false`.  We throw an
    # `UnsupportedIRException` with `ir_levels` bundled in if this fails.
    if debug_config.verify_ir_levels
        try
            Core.Compiler.verify_ir(ir, false, false, verification_lattice)
        catch e
            throw(UnsupportedIRException("IR verification failure", ir))
        end
    end
end

function record_ir!(state::IRTransformationState, name::String, ir::IRCode)
    debug_config = DebugConfig(state)
    verification_lattice = typeinf_lattice(getfield(get_sys(state), :interp))
    @may_timeit debug_config "record_ir!" record_ir!(debug_config, name, ir, verification_lattice)
end

record_ir!(::Nothing, name::String, ir::IRCode) = nothing

"""
    record_mss!(debug_config::DebugConfig,
                name::String,
                mss::Union{SystemStructure, MatchedSystemStructure})

Given a `MatchedSystemStructure` or `SystemStructure` in `mss`, record it into the vector `ss_levels`
under the given `name`.
"""
function record_mss!(debug_config::DebugConfig,
                     name::String,
                     mss::Union{SystemStructure, MatchedSystemStructure})
    if mss isa SystemStructure
        mss = MatchedSystemStructure(mss, Matching(max(nsrcs(mss.graph), ndsts(mss.graph))))
    end
    ss_levels = debug_config.ss_levels
    if ss_levels !== nothing
        push!(ss_levels, SystemStructureRecord(deepcopy(mss), name))
    end
end
function record_mss!(state::IRTransformationState,
                     name::String,
                     mss::Union{SystemStructure, MatchedSystemStructure})
    debug_config = DebugConfig(state)
    @may_timeit debug_config "record_mss!" record_mss!(debug_config, name, mss)
end

Base.show(io::IO, ::MIME"text/plain", sys::IRODESystem) =
    print(io, "IRODESystem for ", getfield(sys, :mi).specTypes)
Base.show(io::IO, sys::IRODESystem) =
    print(io, "IRODESystem for ", getfield(sys, :mi).specTypes)
function Base.show(io::IO, ::MIME"text/plain", state::IRTransformationState)
    print(io, "IRTransformationState for ")
    show(io, MIME"text/plain"(), get_sys(state))
end

function ModelingToolkit.linear_subsys_adjmat!(irs::IRTransformationState)
    graph = irs.structure.graph
    eadj = Vector{Int}[]
    cadj = Vector{Int}[]
    linear_equations = Vector{Int}()
    for (i, inc) in enumerate(getfield(irs.sys, :result).total_incidence)
        isa(inc, Const) && continue
        any(x->x === nonlinear || !isinteger(x), nonzeros(inc.row)) && continue
        (isa(inc.typ, Const) && iszero(inc.typ.val)) || continue
        # Skip any equations involving `t` for now
        # TODO: We may want to adjust our ILS to include equations with `t` in them.
        (1 ∉ rowvals(inc.row)) || continue
        this_eadj = Vector{Int}()
        this_cadj = Vector{Int}()
        for (var, coeff) in zip(rowvals(inc.row), nonzeros(inc.row))
            push!(this_eadj, var - 1)
            push!(this_cadj, coeff)
        end
        p = sortperm(this_eadj)
        push!(eadj, this_eadj[p])
        push!(cadj, this_cadj[p])
        push!(linear_equations, i)
    end
    return SparseMatrixCLIL(nsrcs(graph),
                            ndsts(graph),
                            linear_equations, eadj, cadj)
end

function print_linear_incidence(irs::IRTransformationState)
    for (i, inc) in enumerate(getfield(irs.sys, :result).total_incidence)
        isa(inc, Const) && continue
        any(x->x === nonlinear || !isinteger(x), nonzeros(inc.row)) && continue
        (isa(inc.typ, Const) && iszero(inc.typ.val)) || continue
        # Skip any equations involving `t` for now
        # TODO: We may want to adjust our ILS to include equations with `t` in them.
        (1 ∉ rowvals(inc.row)) || continue
        println(i, " => ", inc)
    end
end

function ModelingToolkit.StructuralTransformations.find_solvables!(state::IRTransformationState; kwargs...)
    # Nothing to be done for now, we did this at system init.
    return nothing
end

# TODO: This should go into ModelingToolkit
function structure_check!(structure::SystemStructure)
    unique_assign = zeros(Int, ndsts(structure.graph))
    for eq in 1:nsrcs(structure.graph)
        ns = 𝑠neighbors(structure.graph, eq)
        if length(ns) == 1
            var = first(ns)
            if unique_assign[var] != 0
                @warn "Equation $eq uniquely determines $var, but so does equation $(unique_assign[var])"
            end
            unique_assign[var] = eq
        end
    end
end
