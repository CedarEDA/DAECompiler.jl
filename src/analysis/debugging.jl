using ModelingToolkit
using ModelingToolkit.BipartiteGraphs

function build_var_names(names::OrderedDict{LevelKey, NameLevel}, var_to_diff)
    var_names = OrderedDict{Int,String}()
    build_var_names!(var_names, names, var_to_diff)
    return var_names
end
function build_var_names!(var_names, names::OrderedDict{LevelKey, NameLevel}, var_to_diff, prefix=String[])
    for name in keys(names)
        name_path = join(vcat(prefix..., name), ".")
        level = names[name]
        if level.children !== nothing
            build_var_names!(var_names, level.children, var_to_diff, [name_path])
        end
        if level.var !== nothing
            var_idx = level.var
            var_names[var_idx] = name_path
            while var_to_diff[var_idx] !== nothing
                var_idx = var_to_diff[var_idx]
                name_path = "D($(name_path))"
                var_names[var_idx] = name_path
            end
        end
    end
end

num_selected_states(prob) = num_selected_states(get_transformed_sys(prob), isa(prob, DAEProblem))
function num_selected_states(tsys::TransformedIRODESystem, isdae)
    mss = ModelingToolkit.MatchedSystemStructure(tsys.state.structure, tsys.var_eq_matching)
    var_assignment, = assign_vars_and_eqs(mss, isdae)
    return length(filter(((state_idx, b),) -> b == 0 && state_idx >= 1, var_assignment))
end

function show_assignments(prob::SciMLBase.AbstractDEProblem)
    sys = get_transformed_sys(prob)
    var_names = build_var_names(sys.state.names, sys.state.structure.var_to_diff)

    @info("Variables:")
    for var_idx in 1:ndsts(sys.state.structure.graph)
        name = get(var_names, var_idx, "<no var name>")
        println("  [$var_idx] $(name)")
    end

    mss = ModelingToolkit.MatchedSystemStructure(sys.state.structure, sys.var_eq_matching)
    @info("Matched system structure:")
    display(mss)

    @info("States mappings:")
    var_assignment, = assign_vars_and_eqs(mss, isa(prob, DAEProblem))
    state_mapping = Dict(state_idx => var_idx for (var_idx, (state_idx, b)) in enumerate(var_assignment) if b == 0)
    for state_idx in 1:(prob.u0 === nothing ? 0 : length(prob.u0))
        var_idx = get(state_mapping, state_idx, 0)
        if var_idx == 0
            println("  u[$state_idx] <no state mapping>")
        else
            name = get(var_names, var_idx, "<no var name>")
            println("  u[$state_idx] $(name)")
        end
    end
    if isa(prob, DAEProblem)
        dstate_mapping = Dict(state_idx => var_idx for (var_idx, (state_idx, b)) in enumerate(var_assignment) if b == 1)
        for dstate_idx in 1:(prob.du0 === nothing ? 0 : length(prob.du0))
            var_idx = get(dstate_mapping, dstate_idx, 0)
            if !prob.differential_vars[dstate_idx]
                println(" du[$dstate_idx] <unused>")
            else
                if var_idx == 0
                    println(" du[$dstate_idx] <no state mapping>")
                else
                    name = get(var_names, var_idx, "<no var name>")
                    println(" du[$dstate_idx] $(name)")
                end
            end
        end
    end
end
show_assignments(sol::SciMLBase.AbstractODESolution) = show_assignments(sol.prob)


function extract_equation(state::IRTransformationState, idx::Integer)
    ir = copy(state.ir)
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:inst]
        if is_known_invoke(stmt, equation!, ir)
            eq_num = idnum(ir.stmts[i][:type])
            if eq_num != idx
                ir.stmts[i] = nothing
            end
        end
    end
    for i = 1:200
        ir = compact!(ir)
        ir = cfg_simplify!(ir)
        ir = peephole_pass!(ir)
    end
    ir
end

const IR_LEVEL_DESCRIPTIONS = (
    "unoptimized" =>
        "Completely unoptimized IR",
    "compute_structure" =>
        "After we have computed the structure of the problem, " *
        "initial mapping of equations to variables, etc...",
    "tearing_schedule!.post_diffractor" =>
        "If we need to differentiate any equations, we run " *
        "diffractor on the equations.  This stores the IR " *
        "after Diffractor has run, but before we run inference again.",
    "tearing_schedule!.pre_tearing" =>
        "After compute_structure (and after diffractor and " *
        "its optimizations, if it was run) but before tearing.",
    "tearing_schedule!" =>
        "Once state selection is done, we need to re-arrange " *
        "our equations according to the tearing schedule.",
    "post_dae_finish.dae" =>
        "Final lowering into something a solver can run (DAE form).",
    "post_dae_finish.ode" =>
        "Final lowering into something a solver can run (ODE form).",
)

const level_descriptions = join((" - `:$k`: $v\n" for (k,v) in IR_LEVEL_DESCRIPTIONS))

"""
    summarize_ir_levels(prob::DAEProblem)

Returns a summary of the IR levels.

The IR code for those levels can be viewed by accessing the appropriately keyed value from
`prob.f.sys.state.ir_levels`.
The keys correpond to the IR levels.
E.g.  `prob.f.sys.state.ir_levels[:unoptimized]`, or `prob.f.sys.state.ir_levels[:post_compute_structure]`.

### Selected level descriptions:

$(level_descriptions)
"""
function summarize_ir_levels(prob::Union{DAEProblem,ODEProblem})
    # First, get the IR levels for the underlying `IRTransformationState`, which
    # contains all passes that occur before DAE or ODE system specialization.
    ir_levels = DebugConfig(prob.f.sys).ir_levels
    if length(ir_levels) <= 2
        @warn("Very few IR levels tracked, did you run with the :store_ir_levels debug flag?")
    end

    println("$(string(typeof(prob).name.name)) with $(length(ir_levels)) IR levels:")

    max_num_statements = maximum(length(level.ir.stmts) for level in ir_levels)
    stmt_width = ceil(Int, log10(max_num_statements))

    for level in ir_levels
        num_stmts = length(level.ir.stmts)
        padded_num_stmts = string(" "^(stmt_width - ceil(Int, log10(num_stmts))), num_stmts)
        println("  - [$(padded_num_stmts)] $(level.name)")

        # Some IR levels are associated with equations and variables;
        # summarize those as well if we've got them:
        if level.name ∈ ("compute_structure", "tearing_schedule!")
            if level.name == "compute_structure"
                # Get pre-pantelides system structure
                graph = getfield(get_sys(prob), :unopt_state).structure.graph
            elseif level.name == "tearing_schedule!"
                # Get post-pantelides system structure
                graph = prob.f.sys.state.structure.graph
            end
            num_eqs = length(graph.fadjlist)
            num_vars = isa(graph.badjlist, Int) ? graph.badjlist : length(graph.badjlist)
            println("       -> With system structure with $(num_eqs) equations, $(num_vars) variables, $(graph.ne) connections")
        end
    end
end

using REPL: AbstractTerminal, REPL
using REPL.TerminalMenus: TerminalMenus

function Cthulhu.descend(prob::SciMLBase.AbstractDEProblem, args...; kwargs...)
    tsys = get_transformed_sys(prob)
    Cthulhu.descend(tsys, args...; kwargs...)
end
function Cthulhu.descend(sys::TransformedIRODESystem, ir::IRCode;
                         optimize::Bool=true, annotate_source::Bool=false, kwargs...)
    interp = getfield(get_sys(sys), :interp)
    mi = get_toplevel_mi_from_ir(ir, get_sys(sys))
    override = Cthulhu.SemiConcreteCallInfo(Cthulhu.MICallInfo(mi, Nothing, CC.Effects()), ir)
    # TODO: Make Cthulhu take in a `config` object into `descend()`
    dead_code_elimination = Cthulhu.CONFIG.dead_code_elimination
    try
        Cthulhu.CONFIG.dead_code_elimination = false
        Cthulhu.descend(interp, mi;
            override, # TODO make this positional argument
            optimize, annotate_source, kwargs...)
    finally
        Cthulhu.CONFIG.dead_code_elimination = dead_code_elimination
    end
end
function Cthulhu.descend(sys::TransformedIRODESystem, pattern::Union{AbstractPattern,AbstractString}="";
                         terminal::AbstractTerminal=default_terminal(), kwargs...)
    debug_config = DebugConfig(sys)
    while true  # loop so exiting Cthulu takes you back to level select
        ir = select_ir(debug_config, pattern; terminal)
        isnothing(ir) && return
        Cthulhu.descend(sys, ir; terminal, kwargs...)
    end
end

function default_terminal()
    isdefined(Base, :active_repl) ? REPL.LineEdit.terminal(Base.active_repl) :  REPL.TerminalMenus.default_terminal()
end

"""
    select_ir(obj, pattern::Union{AbstractPattern,AbstractString}="")

For any `obj` containing a `DebugConfig` somewhere within it,
searches that `DebugConfig` for all recorded levels with name matching the pattern.
If there is more than one then present a menu to chose from.
Returns the selected level's IR.
"""
function select_ir end
select_ir(obj, pattern::Union{AbstractPattern,AbstractString}="";
          terminal::AbstractTerminal=default_terminal()) =
    select_ir(DebugConfig(obj), pattern; terminal)
function select_ir(debug_config::DebugConfig, pattern::Union{AbstractPattern,AbstractString}="";
                   terminal::AbstractTerminal=default_terminal())
    ir_levels = debug_config.ir_levels
    isnothing(ir_levels) && error("IR Levels were not stored, rerun with `debug_config=Dict(:store_ir_levels=>true)`")
    return select_ir(debug_config.ir_levels, pattern; terminal)
end
function select_ir(ir_levels::IRCodeRecords, pattern::Union{AbstractPattern,AbstractString}="";
                   terminal::AbstractTerminal=default_terminal())
    allowed_levels = IRCodeRecord[x for x in ir_levels if contains(x.name, pattern)]
    isempty(allowed_levels) && error("No IR Levels with names matching the pattern \"$pattern\" found.")
    if length(allowed_levels) == 1
        @info "Found only matching level" only(allowed_levels).name
        return only(allowed_levels).ir
    end
    level_names = String[allowed_level.name for allowed_level in allowed_levels]
    push!(level_names, "↩")
    menu = TerminalMenus.RadioMenu(level_names, pagesize=32)
    ir_idx = 1
    ir_idx = TerminalMenus.request(terminal, SELECT_IR_MSG, menu; cursor=ir_idx)
    if ir_idx <= 0 || ir_idx > length(allowed_levels)
        return nothing
    else
        return allowed_levels[ir_idx].ir
    end
end

const SELECT_IR_MSG = "Select ir_level (`q` to quit):"
