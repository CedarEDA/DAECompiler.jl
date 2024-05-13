using OrdinaryDiffEq
using ModelingToolkit
using DAECompiler
using DAECompiler: batch_reconstruct
using ModelingToolkit.SymbolicUtils: BasicSymbolic, issym, isterm
using ModelingToolkit: Symbolics
using SymbolicIndexingInterface
using StateSelection: MatchedSystemStructure
using Sundials

const MTK = ModelingToolkit

const DMTK = Base.get_extension(DAECompiler, :DAECompilerModelingToolkitExt)
isnothing(DMTK) && error("Something went weird loading the DAECompilerModelingToolkitExt")
const access_var = DMTK.access_var
const is_differential = DMTK.is_differential
const isvar = DMTK.isvar
const split_namespaces_var = DMTK.split_namespaces_var

function DAECompiler.IRODESystem(model::MTK.ODESystem; debug_config=(;))
    T = @declare_MTKConnector model
    # We assume no user set parameters for the MTK tests
    T_parameterless = T{@NamedTuple{}}
    DAECompiler.IRODESystem(Tuple{T_parameterless}; debug_config)
end


function reconstruct_helper(ro, sym::Union{Num,BasicSymbolic}, u, p, t)
    replacements = Dict()

    # Collect all variables/observeds used within the symbolic expression,
    # translating them to DAECompiler ScopeRefs
    syms = []
    for var in Symbolics.get_variables.(sym)
        push!(syms, getproperty(ro.sys, access_var(var)))
    end

    # Perform the reconstruction for these variables, then build replacements dict
    out = batch_reconstruct(ro, syms, [zero(u)], [u], p, [t])
    for (idx, var) in enumerate(Symbolics.get_variables(sym))
        replacements[var] = out[idx, 1]
    end

    # Replace all instances of all variables with the concrete values
    # stored within `replacements`; this will result in a concrete value.
    return Symbolics.value(Symbolics.substitute(sym, replacements))
end

# A little type-piracy to handle that we store references to objects differently
function (ro::DAECompiler.ODEReconstructedObserved)(sym::Union{Num,BasicSymbolic}, u, p, t)
    return reconstruct_helper(ro, sym, u, p, Float64(t))
end
function (ro::DAECompiler.DAEReconstructedObserved)(sym::Union{Num,BasicSymbolic}, u, p, t)
    return reconstruct_helper(ro, sym, u, p, Float64(t))
end
function (ro::DAECompiler.DAEReconstructedObserved)(sym::Union{Num,BasicSymbolic}, du, u, p, t)
    return reconstruct_helper(ro, sym, u, p, Float64(t))
end

function get_scoperef_from_symbolic(sys::TransformedIRODESystem, sym::Union{Num, BasicSymbolic})
    sym = Symbolics.get_variables(sym)
    length(sym) == 1 || return nothing
    return getproperty(get_sys(sys), access_var(only(sym)))
end
SymbolicIndexingInterface.is_independent_variable(sys::TransformedIRODESystem, sym) = false

function SymbolicIndexingInterface.is_variable(sys::TransformedIRODESystem, sym::Union{Num, BasicSymbolic})
    sr = get_scoperef_from_symbolic(sys, sym)
    !isnothing(sr) && !isnothing(SciMLBase.sym_to_index(sr, sys))
end

function SymbolicIndexingInterface.variable_index(sys::TransformedIRODESystem, sym::Union{Num, BasicSymbolic})
    SciMLBase.sym_to_index(get_scoperef_from_symbolic(sys, sym), sys)
end

SymbolicIndexingInterface.is_parameter(sys::TransformedIRODESystem, sym) = false

function SymbolicIndexingInterface.is_observed(sys::TransformedIRODESystem, sym)
    # this isn't really true, but it should pass tests that are well formed in MTK
    return !is_variable(sys, sym)
end

#🏴‍☠️🏴‍☠️🏴‍☠️ Begin Really Evil Type Piracy: 🏴‍☠️🏴‍☠️🏴‍☠️

# IRODESystem doesn't subtype MTK.AbstractSystem anymore
# so we introduce a wrapper, to make it subtype
struct MTKAdapterSystem <: MTK.AbstractSystem
    sys::IRODESystem
end
DAECompiler.get_sys(adpater::MTKAdapterSystem) = getfield(adpater, :sys)

# Keep track of the original MTK model, so that we can get at the MTK default values
const sys_map = IdDict{Core.MethodInstance,MTK.AbstractSystem}()
sys_map_key(adapter_sys::MTKAdapterSystem) = sys_map_key(get_sys(adapter_sys))
sys_map_key(sys::IRODESystem) = getfield(sys, :mi)

function MTK.structural_simplify(model::MTK.AbstractSystem)
    # Don't do the MTK structural_simplify at all, instead convert to a IRODEProblem
    daecompiler_sys = IRODESystem(model)
    sys_map[sys_map_key(daecompiler_sys)] = model
    return MTKAdapterSystem(daecompiler_sys)
end


# Given a symbolic expression `sym`, run `substitute()` on it with the defaults from `model`
# until fixed-point
function replace_defaults(sym, model)
    defaults = MTK.defaults(model)
    terms = Symbolics.get_variables(sym)
    last_terms = eltype(terms)[]
    while !isa(sym, Number) && !isequal(terms, last_terms)
        sym = Symbolics.value(Symbolics.substitute(sym, defaults))
        last_terms = terms
        terms = Symbolics.get_variables(sym)
    end
    return sym
end


function state_default_mapping!(prob, du0::Vector, u0::Vector)
    sys = get_sys(prob)
    # Construct the optimized variable matching datastructures
    mss = MatchedSystemStructure(prob.f.sys.state.structure, prob.f.sys.var_eq_matching)
    var_assignment, = DAECompiler.assign_vars_and_eqs(mss, isa(prob, DAEProblem))

    # Given a symbolic expression, return the variable within it.
    # If it is something like D(var), then unwrap the differential and try to
    # re-map the index to its unoptimized index
    function peel_differential_var_index(var)
        # Peel `Num` to its internal symbolic expression
        if isa(var, Num)
            var = Symbolics.value(var)
        end

        if is_differential(var)
            # If this was a differential, peel it and recurse to get the underlying variable index:
            inner_var = only(arguments(var))
            var_idx = peel_differential_var_index(inner_var)
            
            # if the underlying variable is `nothing`, it's not real, so just pass it on.
            if var_idx === nothing
                return nothing
            end

            # Once we have the variable index, push it through our derivative map:
            return prob.f.sys.state.structure.var_to_diff[var_idx]
        else
            # If this is not a differential, just return the index of the variable:
            try
                # depending on if we came here via sys.ref, or via a MTK variable we may or may not already have a ScopeRef
                ref = isa(var, DAECompiler.ScopeRef) ? var : getproperty(sys, access_var(var))
                unopt_idx = DAECompiler._sym_to_index(ref)                
                return unopt_idx.var
            catch
                @error("Unable to index $(var) into $(sys)")
                rethrow()
            end
        end
    end

    

    function set_initial_condition!(var, val, new_u0, new_du0)
        # First, peel away any `Differential()` calls wrapping our variable,
        # and walk the chain of `var_to_diff` accordingly, to get our `unopt_idx`
        unopt_idx = peel_differential_var_index(var)
        if unopt_idx !== nothing
            # If that is not `nothing`, then try to get the optimized index:
            opt_idx, in_du = var_assignment[unopt_idx]
            #@info "setting initial conditions" var val
            if opt_idx != 0
                # If that is not 0 then this is a selected state and we can
                # set it in either `u0` or `du0`!
                if in_du
                    #@info("Setting du0", var, opt_idx, val)
                    new_du0[opt_idx] = val
                else
                    #@info("Setting u0", var, opt_idx, val)
                    new_u0[opt_idx] = val
                end
            end
        end
    end

    new_u0 = nothing
    new_du0 = nothing
    if isa(prob, DAEProblem) && prob.du0 !== nothing
        new_du0 = copy(prob.du0)
    end

    if prob.u0 !== nothing
        new_u0 = copy(prob.u0)
        # First, insert "default" values:
        mtksys = sys_map[sys_map_key(sys)]
        defaults = MTK.defaults(mtksys)
        for var in unknowns(mtksys)
            if var ∈ keys(defaults)
                # Canonicalize default values
                default_val = defaults[var]
                default_val = replace_defaults(default_val, mtksys)
                # @info "setting" var default_val
                set_initial_condition!(var, default_val, new_u0, new_du0)
            end
        end

        # Next, override with values from `u0`
        for (var, val) in u0
            if isvar(var) || is_differential(var)
                set_initial_condition!(var, val, new_u0, new_du0)
            else
                @warn "Initial conditions only supported for variables and state_ddt/Differential of variables. Ignoring." var ModelingToolkit.isparameter(var)
            end
        end

        # Do the same for `du0`
        if isa(prob, DAEProblem) && prob.du0 !== nothing
            for (var, val) in du0
                if isvar(var) || is_differential(var)
                    set_initial_condition!(var, val, new_u0, new_du0)
                else
                    @warn "Initial conditions only supported for variables and state_ddt/Differential of variables. Ignoring." var ModelingToolkit.isparameter(var)
                end
            end
        end
    end

    if new_u0 !== nothing
        prob.u0 .= new_u0
    end
    if new_du0 !== nothing
        prob.du0 .= new_du0
    end
    return prob
end

# Hack in support for initial condition hints
function SciMLBase.ODEProblem(sys_adapt::MTKAdapterSystem, u0::Vector, tspan, p = nothing; kw...)
    sys::IRODESystem = get_sys(sys_adapt)
    isnothing(p) || isempty(p) || error("parameter merging not yet supported")  # to do this we would need to define the MTKConnection type inside ODEProblem then create the IRODESystem from that (viable)
    prob = ODEProblem(sys, nothing, tspan, p; jac=true, initializealg=CustomBrownFullBasicInit(), kw...)
    state_default_mapping!(prob, [], u0)
    return prob
end

function SciMLBase.DAEProblem(sys_adapt::MTKAdapterSystem, du0::Vector, u0::Vector, tspan, p = nothing; kw...)
    sys::IRODESystem = get_sys(sys_adapt)
    # don't use CustomBrownFullBasicInit() as we use IDA to solve DAEs and that handles things fine without it and doesn't support it.
    isnothing(p) || isempty(p) || error("param_merging not yet supported")
    prob = DAEProblem(sys, nothing, nothing, tspan, nothing; jac=true, kw...)
    state_default_mapping!(prob, du0, u0)
    return prob
end


# Monkey-pathching the code in DAECompiler
# because we don't just need to handle names that came in from `sys.x.y` (for which we create scopes)
# but also names that come from variables in scope like references to under other names that use flattened names like `sys₊x₊y` and `x₊y`
# as well as things that want to call getproperty on the MTK system to check some metadata (that we may not even store)
function Base.getproperty(sys::IRODESystem, name::Symbol)
    names = DAECompiler.StructuralAnalysisResult(sys).names
    namespaces = split_namespaces_var(name)
    if haskey(names, namespaces[1])
        # Normal DAECompiler way
        return return get_scope_ref(sys, namespaces)
    elseif length(namespaces) > 1 && haskey(names, namespaces[2])
        # Ignore first namespace it's cos we are not fully consistent with if we include the system name or not
        return return get_scope_ref(sys, namespaces; start_idx=2)
    else  # It could be from the mtksys
        mtksys = sys_map[sys_map_key(sys)]
        if hasproperty(mtksys, name)  # if it is actually from the MTK system (which allows unflattened names)
            return getproperty(mtksys, name)
        end
    end
    throw(Base.KeyError(name))  # should be a UndefRef but key error useful for findout what broke it.
end
function get_scope_ref(sys, names; start_idx=1)
    ref = DAECompiler.ScopeRef(sys, DAECompiler.Scope(DAECompiler.Scope(), names[start_idx]))
    for name in @view names[(start_idx+1):end]
        ref = getproperty(ref, name)
    end
    return ref
end
Base.getproperty(sys::MTKAdapterSystem, name::Symbol) = getproperty(get_sys(sys), name)

Core.eval(OrdinaryDiffEq, quote
    # DFBDF doesn't support things we need like
    # https://github.com/SciML/OrdinaryDiffEq.jl/issues/1109 and  https://github.com/SciML/OrdinaryDiffEq.jl/issues/1926
    DFBDF() = $(IDA)()
end)

# Hack in support for `inputs()`, `parameters()`, etc...
# X-ref: ModelingToolkit.jl/src/systems/abstractsystem.jl:236
for prop in [
    :eqs
    :tag
    :noiseeqs
    :iv
    :unknowns
    :ps
    :tspan
    :name
    :var_to_name
    :ctrls
    :defaults
    :guesses
    :observed
    :tgrad
    :jac
    :ctrl_jac
    :Wfact
    :Wfact_t
    :systems
    :structure
    :op
    :constraints
    :controls
    :loss
    :bcs
    :domain
    :ivs
    :dvs
    :connector_type
    :connections
    :preface
    :torn_matching
    :initializesystem
    :initialization_eqs
    :schedule
    :tearing_state
    :substitutions
    :metadata
    :gui_metadata
    :discrete_subsystems
    :parameter_dependencies
    :solved_unknowns
    :split_idxs
    :parent
#    :index_cache  # Do not delegate this to the mtksys
]
    fname1 = Symbol(:get_, prop)
    fname2 = Symbol(:has_, prop)
    @eval begin
        MTK.$fname1(sys::MTKAdapterSystem) = getfield(sys_map[sys_map_key(sys)], $(QuoteNode(prop)))
        MTK.$fname2(sys::MTKAdapterSystem) = isdefined(sys_map[sys_map_key(sys)], $(QuoteNode(prop)))
    end
end

# We do not cache like that so say we do not have a cache
ModelingToolkit.get_index_cache(sys::MTKAdapterSystem) = nothing
ModelingToolkit.has_index_cache(sys::MTKAdapterSystem) = false
#🏴‍☠️🏴‍☠️🏴‍☠️ END Evil Piracy 🏴‍☠️🏴‍☠️🏴‍☠️
