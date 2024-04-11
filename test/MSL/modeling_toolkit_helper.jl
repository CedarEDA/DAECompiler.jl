using OrdinaryDiffEq
using ModelingToolkit
using DAECompiler
using ModelingToolkit.SymbolicUtils: BasicSymbolic, issym, isterm
using ModelingToolkit: Symbolics
using SymbolicIndexingInterface
using StateSelection: MatchedSystemStructure
using Sundials

const MTK = ModelingToolkit
using DAECompiler.MTKComponents: build_ast, access_var, is_differential, isvar

function DAECompiler.IRODESystem(model::MTK.ODESystem; debug_config=(;))
    T = eval(build_ast(model))
    # HACK we assume no user set parameters for the MTK tests, so just want defaults
    T_parameterless = T{@NamedTuple{}}
    DAECompiler.IRODESystem(Tuple{T_parameterless}; debug_config)
end

#🏴‍☠️🏴‍☠️🏴‍☠️ Begin Really Evil Type Piracy: 🏴‍☠️🏴‍☠️🏴‍☠️

struct MTKAdapter <: MTK.AbstractSystem
    sys::IRODESystem
end

# Keep track of the original sys, so that we can get at the MTK default values
sys_map = IdDict{Core.MethodInstance,MTK.AbstractSystem}()
function MTK.structural_simplify(model::MTK.AbstractSystem)
    # Don't do the MTK structural_simplify at all, instead convert to a IRODEProblem
    daecompiler_sys = IRODESystem(sys)
    sys_map[getfield(daecompiler_sys,:mi)] = sys
    return daecompiler_sys
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
                @assert isvar(var)
                unopt_idx = DAECompiler._sym_to_index(getproperty(sys, access_var(var)))

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
        mtksys = sys_map[getfield(sys, :mi)]
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
function SciMLBase.ODEProblem(sys_adapt::MTKAdapter, u0::Vector, tspan, p = nothing; kw...)
    sys = getfield(sys_adapt, :sys)
    isnothing(p) || isempty(p) || error("parameter merging not yet supported")  # to do this we would need to define the MTKConnection type inside ODEProblem then create the IRODESystem from that (viable)
    prob = ODEProblem(sys, nothing, tspan, p; jac=true, initializealg=CustomBrownFullBasicInit(), kw...)
    state_default_mapping!(prob, [], u0)
    return prob
end

function SciMLBase.DAEProblem(sys::MTKAdapter, du0::Vector, u0::Vector, tspan, p = nothing; kw...)
    sys = getfield(sys, :sys)
    # don't use CustomBrownFullBasicInit() as we use IDA to solve DAEs and that handles things fine without it and doesn't support it.
    isnothing(p) || isempty(p) || error("param_merging not yet supported")
    prob = DAEProblem(sys, nothing, nothing, tspan, nothing; jac=true, kw...)
    state_default_mapping!(prob, du0, u0)
    return prob
end

function drop_leading_namespace(sym, namespace)
    sym_str = string(sym)
    prefix = string(nameof(namespace)) * "₊"
    startswith(sym_str, prefix) || return sym  # didn't start with prefix so no need to drop it
    return Symbol(sym_str[nextind(sym_str, length(prefix)):end])
end

# Monkey-pathching the code in DAECompiler
function Base.getproperty(sys::IRODESystem, name::Symbol)
    mtksys = sys_map[getfield(sys, :mi)]
    names = DAECompiler.StructuralAnalysisResult(sys).names
    if haskey(names, name)
        # Normal DAECompiler way
        return DAECompiler.ScopeRef(sys, DAECompiler.Scope(DAECompiler.Scope(), name))
    elseif haskey(names, drop_leading_namespace(name, mtksys))
        # If it came from something that includes the sys prefix  (probably from the MTK system originally)
        return DAECompiler.ScopeRef(sys, DAECompiler.Scope(DAECompiler.Scope(), drop_leading_namespace(name, mtksys)))
    elseif hasproperty(mtksys, name)  # if it is actually from the MTK system (which allows unflattened names)
        getproperty(mtksys, name)
    else
        throw(Base.KeyError(name))  # should be a UndefRef but key error useful for findout what broke it.
    end
end
Base.getproperty(sys::MTKAdapter, name::Symbol) = getproperty(getfield(sys, :sys), name)

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
        MTK.$fname1(sys::MTKAdapter) = getfield(sys_map[getfield(getfield(sys, :sys),:mi)], $(QuoteNode(prop)))
        MTK.$fname2(sys::MTKAdapter) = isdefined(sys_map[getfield(getfield(sys, :sys),:mi)], $(QuoteNode(prop)))
    end
end

# We do not cache like that so say we do not have a cache
ModelingToolkit.get_index_cache(sys::MTKAdapter) = nothing
ModelingToolkit.has_index_cache(sys::MTKAdapter) = false
#🏴‍☠️🏴‍☠️🏴‍☠️ END Evil Piracy 🏴‍☠️🏴‍☠️🏴‍☠️
