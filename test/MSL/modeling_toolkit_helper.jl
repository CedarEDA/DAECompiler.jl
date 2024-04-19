using ModelingToolkit
using DAECompiler
using ModelingToolkit.SymbolicUtils: BasicSymbolic, issym, isterm
using ModelingToolkit: Symbolics
using SymbolicIndexingInterface
using Sundials

const MTK = ModelingToolkit

_c(x) = x
#_c(x) = nameof(x)  # uncomment this to disable interpolating function objects into AST, so as to get nicer printing ASTs

function declare_vars(model)
    ret = Expr(:block)
    ret.args = map(unknowns(model)) do x
        var_name = access_var(x)
        :($var_name = $(_c(DAECompiler.Intrinsics.variable))($(QuoteNode(var_name))))
    end
    return ret
end

# Declare derivatives ahead of time, as `state_ddt`'s of their primal variable.
function declare_derivatives(state)
    ret = Expr(:block)
    append!(ret.args, filter(!isnothing, map(state.fullvars) do var
        if typeof(operation(var)) == Differential
            var_name = access_var(var)
            return :($var_name = $(_c(DAECompiler.Intrinsics.state_ddt))($(access_var(only(arguments(var))))))
        end
        return nothing
    end))
    return ret
end

access_var(x) = Symbol(repr(x))

"""
Determines if somethings is a variable (i.e. an unknown in MTK v9 terminology)
Base on https://github.com/SciML/ModelingToolkit.jl/blob/master/src/utils.jl#L456-L458
"""
isvar(x) = !ModelingToolkit.isparameter(x) && (!ModelingToolkit.istree(x) || !ModelingToolkit.isparameter(operation(x))) && !ModelingToolkit.isconstant(x)

is_differential(var) = isterm(var) && isa(operation(var), Differential)

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

struct UnsupportedTermException
    term
end
Base.showerror(io::IO, x::UnsupportedTermException) = println(io, "do not know how to make AST for $(x.term)::$(typeof(x.term))")
# using KristofferC's hack to disable backtraces for this, as they are rarely informative
# Instead look at the backtrace of the exeption that caused this one
function Base.showerror(io::IO, ex::UnsupportedTermException, bt; backtrace=true)
    Base.with_output_color(get(io, :color, false) ? Base.error_color() : :nothing, io) do io
        showerror(io, ex)
    end
end

function make_ast(x, model)
    is_unrecognized_t = issym(x) && nameof(x) == :t  #HACK: sometimes the `t` doesn't come out right, idk why. So just capture things that use that as the name
    if x === MTK.t_nounits || is_unrecognized_t
        return :($(_c(DAECompiler.Intrinsics.sim_time))())
    end
    (x === MTK.t_unitful || x === MTK.t) && error("time with units not supported")

    if haskey(MTK.defaults(model), x)  && is_parameter(model, x)
        # Hard code all parameters we encounter to their default value
        return replace_defaults(MTK.defaults(model)[x], model)
    end

    try
        make_ast(operation(x), x, model)
    catch e
        throw(UnsupportedTermException(x))
    end
end
make_ast(x::Real, _) = x
make_ast(::Union{<:BasicSymbolic,<:Differential}, x, _) = access_var(x)
function make_ast(f::Function, x, model)
    if f == getindex
        @assert isvar(x)
        # Because we don't support vector types in DAECompiler, we've lowered every state to
        # its own independent variable; so we need to lower accesses to those variables just
        # using their full names.
        return access_var(x)
    end
    return Expr(:call, _c(f), make_ast.(arguments(x), Ref(model))...)
    #end
end

make_ast(op, x, _) = error("$x :: $op :: $(arguments(x))")

function declare_equation(eq::Equation, model)
    normed_eq = eq.rhs - eq.lhs
    Expr(:call, _c(DAECompiler.Intrinsics.equation!), make_ast(normed_eq, model))
end

function build_ast(model)
    model = MTK.expand_connections(model)
    state = MTK.TearingState(model)
    eqs = MTK.equations(state)

    body = Expr(:block)
    push!(body.args, declare_vars(model))
    push!(body.args, declare_derivatives(state))
    append!(body.args, declare_equation.(eqs, Ref(model)))
    return :(()->$body)
end

using DAECompiler: batch_reconstruct
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

function DAECompiler.IRODESystem(model::MTK.ODESystem)
    fun = eval(build_ast(model))
    debug_config = (;
        store_ir_levels = true,
        verify_ir_levels = true,
        store_ss_levels = true,
    )
    DAECompiler.IRODESystem(Tuple{typeof(fun)}; debug_config)
end

#🏴‍☠️🏴‍☠️🏴‍☠️ Begin Really Evil Type Piracy: 🏴‍☠️🏴‍☠️🏴‍☠️

# Keep track of the original sys, so that we can get at the MTK default values
sys_map = IdDict{Core.MethodInstance,MTK.AbstractSystem}()
function MTK.structural_simplify(sys::MTK.AbstractSystem)
    # Don't do the MTK structural_simplify at all, instead convert to a IRODEProblem
    daecompiler_sys = IRODESystem(sys)
    sys_map[getfield(daecompiler_sys,:mi)] = sys
    return daecompiler_sys
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

                # Oops, this variable is not real
                if unopt_idx[1]
                    return nothing
                end

                # Otherwise, return the index
                return unopt_idx[2]
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
using ModelingToolkit: MatchedSystemStructure
function SciMLBase.ODEProblem(sys::IRODESystem, u0::Vector, tspan, p = nothing; kw...)
    prob = ODEProblem(sys, nothing, tspan, p; jac=true, initializealg=CustomBrownFullBasicInit(), kw...)
    state_default_mapping!(prob, [], u0)
    return prob
end

function SciMLBase.DAEProblem(sys::IRODESystem, du0::Vector, u0::Vector, tspan, p = nothing; kw...)
    # don't use CustomBrownFullBasicInit() as we use IDA to solve DAEs and that handles things fine without it and doesn't support it.
    prob = DAEProblem(sys, nothing, nothing, tspan, p; jac=true, kw...)
    state_default_mapping!(prob, du0, u0)
    return prob
end

function MTK.StructuralTransformations.ODAEProblem{iip}(sys::IRODESystem, u0map, tspan, parammap=nothing; kw...) where iip
    return ODEProblem(sys, u0map, tspan; kw...)
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
    var_obs_names = DAECompiler.StructuralAnalysisResult(sys).var_obs_names
    if haskey(var_obs_names, name)
        # Normal DAECompiler way
        return DAECompiler.ScopeRef(sys, DAECompiler.Scope(DAECompiler.Scope(), name))
    elseif haskey(var_obs_names, drop_leading_namespace(name, mtksys))
        # If it came from something that includes the sys prefix  (probably from the MTK system originally)
        return DAECompiler.ScopeRef(sys, DAECompiler.Scope(DAECompiler.Scope(), drop_leading_namespace(name, mtksys)))
    elseif hasproperty(mtksys, name)  # if it is actually from the MTK system (which allows unflattened names)
        getproperty(mtksys, name)
    else
        throw(Base.KeyError(name))  # should be a UndefRef but key error useful for findout what broke it.
    end
end


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
        MTK.$fname1(sys::IRODESystem) = getfield(sys_map[getfield(sys,:mi)], $(QuoteNode(prop)))
        MTK.$fname2(sys::IRODESystem) = isdefined(sys_map[getfield(sys,:mi)], $(QuoteNode(prop)))
    end
end

# We do not cache like that so say we do not have a cache
ModelingToolkit.get_index_cache(sys::IRODESystem) = nothing
ModelingToolkit.has_index_cache(sys::IRODESystem) = false
#🏴‍☠️🏴‍☠️🏴‍☠️ END Evil Piracy 🏴‍☠️🏴‍☠️🏴‍☠️
