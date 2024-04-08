module MTKComponents
using ..DAECompiler
using ModelingToolkit.SymbolicUtils: BasicSymbolic, issym, isterm
using ModelingToolkit
using ModelingToolkit: Symbolics
const MTK = ModelingToolkit
using SymbolicIndexingInterface
using SciMLBase

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


end  # module