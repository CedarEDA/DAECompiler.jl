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

function declare_parameters(model, struct_name)
    param_names_tuple_expr =  Expr(:tuple, Meta.quot.(access_var.(MTK.parameters(model)))...)
    struct_expr = :(
        struct $struct_name{B<:NamedTuple}
            backing::B
        end
    )
    
    
    constructor_expr =:(
        function $struct_name(; kwargs...)
            unexpected_parameters = setdiff(keys(kwargs), $param_names_tuple_expr)
            isempty(unexpected_parameters) || error("unexpected parameters passed: $unexpected_parameters")
            backing = NamedTuple(kwargs)
            return $struct_name(backing)
        end
    )
    propertynames_expr = :(Base.propertynames(::$struct_name) = $param_names_tuple_expr)

    # build up the getproperty piece by piece
    # we need to do this with a constant foldable function rather than assign values to paraemeters
    # so that it constant folds any parameters not passed so we can alias eliminate them
    # see https://github.com/JuliaComputing/DAECompiler.jl/issues/860
    # We need the conversion-constraint to Number to stop DAECompiler hanging in inference: https://github.com/JuliaComputing/DAECompiler.jl/issues/864
    getproperty_expr = :(@inline function Base.getproperty(this::$struct_name{B}, name::Symbol)::Number where B; end)
    getproperty_body = []
    defaults = MTK.defaults(model)
    for param_sym in MTK.parameters(model)
              param_default = get(defaults, param_sym, nothing)
        param_default === nothing && continue  # no need to have a special case for parameters without defaults, they will error if not present
        param_value = make_ast(param_default, model)
        param_name = Meta.quot(access_var(param_sym))

        push!(getproperty_body, :(
            if name === $param_name
                return if hasfield(B, $param_name)
                    getfield(getfield(this, :backing), $param_name)
                else 
                    $param_value
                end
            end
        ))
    end
    push!(getproperty_body, :( # final "else"
        return getfield(getfield(this, :backing), name)
    ))
    getproperty_expr.args[end].args[end] = Expr(:block, getproperty_body...)
    
    return Expr(:block, struct_expr, constructor_expr, propertynames_expr, getproperty_expr)
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

    if is_parameter(model, x)
        param_name = access_var(x)
        return :(this.$param_name)
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

    struct_name = gensym(nameof(model))
    return quote
        $(declare_parameters(model, struct_name))

        function (this::$struct_name)()
            $(declare_vars(model))
            $(declare_derivatives(state))
            $(declare_equation.(eqs, Ref(model))...)
        end

        $struct_name  # this is the last line so it is the return value of eval'ing this block
    end
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