module MTKComponents
using ..DAECompiler
using DAECompiler.Intrinsics
using ModelingToolkit.SymbolicUtils: BasicSymbolic, issym, isterm
using ModelingToolkit
using ModelingToolkit: Symbolics
const MTK = ModelingToolkit
using SymbolicIndexingInterface
using SciMLBase

export MTKConnector

abstract type MTKConnector end

_c(x) = x
#_c(x) = nameof(x)  # uncomment this to disable interpolating function objects into AST, so as to get nicer printing ASTs

function declare_vars(model, scope_var)
    ret = Expr(:block)
    ret.args = map(unknowns(model)) do x
        var_name = access_var(x)
        :($var_name = $(_c(DAECompiler.Intrinsics.variable))($scope_var($(QuoteNode(var_name)))))
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
        struct $struct_name{B<:NamedTuple} <: $(_c(MTKConnector))
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
    getproperty_expr = :(@inline function Base.getproperty(this::$struct_name{B}, name::Symbol)::Float64 where B; end)
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

function declare_equation(eq::Equation, model, scope_var)
    normed_eq = eq.rhs - eq.lhs
    Expr(:call, _c(DAECompiler.Intrinsics.equation!),
        make_ast(normed_eq, model), 
        Expr(:call, scope_var, Meta.quot(gensym(:mtk_eq)))
    )
end

function build_ast(model)
    model = MTK.expand_connections(model)
    state = MTK.TearingState(model)
    eqs = MTK.equations(state)

    struct_name = gensym(nameof(model))
    return quote
        $(declare_parameters(model, struct_name))

        function (this::$struct_name)()
            # TODO This is broken with the scope changes
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

function drop_leading_namespace(sym, namespace)
    sym_str = string(sym)
    prefix = string(nameof(namespace)) * "₊"
    startswith(sym_str, prefix) || return sym  # didn't start with prefix so no need to drop it
    return Symbol(sym_str[nextind(sym_str, length(prefix)):end])
end

############################################################
"""
    MTKConnector(mtk_component::MTK.ODESystem, ports...)

Declares a connector function that allows you to call a component defined in MTK from DAECompiler.
It takes a component as represented by an ODESystem, and a list of "ports" which correpond to variables in the component,
as identified by their references in the component.
This returns a constructor for a struct that accepts any of the parameters the mtk_component accepted, passed by keyword argument.
The struct itself (once constructed by passing zero or more parameters) accepts in positional arguments in order correponding to each port declared earlier a variable
(or even an expression) respectively  and will impose the connection between that variable in the outer system and the variable inside the port.

Generally if you have parameters you want to be user-setable you will make the struct a field of your DAE system (but make sure it is concetely typed field e.g. by making it type-parametric and passing in an instance)
If on the other hand, you have no parameters, or you don't need them to be user-setable and want to hard code their value, you can instead put it in a `const` global variable.

Example:
```
# At top-level
const foo = ODESystem(...; name=:myfoo) # with parameter `a` and variables `x` and `y`
const FooConn = MTKConnector(foo, foo.x)`
const foo_conn! = FooConn(a=1.5)
#...
function (this::BarCedarSystem)()
    (;outer_x,) = variables()
    foo_conn!(outer_x)
    #...
end
sys = IRODESystem(Tuple{CedarSystem})

sys.myfoo.y  # references the value `y` from within the `foo` that is within the system
```
This 

Note: this (like `include` or `eval`) always runs at top-level scope, even if invoked in a function.
"""
function MTKConnector(mtk_component::MTK.ODESystem, ports...)
    # TODO should this be a macro so that it called `@eval` inside the user's module?
    # We do need to do run time eval, because we can't decide what to construct with just lexical information.
    eval(MTKConnector_AST(mtk_component, ports...))
end

function MTKConnector_AST(model::MTK.ODESystem, ports...)
    model = MTK.expand_connections(model)
    state = MTK.TearingState(model)
    eqs = MTK.equations(state)

    scope_var = gensym(:scope)  # we will put a Scope in a variable with this name, we declare what we will call it first, so can then use it everywhere in generated expressions

    port_names = gensym.(access_var.(ports))
    port_equations = map(ports, port_names) do port_sym, outer_port_name
        # remove namespace as we are currently inside it
        Expr(:call, _c(equation!),
            Expr(:call, _c(-),
                drop_leading_namespace(port_sym, model),# inside of port
                outer_port_name  # outside of port,   
            ),
            Expr(:call, scope_var, Meta.quot(Symbol(:port_, outer_port_name)))
        )
    end


    struct_name = gensym(nameof(model))
    
    return quote
        $(declare_parameters(model, struct_name))

        function (this::$struct_name)($(port_names...))
            $scope_var =  $(_c(Scope))(nothing, $(Meta.quot(nameof(model))))
            $(declare_vars(model, scope_var))
            $(declare_derivatives(state))
            $(declare_equation.(eqs, Ref(model), scope_var)...)

            begin
                $(port_equations...)
            end
        end

        $struct_name  # this is the last line so it is the return value of eval'ing this block
    end
end


end  # module