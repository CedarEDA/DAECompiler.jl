module MTKComponents
using ..DAECompiler
using DAECompiler.Intrinsics
using DAECompiler.Intrinsics: state_ddt
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


split_namespaces_var(x::Symbol) = Symbol.(split(string(x), '₊'))
split_namespaces_var(x) = split_namespaces_var(access_var(x))

_get_scope!(scope, sub_scopes::Dict{Symbol}) = scope
function _get_scope!(parent_scope, sib_scopes::Dict{Symbol}, namespace::Symbol, var_name_tail::Symbol...)
    scope, subscopes = get!(sib_scopes, namespace) do
        Scope(parent_scope, namespace) => Dict{Symbol, Pair}()
    end
    return _get_scope!(scope, subscopes, var_name_tail...)
end
get_scope!(root_scope, scopes, namespaced_var) = _get_scope!(root_scope, scopes, split_namespaces_var(namespaced_var)...)

function declare_vars(model, root_scope)
    ret = Expr(:block)
    scopes = Dict{Symbol, Pair}()
    ret.args = map(unknowns(model)) do var
        scope = get_scope!(root_scope, scopes, var)
        var_name = access_var(var)
        :($var_name = $(_c(variable))($scope))
    end
    return ret
end

# Declare derivatives ahead of time, as `state_ddt`'s of their primal variable.
function declare_derivatives(state)
    ret = Expr(:block)
    append!(ret.args, filter(!isnothing, map(state.fullvars) do var
        if is_differential(var)
            var_name = access_var(var)
            return :($var_name = $(_c(state_ddt))($(access_var(only(arguments(var))))))
        end
        return nothing
    end))
    return ret
end

access_var(x::Num) = access_var(MTK.value(x))
function access_var(x)
    if istree(x) && isequal(arguments(x), [MTK.t_nounits,])
        # then it is something like `x(t)` so we just want to represent it as `x` for conciseness
        nameof(operation(x))
    else # could be anything including something like `τ[2]` but we will treat as a single DAECompler term
        Symbol(repr(x))
    end
end

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
    if isequal(x, MTK.t_nounits)
        return :($(_c(sim_time))())
    end

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

function declare_equation(eq::Equation, model, root_scope)
    normed_eq = eq.rhs - eq.lhs
    Expr(:call, _c(equation!), make_ast(normed_eq, model), Scope(root_scope, gensym(:mtk_eq)))
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
    for port in ports
        port isa MTK.ODESystem && throw(ArgumentError("Port $port is a ODESystem, not a variable (or expression of variables). Perhaps a subsystem. Did you specify a pin rather than the voltage/current across that pin?"))
    end
    # TODO should this be a macro so that it called `@eval` inside the user's module?
    # We do need to do run time eval, because we can't decide what to construct with just lexical information.
    eval(MTKConnector_AST(mtk_component, ports...))
end

"""
    `scope` is an expression for what scope to use for everything within this model
    By default this will introduce a scope for this model that uses the model's name as its name.
    Setting it to `DAECompiler.Intrinsics.root_scope` to not introduce a new subscope for this model.
"""
function MTKConnector_AST(model::MTK.ODESystem, ports...; scope=Scope(DAECompiler.Intrinsics.root_scope, nameof(model)))
    model = MTK.expand_connections(model)
    state = MTK.TearingState(model)
    eqs = MTK.equations(state)



    port_names = gensym.(access_var.(ports))
    port_equations = map(ports, port_names) do port_sym, outer_port_name
        # remove namespace as we are currently inside it
        Expr(:call, _c(equation!),
            Expr(:call, _c(-),
                drop_leading_namespace(access_var(port_sym), model),# inside of port
                outer_port_name  # outside of port,   
            ),
            Scope(scope, Symbol(:port_, outer_port_name))
        )
    end


    struct_name = gensym(nameof(model))
    
    return quote
        $(declare_parameters(model, struct_name))

        function (this::$struct_name)($(port_names...))
            $(declare_vars(model, scope))
            $(declare_derivatives(state))
            $(declare_equation.(eqs, Ref(model), Ref(scope))...)

            begin
                $(port_equations...)
            end
        end

        $struct_name  # this is the last line so it is the return value of eval'ing this block
    end
end


end  # module