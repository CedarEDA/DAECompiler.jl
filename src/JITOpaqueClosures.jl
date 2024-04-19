
"""
    JITOpaqueClosure <: Function

A `JITOpaqueClosure` is a kind of wrapper around an OpaqueClosure provides it with
just in time specialization on the type -- like a normal julia function would have.

To create one you must pass a builder function to the constructor.
This function is conceptually a lot like a generated function.
It takes a set of argument types and an input and returns the `OpaqueClosure` to call.

The JITOpaqueClosure wrapper takes care of memoizing the function to call so the builder
is only called once on each unique set of argument types.

You must provide a first type-parameter, that is the NAME.
This is not used for anything but shows up in the stacktrace and so is useful for debugging.
It can in theory be set to anything, but most usefully is is normally set as a Symbol.
e,g. `JITOpaqueClosure{:foo}(...)`.

When constructing this you can optionally pass in a gold-class signature `G` as a tuple type parameter.
For example, `JITOpaqueClosure{:foo, Tuple{Int, Int}}(builder)`.
If you do this then the corresponding signature (`(::Int, ::Int)`) will be special case to be 
ahead of time compiled, and statically dispatched to.
Signatures not matching this (or all signatures if you don't specify this) will need to look up their method from
an internal cache which is a bit slower and always incurs a dynamic dispatch.


"""
struct JITOpaqueClosure{NAME, G<:Tuple, F, H} <: Function
    builder::F
    goldclass::H  # call this one if signature matches G
    cache::IdDict{Tuple, Function}
    breadcrumbs::Vector{Breadcrumbs.Breadcrumb}
    maybe_compiling::ReentrantLock
end

const NO_GOLDCLASS = Tuple{nothing}  # no type will ever match this signature

JITOpaqueClosure{NAME}(builder) where NAME = JITOpaqueClosure{NAME, NO_GOLDCLASS}(builder)
function JITOpaqueClosure{NAME, G}(builder) where {NAME, G}
    goldclass = build_goldclass(builder, G)
    return JITOpaqueClosure{NAME, G, typeof(builder), typeof(goldclass)}(
        builder, goldclass, (@new_cache IdDict{Tuple, Function}()), copy(breadcrumb_trail("ir_levels")), ReentrantLock()
    )
end
build_goldclass(_, ::Type{NO_GOLDCLASS}) = error
build_goldclass(builder, arg_types::Type{<:Tuple}) = builder(arg_types.parameters...)

function get_method(this::JITOpaqueClosure{<:Any, G}, arg_types) where G
    if !(G<:Type{NO_GOLDCLASS}) && length(arg_types) != length(G.parameters)
        # If we have a goldclass make sure the length matches, if not thow a method error
        return @opaque (args...)->throw(MethodError(this, args))
    end
    @lock this.maybe_compiling get!(this.cache, arg_types) do
        with_breadcrumb_trail("ir_levels"=>this.breadcrumbs) do
            this.builder(arg_types...)
        end
    end
end

function (this::JITOpaqueClosure{<:Any, G})(args...) where G
    if typeof(args) <: G  # special goldclass case that avoid dict lookup because is AOT compiled and
        this.goldclass(args...)
    else
        arg_types = map(typeof, args)
        oc = get_method(this, arg_types)
        return oc(args...)
    end
end