using ExprTools: splitdef
using Base.Meta: isexpr

struct StaledOverrideError <: Exception
    func
    sig
    loc::LineNumberNode
end
function Base.showerror(io::IO, err::StaledOverrideError)
    print(io, "Staled function overload found: ")
    tt = Base.signature_type(err.func, err.sig)
    Base.show_tuple_as_call(io, Symbol(""), tt)
    print(io, " at ", err.loc.file, ":", err.loc.line)
end
struct UnexpectedOverloadError <: Exception
    func
    sig
    loc::LineNumberNode
end
function Base.showerror(io::IO, err::UnexpectedOverloadError)
    print(io, "Unexpected function overload found: ")
    tt = Base.signature_type(err.func, err.sig)
    Base.show_tuple_as_call(io, Symbol(""), tt)
    print(io, " at ", err.loc.file, ":", err.loc.line)
end

"""
    @override function func(args...)
        [...]
    end

Check if a dispatchable method already exists for a given method definition signature.
This macro ensures that the method definition overrides the existing function behavior
by overloading.
"""
macro override(def) with_override_check(def, #=mode=#:override, __source__) end

"""
    @overload function func(args...)
        [...]
    end

Check if there is no method matching a given method definition signature.
This macro ensures that the method definition introduces new function behavior by overloading.
"""
macro overload(def) with_override_check(def, #=mode=#:overload, __source__) end

function with_override_check(@nospecialize(def), mode::Symbol, __source__::LineNumberNode)
    inner = Base.unwrap_macrocalls(def)
    Base.is_function_def(inner) || throw(ArgumentError(lazy"Expected function overload definition, but given $inner"))
    parts = splitdef(inner)
    if !(mode === :override || mode === :overload)
        throw(ArgumentError(lazy"Expected `mode` to be either `:override` or `:overload`, but given $mode"))
    end
    mode = QuoteNode(mode)
    return quote
        let func = $(parts[:name])
            sig = Tuple{$(map(extract_type, parts[:args])...)}
            if $mode === :override
                if !hasmethod(func, sig)
                    throw(StaledOverrideError(func, sig, $(QuoteNode(__source__))))
                end
            elseif $mode === :overload
                if hasmethod(func, sig)
                    throw(UnexpectedOverloadError(func, sig, $(QuoteNode(__source__))))
                end
            end
        end
        Base.@__doc__ $def
    end |> esc
end

function extract_type(@nospecialize arg)
    if isexpr(arg, :kw)
        arg = arg.args[1]
    end
    if isexpr(arg, :(::))
        @assert length(arg.args) in (1,2)
        return arg.args[end]
    elseif isexpr(arg, :macrocall)
        mname = arg.args[1]
        if mname == :var"@nospecialize"
            return extract_type(arg.args[3])
        else
            error(lazy"Unknown macro: $arg")
        end
    elseif isexpr(arg, :...)
        return :(Vararg{$(extract_type(only(arg.args)))})
    elseif isa(arg, Symbol)
        return Any
    end
    error(lazy"Unknown AST: $arg")
end

macro defintrmethod(name, fdef)
	esc(quote
        global $name
        const $name = begin
            @Base.__doc__ @Base.Experimental.overlay $(nothing) $fdef
        end
    end)
end