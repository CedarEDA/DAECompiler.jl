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

"""
    @insert_node_here compact line settings make_odefunction(f)::ODEFunction
    @insert_node_here compact line settings make_odefunction(f)::ODEFunction true
    @insert_node_here compact line settings (:invoke)(ci, args...)::Int true
    @insert_node_here compact line settings (return x)::Int true
"""
macro insert_node_here(compact, line, settings, ex, reverse_affinity = false)
    source = :(LineNumberNode($(__source__.line), $(QuoteNode(__source__.file))))
    line = :($settings.insert_stmt_debuginfo ? $line : $DAECompiler.insert_new_lineinfo!($compact.ir.debuginfo, $source, $compact.result_idx, $line))
    insert_node_here(compact, line, ex, reverse_affinity)
end

function insert_node_here(compact, line, ex, reverse_affinity)
    isexpr(ex, :(::), 2) || throw(ArgumentError("Expected type-annotated expression, got $ex"))
    ex, type = ex.args
    if isexpr(ex, :call) && isa(ex.args[1], QuoteNode)
        # The called "function" is a non-call `Expr` head
        ex = Expr(ex.args[1].value, ex.args[2:end]...)
    end
    compact = esc(compact)
    line = esc(line)
    type = esc(type)
    if isa(ex, Symbol)
        inst_ex = ex
    elseif isexpr(ex, :return)
        inst_ex = :(ReturnNode($(ex.args...)))
    else
        inst_ex = :(Expr($(QuoteNode(ex.head)), $(ex.args...)))
    end
    return quote
        inst = NewInstruction($(esc(inst_ex)), $type, $line)
        insert_node_here!($compact, inst, $(esc(reverse_affinity)))
    end
end

"""
    @sshow stmt
    @sshow length(ir.stmts) typeof(val)

Drop-in replacement for `@show`, but using `jl_safe_printf` to avoid task switches.

This directly prints to C stdout; `stdout` redirects won't have any effect.
"""
macro sshow(exs...)
    blk = Expr(:block)
    for ex in exs
        push!(blk.args, :(Core.println($(sprint(Base.show_unquoted,ex)*" = "),
                                  repr(begin local value = $(esc(ex)) end))))
    end
    isempty(exs) || push!(blk.args, :value)
    return blk
end
