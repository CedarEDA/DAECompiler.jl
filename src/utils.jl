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

"Get the current file location as a `LineNumberNode`."
macro __SOURCE__()
    return :(LineNumberNode($(__source__.line), $(QuoteNode(__source__.file))))
end

"""
    @insert_instruction_here compact line settings make_odefunction(f)::ODEFunction
    @insert_instruction_here compact line settings make_odefunction(f)::ODEFunction true
    @insert_instruction_here compact line settings (:invoke)(ci, args...)::Int true
    @insert_instruction_here compact line settings (return x)::Int true
"""
macro insert_instruction_here(compact, line, settings, ex, reverse_affinity = false)
    source = :(LineNumberNode($(__source__.line), $(QuoteNode(__source__.file))))
    return generate_insert_instruction_here(compact, line, settings, ex, source, reverse_affinity)
end

function generate_insert_instruction_here(compact, line, settings, ex, source, reverse_affinity)
    isexpr(ex, :(::), 2) || throw(ArgumentError("Expected type-annotated expression, got $ex"))
    ex, type = ex.args
    compact = esc(compact)
    settings = esc(settings)
    line = esc(line)
    inst_ex = esc(process_instruction_expr(ex))
    type = esc(type)
    return :(insert_instruction_here!($compact, $line, $settings, $source, $inst_ex, $type; reverse_affinity = $reverse_affinity))
end

function process_instruction_expr(ex)
    if isexpr(ex, :call) && isa(ex.args[1], QuoteNode)
        # The called "function" is a non-call `Expr` head
        ex = Expr(ex.args[1].value, ex.args[2:end]...)
    end
    isa(ex, Symbol) && return ex
    isexpr(ex, :return) && return :($ReturnNode($(ex.args...)))
    return :(Expr($(QuoteNode(ex.head)), $(ex.args...)))
end

function insert_instruction_here!(compact::IncrementalCompact, line, settings::Settings, source::LineNumberNode, args...; reverse_affinity::Bool = false)
    line = maybe_insert_debuginfo!(compact, settings, source, line, compact.result_idx)
    return insert_instruction_here!(compact, line, args...; reverse_affinity)
end

function insert_instruction_here!(compact::IncrementalCompact, settings::Settings, source::LineNumberNode, inst::NewInstruction; reverse_affinity::Bool = false)
    line = maybe_insert_debuginfo!(compact, settings, source, inst.line, compact.result_idx)
    inst_with_source = NewInstruction(inst.stmt, inst.type, inst.info, line, inst.flag)
end

function insert_instruction_here!(compact::IncrementalCompact, line, inst_ex, type; reverse_affinity::Bool = false)
    inst = NewInstruction(inst_ex, type, line)
    return insert_instruction_here!(compact, inst; reverse_affinity)
end

function insert_instruction_here!(compact::IncrementalCompact, inst::NewInstruction; reverse_affinity::Bool = false)
    return insert_node_here!(compact, inst, reverse_affinity)
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
