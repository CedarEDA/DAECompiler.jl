function is_known_invoke(@nospecialize(x), @nospecialize(func), ir::Union{IRCode,IncrementalCompact})
    isexpr(x, :invoke) || return false
    ft = argextype(x.args[2], ir)
    return singleton_type(ft) === func
end

function is_equation_call(@nospecialize(x), ir::Union{IRCode,IncrementalCompact},
        allow_call::Bool=true)
    isexpr(x, :invoke) || (allow_call && isexpr(x, :call)) || return false
    ft = argextype(_eq_function_arg(x), ir)
    return widenconst(ft) === equation
end

function is_known_invoke_or_call(@nospecialize(x), @nospecialize(func), ir::Union{IRCode,IncrementalCompact})
    return is_known_invoke(x, func, ir) || is_known_call(x, func, ir)
end

function _eq_function_arg(stmt::Expr)
    ft_ind = 1 + isexpr(stmt, :invoke)
    return stmt.args[ft_ind]
end

function _eq_val_arg(stmt::Expr)
    ft_ind = 1 + isexpr(stmt, :invoke)
    return stmt.args[ft_ind + 1]
end

struct UnsupportedIRException <: Exception
    msg::String
    ir::IRCode
end

Base.show(io::IO, e::UnsupportedIRException) = print(io, "UnsupportedIRException(", repr(e.msg) ,")")
function Base.showerror(io::IO, e::UnsupportedIRException)
    println(io, "UnsupportedIRException: ", e.msg)
    show(io, e.ir)
end