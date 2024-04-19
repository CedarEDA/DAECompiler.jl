# TODO: release this as a separate package

"""
# Breadcrumbs

This package provides easy tools for annotating a subset of your call stack for
easy, multi-threading-safe logging.  The main definition is the `@breadcrumb`
macro:

    @breadcrumb trail_name function_def

This macro instruments the given function to push its name (and `__source__`)
onto a task-local breadcrumb trail, allowing for multiple tasks to track a
subset of their callstack.  Use `breadcrumb_trail(trail_name)` to look up
the current breadcrumb trail.  Useful for logging deep callstacks where you
only want to highlight a few frames in the callstack.  Example:

    @breadcrumb "hansel" function foo()
        bar()
    end

    bar() = baz()

    @breadcrumb "hansel" function baz()
        println(join([bc.name for bc in breadcrumb_trail("hansel")], "."))
        ...
    end

When you call `foo()`, the `println()` statement will print `foo.baz`m as those
are the only two functions that are tracked by the breadcrumbs.  Note that you
can define multiple breadcrumb trails (even on the same function) in order to
track different pieces of information, as shown in the next example:

    @breadcrumb ["hansel", "gretel"] foo() = bar()

    function bar()
        baz1()
        baz2()
    end

    @breadcrumb "hansel" baz1() = spoon()
    @breadcrumb "gretel" baz2() = spoon()

    @breadcrumb "hansel" function spoon()
        println("hansel: ", join([bc.name for bc in breadcrumb_trail("hansel")], "."))
        println("gretel: ", join([bc.name for bc in breadcrumb_trail("gretel")], "."))
    end

When running `foo`, there will be four lines printed:

    hansel: foo.baz1.spoon
    gretel: foo
    hansel: foo.spoon
    gretel: foo.baz2

Finally, this is all task-local, so it is async-safe, although until
https://github.com/JuliaLang/julia/issues/35757 is solved in a satisfactory
manner, the breadcrumb trail before the `@async` is cut off:

    @breadcrumb "hansel" function foo()
        @sync begin
            @async bar1()
            @async bar2()
        end
    end

    @breadcrumb "hansel" bar1() = baz()
    @breadcrumb "hansel" bar2() = baz()

    @breadcrumb "hansel" function baz()
        println(join([bc.name for bc in breadcrumb_trail("hansel")], "."))
    end

Running `foo()` will result in:

    bar1.baz
    bar2.baz

Although the order of the lines will be permuted, the stacks will not interfere
with eachother.
"""
module Breadcrumbs
using ExprTools

export @breadcrumb, breadcrumb_trail, with_breadcrumb_trail, with_breadcrumb

struct Breadcrumb
    name::Symbol
    ln::LineNumberNode
end

"""
    breadcrumb_trail(trail_name::String) -> Vector{Breadcrumb}

Returns the task-local breadcrumb trail for the given trail name.  If none
exists, creates a new, empty trail.  Note that this method returns a vector of
breadcrumbs which should not be mutated by user code.  See `@breadcrumb` for
an overview of how users should interact with this package.
"""
function breadcrumb_trail(trail_name::String)::Vector{Breadcrumb}
    return get!(task_local_storage(), trail_name) do
        Breadcrumb[]
    end
end

"""
    with_breadcrumb_trail(f, trail_name=>breadcrumbs)

Run `f()` while temporarily replacing the breadcrumb trail of the given `trail_name` with the
`breadcrumbs` that were given.
The old value is restored afterwards.

This is mostly for if you are going to delay the execution of something
that would use the breadcrumb trail. So you can still get the same trail as if you had not delayed that execution.
"""
function with_breadcrumb_trail(f, (trail_name, breadcrumbs))
    old_trail = copy(breadcrumb_trail(trail_name))
    try
        task_local_storage()[trail_name] = breadcrumbs
        f()
    finally
        task_local_storage()[trail_name] = old_trail
    end
end

@inline function push_breadcrumb!(trail_name::String, bc::Breadcrumb)
    push!(breadcrumb_trail(trail_name), bc)
end

@inline function pop_breadcrumb!(trail_name::String)
    pop!(breadcrumb_trail(trail_name))
end

function identify_function(f)  # only works on 0 arg functions but that's what we are dealign with
    file, linenumber = functionloc(f)
    return LineNumberNode(Int(linenumber), file)
end

"""
    with_breadcrumb(f, trail_name, breadcrumb_name)

Adds to given trail, a breadcrumb with the given name, which might be determined programatically.
The file location is determined by the function" which if used do-block ,or lambda form is the callsite.
"""
@inline function with_breadcrumb(f, trail_name::String, breadcrumb_name)
    bc = Breadcrumb(Symbol(breadcrumb_name), identify_function(f))
    push_breadcrumb!(trail_name, bc)
    try
        return f()
    finally
        pop_breadcrumb!(trail_name)
    end
end

macro breadcrumb(trails, func)
    def = splitdef(func)

    # If `trails` is not a vector, wrap it!
    if !isa(trails, Expr) || trails.head != :vect
        trails = :([$(trails)])
    end

    # Unwrap e.g. "Module.Submodule.foo" to "foo"
    func_name = def[:name]
    while func_name isa Expr
        func_name = func_name.args[end].value
    end

    # Create a single Breadcrumb object describing this function:
    bc = Breadcrumb(func_name, __source__)

    # We're going to wrap the function body in a `try-finally` to push/pop breadcrumbs
    # when entering/leaving this function.
    def[:body] = quote
        for trail_name in $trails
            $push_breadcrumb!(trail_name, $bc)
        end
        $(Expr(:tryfinally,
               def[:body],
               quote
                   for trail_name in $trails
                       $pop_breadcrumb!(trail_name)
                   end
               end
        ))
    end

    @assert isa(def[:body].args[1], LineNumberNode)
    def[:body].args[1] = __source__

    return quote
        Base.@__doc__ $(esc(combinedef(def)))
    end
end

end # module

using .Breadcrumbs
