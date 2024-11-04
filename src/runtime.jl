module Intrinsics
    export variable, variables, equation, equation!, observed!,
        singularity_root!, time_periodic_singularity!, zero_crossings!, sim_time, Scope,
        Scope, GenScope, ddt, epsilon

    """
        variable([scope])

    This intrinsic introduces a variable into the system of equations.
    Optionally takes a `scope` that may be used to symbolically refer to this variable
    in solution objects. The passed `scope` never affects the semantics of the system.
    If two variables are given the same `scope`, they remain semantically separate
    (but a warning is generated and they cannot be accessed symbolically in solution objects).

    Variables are not guaranteed to become part "states", i.e. of the integrator state
    given to the numerical integrator. DAECompiler will automatically generate appropriate
    accessors to re-constitute eliminated variables from the integrator state using the
    equations of the system.
    """
    function variable end

    """
        epsilon([name])

    This intrinsic introduces an epsilon into the system. epsilons are semantically 0.0,
    but the user may ask the system for the partial derivative with respect to any epsilon.
    """
    function epsilon end

    """
        equation!(val, [scope])

    This intrinsic marks an equation. The system of equations is considered satisfied when all
    `val`s passed to all `equation`s are equal to zero (up to tolerance).

    Like all intrinsics, the `scope` argument is non-semantic and two `equation!` calls with
    the same `scope` will introduce separate equations. To build up one equation from multiple
    pieces, see the lower-level `equation` (no `!`) intrinsic. Note that `equation!(val, scope)`
    is equivalent to and implemented as `equation(scope)(val)`. It is provided for convenience.
    """
    function equation! end

    """
        sim_time()

    This instrinsic represents the time variable.
    """
    function sim_time end

    """
        state_ddt(val::Number)

    This instrinsic represents the derivative with respect to the time variable.

    !!! Warning
        This intrinsic is for internal use only and is not exported. Users should
        use `ddt` instead.
    """
    function state_ddt end

    """
        singularity_root!(val::Number)

    This instrinsic represents a discontinuity in val.
    """
    function singularity_root! end

    """
        time_periodic_singularity!(offset::Number, period::Number, count::Number)

    This intrinsic represents a singularity periodic in `t`.  It first occurs at `offset`,
    then occurs again once every `period`, for `count` repetitions.
    """
    function time_periodic_singularity! end

    abstract type AbstractScope; end

    struct Scope{T} <: AbstractScope
        parent::AbstractScope
        name::T
        Scope() = new{Union{}}()
        Scope(s::AbstractScope, name::T) where {T} = new{T}(s, name)
    end
    (scope::Scope)(s::Symbol) = Scope(scope, s)
    # Scope(), but with less function calls, so marginally easier on the compiler
    const root_scope = Scope()

    mutable struct ScopeIdentity; end

    struct GenScope{T} <: AbstractScope
        identity::ScopeIdentity
        sc::Scope{T}
        GenScope(sc::Scope{T}) where {T} = new{T}(ScopeIdentity(), sc)
    end
    GenScope(parent::AbstractScope, name::Symbol) =
        GenScope(Scope(parent, name))
    (scope::GenScope)(s::Symbol) = Scope(scope, s)

    @enum EqKind begin
        DiffAlg
        Obs
        ExternalEq
    end

    """
        equation

    An opaque struct that represents the identity of an equation. Calling an equation
    additively adds the called term to the particular equation.
    """
    mutable struct equation
        @noinline equation(::AbstractScope, ::EqKind) = new()
    end

    """
    An invalid equation that is used as a placeholder in codegen to replace `equation()` calls,
    while retaining the type.
    """
    const placeholder_equation = equation(root_scope, DiffAlg)

    @noinline function (eq::equation)(val)
        if eq === placeholder_equation
            error("Internal ERROR: Equation was replaced, but equation call was not.")
        end
        Base.donotdelete(val)
    end

    """
        ddt(x)

    This intrinsic takes the total derivative of its argument with respect to the systems
    independent variable (i.e. `sim_time()`). There are no special semantic restrictions
    on `x`, although any expression that contributes to it must be AD-able (e.g. have registered
    ChainRules).
    """
    function ddt end

    function check_bad_runtime_intrinsic()
        if !haskey(Base.task_local_storage(), :daecompiler_intrinsics_ok)
	    # TODO: Find a way to turn this on and off automatically in the appropriate
	    # circumstances.
            # error("Unprocessed DAECompiler intrinsic detected")
        end
    end

    @noinline function solved_variable(var::Int, val)
        Base.inferencebarrier(error)("Internal placeholder left in final code")
        return nothing
    end

    export StateKind, AssignedDiff, UnassignedDiff, AlgebraicDerivative, Algebraic
    @enum StateKind begin
        AssignedDiff=1
        UnassignedDiff
        AlgebraicDerivative
        Algebraic
    end
    const LastStateKind = Algebraic
    Base.to_index(kind::StateKind) = Int(kind)

    @noinline function state(slot::Int, kind::StateKind)
        Base.inferencebarrier(error)("Internal placeholder left in final code")
        return nothing
    end

    export EquationKind, StateDiff, Explicit, LastEquationKind
    @enum EquationKind begin
        StateDiff=Int(LastStateKind)+1
        Explicit
    end
    const LastEquationKind = Explicit
    Base.to_index(kind::EquationKind) = Int(kind)

    @noinline function contribution(slot::Int, kind::EquationKind, val)
        Base.inferencebarrier(error)("Internal placeholder left in final code")
        return nothing
    end

    # These will get replaced in the optimization pass
    @noinline Base.@assume_effects :nothrow :effect_free :terminates_globally function variable(name::AbstractScope)
        Base.inferencebarrier(check_bad_runtime_intrinsic)()
        Base.inferencebarrier(14.0)::Float64
    end
    @inline variable() = variable(root_scope)
    @noinline Base.@assume_effects :nothrow :effect_free :terminates_globally function epsilon(name::AbstractScope)
        Base.inferencebarrier(check_bad_runtime_intrinsic)()
        Base.inferencebarrier(0.0)::Float64
    end
    @inline epsilon() = epsilon(root_scope)

    @noinline Base.@assume_effects :nothrow :terminates_globally function singularity_root!(val)
        Base.inferencebarrier(check_bad_runtime_intrinsic)()
        Base.donotdelete(val)
    end
    @noinline Base.@assume_effects :nothrow :terminates_globally function time_periodic_singularity!(offset, period, count)
        Base.inferencebarrier(check_bad_runtime_intrinsic)()
        Base.donotdelete(offset)
        Base.donotdelete(period)
        Base.donotdelete(count)
    end
    @noinline Base.@assume_effects :nothrow :effect_free :terminates_globally function state_ddt(val::Number)
        Base.inferencebarrier(check_bad_runtime_intrinsic)()
        Base.inferencebarrier(0.0)::Float64
    end
    @noinline Base.@assume_effects :nothrow :effect_free :terminates_globally function sim_time()
        Base.inferencebarrier(check_bad_runtime_intrinsic)()
        Base.inferencebarrier(0.0)::Float64
    end
    @noinline Base.@assume_effects :nothrow :effect_free :terminates_globally function ddt(x::Number)
        Base.inferencebarrier(check_bad_runtime_intrinsic)()
        return _UNKNOWN_ZERO
    end

    # Helpers (not one of the intrinsics)
    equation() = equation(root_scope)
    equation(scope::AbstractScope) = equation(scope, DiffAlg)
    equation!(val, scope::AbstractScope=root_scope) = equation(scope)(val)
    variable(name::Symbol) = variable(Scope(root_scope, name))
    epsilon(name::Symbol) = epsilon(Scope(root_scope, name))
    equation!(val::Number, name::Symbol) = equation!(val, Scope(root_scope, name))

    observed() = observed(root_scope)
    observed(scope::AbstractScope) = equation(scope, Obs)

    """
        observed!(val::Number, [name])

    This intrinsic creates an observable which is a quantity that isn't evolved with the system
    but which the user just wants to be able to query from the states.
    """
    observed!(val, scope::AbstractScope=root_scope) = observed(scope)(val)
    observed!(val::Number, name::Symbol) = observed!(val, Scope(root_scope, name))

    # These global variables are inserted for the compiler for variables which should be unused
    # but were not IR-eliminable (e.g. due to lack of inlining). The idea with making these globals
    # is that they can be tweaked and if the computed value changes, DAECompiler has a bug somewhere.
    global _VARIABLE_UNAVAILABLE::Float64 = -42.0
    global _VARIABLE_UNASSIGNED::Float64 = -84.0
    global _DIFF_UNUSED::Float64 = -106.0
    global _UNKNOWN_ZERO::Float64 = 0.0
    global const _EQ_UNUSED   = equation()

    """
        variables([scope])

    This is a utility wrapper for `variable` that allows introducing new variables automatically on
    the first `getproperty` access. This is useful for avoiding repeatition of variable names for
    debug purposes.

    # Example

    Instead of manually writing:
    ```
        function f()
            x = variable(:x)
            y = variable(:y)
            z = variable(:z)
            ...
        end
    ```

    you may write
    ```
    function f()
        (; x, y, z) = variables()
    end
    ```

    Or
    ```
    var = variables()
    ```
    and then use `var.x`, `var.y` etc as needed in your code, knowing that each use of `var,.x` will refer to the same variable.
    """
    mutable struct variables{VT}
        const scope::Scope
        nt::NamedTuple{names, T} where {N, names, T<:NTuple{N, VT}}
        variables(s::Scope) = new{Core.Compiler._return_type(variable, Tuple{})}(s, (;))
    end
    variables() = variables(root_scope)
    variables(s::Symbol) = variables(Scope(root_scope, s))

    @noinline _compute_new_nt_type(nt::NamedTuple{names}, s::Symbol) where {names} = NamedTuple{tuple(names::Tuple{Vararg{Symbol}}..., s)}

    Base.setproperty!(mv::variables, s::Symbol, @nospecialize(v)) = Base.setfield!(mv, s, v)
    @eval @inline function Base.getproperty(mv::variables, s::Symbol)
        nt = getfield(mv, :nt)
        if hasfield(typeof(nt), s)
            return getfield(nt, s)
        else
            v = variable(Scope(getfield(mv, :scope), s))
            # This is functionally equivalent to
            # `mv.nt = merge(mv.net, (; s=>v))`, but carefully avoiding compiler limits
            NT = _compute_new_nt_type(nt, s)
            tup = tuple(nt..., v)
            NT = NT{typeof(tup)}
            mv.nt = $(Expr(:splatnew, :NT, :tup))
            return v
        end
    end
end

using .Intrinsics

struct DynamicStateError; end
function Base.showerror(io::IO, err::DynamicStateError)
    print(io, """
        DynamicStateError: An variable or equation was found to have state-dependent
        control dependence. This is currently not supported. If you believe that your
        control flow is not state-dependent, please file a bug.
    """)
end
