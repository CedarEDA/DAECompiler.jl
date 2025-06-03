"""
# DAECompiler.Intrinsics

This module defines the functions that make up the DAECompiler IR. There are a number of
primitive functions that form the basis of the IR, as well as a number of helper functions
that may be used when writing DAECompiler code through the Julia frontend. The basic IR
functions are:

- [`variable`](@ref): Introduces a variable into the system of equations.
- [`equation`](@ref): Constructs an equation of a particular kind
- [`(::equation)(...)`](@ref): Adds a term to an equation
- [`ddt`](@ref): Takes the total derivative of an expression with respect to the independent variable
- [`ett!`](@ref): Declares an edge-triggered transition
- [`pre`](@ref): Returns the left-limit of a variable at an event trigger
- [`threshold`](@ref): Detects zero-crossings of a continuous expression

"""
module Intrinsics

    import ..@defintrmethod
    export Scope, GenScope, continuous, discrete, epsilon, variable, equation, always, initial, initialguess, observed,
        always!, initial!, initialguess!, observed!, sim_time, ett!, pre, threshold, ddt

    abstract type AbstractScope; end

    ###################################### Variable Intrinsics ###################################
    # This section defines the variable intrinsic. There are several different kinds of variables,
    # each of which additionally gets a helper function that introduces that particular variable.
    # However, only `variable` is a recognized primitive of the compiler.

    @enum VarKind begin
        Continuous=1
        Discrete
        Epsilon
        RemovedVar
    end

    """
        variable(kind, scope)

    This intrinsic introduces a variable into the system of equations.
    Optionally takes a `scope` that may be used to symbolically refer to this variable
    in solution objects. The passed `scope` never affects the semantics of the system.
    If two variables are given the same `scope`, they remain semantically separate
    (but a warning is generated and they cannot be accessed symbolically in solution objects).

    There are several kinds of variables that can be introduced. Each has their own helper
    function. Currently the following kinds of variables are supported:

        - [`continuous`](@ref)
        - [`discrete`](@ref)
        - [`epsilon`](@ref)
    """
    function variable end

    """
        continuous([scope])

    Adds a continuous variable to the system. Continous variables are functions of time
    and are evolved by the integrator in between discrete events.

    Continuous variables are not guaranteed to become part "states", i.e. of the integrator state
    given to the numerical integrator. DAECompiler will automatically generate appropriate
    accessors to re-constitute eliminated variables from the integrator state using the
    equations of the system.
    """
    @inline continuous(scope) = variable(Continuous, scope)

    """
        discrete([scope])

    Adds a discrete variable to the system. Discrete variables are piecewise constant, but
    may be changed by discrete events using the `ett!` intrinsic.
    """
    @inline discrete(scope) = variable(Continuous, scope)

    """
        epsilon([name])

    This intrinsic introduces an epsilon variable into the system. epsilons are semantically 0.0,
    but the user may ask the system for the partial derivative with respect to any epsilon.
    """
    @inline epsilon(scope) = variable(Epsilon, scope)

    ###################################### Equation Intrinsics ###################################
    # This section defines the variable intrinsic. There are several different kinds of variables,
    # each of which additionally gets a helper function that introduces that particular variable.
    # However, only `variable` is a recognized primitive of the compiler.

    @enum EqKind begin
        Always=1
        Initial
        InitialGuess
        Observed
        RemovedEq
    end

    #=
    """
        equation

    An opaque struct that represents the identity of an equation. Calling an equation
    additively adds the called term to the particular equation.
    """
    =#
    mutable struct equation
        """
            equation(kind, scope)

        Introduce a new equation of the specified `kind` into the system.
        """
        @defintrmethod eqdef_method (@noinline equation(kind::EqKind, ::AbstractScope) = new())
    end

    """
        (eq::equation)(val)

    This intrinsic adds a term `val` to an equation `eq`.
    """
    (eq::equation)(val)

    # Equation kinds
    """
        always(scope)

    The `always` equation kind declares an equation that must be `≈ 0.0` at all time points.
    """
    @inline always(scope) = equation(Always, scope)

    """
        initial(scope)

    The `initial` equation kind declares an equation that must be `≈ 0.0` at system intialization.
    """
    @inline initial(scope) = equation(Initial, scope)

    """
        initial(scope)

    The `observed` equation kind is not semantically part of the system, but may be queried on the solution object.
    """
    @inline observed(scope) = equation(Observed, scope)

    ###################################### Scope Intrinsics ###################################
    # This section defines `Scope` and `GenScope` which are used to create a hierarchy of scopes.
    # The actual name of the scope is some abstract symbol type. In DAECompiler tests, this is generally
    # `Symbol`, but in general, these should be frontend identifiers that can map back to what the
    # users expect to work with in the interface.

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

    ###################################### Utility Intrinsics ###################################
    # This section defines `Scope` and `GenScope` which are used to create a hierarchy of scopes.
    # The actual name of the scope is some abstract symbol type. In DAECompiler tests, this is generally
    # `Symbol`, but in general, these should be frontend identifiers that can map back to what the
    # users expect to work with in the interface.
    """
        sim_time()

    This instrinsic represents the time variable.
    """
    function sim_time end

    """
        ddt(x)

    This intrinsic takes the total derivative of its argument with respect to the systems
    independent variable (i.e. `sim_time()`). There are no special semantic restrictions
    on `x`, although any expression that contributes to it must be AD-able (e.g. have registered
    ChainRules).
    """
    function ddt end

    """
        ett!(on_change, assign, val)

    Declare an edge-triggered transision. When `on_change` (in the discrete domain)
    changes value (under egality), the discrete variable `assign` is set to `val`.
    The change to `assign` may trigger further edge-triggered transitions. It is
    an error for multiple transitions to trigger simultaneously, unless one such
    transition strictly dominates the other.

    Note: `assign` must be a bare discrete variable (unlike ordinary continuous equations
    where the assignment is inferred or non-linearly solved).
    """
    function ett! end

    """
        pre(x)

    When evalutated in the continuous domain, this intrinsic is equivalent to
    the identity function. When evaluated at event triggers, this function returns
    the value of `x` using the left-limit of all discrete and continuous algebraic
    variables.
    """
    function pre end

    """
        threshold(x)

    Detects zero-crossing of the continuous-domain expression `x`. Returns `true`
    when `x` is about to be `> 0` and false otherwise, generating an edge-trigger
    event at the crossing. Note that the actual value of `x` at the event-trigger
    may have a small difference from `0` (in either direction) according to the
    solver tolerance.
    """
    function threshold end

    ################################### Magic Methods ###################################
    # In this section, we define the methods of the various intrinsics above. It is these
    # methods and only these methods that are recognized by the DAECompiler.

    # These global variables are inserted for the compiler for variables which should be unused
    # but were not IR-eliminable (e.g. due to lack of inlining). The idea with making these globals
    # is that they can be tweaked and if the computed value changes, DAECompiler has a bug somewhere.
    global _VARIABLE_UNAVAILABLE::Float64 = -42.0
    global _VARIABLE_UNASSIGNED::Float64 = -84.0
    global _DIFF_UNUSED::Float64 = -106.0
    global _UNKNOWN_ZERO::Float64 = 0.0

    @defintrmethod variable_method @noinline Base.@assume_effects :nothrow :effect_free :terminates_globally function variable(kind::VarKind, scope::AbstractScope)
        Base.donotdelete(scope)
        Base.inferencebarrier(_VARIABLE_UNAVAILABLE)::Float64
    end
    const m_variable = first(methods(variable))

    """
    An invalid equation that is used as a placeholder in codegen to replace `equation()` calls,
    while retaining the type.
    """
    const placeholder_equation = equation(Always, root_scope)

    const allow_intrinsics = Base.ScopedValues.ScopedValue(true)
    function check_bad_runtime_intrinsic()
        if !allow_intrinsics[]
            error("Unprocessed DAECompiler intrinsic detected")
        end
    end

    @defintrmethod equation_method @noinline function (eq::equation)(val)
        if eq === placeholder_equation
            error("Internal ERROR: Equation was replaced, but equation call was not.")
        end
        Base.donotdelete(val)
        return nothing
    end
    const m_eq_intro = first(methods(equation))
    const m_eq_eval = first(methods(placeholder_equation))

    @defintrmethod sim_time_method @noinline Base.@assume_effects :nothrow :effect_free :terminates_globally function sim_time()
        Base.inferencebarrier(check_bad_runtime_intrinsic)()
        return Base.inferencebarrier(0.0)::Float64
    end

    @defintrmethod ddt_method @noinline Base.@assume_effects :nothrow :effect_free :terminates_globally function ddt(x::Number)
        Base.inferencebarrier(check_bad_runtime_intrinsic)()
        return _UNKNOWN_ZERO
    end

    @defintrmethod ett!_method @noinline function ett!(on_change, assign, val)
        Base.donotdelete(on_change)
        Base.donotdelete(assign)
        Base.donotdelete(val)
        return nothing
    end

    @defintrmethod pre_method @noinline Base.@assume_effects :nothrow :effect_free :terminates_globally function pre(x)
        Base.donotdelete(x)
        return Base.compilerbarrier(:const, x)
    end

    @defintrmethod threshold_method @noinline Base.@assume_effects :nothrow :effect_free :terminates_globally function threshold(x)
        Base.donotdelete(x)
        return Base.compilerbarrier(:const, false)
    end

    ###################################### Helpers ###################################
    # These are not themselves intrinsics, but are just ordinary julia functions that call
    # the intrinsics, but that can be used to make DAECompiler code slightly easier to write.

    continuous(name::Symbol) = continuous(Scope(root_scope, name))
    @inline continuous() = continuous(root_scope)
    epsilon(name::Symbol) = epsilon(Scope(root_scope, name))
    epsilon() = epsilon(root_scope)
    discrete(name::Symbol) = discrete(Scope(root_scope, name))
    discrete() = discrete(root_scope)

    always() = equation(Always, root_scope)
    always!(val, scope::AbstractScope=root_scope) = always(scope)(val)
    always!(val::Number, name::Symbol) = always!(val, Scope(root_scope, name))
    initial() = initial(root_scope)
    initial!(val, scope::AbstractScope=root_scope) = initial(scope)(val)
    initial!(val::Number, name::Symbol) = initial!(val, Scope(root_scope, name))
    observed() = equation(root_scope)
    observed!(val, scope::AbstractScope=root_scope) = observed(scope)(val)
    observed!(val::Number, name::Symbol) = observed!(val, Scope(root_scope, name))
end

using .Intrinsics

module InternalIntrinsics
    export solved_variable, state, contribution!
    export StateKind, AssignedDiff, UnassignedDiff, AlgebraicDerivative, Algebraic, LastStateKind
    export EquationStateKind, StateDiff, Explicit, LastEquationStateKind

    """
        StateKind

    DAECompiler's internal ABI supports being called either from a SciML DAE or ODE (with mass matrix) ABI.
    To support this, we assign all variables that will become states to one of our kinds:

    - `AssignedDiff`:   A differential variable for which we have a variable-equation assignment for its derivative.
                        In ODE form, this variable will have a slot and the corresponding `du` will be set to the residual
                        of the assigned equation.
                        In DAE form, we will generate an implicit `du[i] - red` in the corresponding equation slot.

    - `UnassignedDiff`: A differential variable for which we do NOT have a variable-equation assignment for its derivative.
                        The derivative of this variable will be an algebraic variable of StateKind `AlgebraicDerivative`.
                        In ODE form, the corresponding `du` entry will be set to this algebraic variable, while the (null row
                        mass matrix) entry corresponding to the derivative will be used for one of the remaining unassigned equations.
                        In DAE form, the corresponding `du` entry will be directly used for the algebraic variable.

    - `AlgebraicDerivative`:
                        An algebraic variable that is the derivative of a differential variable. In ODE form, these are appended to the `u`
                        array (like ordinary `Algebraic` variables). In DAE form, these are appended to the `du` array as described above.

    - `Algebraic`:      An algebraic variable that is not the derivative of a differential variable. In ODE form, these are appended to the
                        `u` array (with the correponding mass matrix row being `0`). In DAE form, these are also appended to the `u` array,
                        with the corresponding entry of `differential_vars` being `false`.

    In order to support both SciML ABIs, the DAECompiler ABI, passes separate views for each of these kinds of variables. The ordering in this
    enum corresponds to the ordering in the arguments of the DAECompiler internal ABI.
    """
    @enum StateKind begin
        AssignedDiff=1
        UnassignedDiff
        AlgebraicDerivative
        Algebraic
    end
    const LastStateKind = Algebraic
    Base.to_index(kind::StateKind) = Int(kind)

    """
        EquationStateKind

    Extends `StateKinds` to include the output states of equations. The two equation state kinds are:

    - `StateDiff`:      This is the output state that corresponds to an `AssignedDiff` variable above. In ODE form this is a mass-matrix with non-zero entry.
                        In DAE form, we reserve a slot in the `out` array for the explicit residual and the toplevel adapter to DAE form then manually applies
                        the mass matrix.
    - `Explicit`:       This is the output state corresponds to an unassigned equation. In ODE form, this corresponds to a mass matrix with zero row. In DAE form,
                        it is any entry of the output array.
    """
    @enum EquationStateKind begin
        StateDiff=Int(LastStateKind)+1
        Explicit
    end
    const LastEquationStateKind = Explicit
    Base.to_index(kind::EquationStateKind) = Int(kind)

    ############################### Internal Intrinsics ###################################
    # These internal intrinsics are not part of the DAECompiler intrinsic set, but get
    # inserted as markers by various optimization passes

    """
        state(slot::Int, kind::StateKind)

    Read the current time-step value of the specified state `slot` for the specfified
    `kind`. Will be replaced by an appropriate array reference during final codegen.
    """
    @noinline function state(slot::Int, kind::StateKind)
        Base.inferencebarrier(error)("Internal placeholder left in final code")
        return nothing
    end

    """
        solved_variable(var::Int, val)

    Part of the post-scheduled IR representation. This intrinsic is used to mark that the
    value with id `var` has value `val`. This is used if codegen passes need to compute the
    value of a particular variable directly. For any dependent variables within the IR, the
    reference to the variable will have already been replaced by `val`.
    """
    @noinline function solved_variable(var::Int, val)
        Base.inferencebarrier(error)("Internal placeholder left in final code")
        return nothing
    end

    """
        assign_var(var, val)

    Generally equivalent to `val - var`, except that `var` and val need not be of Float64 type.
    Used to rewrite complicated non-linear expressions to give them a numbered identity.
    """
    @noinline function assign_var(var, val)
        Base.inferencebarrier(error)("Internal placeholder left in final code")
        return val
    end


    """
        contribution!(slot::Int, kind::EquationStateKind, val)

    Add `val` to the residual `slot` of the specified `kind`. Will be replaced by an appropriate
    array `+=` operation during final codegen.
    """
    @noinline function contribution!(slot::Int, kind::EquationStateKind, val)
        Base.inferencebarrier(error)("Internal placeholder left in final code")
        return nothing
    end
end
using .InternalIntrinsics

struct DynamicStateError; end
function Base.showerror(io::IO, err::DynamicStateError)
    print(io, """
        DynamicStateError: An variable or equation was found to have state-dependent
        control dependence. This is currently not supported. If you believe that your
        control flow is not state-dependent, please file a bug.
    """)
end
