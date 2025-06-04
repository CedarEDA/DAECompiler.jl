############################### lattice.jl #####################################

using SparseArrays

########################## EqStructureLattice ####################################
"""
    struct EqStructureLattice <: Compiler.AbstractLattice

This lattice implements the `AbstractLattice` interface. It adjoins `Incidence` and `Eq`.

The EqStructureLattice is one of the key places where DAECompiler extends `Compiler`.
In compiler parlance, the EqStructureLattice type system can be
thought of as a taint analysis sourced at `variable` (with sinks at `variable`,
`equation`, etc.). In dynamical systems parlance, one might consider this
analysis to compute a structural jacobian.

One particular feature of this lattice, is the control-dependent taint of phi
nodes. For example, in
```
top:
    br @a if not %cond

@a:
    ...
    goto @join
@b:
    ...
    goto @join

@join:
    %phi = ϕ(%a, %b)
```
the taint of %phi depends not only on `%a` and `%b`, but also on the taint of
the branch condition `%cond`. This is a common feature of taint analysis, but
is somewhat unusual from the perspective of other Julia type lattices.
"""
struct EqStructureLattice <: Compiler.AbstractLattice; end
Compiler.widenlattice(::EqStructureLattice) = Compiler.ConstsLattice()
Compiler.is_valid_lattice_norec(::EqStructureLattice, @nospecialize(v)) = isa(v, Incidence) || isa(v, Eq) || isa(v, PartialScope) || isa(v, PartialKeyValue)
Compiler.has_extended_unionsplit(::EqStructureLattice) = true

############################## Linearity #######################################

"""
    Base.@kwdef struct Linearity
        time_dependent::Bool = true
        state_dependent::Bool = true
        nonlinear::Bool = true
    end

Together with `Float64` values in `Incidence`, this struct expresses linearity information:
- Linear, coefficient is a known constant (a `Float64` value).
- Linear, coefficient is an unknown constant (`linear`).
- Linear, coefficient is unknown with a known state or time dependence (`linear_state_dependent`, `linear_time_dependent`).
- Linear, coefficient is unknown with both state and time dependence (`linear_time_and_state_dependent`).
- Nonlinear (`nonlinear` - always taints state and time dependence currently).

A known constant-coefficient contains the highest level of information. `nonlinear` contains the lowest
(and most conservative) level of information, and should be assumed used by default or if uncertain.
"""
Base.@kwdef struct Linearity
    time_dependent::Bool = true
    state_dependent::Bool = true
    nonlinear::Bool = true
    function Linearity(time_dependent, state_dependent, nonlinear)
        if nonlinear && (!time_dependent || !state_dependent)
            throw(ArgumentError("Modeling of state or time independence is not supported for nonlinearities"))
        end
        new(time_dependent, state_dependent, nonlinear)
    end
end

"The variable is used linearly, with an unknown constant."
const linear = Linearity(time_dependent = false, state_dependent = false, nonlinear = false)
"The variable is used linearly, with an unknown constant that may depend on time."
const linear_time_dependent = Linearity(state_dependent = false, nonlinear = false)
"The variable is used linearly, with an unknown constant that may depend on time and on other states."
const linear_state_dependent = Linearity(time_dependent = false, nonlinear = false)
"The variable is used linearly, with an unknown constant that may depend on time and on other states."
const linear_time_and_state_dependent = Linearity(nonlinear = false)
"The variable is used nonlinearly, with a possible dependence on time and other states."
const nonlinear = Linearity()

join_linearity(a::Linearity, b::Real) = a
join_linearity(a::Real, b::Linearity) = b
join_linearity(a::Real, b::Real) = a == b ? a : linear
function join_linearity(a::Linearity, b::Linearity)
    (a.nonlinear | b.nonlinear) && return nonlinear
    return Linearity(; time_dependent = a.time_dependent | b.time_dependent, state_dependent = a.state_dependent | b.state_dependent, nonlinear = false)
end

Base.iszero(::Linearity) = false
Base.zero(::Type{Union{Linearity, Float64}}) = 0.
for f in (:+, :-)
    @eval begin
        Base.$f(a::Real, b::Linearity) = b
        Base.$f(a::Linearity, b::Real) = a
        Base.$f(a::Linearity, b::Linearity) = join_linearity(a, b)
        Base.$f(a::Linearity) = a
    end
end

Base.:(*)(a::Real, b::Linearity) = iszero(a) ? a : b
Base.:(*)(a::Linearity, b::Real) = iszero(b) ? b : a
Base.div(a::Real, b::Linearity) = iszero(a) ? a : nonlinear
Base.div(a::Linearity, b::Real) = a
Base.div(a::Linearity, b::Linearity) = nonlinear
Base.:(/)(a::Real, b::Linearity) = iszero(a) ? a : nonlinear
Base.:(/)(a::Linearity, b::Real) = a
Base.:(/)(a::Linearity, b::Linearity) = nonlinear
Base.rem(a::Real, b::Linearity) = b
Base.rem(a::Linearity, b::Real) = a
Base.rem(a::Linearity, b::Linearity) = a
Base.abs(a::Linearity) = nonlinear
Base.isone(a::Linearity) = false

Base.Broadcast.broadcastable(x::Linearity) = Ref(x)

############################## Incidence #######################################
# TODO: Just use Infinities.jl here?
const MAX_EQS = div(typemax(Int), 2)

const IncidenceValue = Union{Float64, Linearity}
const IncidenceVector = SparseVector{IncidenceValue, Int}

is_non_incidence_type(@nospecialize(type)) = type === Union{} || Base.issingletontype(type)

"""
    struct Incidence

An element of the `EqStructureLattice` that sits between `Const` and `Type` in the
lattice hierarchy. In particular, `Const(::T) ⊑ Incidence(T, {...}) ⊑ T`, where
`{...}` is a set of incidence variables. Lattice operations among the `Incidence`
elements are defined by subset inclusion. Note that in particular this implies
that plain `T` lattice elements have unknown incidence and `Const` lattice elements
have no incidence. A lattice element of type `T` that is known to be state-independent
would have lattice element `Incidence(T, {})`.

For convenience, you may want to use the `incidence"..."` string macro to construct an
[`Incidence`](@ref) from its printed representation.
"""
struct Incidence
    # Considered additive to `row`. In particular, if the `typ` is Float64,
    # the linear combination in `row` will be inhomogeneous.
    typ::Union{Type, Const}
    row::IncidenceVector

    function Incidence(@nospecialize(type), row::AbstractVector)
        if is_non_incidence_type(type)
            throw(DomainError(type, "Invalid type for Incidence"))
        end
        if !isa(row, SparseVector)
            vec, row = row, _zero_row()
            for (i, val) in enumerate(vec)
                row[i] = val
            end
        else
            row = convert(IncidenceVector, row)
        end
        time = row[1]
        if in(time, (linear_time_dependent, linear_time_and_state_dependent))
            throw(ArgumentError("Time incidence cannot be both linear and time-dependent, otherwise it would be nonlinear"))
        end
        for (i, coeff) in zip(rowvals(row), nonzeros(row))
            isa(coeff, Linearity) || continue
            coeff.nonlinear && continue
            if coeff.time_dependent && !in(1, rowvals(row))
                throw(ArgumentError("Time-dependent incidence annotation for $(subscript_state(i)) is inconsistent with an absence of time incidence"))
            end
            if coeff.state_dependent && !any(x -> x != 1, rowvals(row))
                throw(ArgumentError("State-dependent incidence annotation for $(subscript_state(i)) is inconsistent with an absence of state incidence"))
            end
            if i > 1 && coeff.time_dependent && (isa(time, Float64) || !time.state_dependent)
                throw(ArgumentError("Time-dependent state incidence for $(subscript_state(i)) is inconsistent with an absence of state dependence for time"))
            end
        end
        return new(type, row)
    end
end
Incidence(row::IncidenceVector) = Incidence(Float64, row)

function subscript(n::Integer)
    @assert n >= 0
    map(c->Char('₀' + (c - '0')), repr(n))
end

function subscript_state(i)
    if i == 1
        return "t"
    else
        return string("u", subscript(i-1))
    end
end

function Base.show(io::IO, inc::Incidence)
    print(io, "Incidence(")
    first = true
    print_plusminus(io, minus=false) = if first
        first = false
        minus && print(io, "-")
    else
        print(io, minus ? " - " : " + ")
    end
    if isa(inc.typ, Const) && isa(inc.typ.val, Float64)
        if !iszero(inc.typ.val)
            print(io, inc.typ.val)
            first = false
        end
    else
        if inc.typ === Float64
            print(io, 'a')
            !isempty(rowvals(inc.row)) && print_plusminus(io)
        else
            print(io, inc.typ)
            !isempty(rowvals(inc.row)) && print(io, ", ")
        end
    end
    time = inc.row[1]
    is_grouped(v, i) = isa(v, Linearity) && (v.state_dependent || (v.time_dependent || i == 1) && in(time, (linear_state_dependent, nonlinear)))
    function propto(linearity::Linearity)
        str = "∝"
        linearity.time_dependent && (str *= 'ₜ')
        linearity.state_dependent && (str *= 'ₛ')
        return str
    end
    for (i, v) in zip(rowvals(inc.row), nonzeros(inc.row))
        is_grouped(v, i) && continue
        if isa(v, Float64)
            print_plusminus(io, v < 0)
            if abs(v) != 1
                print(io, abs(v))
            end
        else
            !first && print(io, " + ")
            first = false
            if is_grouped(inc.row[1], 1) && v.time_dependent
                print(io, propto(inc.row[1]::Linearity), 't', " * ")
            else # unknown constant coefficient
                print(io, propto(v))
            end
        end
        print(io, subscript_state(i))
    end
    first_grouped = true
    if any(i -> is_grouped(inc.row[i], i), rowvals(inc.row))
        for (i, v) in zip(rowvals(inc.row), nonzeros(inc.row))
            !is_grouped(v, i) && continue
            if first_grouped
                print_plusminus(io)
                first_grouped = false
                print(io, "f(")
            else
                print(io, ", ")
            end
            !v.nonlinear && print(io, propto(v))
            print(io, subscript_state(i))
        end
        if !first_grouped
            print(io, ")")
        end
    end
    print(io, ")")
end

"""
    incidence"a + f(∝ₛt, u₁)"
    incidence"Incidence(a + f(∝ₛt, u₁))" # you may copy-paste straight from its printed output

Construct an [`Incidence`](@ref) from its printed representation.
"""
macro incidence_str(str) generate_incidence(str) end

function generate_incidence(str::String)
    if startswith(str, "Incidence(") && endswith(str, ')')
        # Support `incidence"Incidence(...)"` so the user doesn't have to
        # manually remove the `Incidence` call when copy-pasting.
        str = str[11:(end - 1)]
    end
    str = replace(str, '∝' => '~')
    ex = Meta.parse(str)
    generate_incidence(ex)
end

function generate_incidence(ex)
    T = nothing
    if isexpr(ex, :tuple, 2)
        T, ex = ex.args[1], ex.args[2]
    end
    generate_incidence(T, ex)
end

function generate_incidence(T, ex)
    if isexpr(ex, :call) && ex.args[1] === :+
        terms = ex.args[2:end]
    else
        terms = Any[ex]
    end
    pairs = Dict{Int,Any}()
    for term in terms
        if term === :a
            T === nothing || throw(ArgumentError("The incidence type must not be provided if a constant `Float64` term is already present"))
            T = Float64
            continue
        elseif isa(term, Float64)
            T === nothing || throw(ArgumentError("The incidence type must not be provided if a literal `Float64` term is already present"))
            T = Const(term)
            continue
        end

        @assert isa(term, Symbol) || isexpr(term, :call)

        ispropto(x) = isexpr(x, :call, 2) && startswith(string(x.args[1]), '~')

        if isa(term, Symbol)
            i = parse_variable(string(term))
            pairs[i] = 1.0
        elseif isexpr(term, :call, 3) && term.args[1] === :*
            factor = parse(Float64, string(term.args[2]))
            i = parse_variable(string(term.args[3]))
            pairs[i] = factor
        elseif ispropto(term)
            coefficient, variable = separate_coefficient_and_variable(term)
            coefficient = parse_coefficient(coefficient)
            i = parse_variable(variable)
            pairs[i] = coefficient
        elseif isexpr(term, :call) && term.args[1] === :f
            for argument in @view term.args[2:end]
                if ispropto(argument)
                    coefficient, variable = separate_coefficient_and_variable(argument)
                    coefficient = parse_coefficient(coefficient)
                    i = parse_variable(variable)
                else
                    i = parse_variable(string(argument))
                    coefficient = nonlinear
                end
                pairs[i] = coefficient
            end
        else
            throw(ArgumentError("Unrecognized call to function '$(term.args[1])'"))
        end
    end
    values = IncidenceValue[]
    for i in 1:maximum(keys(pairs); init = 0)
        val = get(pairs, i, 0.0)
        isa(val, Pair) && (val = val.second)
        push!(values, val)
    end
    T = something(T, Const(0.0))
    :(Incidence($T, IncidenceValue[$(values...)]))
end

function separate_coefficient_and_variable(term::Expr)
    str = string(term)
    i = findfirst(in(('t', 'u')), str)::Int
    @view(str[1:prevind(str, i)]), @view(str[i:end])
end

function parse_coefficient(coefficient::AbstractString)
    matched = match(r"^~(ₜ)?(ₛ)?$", coefficient)
    @assert matched !== nothing
    time_dependent = matched.captures[1] !== nothing
    state_dependent = matched.captures[2] !== nothing
    return Linearity(; time_dependent, state_dependent, nonlinear = false)
end

function parse_variable(term)
    term == "t" && return 1
    matched = match(r"^u([₀₁₂₃₄₅₆₇₈₉]+)$", term)
    @assert matched !== nothing
    capture = matched[1]
    return 1 + parse(Int, map(subscript_to_number, capture))
end

subscript_to_number(char) = Char(48 + (UInt32(char) - 8320))

_zero_row() = IncidenceVector(MAX_EQS, Int[], IncidenceValue[])
const _ZERO_ROW = _zero_row()
const _ZERO_CONST = Const(0.0)
Base.zero(::Type{Incidence}) = Incidence(_ZERO_CONST, _zero_row())
function Incidence(T::Union{Type, Compiler.Const} = Float64)
    vec = Incidence(T, _zero_row())
    vec
end

function Incidence(T::PartialStruct)
    PartialStruct(T.typ, Any[(isa(f, Const) || is_non_incidence_type(f)) ? f : Incidence(f) for f in T.fields])
end

Base.:(==)(a::Incidence, b::Incidence) = a.typ === b.typ &&
    a.row.nzind == b.row.nzind && a.row.nzval == b.row.nzval
Base.:(+)(a::Incidence, b::Incidence) = tfunc(Val{Core.Intrinsics.add_float}(), a, b)
Base.:(-)(a::Incidence, b::Incidence) = tfunc(Val{Core.Intrinsics.sub_float}(), a, b)
Base.:(+)(a::Const, b::Incidence) = tfunc(Val{Core.Intrinsics.add_float}(), a, b)
Base.:(+)(a::Incidence, b::Const) = tfunc(Val{Core.Intrinsics.add_float}(), a, b)

function Incidence(v::Int)
    row = _zero_row()
    row[v+1] = 1.0
    return Incidence(_ZERO_CONST, row)
end
function Incidence(T::Union{Type, Compiler.Const}, v::Int)
    T === Float64 && return Incidence(v)
    row = _zero_row()
    row[v+1] = nonlinear
    return Incidence(T, row)
end

"Identify the id number of an equation or variable"
idnum(a::Incidence) = only(findall(!iszero, a.row)) - 1  # Inverse of above constructor.

"Identify of an epsilon from the incidence"
function epsnum(a::Incidence)
    if length(a.eps) != 1
        error("Expected exactly one epsilon, but got $a")
    end
    return only(a.eps)
end

Base.copy(inc::Incidence) = Incidence(inc.typ, copy(inc.row))
Base.isempty(inc::Incidence) = iszero(inc.row) # slightly weaker than iszero as that also requires it to be Const(0.0)

#################################### Eq ########################################
struct Eq
    id::Int
end

idnum(x::Eq) = x.id

########################### Dependence Queries ################################
"returns true if the incidence of this data is known and can be directly processed by `has_dependence` etc"
has_simple_incidence_info(::Incidence) = true
has_simple_incidence_info(::Const) = true
has_simple_incidence_info(::Any) = false

"""
returns true if statement may depend on time or state.
NOTE: this ignores epsilon dependency, which is not normally relevant
To include that you must do `has_dependence(x) || has_epsilon_dependence(x)`
"""
function has_dependence(typ::Incidence)
    _, vals = findnz(typ.row)
    return any(!iszero, vals)  # includes vacuous truth
end
has_dependence(::Const) = false
has_dependence(ps::PartialStruct) = any(has_dependence, ps.fields)
has_dependence(::Type) = true
has_dependence(::Eq) = false


"""
returns true if statement may depend on time or state.
NOTE: this ignores epsilon dependency, which is not normally relevant
To include that you must do `has_dependence(x) || has_epsilon_dependence(x)`
"""
function has_dependence_other_than(typ::Incidence, allowed::BitSet)
    return any(x->!(x-1 in allowed), rowvals(typ.row))
end
has_dependence_other_than(::Const, allowed::BitSet) = false
has_dependence_other_than(ps::PartialStruct, allowed::BitSet) = any(f->has_dependence_other_than(f, allowed), ps.fields)
has_dependence_other_than(::Type, allowed::BitSet) = true
has_dependence_other_than(::Eq, allowed::BitSet) = false

"returns true if statement may depend on time"
function has_time_dependence(typ::Incidence)
    time_coef = typ.row[1]
    return !iszero(time_coef)
end
has_time_dependence(::Const) = false

"returns true if statement may depend on a epsilon"
function has_epsilon_dependence(typ::Incidence)
    return !isempty(typ.eps)
end
has_epsilon_dependence(::Const) = false


"returns true if and only if the statement dependes on time only"
function has_only_time_dependence(typ::Incidence)
    has_time_dependence(typ) || return false
    return iszero(typ.row[2:end])
end
has_only_time_dependence(::Const) = false

"returns true if statement may depend on any listed state identified by var_num"
function has_state_dependence(typ::Incidence, state_var_nums)
    for var_num in state_var_nums
        !iszero(typ.row[var_num+1]) && return true
    end
    return false
end
has_state_dependence(::Const, _) = false


################################ PartialScope ###################################
"""
    struct PartialScope

This lattice element is an abstraction of `Intrinsics.AbstractScope` and
represents that the scope is passed in either as an argument (for `id::Int`),
or in a `ScopedValue` (for `id::ScopedValue`). In the latter case, the point
of determination is function entry, so modification of the ScopedValue inside
the function does not affect the value.
"""
struct PartialScope
    id::Union{Int, Base.ScopedValues.ScopedValue}
end

idnum(x::PartialScope) = x.id

############################### PartialKeyValue ################################
"""
    struct PartialKeyValue

This lattice element enriches the contained lattice element `typ` with additional
information how the lattice element in question behaves under operations from
the `Base.OptimizedGenerics.KeyValue` interface. In particular, `KeyValue.get`
with a constant `key::Const` on a collection with this lattice element may be inferred
to `vals[key.val]` and accordinly `KeyValue.set` may produce this lattice element
when called with a constant key. `parent` chains to lattice element of the previous
collection, which may itself be a `PartialKeyValue`. It is not legal for `typ`
to be a `PartialKeyValue` and in general `typ` need not match `parent.typ` (
it may have been narrowed by a PiNode in the common case, but `KeyValue.set`
is also not required to be type-preserving).
"""
struct PartialKeyValue
    typ::Any
    parent::Any
    vals::IdDict{Any, Any}
end
PartialKeyValue(typ) = PartialKeyValue(typ, typ, IdDict{Any, Any}())

function getkeyvalue_tfunc(𝕃::Compiler.AbstractLattice,
        @nospecialize(collection), @nospecialize(key))
    isa(key, Const) || return Tuple{Any}
    if haskey(collection.vals, key.val)
        return Compiler.tuple_tfunc(𝕃, Any[collection.vals[key.val]])
    end
    error()
end

######################### AbstractLattice interface ############################

Compiler.widenconst(inc::Incidence) = widenconst(inc.typ)
Compiler.widenconst(::Eq) = equation
Compiler.widenconst(::PartialScope) = Scope
Compiler.widenconst(pkv::PartialKeyValue) = widenconst(pkv.typ)
Compiler.:⊑(inc::Incidence, inc2) = Compiler.:⊑(inc2, Float64) && !isa(inc2, Const)

function Compiler._uniontypes(x::Incidence, ts::Vector{Any})
    u = x.typ
    if isa(u, Union)
        Compiler.push!(ts, is_non_incidence_type(u.a) ? u.a : Incidence(u.a, x.row, x.eps))
        Compiler.push!(ts, is_non_incidence_type(u.b) ? u.b : Incidence(u.b, x.row, x.eps))
        return ts
    else
        Compiler.push!(ts, x)
        return ts
    end
end

function Compiler.widenlattice(🥬::EqStructureLattice, ps::Compiler.PartialStruct)
    wc = widenconst(ps)
    if is_all_inc_or_const(ps.fields)
        widened = aggressive_incidence_join(wc, ps.fields)
        wc !== widened && return widened
    end
    return Compiler.widenlattice(Compiler.widenlattice(🥬), ps)
end

function Compiler.:⊑(🥬::Compiler.PartialsLattice{EqStructureLattice}, @nospecialize(a), @nospecialize(b))
    if isa(a, PartialStruct)
        if isa(b, Incidence)
            isempty(b) || return false
            bjoin = aggressive_incidence_join(a.typ, a.fields)
            isa(bjoin, Incidence) || return false
            isempty(bjoin) || return false
            return Compiler.:⊑(🥬, a, b.typ)
        end
    end
    return @invoke Compiler.:⊑(🥬::Compiler.PartialsLattice, a::Any, b::Any)
end

function Compiler.:⊑(🥬::EqStructureLattice, @nospecialize(a), @nospecialize(b))
    if isa(a, PartialKeyValue)
        if isa(b, PartialKeyValue)
            return a.vals === b.vals
        end
        a = a.typ
    end
    isa(b, PartialKeyValue) && return a === Union{}
    if isa(a, Incidence)
        if isa(b, Incidence)
            return a == b
        elseif isa(b, Eq)
            return false
        end
        return Compiler.:⊑(Compiler.widenlattice(🥬), widenconst(a), b)
    elseif isa(b, Incidence)
        return isa(a, Eq) || a === Union{}
    end
    if isa(a, Eq)
        if isa(b, Eq)
            return a === b
        end
        return Compiler.:⊑(Compiler.widenlattice(🥬), widenconst(a), b)
    end
    isa(b, Eq) && return a === Union{}
    if isa(a, PartialScope)
        if isa(b, PartialScope)
            return a === b
        end
        return Compiler.:⊑(Compiler.widenlattice(🥬), widenconst(a), b)
    end
    isa(b, PartialScope) && return a === Union{}
    Compiler.:⊑(Compiler.widenlattice(🥬), a, b)
end

function Compiler.is_lattice_equal(🥬::EqStructureLattice, @nospecialize(a), @nospecialize(b))
    if isa(a, Incidence)
        isa(b, Incidence) || return false
        return a == b
    elseif isa(b, Incidence)
        return false
    end
    if isa(a, Eq) || isa(b, Eq)
        return a === b
    end
    if isa(a, PartialKeyValue) || isa(b, PartialKeyValue)
        return a === b
    end
    Compiler.is_lattice_equal(Compiler.widenlattice(🥬), a, b)
end

function Compiler.tmeet(🥬::EqStructureLattice, @nospecialize(a), @nospecialize(b::Type))
    if isa(a, Incidence)
        meet = Compiler.tmeet(Compiler.widenlattice(🥬), a.typ, b)
        meet == Union{} && return Union{}
        Base.issingletontype(meet) && return meet
        return Incidence(meet, copy(a.row))
    elseif isa(a, Eq)
        meet = Compiler.tmeet(Compiler.widenlattice(🥬), equation, b)
        meet == Union{} && return Union{}
        return a
    elseif isa(a, PartialKeyValue)
        return PartialKeyValue(Compiler.tmeet(Compiler.widenlattice(🥬), a.typ, b),
            a.parent, a.vals)
    elseif isa(a, PartialScope)
        meet = Compiler.tmeet(Compiler.widenlattice(🥬), Scope, b)
        meet === Union{} && return Union{}
        return a
    end
    return Compiler.tmeet(Compiler.widenlattice(🥬), a, b)
end

function Compiler._getfield_tfunc(🥬::EqStructureLattice, @nospecialize(s00), @nospecialize(name), setfield::Bool)
    if isa(name, Incidence)
        name = name.typ
    end
    if isa(s00, Incidence)
        if s00.typ == Union{}
            return Union{}
        end
        rt = Compiler._getfield_tfunc(Compiler.widenlattice(🥬), s00.typ, name, setfield)
        if rt == Union{}
            return Union{}
        end
        if isempty(s00)
            return Incidence(rt)
        end
        return Incidence(rt, copy(s00.row), copy(s00.eps))
    end
    return Compiler._getfield_tfunc(Compiler.widenlattice(🥬), s00, name, setfield)
end

function Compiler.has_nontrivial_extended_info(🥬::EqStructureLattice, @nospecialize(a))
    isa(a, Incidence) && return true
    isa(a, Eq) && return true
    isa(a, PartialScope) && return true
    isa(a, PartialKeyValue) && return true
    return Compiler.has_nontrivial_extended_info(Compiler.widenlattice(🥬), a)
end

function Compiler.is_const_prop_profitable_arg(🥬::EqStructureLattice, @nospecialize(a))
    isa(a, Incidence) && return true
    isa(a, Eq) && return true
    return Compiler.has_nontrivial_extended_info(Compiler.widenlattice(🥬), a)
end

# TODO: We really shouldn't propagate Incidence interprocedurally, but ok for now
function Compiler.is_forwardable_argtype(🥬::EqStructureLattice, @nospecialize(a))
    isa(a, Incidence) && return true
    isa(a, Eq) && return true
    return Compiler.is_forwardable_argtype(Compiler.widenlattice(🥬), a)
end

function Compiler.widenreturn(🥬::EqStructureLattice, @nospecialize(a), info::Compiler.BestguessInfo)
    isa(a, Incidence) && return a
    isa(a, Eq) && return a
    return Compiler.widenreturn(Compiler.widenlattice(🥬), a, info)
end

function Compiler.widenreturn_noslotwrapper(🥬::EqStructureLattice, @nospecialize(a), info::Compiler.BestguessInfo)
    isa(a, Incidence) && return a
    isa(a, Eq) && return a
    return Compiler.widenreturn_noslotwrapper(Compiler.widenlattice(🥬), a, info)
end

function Compiler.tmerge(🥬::EqStructureLattice, @nospecialize(a), @nospecialize(b))
    if isa(b, Incidence) && !isa(a, Incidence)
        (a, b) = (b, a)
    end
    if isa(a, Incidence)
        if isa(b, Incidence)
            merged_typ = Compiler.tmerge(Compiler.widenlattice(🥬), a.typ, b.typ)
            row = _zero_row()
            for i in union(rowvals(a.row), rowvals(b.row))
                row[i] = join_linearity(a.row[i], b.row[i])
            end
            return Incidence(merged_typ, row)
        elseif isa(b, Const)
            # Const has no incidence taint
            typ = Compiler.tmerge(Compiler.widenlattice(🥬), a.typ, b)
            r = copy(a)
            for i in rowvals(r.row)
                r.row[i] = nonlinear
            end
            return Incidence(typ, r.row)
        else
            a = widenconst(a)
        end
    end
    if isa(a, PartialKeyValue)
        if isa(b, PartialKeyValue)
            if a.vals === b.vals && a.parent === b.parent
                return PartialKeyValue(
                    Compiler.tmerge(Compiler.widenlattice(🥬), a.typ, b.typ),
                    a.parent, a.vals)
            end
        end
        a = widenconst(a)
    end
    if isa(b, PartialKeyValue)
        b = widenconst(b)
    end
    if isa(a, Const) && isa(b, Const)
        return Incidence(Compiler.tmerge(Compiler.widenlattice(🥬), a, b))
    end
    return Compiler.tmerge(Compiler.widenlattice(🥬), a, b)
end

function Compiler.tmerge_field(🥬::EqStructureLattice, @nospecialize(a), @nospecialize(b))
    if isa(a, PartialStruct) || isa(b, PartialStruct)
        # TODO: This is non-convergent in general, but we do need to merge any
        #       PartialStructs that have Incidences in them in order to keep our
        #       tracking precise, so let's leave this for the time being until it
        #       causes problems.
        a === b && return a
        return Compiler.tmerge_partial_struct(Compiler.PartialsLattice(🥬), a, b)
    else
        return Compiler.tmerge(🥬, a, b)
    end
end

################################ utilities #####################################

function _aggressive_incidence_join(@nospecialize(rt), argtypes::Vector{Any})
    local inci = Incidence(rt)
    found_any = false
    for a in argtypes
        if isa(a, PartialStruct)
            a = aggressive_incidence_join(Any, a.fields)
        end
        if isa(a, PartialKeyValue)
            a = a.typ
        end
        if isa(a, Const)
            found_any = true
        elseif isa(a, Incidence)
            if !found_any
                union!(inci.eps, a.eps)
                inci.row .= a.row
                found_any = true
                continue
            end
            union!(inci.eps, a.eps)  # must allow both
            for (i, v) in zip(rowvals(a.row), nonzeros(a.row))
                # as long as they are equal then it is correct for both so nothing to do
                if inci.row[i] != v
                    # Otherwise it can't be either but must allow both.
                    inci.row[i] = join_linearity(inci.row[i], v)
                end
            end
            # and the the other way: catch places that `rr` is nonzero but `aa` is zero
            for (i, v) in zip(rowvals(inci.row), nonzeros(inci.row))
                if a.row[i] != v
                    inci.row[i] = join_linearity(a.row[i], v)
                end
            end
        end
    end
    found_any && return inci
    return nothing
end

"""
    aggressive_incidence_join(rt, argtypes)

Combines multiple `Incidence`s to find one that is correct for all of them.
"""
function aggressive_incidence_join(@nospecialize(rt), argtypes::Vector{Any})
    inci = _aggressive_incidence_join(rt, argtypes)
    inci === nothing && return rt
    return inci
end
