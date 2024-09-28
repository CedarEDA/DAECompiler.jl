############################### lattice.jl #####################################

using SparseArrays

########################## DAELattice ####################################
"""
    struct DAELattice <: CC.AbstractLattice

This lattice implements the `AbstractLattice` interface. It adjoins `Incidence` and `Eq`.

The DAELattice is one of the key places where DAECompiler extends
Core.Compiler. In compiler parlance, the DAELattice type system can be
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
struct DAELattice <: CC.AbstractLattice; end
CC.widenlattice(::DAELattice) = CC.ConstsLattice()
CC.is_valid_lattice_norec(::DAELattice, @nospecialize(v)) = isa(v, Incidence) || isa(v, Eq) || isa(v, PartialScope) || isa(v, PartialKeyValue)
CC.has_extended_unionsplit(::DAELattice) = true

############################## NonLinear #######################################

"""
    struct NonLinear

This singleton number is similar to `missing` in that arithmatic with it is
saturating. When used as a coefficient in the Incidence lattice, it indicates
that the corresponding variable does not have a (constant) linear coefficient.
This may either mean that the variable in question has a non-constant linear
coefficient or that the variable is used non-linearly. We do not currently
distinguish the two situations.
"""
struct NonLinear; end
Base.iszero(::NonLinear) = false
Base.zero(::Type{Union{NonLinear, Float64}}) = 0.
for f in (:+, :-)
    @eval begin
        Base.$f(a::Real, b::NonLinear) = b
        Base.$f(a::NonLinear, b::Real) = a
        Base.$f(a::NonLinear, b::NonLinear) = nonlinear
        Base.$f(::NonLinear) = nonlinear
    end
end

Base.:(*)(a::Real, b::NonLinear) = iszero(a) ? a : b
Base.:(*)(a::NonLinear, b::Real) = iszero(b) ? b : a
Base.:(*)(a::NonLinear, b::NonLinear) = nonlinear
Base.div(a::Real, b::NonLinear) = iszero(a) ? a : b
Base.div(a::NonLinear, b::Real) = a
Base.div(a::NonLinear, b::NonLinear) = a
Base.:(/)(a::Real, b::NonLinear) = iszero(a) ? a : b
Base.:(/)(a::NonLinear, b::Real) = a
Base.:(/)(a::NonLinear, b::NonLinear) = a
Base.rem(a::Real, b::NonLinear) = b
Base.rem(a::NonLinear, b::Real) = a
Base.rem(a::NonLinear, b::NonLinear) = a
Base.abs(a::NonLinear) = nonlinear
Base.isone(a::NonLinear) = false

const nonlinear = NonLinear.instance
Base.Broadcast.broadcastable(::NonLinear) = Ref(nonlinear)

############################## Incidence #######################################
# TODO: Just use Infinities.jl here?
const MAX_EQS = div(typemax(Int), 2)

# For now, we only track exact, integer linearities, because that's what
# MTK can handle, so `nonlinear` includes linear operations with floating
# point values.
const IncidenceVector = SparseVector{Union{Float64, NonLinear}, Int}

is_non_incidence_type(@nospecialize(type)) = type === Union{} || Base.issingletontype(type)

"""
    struct Incidence

An element of the `DAELattice` that sits between `Const` and `Type` in the
lattice hierarchy. In particular, `Const(::T) ⊑ Incidence(T, {...}) ⊑ T`, where
`{...}` is a set of incidence variables. Lattice operations among the `Incidence`
elements are defined by subset inclusion. Note that in particular this implies
that plain `T` lattice elements have unknown incidence and `Const` lattice elements
have no incidence. A lattice element of type `T` that is known to be state-independent
would have lattice element `Incidence(T, {})`.
"""
struct Incidence
    # Considered additive to `row`. In particular, if the `typ` is Float64,
    # the linear combination in `row` will be inhomogeneous.
    typ::Union{Type, Const}
    row::IncidenceVector
    eps::BitSet

    function Incidence(@nospecialize(type), row, eps::BitSet)
        if is_non_incidence_type(type)
            throw(DomainError(type, "Invalid type for Incidence"))
        end
        return new(type, row, eps)
    end
end
Incidence(row::IncidenceVector, eps::BitSet) = Incidence(Float64, row, eps)

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
    if isa(inc.typ, Const) && isa(inc.typ.val, Float64)
        if !iszero(inc.typ.val)
            print(io, inc.typ.val)
            first = false
        end
    else inc.typ !== Float64
        print(io, inc.typ, ", ")
    end
    print_plusminus(io, minus=false) = if first
            first = false
            minus && print(io, "-")
        else
            print(io, minus ? " - " : " + ")
        end
    for (i, v) in zip(rowvals(inc.row), nonzeros(inc.row))
        v !== nonlinear || continue
        print_plusminus(io, v < 0)
        if abs(v) != 1 || i == 1
            print(io, abs(v))
        end
        print(io, subscript_state(i))
    end
    first_nonlinear = true
    for (i, v) in zip(rowvals(inc.row), nonzeros(inc.row))
        v === nonlinear || continue
        if first_nonlinear
            print_plusminus(io)
            first_nonlinear = false
            print(io, "f(")
        else
            print(io, ", ")
        end
        print(io, i == 1 ? "t" : subscript_state(i))
    end
    if !first_nonlinear
        print(io, ")")
    end
    if !isempty(inc.eps)
        for (i, var) in enumerate(inc.eps)
            print_plusminus(io)
            print(io, "c", subscript(i), "ε", subscript(i))
        end
    end
    print(io, ")")
end

_zero_row() = IncidenceVector(MAX_EQS, Int[], Union{Float64, NonLinear}[])
const _ZERO_ROW = _zero_row()
const _ZERO_CONST = Const(0.0)
Base.zero(::Type{Incidence}) = Incidence(_ZERO_CONST, _zero_row(), BitSet())
function Incidence(T::Union{Type, CC.Const} = Float64)
    vec = Incidence(T, _zero_row(), BitSet())
    vec
end

function Incidence(T::PartialStruct)
    PartialStruct(T.typ, Any[(isa(f, Const) || is_non_incidence_type(f)) ? f : Incidence(f) for f in T.fields])
end

Base.:(==)(a::Incidence, b::Incidence) = a.typ === b.typ &&
    a.row.nzind == b.row.nzind && a.row.nzval == b.row.nzval && a.eps == b.eps
Base.:(+)(a::Incidence, b::Incidence) = tfunc(Val{Core.Intrinsics.add_float}(), a, b)
Base.:(+)(a::Const, b::Incidence) = tfunc(Val{Core.Intrinsics.add_float}(), a, b)
Base.:(+)(a::Incidence, b::Const) = tfunc(Val{Core.Intrinsics.add_float}(), a, b)

function Incidence(v::Int)
    row = _zero_row()
    row[v+1] = 1.0
    return Incidence(_ZERO_CONST, row, BitSet())
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

Base.copy(inc::Incidence) = Incidence(inc.typ, copy(inc.row), copy(inc.eps))
Base.isempty(inc::Incidence) = iszero(inc.row) && isempty(inc.eps)  # slightly weaker than iszero as that also requires it to be Const(0.0)

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

function getkeyvalue_tfunc(𝕃::Core.Compiler.AbstractLattice,
        @nospecialize(collection), @nospecialize(key))
    isa(key, Const) || return Tuple{Any}
    if haskey(collection.vals, key.val)
        return CC.tuple_tfunc(𝕃, Any[collection.vals[key.val]])
    end
    error()
end

######################### AbstractLattice interface ############################

CC.widenconst(inc::Incidence) = widenconst(inc.typ)
CC.widenconst(::Eq) = equation
CC.widenconst(::PartialScope) = Scope
CC.widenconst(pkv::PartialKeyValue) = widenconst(pkv.typ)
CC.:⊑(inc::Incidence, inc2) = CC.:⊑(inc2, Float64) && !isa(inc2, Const)

function CC._uniontypes(x::Incidence, ts)
    u = x.typ
    if isa(u, Union)
        CC.push!(ts, is_non_incidence_type(u.a) ? u.a : Incidence(u.a, x.row, x.eps))
        CC.push!(ts, is_non_incidence_type(u.b) ? u.b : Incidence(u.b, x.row, x.eps))
        return ts
    else
        CC.push!(ts, x)
        return ts
    end
end

function CC.widenlattice(🥬::DAELattice, ps::CC.PartialStruct)
    wc = widenconst(ps)
    if is_all_inc_or_const(ps.fields)
        widened = aggressive_incidence_join(wc, ps.fields)
        wc !== widened && return widened
    end
    return CC.widenlattice(CC.widenlattice(🥬), ps)
end

function CC.:⊑(🥬::CC.PartialsLattice{DAELattice}, @nospecialize(a), @nospecialize(b))
    if isa(a, PartialStruct)
        if isa(b, Incidence)
            isempty(b) || return false
            bjoin = aggressive_incidence_join(a.typ, a.fields)
            isa(bjoin, Incidence) || return false
            isempty(bjoin) || return false
            return CC.:⊑(🥬, a, b.typ)
        end
    end
    return @invoke CC.:⊑(🥬::CC.PartialsLattice, a::Any, b::Any)
end

function CC.:⊑(🥬::DAELattice, @nospecialize(a), @nospecialize(b))
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
        return CC.:⊑(CC.widenlattice(🥬), widenconst(a), b)
    elseif isa(b, Incidence)
        return isa(a, Eq) || a === Union{}
    end
    if isa(a, Eq)
        if isa(b, Eq)
            return a === b
        end
        return CC.:⊑(CC.widenlattice(🥬), widenconst(a), b)
    end
    isa(b, Eq) && return a === Union{}
    if isa(a, PartialScope)
        if isa(b, PartialScope)
            return a === b
        end
        return CC.:⊑(CC.widenlattice(🥬), widenconst(a), b)
    end
    isa(b, PartialScope) && return a === Union{}
    CC.:⊑(CC.widenlattice(🥬), a, b)
end

function CC.is_lattice_equal(🥬::DAELattice, @nospecialize(a), @nospecialize(b))
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
    CC.is_lattice_equal(CC.widenlattice(🥬), a, b)
end

function CC.tmeet(🥬::DAELattice, @nospecialize(a), @nospecialize(b::Type))
    if isa(a, Incidence)
        meet = CC.tmeet(CC.widenlattice(🥬), a.typ, b)
        meet == Union{} && return Union{}
        Base.issingletontype(meet) && return meet
        return Incidence(meet, copy(a.row), copy(a.eps))
    elseif isa(a, Eq)
        meet = CC.tmeet(CC.widenlattice(🥬), equation, b)
        meet == Union{} && return Union{}
        return a
    elseif isa(a, PartialKeyValue)
        return PartialKeyValue(CC.tmeet(CC.widenlattice(🥬), a.typ, b),
            a.parent, a.vals)
    elseif isa(a, PartialScope)
        meet = CC.tmeet(CC.widenlattice(🥬), Scope, b)
        meet === Union{} && return Union{}
        return a
    end
    return CC.tmeet(CC.widenlattice(🥬), a, b)
end

function CC._getfield_tfunc(🥬::DAELattice, @nospecialize(s00), @nospecialize(name), setfield::Bool)
    if isa(name, Incidence)
        name = name.typ
    end
    if isa(s00, Incidence)
        if s00.typ == Union{}
            return Union{}
        end
        rt = CC._getfield_tfunc(CC.widenlattice(🥬), s00.typ, name, setfield)
        if rt == Union{}
            return Union{}
        end
        if isempty(s00)
            return Incidence(rt)
        end
        return Incidence(rt, copy(s00.row), copy(s00.eps))
    end
    return CC._getfield_tfunc(CC.widenlattice(🥬), s00, name, setfield)
end

function CC.has_nontrivial_extended_info(🥬::DAELattice, @nospecialize(a))
    isa(a, Incidence) && return true
    isa(a, Eq) && return true
    isa(a, PartialScope) && return true
    isa(a, PartialKeyValue) && return true
    return CC.has_nontrivial_extended_info(CC.widenlattice(🥬), a)
end

function CC.is_const_prop_profitable_arg(🥬::DAELattice, @nospecialize(a))
    isa(a, Incidence) && return true
    isa(a, Eq) && return true
    return CC.has_nontrivial_extended_info(CC.widenlattice(🥬), a)
end

# TODO: We really shouldn't propagate Incidence interprocedurally, but ok for now
function CC.is_forwardable_argtype(🥬::DAELattice, @nospecialize(a))
    isa(a, Incidence) && return true
    isa(a, Eq) && return true
    return CC.is_forwardable_argtype(CC.widenlattice(🥬), a)
end

function CC.widenreturn(🥬::DAELattice, @nospecialize(a), info::CC.BestguessInfo)
    isa(a, Incidence) && return a
    isa(a, Eq) && return a
    return CC.widenreturn(CC.widenlattice(🥬), a, info)
end

function CC.widenreturn_noslotwrapper(🥬::DAELattice, @nospecialize(a), info::CC.BestguessInfo)
    isa(a, Incidence) && return a
    isa(a, Eq) && return a
    return CC.widenreturn_noslotwrapper(CC.widenlattice(🥬), a, info)
end

function CC.tmerge(🥬::DAELattice, @nospecialize(a), @nospecialize(b))
    if isa(b, Incidence) && !isa(a, Incidence)
        (a, b) = (b, a)
    end
    if isa(a, Incidence)
        if isa(b, Incidence)
            merged_typ = CC.tmerge(CC.widenlattice(🥬), a.typ, b.typ)
            row = _zero_row()
            for i in union(rowvals(a.row), rowvals(b.row))
                if a.row[i] == b.row[i]
                    row[i] = a.row[i]
                else
                    row[i] = nonlinear
                end
            end
            return Incidence(merged_typ, row, union(a.eps, b.eps))
        elseif isa(b, Const)
            # Const has no incidence taint
            typ = CC.tmerge(CC.widenlattice(🥬), a.typ, b)
            r = copy(a)
            for i in rowvals(r.row)
                r.row[i] = nonlinear
            end
            return Incidence(typ, r.row, copy(a.eps))
        else
            a = widenconst(a)
        end
    end
    if isa(a, PartialKeyValue)
        if isa(b, PartialKeyValue)
            if a.vals === b.vals && a.parent === b.parent
                return PartialKeyValue(
                    CC.tmerge(CC.widenlattice(🥬), a.typ, b.typ),
                    a.parent, a.vals)
            end
        end
        a = widenconst(a)
    end
    if isa(b, PartialKeyValue)
        b = widenconst(b)
    end
    if isa(a, Const) && isa(b, Const)
        return Incidence(CC.tmerge(CC.widenlattice(🥬), a, b))
    end
    return CC.tmerge(CC.widenlattice(🥬), a, b)
end

function CC.tmerge_field(🥬::DAELattice, @nospecialize(a), @nospecialize(b))
    if isa(a, PartialStruct) || isa(b, PartialStruct)
        # TODO: This is non-convergent in general, but we do need to merge any
        #       PartialStructs that have Incidences in them in order to keep our
        #       tracking precise, so let's leave this for the time being until it
        #       causes problems.
        a === b && return a
        return CC.tmerge_partial_struct(CC.PartialsLattice(🥬), a, b)
    else
        return CC.tmerge(🥬, a, b)
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
                    # Otherwise it can't be either but must allow both. We would ideally represent this as
                    # `LinearUnion{rr[i], v}` or `Linear`, but we don't have lattice elements like that
                    # `NonLinear` is our more general representation
                    inci.row[i] = nonlinear
                end
            end
            # and the the other way: catch places that `rr` is nonzero but `aa` is zero
            for (i, v) in zip(rowvals(inci.row), nonzeros(inci.row))
                if a.row[i] != v
                    # mix of nonlinear and linear, or again: a mix of two different linear coefficients
                    inci.row[i] = nonlinear
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
