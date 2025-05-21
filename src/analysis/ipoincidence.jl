
struct CalleeMapping
    var_coeffs::Vector{Any}
    eqs::Vector{Int}
    applied_scopes::Vector{Any}
end

struct CallerMappingState
    result::DAEIPOResult
    caller_var_to_diff::DiffGraph
    caller_varclassification::Vector{VarEqClassification}
    caller_varkind::Union{Vector{Intrinsics.VarKind}, Nothing}
    caller_eqclassification::Vector{VarEqClassification}
end

function compute_missing_coeff!(coeffs, (;result, caller_var_to_diff, caller_varclassification, caller_varkind)::CallerMappingState, v)
    # First find the rootvar, and if we already have a coeff for it
    # apply the derivatives.
    ndiffs = 0
    calle_inv = invview(result.var_to_diff)
    while calle_inv[v] !== nothing && !isassigned(coeffs, v)
        ndiffs += 1
        v = calle_inv[v]
    end

    if !isassigned(coeffs, v)
        @assert v > result.nexternalargvars # Arg vars should have already been mapped
        # Reached the root and it's an internal variable. We need to allocate
        # it in the caller now
        coeffs[v] = Incidence(add_vertex!(caller_var_to_diff))
        push!(caller_varclassification, result.varclassification[v] == External ? Owned : CalleeInternal)
        push!(caller_varkind, result.varkinds[v])
    end
    thisinc = coeffs[v]

    for _ = 1:ndiffs
        dv = result.var_to_diff[v]
        coeffs[dv] = structural_inc_ddt(caller_var_to_diff, caller_varclassification, caller_varkind, thisinc)
        v = dv
    end

    return nothing
end

apply_linear_incidence(𝕃, ret::Type, caller::CallerMappingState, mapping::CalleeMapping) = ret
apply_linear_incidence(𝕃, ret::Const, caller::CallerMappingState, mapping::CalleeMapping) = ret
function apply_linear_incidence(𝕃, ret::Incidence, caller::Union{CallerMappingState, Nothing}, mapping::CalleeMapping)
    # Substitute variables returned by the callee with the incidence defined by the caller.
    # The composition will be additive in the constant terms, and multiplicative for linear coefficients.
    caller_variables = mapping.var_coeffs

    typ = ret.typ
    row = _zero_row()

    used_caller_variables = Int[]
    for i in rowvals(ret.row)
        i == 1 && continue # skip time
        v = i - 1
        if !isassigned(caller_variables, v)
            compute_missing_coeff!(caller_variables, caller::CallerMappingState, v)
        end
        substitution = caller_variables[i - 1]
        isa(substitution, Incidence) || continue
        for j in rowvals(substitution.row)
            push!(used_caller_variables, j)
        end
    end
    did_use_time = in(1, used_caller_variables)

    for (i, coeff) in zip(rowvals(ret.row), nonzeros(ret.row))
        # Time dependence persists as itself
        if i == 1
            row[i] = coeff
            continue
        end

        v = i - 1
        substitution = caller_variables[v]
        if isa(substitution, Incidence)
            # Distribute the coefficient to all terms.
            # Because the coefficient is expressed in the reference of the callee,
            # state dependence must be processed carefully.
            typ = compose_additive_term(typ, substitution.typ, coeff)
            for (j, substitute) in zip(rowvals(substitution.row), nonzeros(substitution.row))
                row[j] === nonlinear && continue # no more information to be gained
                if substitute === nonlinear || coeff === nonlinear
                    row[j] = nonlinear
                elseif isa(coeff, Float64)
                    row[j] += coeff * substitute
                else
                    time_dependent = coeff.time_dependent
                    state_dependent = false
                    if isa(substitute, Linearity)
                        time_dependent |= substitute.time_dependent
                        state_dependent |= substitute.state_dependent
                    end
                    if coeff.state_dependent
                        if coeff.time_dependent && did_use_time
                            # The term is at least bilinear in another state, and this state
                            # from the callee may alias time from the caller, so we must mark
                            # time as nonlinear.
                            row[1] = nonlinear
                        end
                        if count(==(j), used_caller_variables) > 1
                            # The term is at least bilinear in another state, but we don't
                            # know which state, so we must fall back to nonlinear.
                            row[j] = nonlinear
                            continue
                        end
                        # We'll only be state-dependent if variables from the callee
                        # map to at least one other variable than `j`.
                        state_dependent |= length(used_caller_variables) - did_use_time > 1
                        # If another state may contain time, we may be time-dependent too.
                        time_dependent |= did_use_time
                    end
                    j == 1 && (time_dependent = false)
                    row[j] += Linearity(; nonlinear = false, state_dependent, time_dependent)
                end
            end
        elseif isa(substitution, Const)
            typ = compose_additive_term(typ, substitution, coeff)
        else
            return widenconst(typ) # unknown lattice element, we should widen
        end
    end

    return Incidence(typ, row)
end

function compose_additive_term(@nospecialize(a), @nospecialize(b), coeff)
    isa(a, Const) || return widenconst(a)
    isa(b, Const) || return widenconst(a)
    isa(coeff, Linearity) && return widenconst(a)
    val = a.val + b.val * coeff
    isa(val, Float64) || return widenconst(a)
    return Const(val)
end

function apply_linear_incidence(𝕃, ret::Eq, caller::CallerMappingState, mapping::CalleeMapping)
    eq_mapping = mapping.eqs[ret.id]
    if eq_mapping == 0
        error("I removed these from StructuralRefiner for conceptual reasons - if we hit these, lets revisit")
        #push!(caller_eqclassification, Owned)
        #push!(caller_eqkinds, result.eqkinds[ret.id])
        mapping.eqs[ret.id] = eq_mapping = length(caller_eqclassification)
    end
    return Eq(eq_mapping)
end

function apply_linear_incidence(𝕃, ret::PartialStruct, caller::CallerMappingState, mapping::CalleeMapping)
    return PartialStruct(𝕃, ret.typ, Any[apply_linear_incidence(𝕃, f, caller, mapping) for f in ret.fields])
end


function process_template!(𝕃, coeffs, eq_mapping, applied_scopes, argtypes, template_argtypes)
    for (arg, template) in zip(argtypes, template_argtypes)
        if isa(template, Incidence)
            if isempty(template)
                # @assert iszero(arg)
                continue
            end
            (idxs, vals) = findnz(template.row)
            @assert only(vals) == 1.0
            @assert !isassigned(coeffs, only(idxs)-1)
            coeffs[only(idxs)-1] = arg
        elseif isa(template, Eq)
            @assert isa(arg, Eq)
            eq_mapping[idnum(template)] = idnum(arg)
        elseif Compiler.is_const_argtype(template)
            #@CC.show (arg, template)
            #@assert CC.is_lattice_equal(DAE_LATTICE, arg, template)
        elseif isa(template, PartialScope)
            id = idnum(template)
            (id > length(applied_scopes)) && resize!(applied_scopes, id)
            if isa(arg, Const)
                @assert isa(arg.val, Union{Scope, Nothing})
                applied_scopes[id] = arg.val
            elseif isa(arg, PartialScope)
                applied_scopes[id] = arg
            else
                applied_scopes[id] = arg
            end
        elseif isa(template, PartialStruct)
            if isa(arg, PartialStruct)
                fields = arg.fields
            else
                fields = Any[Compiler.getfield_tfunc(𝕃, arg, Const(i)) for i = 1:length(template.fields)]
            end
            process_template!(𝕃, coeffs, eq_mapping, applied_scopes, fields, template.fields)
        else
            @show (arg, template, template_argtypes)
            error()
        end
    end
end

function CalleeMapping(𝕃::Compiler.AbstractLattice, argtypes::Vector{Any}, callee_result::DAEIPOResult)
    applied_scopes = Any[]
    coeffs = Vector{Any}(undef, length(callee_result.var_to_diff))
    eq_mapping = fill(0, length(callee_result.total_incidence))

    process_template!(𝕃, coeffs, eq_mapping, applied_scopes, argtypes, callee_result.argtypes)

    return CalleeMapping(coeffs, eq_mapping, applied_scopes)
end


struct MappingInfo <: Compiler.CallInfo
    info::Any
    result::DAEIPOResult
    mapping::CalleeMapping
end

function _make_argument_lattice_elem(𝕃, which::Argument, @nospecialize(argt), add_variable!, add_equation!, add_scope!)
    if isa(argt, Const)
        #@assert !isa(argt.val, Scope) # Shouldn't have been forwarded
        return argt
    elseif isa(argt, Type) && argt <: Intrinsics.AbstractScope
        return PartialScope(add_scope!(which))
    elseif isa(argt, Type) && argt == equation
        return Eq(add_equation!(which))
    elseif is_non_incidence_type(argt)
        return argt
    elseif Compiler.isprimitivetype(argt)
        inc = Incidence(add_variable!(which))
        return argt === Float64 ? inc : Incidence(argt, inc.row)
    elseif isa(argt, PartialStruct)
        return PartialStruct(𝕃, argt.typ, Any[make_argument_lattice_elem(𝕃, which, f, add_variable!, add_equation!, add_scope!) for f in argt.fields])
    elseif isabstracttype(argt) || ismutabletype(argt) || !isa(argt, DataType)
        return nothing
    else
        fields = Any[]
        any = false
        # TODO: This doesn't handle recursion
        if Base.datatype_fieldcount(argt) === nothing
            return nothing
        end
        for i = 1:length(fieldtypes(argt))
            # TODO: Can we make this lazy?
            ft = fieldtype(argt, i)
            mft = _make_argument_lattice_elem(𝕃, which, ft, add_variable!, add_equation!, add_scope!)
            if mft === nothing
                push!(fields, Incidence(ft))
            else
                any = true
                push!(fields, mft)
            end
        end
        return any ? PartialStruct(𝕃, argt, fields) : nothing
    end
end

function make_argument_lattice_elem(𝕃, which::Argument, @nospecialize(argt), add_variable!, add_equation!, add_scope!)
    mft = _make_argument_lattice_elem(𝕃, which, argt, add_variable!, add_equation!, add_scope!)
    mft === nothing ? Incidence(argt) : mft
end
