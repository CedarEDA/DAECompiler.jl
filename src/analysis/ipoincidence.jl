
struct CalleeMapping
    var_coeffs::Vector{Any}
    eqs::Vector{Int}
    applied_scopes::Vector{Any}
end

struct CallerMappingState
    callee_result::DAEIPOResult
    caller_var_to_diff::DiffGraph
    caller_varclassification::Vector{VarEqClassification}
    caller_varkind::Union{Vector{Intrinsics.VarKind}, Nothing}
    caller_eqclassification::Vector{VarEqClassification}
    caller_eqkinds::Union{Vector{Intrinsics.EqKind}, Nothing}
end

function compute_missing_coeff!(coeffs, (;callee_result, caller_var_to_diff, caller_varclassification, caller_varkind)::CallerMappingState, v)
    # First find the rootvar, and if we already have a coeff for it
    # apply the derivatives.
    ndiffs = 0
    callee_inv = invview(callee_result.var_to_diff)
    while callee_inv[v] !== nothing && !isassigned(coeffs, v)
        ndiffs += 1
        v = callee_inv[v]
    end

    if !isassigned(coeffs, v)
        @assert v > callee_result.nexternalargvars # Arg vars should have already been mapped
        # Reached the root and it's an internal variable. We need to allocate
        # it in the caller now
        coeffs[v] = Incidence(add_vertex!(caller_var_to_diff))
        push!(caller_varclassification, callee_result.varclassification[v] == External ? Owned : CalleeInternal)
        push!(caller_varkind, callee_result.varkinds[v])
    end
    thisinc = coeffs[v]

    for _ = 1:ndiffs
        dv = callee_result.var_to_diff[v]
        coeffs[dv] = structural_inc_ddt(caller_var_to_diff, caller_varclassification, caller_varkind, thisinc)
        v = dv
    end

    return nothing
end

apply_linear_incidence!(mapping::CalleeMapping, 𝕃, ret::Type, caller::CallerMappingState) = ret
apply_linear_incidence!(mapping::CalleeMapping, 𝕃, ret::Const, caller::CallerMappingState) = ret
function apply_linear_incidence!(mapping::CalleeMapping, 𝕃, ret::Incidence, caller::Union{CallerMappingState, Nothing})
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
                        if j == 1 # time
                            state_dependent |= length(used_caller_variables) > 1
                        else # state
                            state_dependent |= length(used_caller_variables) - did_use_time > 1
                        end
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
    isa(coeff, Linearity) && return b.val == 0 ? a : widenconst(a)
    val = a.val + b.val * coeff
    isa(val, Float64) || return widenconst(a)
    return Const(val)
end

function apply_linear_incidence!(mapping::CalleeMapping, 𝕃, ret::Eq, caller::CallerMappingState)
    eq_mapping = mapping.eqs[ret.id]
    if eq_mapping == 0
        push!(caller.caller_eqclassification, Owned)
        push!(caller.caller_eqkinds, caller.callee_result.eqkinds[ret.id])
        mapping.eqs[ret.id] = eq_mapping = length(caller.caller_eqclassification)
    end
    return Eq(eq_mapping)
end

function apply_linear_incidence!(mapping::CalleeMapping, 𝕃, ret::PartialStruct, caller::CallerMappingState)
    return PartialStruct(𝕃, ret.typ, Any[apply_linear_incidence!(mapping, 𝕃, f, caller) for f in ret.fields])
end

function CalleeMapping(𝕃::AbstractLattice, argtypes::Vector{Any}, callee_ci::CodeInstance, callee_result::DAEIPOResult)
    caller_argtypes = Compiler.va_process_argtypes(𝕃, argtypes, callee_ci.inferred.nargs, callee_ci.inferred.isva)
    callee_argtypes = callee_ci.inferred.ir.argtypes
    argmap = ArgumentMap(callee_argtypes)
    nvars = length(callee_result.var_to_diff)
    neqs = length(callee_result.total_incidence)
    @assert length(argmap.variables) ≤ nvars
    @assert length(argmap.equations) ≤ neqs

    applied_scopes = Any[]
    coeffs = Vector{Any}(undef, nvars)
    eq_mapping = fill(0, neqs)
    mapping = CalleeMapping(coeffs, eq_mapping, applied_scopes)

    fill_callee_mapping!(mapping, argmap, caller_argtypes, 𝕃)
    return mapping
end

function fill_callee_mapping!(mapping::CalleeMapping, argmap::ArgumentMap, argtypes::Vector{Any}, 𝕃::AbstractLattice)
    for (i, index) in enumerate(argmap.variables)
        type = get_fieldtype(argtypes, index, 𝕃)
        mapping.var_coeffs[i] = type
    end
    for (i, index) in enumerate(argmap.equations)
        eq = get_fieldtype(argtypes, index, 𝕃)::Eq
        mapping.eqs[i] = eq.id
    end
end

function get_fieldtype(argtypes::Vector{Any}, index::CompositeIndex, 𝕃::AbstractLattice = Compiler.fallback_lattice)
    @assert !isempty(index)
    index = copy(index)
    type = argtypes[popfirst!(index)]
    while !isempty(index)
        type = Compiler.getfield_tfunc(𝕃, type, Const(popfirst!(index)))
    end
    return type
end

struct MappingInfo <: Compiler.CallInfo
    info::Any
    result::DAEIPOResult
    mapping::CalleeMapping
end
