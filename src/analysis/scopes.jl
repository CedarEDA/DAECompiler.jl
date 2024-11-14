
struct ScopeDictEntry
    isvar::Union{Bool, Missing}
    idx::Int
    children::OrderedDict{Any, ScopeDictEntry}
    ScopeDictEntry() = new(missing, 0)
    ScopeDictEntry(isvar::Bool, idx::Int) = new(isvar, idx)
    ScopeDictEntry(children::OrderedDict{Any, ScopeDictEntry}) = new(missing, 0, children)
end

function record_scope!(ir::IRCode,
                       warnings::Vector{UnsupportedIRException}, names::OrderedDict,
                       scope::Union{Intrinsics.Scope, Intrinsics.GenScope, PartialStruct, PartialScope},
                       entry::ScopeDictEntry)
    stack = sym_stack(scope)
    name_dict = walk_dict(names, stack)
    if haskey(name_dict, stack[1])
        new = get_inline_backtrace(ir, varssa[idx])
        existing = get_inline_backtrace(ir, varssa[lens(names[stack[1]])])

        io = IOBuffer()
        Base.show_backtrace(io, new)
        print(io, "\n")
        Base.show_backtrace(io, existing)
        push!(warnings, UnsupportedIRException("Duplicate definition for scope $scope" * String(take!(io)), ir))
        name_dict[stack[1]] = ScopeDictEntry()
    else
        name_dict[stack[1]] = entry
    end
end