@enum GenerationMode begin
    DAE
    DAENoInit

    ODE
    ODENoInit

    # These are primarily for debug
    InitUncompress
end

struct Settings
    mode::GenerationMode
    force_inline_all::Bool
    insert_stmt_debuginfo::Bool
    insert_ssa_debuginfo::Bool
    skip_optimizations::Bool
end
Settings(; mode::GenerationMode=DAE, force_inline_all::Bool=false, insert_stmt_debuginfo::Bool=false, insert_ssa_debuginfo::Bool=false, skip_optimizations::Bool = false) = Settings(mode, force_inline_all, insert_stmt_debuginfo, insert_ssa_debuginfo, skip_optimizations)
