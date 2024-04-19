# Insert `CedarEDA` into our load path so we use its manifest for all the `CedarEDA` deps
if !(dirname(@__DIR__) ∈ Base.LOAD_PATH)
    insert!(Base.LOAD_PATH, 2, dirname(@__DIR__))
end

using Documenter, Base64

# Normalize the documenter key if it's not already base64-encoded
documenter_key = get(ENV, "DOCUMENTER_KEY", "")
try
    base64decode(documenter_key)
catch e
    if isa(e, ArgumentError)
        if !endswith(documenter_key, "\n")
            global documenter_key = string(documenter_key, "\n")
        end
        ENV["DOCUMENTER_KEY"] = base64encode(documenter_key)
    else
        rethrow(e)
    end
end

@info("Running any `0_setup.jl` files in the `src/` tree...")
# Ensure that we've run any `0_setup.jl` files.
# We run them in parallel for maximum efficiency.
function find_setup_files(root)
    setup_files = String[]
    for (root, dirs, files) in walkdir(root)
        if "0_setup.jl" ∈ files
            push!(setup_files, joinpath(root, "0_setup.jl"))
        end
    end
    return setup_files
end
Threads.@threads for setup_file in find_setup_files(joinpath(@__DIR__, "src/"))
    run(Cmd(`$(Base.julia_cmd()) --project=$(Base.active_project()) $(setup_file)`; dir=dirname(setup_file)))
end

makedocs(;
    sitename = "DAECompiler",
    authors="JuliaHub, Inc.",
    format=Documenter.HTML(; edit_link=nothing, sidebar_sitename=false, ansicolor=true),
    clean=true,
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/JuliaComputing/DAECompiler.jl.git",
    branch = "docs",
    target = "build",
)
