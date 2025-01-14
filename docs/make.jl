using Documenter
using Literate
using StochasticWeatherGenerators

examples_jl_path = joinpath(dirname(@__DIR__), "examples")
examples_md_path = joinpath(@__DIR__, "src", "examples")

for file in readdir(examples_md_path)
    if endswith(file, ".md")
        rm(joinpath(examples_md_path, file))
    end
end

for file in readdir(examples_jl_path)
    if !startswith(file, "utilities")
       Literate.markdown(joinpath(examples_jl_path, file), examples_md_path, mdstrings=true)
    end
end

pages = [
    "Home" => "index.md",
    "API" => "api.md",
    "Tutorials" => [
        "Paper" => joinpath("examples", "tuto_paper.md"),
        "Add weather variables and crop model application" => joinpath("examples", "tuto_add_station_variable.md")
    ]
]

fmt = Documenter.HTML(;
    prettyurls=true, #get(ENV, "CI", "false") == "true",
    repolink="https://github.com/dmetivie/StochasticWeatherGenerators.jl",
    canonical="https://dmetivie.github.io/StochasticWeatherGenerators.jl",
    assets=String[],
)

makedocs(
    sitename = "StochasticWeatherGenerators",
    authors = "David Métivier",
    format = fmt,
    modules = [StochasticWeatherGenerators],
    pages = pages
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(; repo="github.com/dmetivie/StochasticWeatherGenerators.jl", devbranch="master")