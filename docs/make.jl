using Documenter
using Literate
using StochasticWeatherGenerator


examples_jl_path = joinpath(dirname(@__DIR__), "examples")
examples_md_path = joinpath(@__DIR__, "src", "examples")

for file in readdir(examples_md_path)
    if endswith(file, ".md")
        rm(joinpath(examples_md_path, file))
    end
end

for file in readdir(examples_jl_path)
    if !startswith(file, "geo")
       Literate.markdown(joinpath(examples_jl_path, file), examples_md_path, mdstrings=true)
    end
end

pages = [
    "Home" => "index.md",
    "Tutorials" => [
        "Paper" => joinpath("examples", "tuto_paper.md")
    ]
]

fmt = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    repolink="https://github.com/dmetivie/StochasticWeatherGenerator.jl",
    canonical="https://dmetivie.github.io/StochasticWeatherGenerator.jl",
    assets=String[],
)

makedocs(
    sitename = "StochasticWeatherGenerator",
    authors = "David Métivier",
    format = fmt,
    modules = [StochasticWeatherGenerator]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(; repo="github.com/dmetivie/StochasticWeatherGenerator.jl", devbranch="master")