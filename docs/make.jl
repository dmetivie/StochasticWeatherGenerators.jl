using DocumenterCitations
using Documenter
using Literate
using StochasticWeatherGenerators
using SmoothPeriodicStatsModels

using Pkg

ENV["JULIA_DEBUG"] = "Documenter"

PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
PkgVERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/dmetivie/StochasticWeatherGenerators.jl"

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "biblio_swg.bib"),
    style=:authoryear #:numeric  # default

)

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

SUBSECTION_MODELS = ["Rainfall" => joinpath("models", "rain.md"), "Temperature & Others" => joinpath("models", "others.md")]

pages = [
    "Home" => "index.md",
    "📘 Models" => SUBSECTION_MODELS,        
    "📅 Weather Data" => "data.md",
    "📎 Tutorials" => [
        "Multisite rainfall HMM based SWG (paper) " => joinpath("examples", "tuto_paper.md"),
        "Multivariate SWG: Application to crop model" => joinpath("examples", "tuto_add_station_variable.md")
        ],
    "Utilities" => "api.md",
]

fmt = Documenter.HTML(
        prettyurls=true,
        repolink="https://github.com/dmetivie/StochasticWeatherGenerators.jl",
        canonical="https://dmetivie.github.io/StochasticWeatherGenerators.jl",
        assets=String["assets/citations.css"],
        footer="[$NAME.jl]($GITHUB) v$PkgVERSION docs powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl)."
    )

makedocs(
    sitename = "StochasticWeatherGenerators.jl",
    authors = "David Métivier",
    format = fmt,
    modules = [StochasticWeatherGenerators],
    pages = pages,
    plugins=[bib]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(; repo="github.com/dmetivie/StochasticWeatherGenerators.jl", devbranch="master")