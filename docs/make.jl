using Documenter
using StochasticWeatherGenerator

makedocs(
    sitename = "StochasticWeatherGenerator",
    format = Documenter.HTML(),
    modules = [StochasticWeatherGenerator]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
