using Markdown#hide
cd(@__DIR__)#hide

md"""
# Multisite daily Stochastic Weather Generator
"""

md"""
This tutorial describes the Stochastic Weather Generator describes in *Interpretable Seasonal Hidden Markov Model for spatio-temporal stochastic rain generator in France* by Emmanuel Gobet, David Métivier and Sylvie Parey.
It provides a step by step construction of the Seasonal Hidden Markov Model (SHMM), the interpretation of the hidden states as Weather regimes over France and eventually the validation of the model with simulations.
"""

md"""
## Set up
"""

md"""
### Package and functions
"""

using CSV, JLD, DelimitedFiles # File Read/Load/Save

using DataFrames, DataFramesMeta # DataFrames

using Dates

using StatsBase, Random

using Distributions


md"""
The two main package are not yet registered to the official Julia registry. 
They can be either `add`ed through [my local Julia registry](https://github.com/dmetivie/MyJuliaRegistry) with the [LocalRegistry.jl](https://github.com/GunnarFarneback/LocalRegistry.jl).
Or simply `add`ed via url i.e.
```julia
import Pkg
Pkg.add("https://github.com/dmetivie/SmoothPeriodicStatsModels.jl")
Pkg.add("https://github.com/dmetivie/StochasticWeatherGenerators.jl")
```
"""

using SmoothPeriodicStatsModels # Name might change. Small collection of smooth periodic models e.g. AR, HMM

using StochasticWeatherGenerators # interface to use with SmoothPeriodicStatsModels.jl

#-
save_tuto_path = "../../assets/tuto_1"

Random.seed!(1234)

md"""
### Settings for plotting

Some settings to have nice plots and save cropped version of plots (necessary in some cases).
"""

using StatsPlots, LaTeXStrings
using StatsPlots.PlotMeasures # To play with margin in Plots

gr() # plotly() # for interactive plots
default(thickness_scaling=1.2, fontfamily="Computer Modern", linewidth=2, label=nothing, size=(1000, 600))
scalefontsizes(1.5)
cur_colors = get_color_palette(:auto, 100);
my_palette(K) = palette(vcat(cur_colors[1], [cur_colors[c] for c in 3:4], cur_colors[2]), K)

file_for_plot_utilities = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/plot_utilities.jl")
include(file_for_plot_utilities)

md"""
To plot maps, we use `GeoMakie.jl` + a hack with `NaturalEarth.jl`. This is still experimental.
I used `cartopy` before with `PyCall.jl` which works very well.

For the following code to work you will need to add the following packages
```julia
import Pkg
Pkg.add("HTTP", "JSON3", "GeoMakie", "CairoMakie")
```
"""

file_for_maps_with_geomakie = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/geo_makie_features.jl") # download file from a GitHub repo
include(file_for_maps_with_geomakie)

md"""
### Global Parameters
"""
plot(0:5)