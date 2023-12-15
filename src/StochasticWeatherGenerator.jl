module StochasticWeatherGenerator

# Trigonometric part

using StatsBase, Distributions
using DataFrames, DataFramesMeta, Dates
using CSV, Printf # File Read/Load
using LinearAlgebra: tril 
using Copulas # for correlated generation
using RollingFunctions # rollmean for climate indices

# using RecipesBase#: @userplot, @recipe, @series
# import RecipesBase: @userplot
# include("plot_recipes.jl")
import SmoothPeriodicStatsModels: fit_mle_stations

# Data part
include("utilities.jl")
include("data/extracting.jl")
include("data/stations.jl")

# Rain part
include("rain/correlations.jl")
include("rain/generation.jl")

# Temperature
include("AR1/fit_mle.jl")
include("temperature/fit_mle.jl")

# Climate
include("climate_indices.jl")

# ## Rain
export rand_rain, fit_mle_RR, cov_rain
export joint_rain, Σ_Spearman2Pearson, Σ_Kendall2Pearson, corTail

# ## AR1
export cov_ar1, fit_AR1

# ## Data
export select_in_range_df, collect_data_ECA, shortname, collect_data_ECA!
# ### Geo
export distance_x_to_y, distance_x_to_all_stations, dms_to_dd

# ## Generic
export vec_triu, onefy, zerofy
export my_color
export whole_year, dayofyear_Leap

# ## Climate 
export longuest_spell, pmf_spell#, cartopy_map_with_stations, 
export VCX3, cum_monthly

# ### Plot

end
