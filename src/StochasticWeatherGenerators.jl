module StochasticWeatherGenerators

# Trigonometric part

using StatsBase, Distributions
using DataFrames, DataFramesMeta, Dates
using CSV, Printf # File Read/Load
using LinearAlgebra
using Copulas # for correlated generation
using NearestCorrelationMatrix: nearest_cor!, nearest_cor
using RollingFunctions # rollmean for climate indices

# For WGEN Markov models
using NonlinearSolve
using OrderedCollections

# missings
using Impute: Interpolate, impute!

# using RecipesBase#: @userplot, @recipe, @series
# import RecipesBase: @userplot
# include("plot_recipes.jl")

# Data part
include("utilities.jl")
include("data/extracting.jl")
include("data/stations.jl")

# Rain part
include("rain/fit_mle.jl")
include("rain/correlations.jl")
include("rain/generation.jl")
include("rain/wgen.jl")
# Temperature
include("AR1/fit_mle.jl")
include("temperature/TN.jl")


# Climate
include("climate_indices.jl")

# ## Rain
export rand_RR, fit_mle_RR, cov_RR, cor_RR, fit_mle_RO
export mix_ini
export joint_rain, Σ_Spearman2Pearson, Σ_Kendall2Pearson, corTail
export fit_markov_chain, simulate_markov_gaussian, fit_Ω, wgen, fit_wgen

# ## AR1
export cov_ar1, fit_AR1
export fit_TN, rand_cond
export cor_groupby, cor_groupbyTXTN

# ## Data
export select_in_range_df, collect_data_ECA, collect_data_INRAE, shortname, collect_data_ECA!, collect_data_MeteoFrance, clean_data, download_data_MeteoFrance, isvalid_data_ECA
# ### Geo
export distance_x_to_y, distance_x_to_all_stations, dms_to_dd, haversine, LLA

# ## Generic
export vec_triu, onefy, zerofy
export my_color
export whole_year, dayofyear_Leap

# ## Climate 
export longuest_spell, pmf_spell#, cartopy_map_with_stations, 
export VCX3, monthly_agg

# ### Plot

end
