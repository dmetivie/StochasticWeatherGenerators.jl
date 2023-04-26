module StochasticWeatherGenerator

using Distributions
# Trigonometric part

using ArgCheck
using LogExpFunctions: logsumexp!
import BinaryHierarchicalPeriodicHiddenMarkovModels: polynomial_trigo, interleave2
import Distributions: fit_mle
using Optimization
using JuMP, Ipopt

# 

abstract type AbstractMLE end

include("TrigonometricMixture/utilities.jl")
include("TrigonometricMixture/fit_mle_trig_EM.jl")
include("TrigonometricMixture/fit_mle_trig_Optim.jl")

export OptimMLE
export μₜ, σₜ, αₜ

using StatsBase
using DataFrames, DataFramesMeta, Dates
using CSV, Printf # File Read/Load
using LinearAlgebra: tril 
using Copulas # for rain generation
# Data part
include("utilities.jl")
include("data/extracting.jl")
include("data/stations.jl")

# Rain part
include("rain/correlations.jl")
include("rain/generation.jl")

export rand_rain
export joint_rain, Σ_Spearman2Pearson, Σ_Kendall2Pearson, corTail
export select_in_range_df, collect_data_ECA, shortname
export cartopy_map_with_stations, distance_x_to_y, distance_x_to_all_stations, dms_to_dd
export vec_triu, onefy, zerofy
export whole_year, dayofyear_Leap
export my_color

end
