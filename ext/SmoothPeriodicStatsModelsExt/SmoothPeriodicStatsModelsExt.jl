module SmoothPeriodicStatsModelsExt

using StochasticWeatherGenerators

using SmoothPeriodicStatsModels: αₜ, σₜ, μₜ, ρₜ
using SmoothPeriodicStatsModels: AR1, SmoothAR1, SmoothSeasonalMixture, mixtures, parameters,
    fit_mle
# The generic time-varying layer, re-exported by SmoothPeriodicStatsModels: a seasonal model is
# a `ParamSpec` of bases and links, and its likelihood is one of the ready-made objectives.
using SmoothPeriodicStatsModels: seasonal_family, TrigBasis, PeriodicDesign, PeriodicNLL,
    TimeVaryingFit, group_by_time, ncoef
using SmoothPeriodicStatsModels: fit_loss_optim
import SmoothPeriodicStatsModels: fit_mle_RO

using StatsBase, Distributions
using Random: AbstractRNG, GLOBAL_RNG
using DataFrames, DataFramesMeta, Dates
using LinearAlgebra

using Copulas # for correlated generation

include("fit_mle_TN.jl")
include("fit_mle_rain.jl")
include("fit_mle_AR.jl")

end # module
