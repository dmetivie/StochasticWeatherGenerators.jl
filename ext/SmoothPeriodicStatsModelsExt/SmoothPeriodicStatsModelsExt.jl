module SmoothPeriodicStatsModelsExt

using StochasticWeatherGenerators

using SmoothPeriodicStatsModels: αₜ, σₜ, μₜ, ρₜ
using SmoothPeriodicStatsModels: AR1, fit_mle_AR1
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
