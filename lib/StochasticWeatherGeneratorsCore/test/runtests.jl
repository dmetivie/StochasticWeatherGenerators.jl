using StochasticWeatherGeneratorsCore
using Test, Dates, Random, Statistics
using DataFrames: DataFrame
import Tables

const SWGC = StochasticWeatherGeneratorsCore

include("helpers.jl")

@testset "StochasticWeatherGeneratorsCore" begin
    include("calendar.jl")
    include("rng.jl")
    include("weatherdata.jl")
    include("window.jl")
end
