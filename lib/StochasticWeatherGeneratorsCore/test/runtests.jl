using StochasticWeatherGeneratorsCore
using Test, Dates, Random, Statistics
using DataFrames: DataFrame
import Tables

const SWGC = StochasticWeatherGeneratorsCore

include("helpers.jl")
include("toy_class.jl")
using .ToyGenerators

@testset "StochasticWeatherGeneratorsCore" begin
    include("calendar.jl")
    include("rng.jl")
    include("weatherdata.jl")
    include("window.jl")
    include("fit.jl")
    include("simulate.jl")
    include("tables.jl")

    @testset "toy classes use the public API only" begin
        src = read(joinpath(@__DIR__, "toy_class.jl"), String)
        used = unique(Symbol(m.captures[1]) for m in eachmatch(r"SWGC\.(\w+!?)", src))
        @test !isempty(used)
        @test filter(s -> !Base.ispublic(SWGC, s), used) == Symbol[]
        @test !occursin("StochasticWeatherGeneratorsCore.", src)
    end
end
