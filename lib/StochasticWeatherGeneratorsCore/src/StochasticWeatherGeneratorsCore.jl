"""
    StochasticWeatherGeneratorsCore

Interface of the StochasticWeatherGenerators packages: weather data, calendars, the generator protocol,
ensemble simulation, Tables.jl output and persistence.
"""
module StochasticWeatherGeneratorsCore

using Dates
using Random
using Statistics: median
import Tables

export WeatherData, Leap366, NoLeap365, Day360
export dates, stations, variables, coords, distances, check_complete

public AbstractCalendar, period, period_index, calendar, kinds, sites, stream, RNG_DERIVATION, default_window,
    Haversine

include("calendar.jl")
include("rng.jl")
include("variables.jl")
include("weatherdata.jl")
include("sites.jl")
include("window.jl")

end
