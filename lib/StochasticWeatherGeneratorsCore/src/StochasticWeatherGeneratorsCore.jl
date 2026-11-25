"""
    StochasticWeatherGeneratorsCore

Interface of the StochasticWeatherGenerators packages: weather data, calendars, the generator protocol,
ensemble simulation, Tables.jl output and persistence.
"""
module StochasticWeatherGeneratorsCore

using Dates
using Random
using Statistics: mean, median
import StatsAPI
import StatsAPI: fit, params, loglikelihood, dof, nobs, aic, aicc, bic
import JSON
import Tables

export WeatherData, WeatherSimulation, FittedGenerator, Leap366, NoLeap365, Day360
export fit, simulate, decode, spec, fitinfo, params, loglikelihood, aic, aicc, bic, dof, nobs, likelihood_kind
export members, dates, stations, variables, coords, distances, check_complete
export transitions, correlation, regime_model, save_generator, load_generator

# class-author (T2) protocol and helpers
public AbstractGenerator, AbstractCalendar, AbstractRegimeSource, resolve, fit_generator, simulate_member!,
    requested_variables, declared_kinds, capabilities, timestep, regime_source, nregimes, component,
    simulated_variables, bic_by, FitInfo, SimContext, SimBuffer, period, period_index, period_indices, calendar,
    kinds, sites, stream, RNG_DERIVATION, default_window, Haversine, partitions, member, provenance, regimes,
    nsteps, nburnin, past, given, to_plain, from_plain, register_type, default_solver

include("calendar.jl")
include("rng.jl")
include("variables.jl")
include("weatherdata.jl")
include("sites.jl")
include("window.jl")
include("interface.jl")
include("fitinfo.jl")
include("fitted.jl")
include("fit.jl")
include("simcontext.jl")
include("weathersimulation.jl")
include("tables.jl")
include("simulate.jl")
include("persistence.jl")
include("solver.jl")

function __init__()
    _register_core_types()
end

end
