# FitInfo: what a fit saw and reported.

"""
    FitInfo(data::WeatherData; loglikelihood = nothing, dof = nothing, nobs = nothing, history = nothing)

Record of a fit: the window data layout, the site metadata and what the class reports.

Classes build it from the window data; [`fit`](@ref) then fills `seed`, `solver` and `versions`.
Read-only properties:
- `dates` (the fit window), `stations`, `variables` (canonical), `names` (original), `kinds`, `calendar`,
  `datecol`, `stationcol`;
- `sites` (station table), `distances` (D×D matrix in km, or `nothing` without coordinates), `crs`;
- `seed`, `rng_derivation`, `solver`, `loglikelihood`, `dof`, `nobs`, `history`, `versions`.

Experimental: may change in minor releases.
"""
struct FitInfo{D,S,C<:AbstractCalendar}
    dates::Vector{D}
    stations::Vector{S}
    variables::Vector{Symbol}
    names::Vector{Symbol}
    kinds::Vector{Symbol}
    calendar::C
    datecol::Symbol
    stationcol::Union{Symbol,Nothing}
    sites::NamedTuple
    distances::Union{Matrix{Float64},Nothing}
    crs::Symbol
    seed::Union{UInt64,Nothing}
    rng_derivation::Int
    solver::String
    loglikelihood::Union{Float64,Nothing}
    dof::Union{Int,Nothing}
    nobs::Union{Int,Nothing}
    history::Any
    versions::Dict{String,String}
end

_maybe(::Type, ::Nothing) = nothing
_maybe(::Type{T}, x) where {T} = convert(T, x)

function FitInfo(data::WeatherData; loglikelihood = nothing, dof = nothing, nobs = nothing, history = nothing)
    c = coords(data)
    dist = c === nothing ? nothing : _distances(c, _default_metric(data.crs))
    return FitInfo(copy(data.dates), copy(data.stations), copy(data.vars), copy(data.names), copy(data.kinds),
        data.calendar, data.datecol, data.stationcol, sites(data), dist, data.crs, nothing, RNG_DERIVATION, "",
        _maybe(Float64, loglikelihood), _maybe(Int, dof), _maybe(Int, nobs), history, Dict{String,String}())
end

# FitInfo with the fields `fit` owns replaced
function _complete(info::FitInfo; seed, solver, versions)
    return FitInfo(info.dates, info.stations, info.variables, info.names, info.kinds, info.calendar, info.datecol,
        info.stationcol, info.sites, info.distances, info.crs, seed, RNG_DERIVATION, solver, info.loglikelihood,
        info.dof, info.nobs, info.history, versions)
end
