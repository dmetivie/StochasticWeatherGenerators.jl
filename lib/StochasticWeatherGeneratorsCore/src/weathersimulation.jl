# WeatherSimulation: an ensemble of simulated members.

"""
    WeatherSimulation

An ensemble returned by [`simulate`](@ref): one D×N×M array per variable (stations × days × members).

- `sims[:TX]` (aliases allowed) is the D×N×M array; `sims[:regime]` the N×M regime path, for classes with regimes.
- `members(sims)`, `dates(sims)`, `stations(sims)`, `variables(sims)`, `provenance(sims)`;
  `member(sims, m)` returns member `m` as a [`WeatherData`](@ref).
- It is a Tables.jl table with rows member → station → date and columns `member`, date, station, the
  variables under their input names, and `regime`; see also [`partitions`](@ref).
"""
struct WeatherSimulation{T<:AbstractFloat,D,S,C<:AbstractCalendar,R<:Union{Matrix{Int},Nothing}}
    vars::Vector{Symbol}
    names::Vector{Symbol}
    kinds::Vector{Symbol}
    arrays::Vector{Array{T,3}}
    regimes::R
    members::Vector{Int}
    dates::Vector{D}
    stations::Vector{S}
    datecol::Symbol
    stationcol::Union{Symbol,Nothing}
    sitemeta::NamedTuple
    crs::Symbol
    calendar::C
    provenance::NamedTuple
end

"""
    members(sims::WeatherSimulation)

Return the member numbers of `sims`.
"""
members(s::WeatherSimulation) = s.members
dates(s::WeatherSimulation) = s.dates
stations(s::WeatherSimulation) = s.stations
variables(s::WeatherSimulation) = s.vars
kinds(s::WeatherSimulation) = NamedTuple{Tuple(s.vars)}(Tuple(s.kinds))
calendar(s::WeatherSimulation) = s.calendar
sites(s::WeatherSimulation) = merge((station = s.stations,), s.sitemeta)
coords(s::WeatherSimulation) = _coords(s.sitemeta, s.crs)

"""
    provenance(sims::WeatherSimulation)

Return how `sims` was drawn: `(; seed, rng_derivation, members, dates, stations, burnin, generator)`.

Experimental: may change in minor releases.
"""
provenance(s::WeatherSimulation) = s.provenance

function regimes(s::WeatherSimulation)
    s.regimes === nothing && throw(ArgumentError("this simulation has no regimes"))
    return s.regimes
end

function Base.getindex(s::WeatherSimulation, v::Union{Symbol,AbstractString})
    Symbol(v) === :regime && return regimes(s)
    return s.arrays[_varindex_or_error(s.vars, s.names, v; what = "simulation")]
end

function member(s::WeatherSimulation{T}, m::Integer) where {T}
    j = findfirst(==(m), s.members)
    j === nothing && throw(ArgumentError("no member $m; members are $(_rangestring(s.members))"))
    vals = Matrix{Union{Missing,T}}[Matrix{Union{Missing,T}}(A[:, :, j]) for A in s.arrays]
    return WeatherData(copy(s.dates), copy(s.stations), copy(s.vars), copy(s.names), copy(s.kinds), vals,
        s.sitemeta, s.crs, s.calendar, s.datecol, s.stationcol)
end

_rangestring(v) = (length(v) > 1 && v == first(v):last(v)) ? "$(first(v)):$(last(v))" : join(v, ", ")

function Base.show(io::IO, s::WeatherSimulation)
    print(io, "WeatherSimulation(", length(s.members), " members × ", length(s.stations), " stations × ",
        length(s.dates), " days, ", join(s.names, ", "), ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::WeatherSimulation{T}) where {T}
    println(io, "WeatherSimulation{$T}: ", length(s.members), " member", length(s.members) == 1 ? "" : "s", " × ",
        length(s.stations), " stations × ", length(s.dates), " days (", first(s.dates), " to ", last(s.dates), ")")
    print(io, "  variables: ", join(s.names, ", "), s.regimes === nothing ? "" : ", regime")
    print(io, "\n  members: ", _rangestring(s.members), ", seed ", s.provenance.seed, ", generator ",
        s.provenance.generator)
end
