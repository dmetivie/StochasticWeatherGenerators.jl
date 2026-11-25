# What a class sees when it simulates one member.

"""
    SimContext

The simulation context of one member, passed to [`simulate_member!`](@ref).

Accessors: `dates(ctx)` (the output dates), `period_indices(ctx)` (one per step, burn-in included),
`nsteps(ctx)`, `nburnin(ctx)`, `member(ctx)`, `stream(ctx, block)`, `past(ctx)`, `given(ctx)`,
`stations(ctx)` and `calendar(ctx)`.

Experimental: may change in minor releases.
"""
struct SimContext{D,S,C<:AbstractCalendar,P,G<:NamedTuple}
    dates::Vector{D}
    period_indices::Vector{Int}
    nburnin::Int
    member::Int
    seed::UInt64
    past::P
    given::G
    stations::Vector{S}
    calendar::C
end

_with_member(c::SimContext, m::Int) =
    SimContext(c.dates, c.period_indices, c.nburnin, m, c.seed, c.past, c.given, c.stations, c.calendar)

dates(c::SimContext) = c.dates
stations(c::SimContext) = c.stations
calendar(c::SimContext) = c.calendar

"""
    period_indices(ctx::SimContext)

Return the period index of each simulated step, burn-in steps first.

Experimental: may change in minor releases.
"""
period_indices(c::SimContext) = c.period_indices

"""
    nsteps(ctx::SimContext)

Return the number of steps to simulate: the burn-in steps plus the output dates.

Experimental: may change in minor releases.
"""
nsteps(c::SimContext) = length(c.period_indices)

"""
    nburnin(ctx::SimContext)

Return the number of burn-in steps that precede the output dates.

Experimental: may change in minor releases.
"""
nburnin(c::SimContext) = c.nburnin

"""
    member(ctx::SimContext)
    member(sims::WeatherSimulation, m)

Return the member number of `ctx`, or member `m` of `sims` as a [`WeatherData`](@ref).

Experimental: may change in minor releases.
"""
member(c::SimContext) = c.member

stream(c::SimContext, block = 0) = stream(c.seed, c.member, block)

"""
    past(ctx::SimContext)

Return the initial state given to `simulate`: a [`WeatherData`](@ref) ending the day before the first date
(stations in the generator's order), a date for the class to interpret, or `nothing`.

Experimental: may change in minor releases.
"""
past(c::SimContext) = c.past

"""
    given(ctx::SimContext)

Return the conditioning values as a NamedTuple: `regime` (a vector of length N) and D×N variable matrices
under canonical names; burn-in is 0 when it is not empty.

Experimental: may change in minor releases.
"""
given(c::SimContext) = c.given

"""
    SimBuffer

The output buffer of one member: `buf[var]` is a writable D×`nsteps` matrix for each simulated variable
(aliases allowed), and `regimes(buf)` the regime vector for classes with regimes. Values start as `NaN`
(regime 0) and must all be written.

Experimental: may change in minor releases.
"""
struct SimBuffer{A<:AbstractMatrix{Float64},R}
    vars::Vector{Symbol}
    names::Vector{Symbol}
    arrays::Vector{A}
    regimes::R
end

Base.getindex(b::SimBuffer, v::Union{Symbol,AbstractString}) =
    b.arrays[_varindex_or_error(b.vars, b.names, v; what = "simulation buffer")]
variables(b::SimBuffer) = b.vars

"""
    regimes(x)

Return the regime path of a [`SimBuffer`](@ref) (writable vector) or of a [`WeatherSimulation`](@ref) (N×M).

Experimental: may change in minor releases.
"""
function regimes(b::SimBuffer)
    b.regimes === nothing && throw(ArgumentError("this generator has no regimes (capability `regimes` is false)"))
    return b.regimes
end
