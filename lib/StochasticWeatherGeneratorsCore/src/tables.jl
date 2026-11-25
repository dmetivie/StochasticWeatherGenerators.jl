# Tables.jl view of a WeatherSimulation: lazy columns, rows member → station → date.

# lazy column: row i maps to (member j, station s, day n) with n fastest, then s, then j
struct SimColumn{T,F} <: AbstractVector{T}
    f::F
    nd::Int
    nn::Int
    nm::Int
end
SimColumn{T}(f::F, nd, nn, nm) where {T,F} = SimColumn{T,F}(f, nd, nn, nm)

Base.size(c::SimColumn) = (c.nd * c.nn * c.nm,)
Base.IndexStyle(::Type{<:SimColumn}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(c::SimColumn, i::Int)
    @boundscheck checkbounds(c, i)
    k = i - 1
    n = k % c.nn
    k = k ÷ c.nn
    s = k % c.nd
    j = k ÷ c.nd
    return c.f(j + 1, s + 1, n + 1)
end

struct _MemberAt
    members::Vector{Int}
end
(f::_MemberAt)(j, s, n) = f.members[j]
struct _DateAt{V}
    dates::V
end
(f::_DateAt)(j, s, n) = f.dates[n]
struct _StationAt{V}
    stations::V
end
(f::_StationAt)(j, s, n) = f.stations[s]
struct _ValueAt{A}
    A::A
end
(f::_ValueAt)(j, s, n) = f.A[s, n, j]
struct _RegimeAt
    R::Matrix{Int}
end
(f::_RegimeAt)(j, s, n) = f.R[n, j]

Tables.istable(::Type{<:WeatherSimulation}) = true
Tables.columnaccess(::Type{<:WeatherSimulation}) = true

function Tables.columns(s::WeatherSimulation{T,D,S}) where {T,D,S}
    nd, nn, nm = length(s.stations), length(s.dates), length(s.members)
    cols = Pair{Symbol,AbstractVector}[:member => SimColumn{Int}(_MemberAt(s.members), nd, nn, nm),
        s.datecol => SimColumn{D}(_DateAt(s.dates), nd, nn, nm)]
    s.stationcol === nothing || push!(cols, s.stationcol => SimColumn{S}(_StationAt(s.stations), nd, nn, nm))
    for (n, A) in zip(s.names, s.arrays)
        push!(cols, n => SimColumn{T}(_ValueAt(A), nd, nn, nm))
    end
    s.regimes === nothing || push!(cols, :regime => SimColumn{Int}(_RegimeAt(s.regimes), nd, nn, nm))
    return (; cols...)
end

"""
    partitions(sims::WeatherSimulation; chunk = nothing)

Return the partitions of `sims` as NamedTuple column tables, member-major.

Without `chunk` there is one partition per (member, station), rows by date; this is `Tables.partitions(sims)`.
With `chunk = n` there is one partition per (member, block of `n` days), rows station → date.

Experimental: may change in minor releases.
"""
function partitions(s::WeatherSimulation; chunk = nothing)
    nd, nn, nm = length(s.stations), length(s.dates), length(s.members)
    if chunk === nothing
        return (_station_part(s, j, i) for j in 1:nm for i in 1:nd)
    end
    (chunk isa Integer && chunk >= 1) || throw(ArgumentError("`chunk` must be a positive number of days, got $chunk"))
    return (_chunk_part(s, j, r) for j in 1:nm for r in Iterators.partition(1:nn, Int(chunk)))
end

Tables.partitions(s::WeatherSimulation) = partitions(s)

function _station_part(s::WeatherSimulation, j, i)
    nn = length(s.dates)
    cols = Pair{Symbol,AbstractVector}[:member => fill(s.members[j], nn), s.datecol => copy(s.dates)]
    s.stationcol === nothing || push!(cols, s.stationcol => fill(s.stations[i], nn))
    for (n, A) in zip(s.names, s.arrays)
        push!(cols, n => A[i, :, j])
    end
    s.regimes === nothing || push!(cols, :regime => s.regimes[:, j])
    return (; cols...)
end

function _chunk_part(s::WeatherSimulation, j, r)
    nd, len = length(s.stations), length(r)
    cols = Pair{Symbol,AbstractVector}[:member => fill(s.members[j], nd * len), s.datecol => repeat(s.dates[r], nd)]
    s.stationcol === nothing || push!(cols, s.stationcol => repeat(s.stations, inner = len))
    for (n, A) in zip(s.names, s.arrays)
        push!(cols, n => vec(permutedims(A[:, r, j])))
    end
    s.regimes === nothing || push!(cols, :regime => repeat(s.regimes[r, j], nd))
    return (; cols...)
end
