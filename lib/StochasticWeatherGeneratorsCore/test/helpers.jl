# Shared test helpers: a 360-day date type and synthetic station tables.

# minimal 360-day date-like value (stands in for CFTime.DateTime360Day)
struct D360 <: Dates.AbstractTime
    y::Int
    m::Int
    d::Int
end
Dates.year(x::D360) = x.y
Dates.month(x::D360) = x.m
Dates.day(x::D360) = x.d
Base.isless(a::D360, b::D360) = isless((a.y, a.m, a.d), (b.y, b.m, b.d))

d360_dates(y1, y2) = [D360(y, m, d) for y in y1:y2 for m in 1:12 for d in 1:30]

# wide table with one row per (station, day); TX seasonal, TN = TX - 8, RR ≥ 0
function station_table(; stations = [10, 20, 30], dates = Date(2000):Day(1):Date(2001, 12, 31), seed = 1,
        coords = true)
    rng = Xoshiro(seed)
    dts = collect(dates)
    n, nd = length(dts), length(stations)
    doy = [Dates.dayofyear(d) for d in dts]
    tx = vcat([15 .+ 8 .* sin.(2π .* (doy .- 100) ./ 365) .+ 2 .* randn(rng, n) .+ s for s in 1:nd]...)
    cols = (DATE = repeat(dts, nd), STAID = repeat(stations, inner = n), TX = tx, TN = tx .- 8 .- rand(rng, nd * n),
        RR = max.(0.0, 3 .* randn(rng, nd * n)))
    coords || return cols
    return merge(cols, (LAT = repeat(45.0 .+ (1:nd), inner = n), LON = repeat(2.0 .+ (1:nd) ./ 2, inner = n)))
end

# long layout of a wide table built by `station_table`
function long_table(t; vars = (:TX, :TN, :RR))
    n, nv = length(t.DATE), length(vars)
    return (DATE = repeat(t.DATE, nv), STAID = repeat(t.STAID, nv), LAT = repeat(t.LAT, nv), LON = repeat(t.LON, nv),
        variable = repeat(String.(collect(vars)), inner = n), value = vcat((t[v] for v in vars)...))
end

# subset the rows of a column table
rows(t::NamedTuple, idx) = map(c -> c[idx], t)
