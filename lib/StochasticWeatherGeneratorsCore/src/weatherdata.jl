# WeatherData: daily multisite multivariable data on a calendar, stored as stations × days matrices.

"""
    WeatherData(table; date = :auto, station = :auto, variables = nothing, kinds = (;),
                calendar = Leap366(), crs = :lonlat, verbose = true)
    WeatherData(A::AbstractArray{<:Real,3}, dates, stations = 1:size(A, 1); vars, coords = nothing,
                calendar = Leap366(), kinds = (;), crs = :lonlat)
    WeatherData(values::NamedTuple, dates; stations = 1:D, coords = nothing,
                calendar = Leap366(), kinds = (;), crs = :lonlat)

Daily weather data for D stations, N consecutive days and V variables, read from any Tables.jl table.

The table is wide (one column per variable) or long (`variable` and `value` columns).
- The date column is `date`/`time` (any case) or the first column of dates; `DateTime` values must be at midnight.
- The station column is `station`, `STAID`, `site` or `id`; without one the data is a single site with id 1.
- Variables are recognised by name (any case): RR/PRCP/pr (`:rain`), TX/TMAX/tasmax (`:temperature`),
  TN/TMIN/tasmin (`:lower_temperature`), QQ/rsds (`:radiation`), ETPP/ET0/PET (`:evapotranspiration`)
  and FG (`:wind`). They are stored under the canonical name (`TMAX` becomes `:TX`) and written back under the
  original name. Aliases do not convert units.
- Other columns are dropped with one info line, unless declared in `kinds`, e.g. `kinds = (WIND = :positive,)`.
  Valid kinds are the six above plus `:positive`, `:continuous` and `:custom`.
- `variables` keeps exactly these variables (aliases allowed); undeclared ones get kind `:continuous`.
- Site metadata columns LAT/LON/ALT (or X/Y/ALT with `crs = :projected`) are kept per station.
- `NaN` becomes `missing` (with a warning), days absent from the table become `missing`, and ±Inf, non-numeric
  variables and duplicate (date, station) rows are errors.

The array forms take a D×N×V array `A` (gridded data) or a NamedTuple of D×N matrices (or N vectors for one site);
`coords` is a NamedTuple of per-station vectors such as `(LAT = lat, LON = lon)`.
"""
struct WeatherData{D,S,T,C<:AbstractCalendar}
    dates::Vector{D}
    stations::Vector{S}
    vars::Vector{Symbol}
    names::Vector{Symbol}
    kinds::Vector{Symbol}
    values::Vector{Matrix{Union{Missing,T}}}
    sitemeta::NamedTuple
    crs::Symbol
    calendar::C
    datecol::Symbol
    stationcol::Union{Symbol,Nothing}
    function WeatherData{D,S,T,C}(dates, stations, vars, names, kinds, values, sitemeta, crs, calendar,
            datecol, stationcol) where {D,S,T,C}
        nd, nn = length(stations), length(dates)
        length(vars) == length(names) == length(kinds) == length(values) ||
            throw(DimensionMismatch("vars, names, kinds and values must have the same length"))
        for M in values
            size(M) == (nd, nn) || throw(DimensionMismatch("each variable must be a $nd×$nn (stations × days) matrix"))
        end
        allunique(vars) || throw(ArgumentError("duplicate variables $(vars)"))
        allunique(stations) || throw(ArgumentError("duplicate station ids"))
        _checkcrs(crs)
        for (k, v) in pairs(sitemeta)
            length(v) == nd || throw(DimensionMismatch("site metadata $k must have one value per station"))
        end
        return new{D,S,T,C}(dates, stations, vars, names, kinds, values, sitemeta, crs, calendar, datecol, stationcol)
    end
end

WeatherData(dates::Vector{D}, stations::Vector{S}, vars, names, kinds, values::Vector{Matrix{Union{Missing,T}}},
    sitemeta, crs, calendar::C, datecol, stationcol) where {D,S,T,C} =
    WeatherData{D,S,T,C}(dates, stations, vars, names, kinds, values, sitemeta, crs, calendar, datecol, stationcol)

_checkcrs(crs) = crs in (:lonlat, :projected) ||
                 throw(ArgumentError("`crs` must be :lonlat or :projected, got $(repr(crs))"))

# ---------------------------------------------------------------- table constructor

function WeatherData(table; date = :auto, station = :auto, variables = nothing, kinds::NamedTuple = (;),
        calendar::AbstractCalendar = Leap366(), crs::Symbol = :lonlat, verbose::Bool = true,
        algkinds::NamedTuple = (;))
    Tables.istable(table) || throw(ArgumentError(
        "expected a Tables.jl table (e.g. a DataFrame or a NamedTuple of vectors), got a $(typeof(table))"))
    _checkcrs(crs)
    return _weatherdata(Tables.columns(table), date, station, variables, kinds, calendar, crs, verbose, algkinds)
end

# not specialized on the table type; row loops live in specialized helpers
function _weatherdata(@nospecialize(cols), @nospecialize(date), @nospecialize(station), @nospecialize(variables),
        @nospecialize(kinds), calendar::AbstractCalendar, crs::Symbol, verbose::Bool, @nospecialize(algkinds))
    colnames = Symbol[Symbol(n) for n in Tables.columnnames(cols)]
    isempty(colnames) && throw(ArgumentError("the table has no columns"))
    nrow = length(Tables.getcolumn(cols, colnames[1]))
    nrow == 0 && throw(ArgumentError("the table has no rows"))

    datecol = _find_datecol(cols, colnames, date)
    stationcol = _find_stationcol(colnames, station, datecol)
    rawdates = _todates(Tables.getcolumn(cols, datecol), datecol)
    steps = _steps(calendar, rawdates)
    smin, smax = extrema(steps)
    axis = _date_axis(calendar, rawdates, steps, smin, smax)
    tidx = steps .- (smin - 1)

    if stationcol === nothing
        stations, sidx = [1], ones(Int, nrow)
    else
        stations, sidx = _station_index(Tables.getcolumn(cols, stationcol), stationcol)
    end

    sitekeys = SITE_KEYS[crs]
    sitecols, sitecanon = Symbol[], Symbol[]
    for n in colnames
        (n === datecol || n === stationcol) && continue
        c = get(SITE_ALIASES, _upper(n), nothing)
        (c !== nothing && c in sitekeys) || continue
        j = findfirst(==(c), sitecanon)
        j === nothing || throw(ArgumentError("columns $(sitecols[j]) and $n both give site metadata $c; drop one"))
        push!(sitecols, n)
        push!(sitecanon, c)
    end
    sitemeta = _sitemeta(cols, sitecols, sitecanon, sidx, stations, sitekeys)

    reserved = Symbol[datecol, sitecols...]
    stationcol === nothing || push!(reserved, stationcol)
    ivar = findfirst(n -> _upper(n) == "VARIABLE" && n ∉ reserved, colnames)
    ival = findfirst(n -> _upper(n) == "VALUE" && n ∉ reserved, colnames)
    long = ivar !== nothing && ival !== nothing
    if long
        varcol = Tables.getcolumn(cols, colnames[ivar])
        any(ismissing, varcol) && throw(ArgumentError("the column $(colnames[ivar]) has missing variable names"))
        rowvars = _rowvars(varcol)
        candidates = unique(rowvars)
        others = setdiff(colnames, reserved, (colnames[ivar], colnames[ival]))
    else
        candidates = setdiff(colnames, reserved)
        others = Symbol[]
    end
    names, vars, kindv, dropped = _select_variables(candidates, variables, kinds, algkinds, colnames)
    dropped = vcat(others, dropped)
    if verbose && !isempty(dropped)
        fix = variables === nothing ?
              "to keep one, declare its kind, e.g. `kinds = ($(first(dropped)) = :continuous,)`" :
              "they are not in `variables`"
        @info "Dropping columns $(join(dropped, ", ")): $fix."
    end

    nd, nn = length(stations), length(axis)
    if long
        valcol = Tables.getcolumn(cols, colnames[ival])
        _isnumeric(valcol) || throw(ArgumentError(
            "the value column $(colnames[ival]) is not numeric (eltype $(eltype(valcol))); convert it first"))
        vidx = _vidx(rowvars, names)
        _check_duplicates(sidx, tidx, vidx, nd, nn, length(names), axis, stations, names)
        T = _valtype([eltype(valcol)])
        mats = [Matrix{Union{Missing,T}}(missing, nd, nn) for _ in names]
        nans = zeros(Int, length(names))
        _fill_long!(mats, valcol, vidx, sidx, tidx, nans, names, stations, axis)
    else
        valcols = [Tables.getcolumn(cols, n) for n in names]
        for (n, c) in zip(names, valcols)
            _isnumeric(c) || throw(ArgumentError(
                "the column $n is not numeric (eltype $(eltype(c))); convert it, or drop it from the table"))
        end
        _check_duplicates(sidx, tidx, nothing, nd, nn, 1, axis, stations, names)
        T = _valtype(eltype.(valcols))
        mats = [Matrix{Union{Missing,T}}(missing, nd, nn) for _ in names]
        nans = zeros(Int, length(names))
        for v in eachindex(names)
            nans[v] = _fill_wide!(mats[v], valcols[v], sidx, tidx, names[v], stations, axis)
        end
    end
    _warn_nan(names, nans)
    _warn_kelvin(names, kindv, mats)
    return WeatherData(axis, stations, vars, names, kindv, mats, sitemeta, crs, calendar, datecol, stationcol)
end

_steps(cal, rawdates) = Int[stepnumber(cal, d) for d in rawdates]
_rowvars(varcol) = Symbol[Symbol(string(x)) for x in varcol]
_vidx(rowvars, names) = Int[something(findfirst(==(v), names), 0) for v in rowvars]

function _find_datecol(@nospecialize(cols), colnames, @nospecialize(date))
    if date === :auto
        i = findfirst(n -> _upper(n) in DATE_NAMES, colnames)
        i === nothing &&
            (i = findfirst(n -> nonmissingtype(eltype(Tables.getcolumn(cols, n))) <: Dates.AbstractTime, colnames))
        i === nothing && throw(ArgumentError("no date column among $(join(colnames, ", ")); " *
                                             "name it `date` or pass `date = :mycolumn`"))
        return colnames[i]
    end
    d = Symbol(date)
    d in colnames || throw(ArgumentError("no column $d; columns are $(join(colnames, ", "))"))
    return d
end

function _find_stationcol(colnames, station, datecol)
    station === nothing && return nothing
    if station === :auto
        for cand in STATION_NAMES
            i = findfirst(n -> n !== datecol && _upper(n) == cand, colnames)
            i === nothing || return colnames[i]
        end
        return nothing
    end
    s = Symbol(station)
    s in colnames || throw(ArgumentError("no column $s; columns are $(join(colnames, ", "))"))
    return s
end

function _todates(col, name)
    any(ismissing, col) && throw(ArgumentError("the date column $name has missing values"))
    D = nonmissingtype(eltype(col))
    if D <: DateTime
        for d in col
            d == DateTime(Date(d)) || throw(ArgumentError(
                "the date column $name has times other than midnight ($d); aggregate to daily values first"))
        end
        return Date[Date(d) for d in col]
    end
    if !(D <: Dates.AbstractTime)
        all(x -> x isa Dates.AbstractTime, col) || throw(ArgumentError(
            "the date column $name holds $(D) values, not dates; parse them first, e.g. `Date.(df.$name)`"))
        D = typeof(first(col))
    end
    return collect(D, col)
end

function _date_axis(cal, rawdates::AbstractVector{D}, steps, smin, smax) where {D}
    n = smax - smin + 1
    axis = Vector{D}(undef, n)
    filled = falses(n)
    for (d, s) in zip(rawdates, steps)
        i = s - smin + 1
        filled[i] || (axis[i] = d; filled[i] = true)
    end
    for i in findall(!, filled)
        d = _date_of_step(cal, smin + i - 1, D)
        d === nothing && throw(ArgumentError(
            "the dates have a gap after $(axis[i-1]) that $(cal) cannot fill with $(D) values; " *
            "$(cal) data must be gap-free"))
        axis[i] = d
    end
    return axis
end

function _station_index(col, name)
    any(ismissing, col) && throw(ArgumentError("the station column $name has missing values"))
    S = nonmissingtype(eltype(col))
    stations = S[]
    lookup = Dict{S,Int}()
    sidx = Vector{Int}(undef, length(col))
    for (r, x) in enumerate(col)
        sidx[r] = get!(lookup, x) do
            push!(stations, x)
            length(stations)
        end
    end
    return stations, sidx
end

function _sitemeta(@nospecialize(cols), sitecols, sitecanon, sidx, stations, sitekeys)
    entries = Pair{Symbol,Vector{Float64}}[]
    for key in sitekeys
        j = findfirst(==(key), sitecanon)
        j === nothing && continue
        name = sitecols[j]
        push!(entries, key => _site_values(Tables.getcolumn(cols, name), name, sidx, stations))
    end
    return (; entries...)
end

function _site_values(col, name, sidx, stations)
    _isnumeric(col) || throw(ArgumentError("the site column $name is not numeric (eltype $(eltype(col)))"))
    vals = fill(NaN, length(stations))
    set = falses(length(stations))
    for (r, x) in enumerate(col)
        ismissing(x) && continue
        s = sidx[r]
        if set[s]
            vals[s] == x || throw(ArgumentError("$name varies within station $(stations[s]) ($(vals[s]) and $x); " *
                                                "site metadata must be constant per station"))
        else
            vals[s] = x
            set[s] = true
        end
    end
    all(set) || throw(ArgumentError("$name is missing for station $(stations[findfirst(!, set)])"))
    return vals
end

function _isnumeric(col)
    T = nonmissingtype(eltype(col))
    T <: Real && return true
    T === Any || return false
    return all(x -> ismissing(x) || x isa Real, col)
end

function _valtype(types)
    T = Union{}
    for t in types
        t = nonmissingtype(t)
        T = promote_type(T, t <: AbstractFloat ? t : Float64)
    end
    return T === Union{} ? Float64 : T
end

function _select_variables(candidates::Vector{Symbol}, @nospecialize(variables), @nospecialize(kinds), @nospecialize(algkinds), colnames)
    canon = Symbol[something(_alias(c), c) for c in candidates]
    function lookup(nt, c, cn)
        for (k, v) in pairs(nt)
            (k === c || k === cn || _alias(k) === cn) && return v, k
        end
        return nothing, nothing
    end
    used = Set{Symbol}()
    kindof = Vector{Union{Nothing,Symbol}}(nothing, length(candidates))
    for (i, c) in enumerate(candidates)
        k, key = lookup(kinds, c, canon[i])
        if k !== nothing
            push!(used, key)
            kindof[i] = _checkkind(c, k)
            continue
        end
        k, _ = lookup(algkinds, c, canon[i])
        if k !== nothing
            kindof[i] = _checkkind(c, k)
            continue
        end
        kindof[i] = get(ALIAS_KINDS, canon[i], nothing)
    end
    absent = [k for k in keys(kinds) if k ∉ used]
    isempty(absent) || throw(ArgumentError("`kinds` names $(join(absent, ", ")), which is not a variable column; " *
                                           "columns are $(join(colnames, ", "))"))
    if variables === nothing
        keep = findall(!isnothing, kindof)
    else
        req = variables isa Union{Symbol,AbstractString} ? [Symbol(variables)] : Symbol[Symbol(v) for v in variables]
        keep = Int[]
        for r in req
            rc = something(_alias(r), r)
            i = findfirst(==(r), candidates)
            i === nothing && (i = findfirst(==(rc), canon))
            i === nothing && throw(ArgumentError("no column for variable $r; columns are $(join(colnames, ", "))"))
            i in keep || push!(keep, i)
            kindof[i] === nothing && (kindof[i] = :continuous)
        end
    end
    isempty(keep) && throw(ArgumentError(
        "no weather variable found among $(join(candidates, ", ")); use known names (RR, TX, TN, QQ, ETPP, FG " *
        "or aliases such as PRCP, TMAX) or declare kinds, e.g. `kinds = (WIND = :positive,)`"))
    for (a, i) in enumerate(keep), j in keep[a+1:end]
        canon[i] === canon[j] && throw(ArgumentError(
            "columns $(candidates[i]) and $(candidates[j]) both map to $(canon[i]); drop or rename one"))
    end
    dropped = candidates[setdiff(eachindex(candidates), keep)]
    return candidates[keep], canon[keep], Symbol[kindof[i] for i in keep], dropped
end

function _check_duplicates(sidx, tidx, vidx, nd, nn, nv, axis, stations, names)
    seen = falses(nd, nn, nv)
    first3 = String[]
    count = 0
    for r in eachindex(sidx)
        v = vidx === nothing ? 1 : vidx[r]
        v == 0 && continue
        s, t = sidx[r], tidx[r]
        if seen[s, t, v]
            count += 1
            length(first3) < 3 &&
                push!(first3, vidx === nothing ? "($(axis[t]), $(stations[s]))" : "($(axis[t]), $(stations[s]), $(names[v]))")
        else
            seen[s, t, v] = true
        end
    end
    count == 0 && return nothing
    what = vidx === nothing ? "(date, station)" : "(date, station, variable)"
    throw(ArgumentError("$count duplicate $what rows, e.g. $(join(first3, ", ")); remove or aggregate them"))
end

function _checkvalue(x, name, station, date)
    isinf(x) && throw(ArgumentError(
        "infinite value $x in $name at station $station on $date; replace it by `missing` or a finite value"))
    return nothing
end

function _fill_wide!(M, col, sidx, tidx, name, stations, axis)
    nans = 0
    for r in eachindex(col, sidx)
        x = col[r]
        ismissing(x) && continue
        if x isa AbstractFloat
            isnan(x) && (nans += 1; continue)
            _checkvalue(x, name, stations[sidx[r]], axis[tidx[r]])
        end
        M[sidx[r], tidx[r]] = x
    end
    return nans
end

function _fill_long!(mats, col, vidx, sidx, tidx, nans, names, stations, axis)
    for r in eachindex(col, sidx)
        v = vidx[r]
        v == 0 && continue
        x = col[r]
        ismissing(x) && continue
        if x isa AbstractFloat
            isnan(x) && (nans[v] += 1; continue)
            _checkvalue(x, names[v], stations[sidx[r]], axis[tidx[r]])
        end
        mats[v][sidx[r], tidx[r]] = x
    end
    return nothing
end

function _warn_nan(names, nans)
    sum(nans) == 0 && return nothing
    detail = join(("$n: $c" for (n, c) in zip(names, nans) if c > 0), ", ")
    @warn "$(sum(nans)) NaN values were read as missing ($detail)."
    return nothing
end

function _warn_kelvin(names, kinds, mats)
    for (n, k, M) in zip(names, kinds, mats)
        k in (:temperature, :lower_temperature) || continue
        x = collect(skipmissing(M))
        isempty(x) && continue
        m = median(x)
        m > 100 && @warn "$n looks like Kelvin (median $(round(m; digits = 1))); convert it to °C."
    end
    return nothing
end

# ---------------------------------------------------------------- array constructors

function WeatherData(A::AbstractArray{<:Union{Missing,Real},3}, dates::AbstractVector,
        stations::AbstractVector = 1:size(A, 1); vars, coords = nothing, calendar::AbstractCalendar = Leap366(),
        kinds::NamedTuple = (;), crs::Symbol = :lonlat)
    nv = size(A, 3)
    length(vars) == nv || throw(DimensionMismatch("`vars` has $(length(vars)) names but A has $nv variables"))
    mats = [A[:, :, v] for v in 1:nv]
    return _from_arrays(mats, Symbol.(collect(vars)), dates, stations, coords, calendar, kinds, crs)
end

function WeatherData(values::NamedTuple, dates::AbstractVector; stations = nothing, coords = nothing,
        calendar::AbstractCalendar = Leap366(), kinds::NamedTuple = (;), crs::Symbol = :lonlat)
    isempty(values) && throw(ArgumentError("no variables given"))
    mats = [v isa AbstractVector ? reshape(v, 1, :) : v for v in values]
    all(m -> m isa AbstractMatrix, mats) || throw(ArgumentError("each variable must be a D×N matrix or an N vector"))
    nd = size(first(mats), 1)
    st = stations === nothing ? (1:nd) : stations
    return _from_arrays(mats, collect(Symbol, keys(values)), dates, st, coords, calendar, kinds, crs)
end

function _from_arrays(mats, names::Vector{Symbol}, dates, stations, coords, calendar, kinds, crs)
    _checkcrs(crs)
    isempty(dates) && throw(ArgumentError("no dates given"))
    dts = _todates(dates, :dates)
    check_consecutive(calendar, dts)
    nd, nn = length(stations), length(dts)
    for (n, M) in zip(names, mats)
        size(M) == (nd, nn) || throw(DimensionMismatch("$n is $(join(size(M), "×")), expected $nd×$nn (stations × days)"))
        _isnumeric(M) || throw(ArgumentError("$n is not numeric (eltype $(eltype(M)))"))
    end
    sel, vars, kindv, _ = _select_variables(names, names, kinds, (;), names)
    T = _valtype(eltype.(mats))
    st = collect(stations)
    vals = Matrix{Union{Missing,T}}[]
    nans = zeros(Int, length(names))
    for (v, M) in enumerate(mats)
        out = Matrix{Union{Missing,T}}(missing, nd, nn)
        for j in 1:nn, i in 1:nd
            x = M[i, j]
            ismissing(x) && continue
            if x isa AbstractFloat
                isnan(x) && (nans[v] += 1; continue)
                _checkvalue(x, names[v], st[i], dts[j])
            end
            out[i, j] = x
        end
        push!(vals, out)
    end
    _warn_nan(names, nans)
    _warn_kelvin(names, kindv, vals)
    meta = _coords_meta(coords, crs, nd)
    return WeatherData(dts, st, vars, sel, kindv, vals, meta, crs, calendar, :date, :station)
end

function _coords_meta(coords, crs, nd)
    coords === nothing && return (;)
    coords isa NamedTuple || throw(ArgumentError("`coords` must be a NamedTuple such as (LAT = lat, LON = lon)"))
    sitekeys = SITE_KEYS[crs]
    found = Dict{Symbol,Vector{Float64}}()
    for (k, v) in pairs(coords)
        c = get(SITE_ALIASES, _upper(k), nothing)
        (c !== nothing && c in sitekeys) || throw(ArgumentError(
            "unknown site metadata $k for crs = :$crs; use $(join(sitekeys, ", "))"))
        length(v) == nd || throw(DimensionMismatch("coords.$k has $(length(v)) values for $nd stations"))
        found[c] = Float64.(collect(v))
    end
    return (; (k => found[k] for k in sitekeys if haskey(found, k))...)
end

# ---------------------------------------------------------------- accessors

"""
    dates(x)

Return the dates of `x` (data, fitted generator window, simulation or simulation context).
"""
dates(d::WeatherData) = d.dates

"""
    stations(x)

Return the station ids of `x`.
"""
stations(d::WeatherData) = d.stations

"""
    variables(x)

Return the canonical variable names of `x` (`:TX` for a `TMAX` column).
"""
variables(d::WeatherData) = d.vars

"""
    kinds(x)

Return the kind of each variable of `x` as a NamedTuple, e.g. `(RR = :rain, TX = :temperature)`.

Experimental: may change in minor releases.
"""
kinds(d::WeatherData) = NamedTuple{Tuple(d.vars)}(Tuple(d.kinds))

"""
    calendar(x)

Return the calendar of `x`.

Experimental: may change in minor releases.
"""
calendar(d::WeatherData) = d.calendar

"""
    sites(x)

Return the station table of `x`: a NamedTuple of columns `station` and the site metadata (LAT, LON, ALT, X, Y).

Experimental: may change in minor releases.
"""
sites(d::WeatherData) = merge((station = d.stations,), d.sitemeta)

"""
    coords(x)

Return the (lon, lat) pair of each station, or (x, y) with `crs = :projected`, or `nothing` without coordinates.
"""
coords(d::WeatherData) = _coords(d.sitemeta, d.crs)

function _coords(meta::NamedTuple, crs)
    a, b = crs === :lonlat ? (:LON, :LAT) : (:X, :Y)
    (haskey(meta, a) && haskey(meta, b)) || return nothing
    return [(x, y) for (x, y) in zip(meta[a], meta[b])]
end

Base.getindex(d::WeatherData, v::Union{Symbol,AbstractString}) = d.values[_varindex_or_error(d.vars, d.names, v)]

function Base.getindex(d::WeatherData, r::AbstractRange)
    eltype(r) <: Real && throw(ArgumentError("index days by dates, e.g. `data[Date(2000):Date(2000, 12, 31)]`"))
    isempty(r) && throw(ArgumentError("empty date range"))
    step(r) == Day(1) || throw(ArgumentError("the date range must have a step of one day, got $(step(r))"))
    return _subset(d, _dayrange(d, first(r), last(r)))
end

# day indices of the dates a to b in `d`
function _dayrange(d::WeatherData, a, b)
    s0 = stepnumber(d.calendar, first(d.dates))
    i, j = stepnumber(d.calendar, a) - s0 + 1, stepnumber(d.calendar, b) - s0 + 1
    i <= j || throw(ArgumentError("the window $a to $b is empty"))
    (1 <= i && j <= length(d.dates)) || throw(ArgumentError(
        "the window $a to $b is outside the data ($(first(d.dates)) to $(last(d.dates)))"))
    return i:j
end

_subset(d::WeatherData, idx::AbstractUnitRange) = WeatherData(d.dates[idx], copy(d.stations), copy(d.vars),
    copy(d.names), copy(d.kinds), [M[:, idx] for M in d.values], d.sitemeta, d.crs, d.calendar, d.datecol, d.stationcol)

function _select_vars(d::WeatherData, req)
    idx = [_varindex_or_error(d.vars, d.names, v) for v in req]
    unique!(idx)
    return WeatherData(d.dates, d.stations, d.vars[idx], d.names[idx], d.kinds[idx], d.values[idx], d.sitemeta,
        d.crs, d.calendar, d.datecol, d.stationcol)
end

function _select_stations(d::WeatherData, idx)
    meta = (; (k => v[idx] for (k, v) in pairs(d.sitemeta))...)
    return WeatherData(d.dates, d.stations[idx], d.vars, d.names, d.kinds, [M[idx, :] for M in d.values], meta,
        d.crs, d.calendar, d.datecol, d.stationcol)
end

function Base.:(==)(a::WeatherData, b::WeatherData)
    return all(isequal(getfield(a, f), getfield(b, f)) for f in fieldnames(WeatherData))
end
Base.isequal(a::WeatherData, b::WeatherData) = a == b
Base.hash(d::WeatherData, h::UInt) = hash(d.values, hash(d.vars, hash(d.stations, hash(d.dates, h))))

# ---------------------------------------------------------------- merge

"""
    merge(d1::WeatherData, d2::WeatherData)
    merge(d1::WeatherData, table; kw...)

Return the data of the stations of `d1` and `d2` over the union of their dates and variables.

Both must use the same calendar and crs, and share no station; absent values are `missing`.
"""
function Base.merge(d1::WeatherData, d2::WeatherData)
    typeof(d1.calendar) == typeof(d2.calendar) ||
        throw(ArgumentError("cannot merge data on $(d1.calendar) and $(d2.calendar)"))
    d1.crs === d2.crs || throw(ArgumentError("cannot merge data with crs :$(d1.crs) and :$(d2.crs)"))
    common = intersect(d1.stations, d2.stations)
    isempty(common) || throw(ArgumentError(
        "stations $(join(common, ", ")) are in both datasets; merge adds stations, drop them from one side"))
    cal = d1.calendar
    s1 = stepnumber.(Ref(cal), (first(d1.dates), last(d1.dates)))
    s2 = stepnumber.(Ref(cal), (first(d2.dates), last(d2.dates)))
    smin, smax = min(s1[1], s2[1]), max(s1[2], s2[2])
    D = promote_type(eltype(d1.dates), eltype(d2.dates))
    alld = vcat(convert(Vector{D}, d1.dates), convert(Vector{D}, d2.dates))
    steps = vcat(collect(s1[1]:s1[2]), collect(s2[1]:s2[2]))
    axis = _date_axis(cal, alld, steps, smin, smax)
    vars, names, kindv = copy(d1.vars), copy(d1.names), copy(d1.kinds)
    for (v, n, k) in zip(d2.vars, d2.names, d2.kinds)
        i = findfirst(==(v), vars)
        if i === nothing
            push!(vars, v); push!(names, n); push!(kindv, k)
        elseif kindv[i] !== k
            throw(ArgumentError("$v has kind :$(kindv[i]) in the first dataset and :$k in the second"))
        end
    end
    T = promote_type(nonmissingtype(eltype(eltype(d1.values))), nonmissingtype(eltype(eltype(d2.values))))
    n1, nd = length(d1.stations), length(d1.stations) + length(d2.stations)
    o1, o2 = s1[1] - smin, s2[1] - smin
    vals = Matrix{Union{Missing,T}}[]
    for v in vars
        M = Matrix{Union{Missing,T}}(missing, nd, length(axis))
        i = findfirst(==(v), d1.vars)
        i === nothing || (M[1:n1, o1 .+ (1:length(d1.dates))] = d1.values[i])
        j = findfirst(==(v), d2.vars)
        j === nothing || (M[n1+1:nd, o2 .+ (1:length(d2.dates))] = d2.values[j])
        push!(vals, M)
    end
    keep = intersect(keys(d1.sitemeta), keys(d2.sitemeta))
    meta = (; (k => vcat(d1.sitemeta[k], d2.sitemeta[k]) for k in keep)...)
    lost = setdiff(union(keys(d1.sitemeta), keys(d2.sitemeta)), keep)
    isempty(lost) || @info "Dropping site metadata $(join(lost, ", ")): it is not given for every station."
    S = promote_type(eltype(d1.stations), eltype(d2.stations))
    st = vcat(convert(Vector{S}, d1.stations), convert(Vector{S}, d2.stations))
    stationcol = something(d1.stationcol, d2.stationcol, :station)
    return WeatherData(axis, st, vars, names, kindv, vals, meta, d1.crs, cal, d1.datecol, stationcol)
end

function Base.merge(d1::WeatherData, table; verbose::Bool = true, kw...)
    algkinds = (; (v => k for (v, k) in zip(d1.vars, d1.kinds))...)
    d2 = WeatherData(table; calendar = d1.calendar, crs = d1.crs, verbose, algkinds, kw...)
    return merge(d1, d2)
end

# ---------------------------------------------------------------- Tables.jl output (rows station → date)

Tables.istable(::Type{<:WeatherData}) = true
Tables.columnaccess(::Type{<:WeatherData}) = true
function Tables.columns(d::WeatherData)
    nd, nn = length(d.stations), length(d.dates)
    cols = Pair{Symbol,AbstractVector}[d.datecol => repeat(d.dates, nd)]
    d.stationcol === nothing || push!(cols, d.stationcol => repeat(d.stations, inner = nn))
    for (n, M) in zip(d.names, d.values)
        push!(cols, n => vec(permutedims(M)))
    end
    return (; cols...)
end

function Base.show(io::IO, d::WeatherData)
    print(io, "WeatherData(", length(d.stations), " stations × ", length(d.dates), " days, ",
        join(d.names, ", "), ")")
end

function Base.show(io::IO, ::MIME"text/plain", d::WeatherData)
    nd, nn = length(d.stations), length(d.dates)
    println(io, "WeatherData: $nd station$(nd == 1 ? "" : "s") × $nn days ($(first(d.dates)) to $(last(d.dates))), ",
        d.calendar)
    vs = [n === v ? "$n ($k)" : "$n → $v ($k)" for (n, v, k) in zip(d.names, d.vars, d.kinds)]
    print(io, "  variables: ", join(vs, ", "))
    isempty(d.sitemeta) || print(io, "\n  site metadata: ", join(keys(d.sitemeta), ", "), " (crs :", d.crs, ")")
    nmiss = sum(M -> count(ismissing, M), d.values)
    nmiss > 0 && print(io, "\n  missing values: ", nmiss)
end
