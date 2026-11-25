# Fit windows and the complete-data check.

_datecode(d::Date) = "Date($(Dates.year(d)), $(Dates.month(d)), $(Dates.day(d)))"
_datecode(d) = repr(d)

_window_value(d::WeatherData{<:Date}, i, j) = d.dates[i]:Day(1):d.dates[j]
_window_value(d::WeatherData, i, j) = (d.dates[i], d.dates[j])

# day indices of `window` in `d`
_window_indices(d::WeatherData, ::Nothing) = _window_indices(d, default_window(d; verbose = false))
function _window_indices(d::WeatherData, w::AbstractRange)
    isempty(w) && throw(ArgumentError("the window is empty"))
    step(w) == Day(1) || throw(ArgumentError("the window must have a step of one day, got $(step(w))"))
    return _dayrange(d, first(w), last(w))
end
_window_indices(d::WeatherData, w::Union{Tuple{Any,Any},Pair}) = _dayrange(d, first(w), last(w))
_window_indices(::WeatherData, w) = _not_a_window(w)
_window_indices(::WeatherData, w::AbstractRange{<:Real}) = _not_a_window(w)
_not_a_window(w) = throw(ArgumentError(
    "a window is a date range such as `Date(1990):Date(2019, 12, 31)` or a pair of dates, got $(repr(w))"))

# first and last day index where station s has a value for every variable; (0, -1) when none
function _station_span(d::WeatherData, s)
    f, l = 1, length(d.dates)
    for M in d.values
        row = view(M, s, :)
        i = findfirst(!ismissing, row)
        i === nothing && return 0, -1
        f, l = max(f, i), min(l, findlast(!ismissing, row))
    end
    return f, l
end

"""
    default_window(data; verbose = true)

Return the range of dates where every station has values, from each station's first to last day with every variable.

It prints one info line naming the limiting stations when the window is shorter than the data.

Experimental: may change in minor releases.
"""
function default_window(d::WeatherData; verbose::Bool = true)
    spans = [_station_span(d, s) for s in eachindex(d.stations)]
    a, b = maximum(first, spans), minimum(last, spans)
    if any(sp -> sp[1] == 0, spans) || a > b
        lines = [sp[1] == 0 ? "  $st: no day with every variable" : "  $st: $(d.dates[sp[1]]) to $(d.dates[sp[2]])"
                 for (st, sp) in zip(d.stations, spans)]
        throw(ArgumentError("the stations share no period with every variable; their ranges are:\n" *
                            join(lines, "\n") * "\nDrop stations or variables, or pass `window =`."))
    end
    if verbose && (a > 1 || b < length(d.dates))
        why = String[]
        a > 1 && push!(why, "starts with station $(d.stations[argmax(first.(spans))])")
        b < length(d.dates) && push!(why, "ends with station $(d.stations[argmin(last.(spans))])")
        @info "Window $(d.dates[a]) to $(d.dates[b]), common to all stations ($(join(why, ", ")))."
    end
    return _window_value(d, a, b)
end

"""
    check_complete(data, window = default_window(data))

Throw an error naming the station, variable and dates of the missing values of `data` within `window`.
"""
function check_complete(d::WeatherData, window = nothing)
    idx = _window_indices(d, window)
    gaps = Tuple{Int,Int,Int}[]   # (variable, station, count)
    for (v, M) in enumerate(d.values), s in eachindex(d.stations)
        c = count(ismissing, view(M, s, idx))
        c > 0 && push!(gaps, (v, s, c))
    end
    isempty(gaps) && return nothing
    v, s, c = first(gaps)
    row = view(d.values[v], s, idx)
    shown = [d.dates[idx[i]] for i in findall(ismissing, row)[1:min(5, c)]]
    name = d.names[v]
    msg = "$name has $c missing day$(c == 1 ? "" : "s") at station $(d.stations[s]) in the window " *
          "$(d.dates[first(idx)]) to $(d.dates[last(idx)]), e.g. $(join(shown, ", "))."
    if length(gaps) > 1
        others = ["($(d.names[g[1]]), $(d.stations[g[2]]))" for g in gaps[2:min(end, 6)]]
        msg *= "\nAlso affected (variable, station): $(join(others, ", "))" * (length(gaps) > 6 ? ", …" : "") * "."
    end
    msg *= "\nFix: impute them, e.g. `read_stations(...; impute = [:$name])` or Impute.jl"
    run = _longest_complete_run(d)
    if run === nothing
        msg *= ", since no day is complete."
    else
        msg *= ", or fit on a gap-free window, e.g. `window = $(_datecode(d.dates[first(run)])):" *
               "$(_datecode(d.dates[last(run)]))` (the longest gap-free run)."
    end
    throw(ArgumentError(msg))
end

function _longest_complete_run(d::WeatherData)
    ok = trues(length(d.dates))
    for M in d.values, t in eachindex(ok)
        ok[t] && any(ismissing, view(M, :, t)) && (ok[t] = false)
    end
    best, start, bstart = 0, 0, 0
    for t in eachindex(ok)
        if ok[t]
            start == 0 && (start = t)
            if t - start + 1 > best
                best, bstart = t - start + 1, start
            end
        else
            start = 0
        end
    end
    return best == 0 ? nothing : bstart:(bstart + best - 1)
end
