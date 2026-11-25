# fit: data in, FittedGenerator out.

"""
    fit(data, alg::AbstractGenerator; window = nothing, seed = nothing, solver = nothing, verbose = true)

Fit generator class `alg` to `data` (a Tables.jl table or a [`WeatherData`](@ref)) and return a
[`FittedGenerator`](@ref).

- `window` is a date range such as `Date(1986):Date(2019, 12, 31)`; the default is the period common to all
  stations. Days before the window are passed to the class as `past` only.
- The data must be complete in the window, unless the class handles missing values.
- `seed` (any integer) sets the fit streams; `nothing` draws one, recorded in `fitinfo(g).seed`.
- `solver` is the optimizer used by classes that need one; `nothing` lets them call `default_solver()`.
"""
function StatsAPI.fit(data, alg::AbstractGenerator; window = nothing, seed = nothing, solver = nothing,
        verbose::Bool = true)
    timestep(alg) == Day(1) || throw(ArgumentError(
        "$(_classname(alg)) has time step $(timestep(alg)); only daily (`Day(1)`) generators are supported"))
    kind = likelihood_kind(alg)
    kind in LIKELIHOOD_KINDS || throw(ArgumentError(
        "$(_classname(alg)) declares likelihood_kind :$kind; valid kinds are $(join(repr.(LIKELIHOOD_KINDS), ", "))"))
    s = seed === nothing ? rand(Random.default_rng(), UInt64) : seedword(seed)

    wd = _fit_data(data, alg, verbose)
    idx = _window_indices(wd, window === nothing ? default_window(wd; verbose) : window)
    _hascap(alg, :handles_missing) || check_complete(wd, (wd.dates[first(idx)], wd.dates[last(idx)]))
    wdata = _subset(wd, idx)
    pst = first(idx) > 1 ? _subset(wd, 1:first(idx)-1) : nothing

    alg2 = resolve(alg, wdata)
    alg2 isa AbstractGenerator || throw(ArgumentError(
        "`resolve` for $(_classname(alg)) returned a $(typeof(alg2)); it must return an AbstractGenerator"))
    g = fit_generator(wdata, alg2; past = pst, seed = s, rng = stream(s, :fit, 0), solver, verbose)
    g isa FittedGenerator || throw(ArgumentError(
        "`fit_generator` for $(_classname(alg2)) returned a $(typeof(g)); it must return a FittedGenerator"))
    solvername = solver === nothing ? "default" : string(nameof(typeof(solver)))
    info = _complete(g.info; seed = s, solver = solvername, versions = _versions(g.spec))
    return FittedGenerator(g.spec, g.params, info)
end

function _fit_data(data, alg, verbose)
    req = requested_variables(alg)
    if data isa WeatherData
        return req === nothing ? data : _select_vars(data, req)
    end
    return WeatherData(data; variables = req, algkinds = declared_kinds(alg), verbose)
end

function _versions(alg)
    v = Dict{String,String}("julia" => string(VERSION))
    v["StochasticWeatherGeneratorsCore"] = string(pkgversion(@__MODULE__))
    mod = Base.moduleroot(parentmodule(typeof(alg)))
    pv = pkgversion(mod)
    pv === nothing || (v[string(nameof(mod))] = string(pv))
    return v
end
