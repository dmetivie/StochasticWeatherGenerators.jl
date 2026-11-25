# FittedGenerator: a fitted class, its parameters and its FitInfo.

"""
    FittedGenerator(spec::AbstractGenerator, params, info::FitInfo)

A fitted weather generator: the resolved class `spec`, its fitted `params` and the [`FitInfo`](@ref) record.

[`fit`](@ref) returns it; [`simulate`](@ref) draws from it.
"""
struct FittedGenerator{G<:AbstractGenerator,P} <: StatsAPI.StatisticalModel
    spec::G
    params::P
    info::FitInfo
end

"""
    spec(g::FittedGenerator)

Return the resolved class (model specification) of `g`.
"""
spec(g::FittedGenerator) = g.spec

"""
    params(g::FittedGenerator)

Return the fitted parameters of `g`, in the class's own layout.
"""
StatsAPI.params(g::FittedGenerator) = g.params

"""
    fitinfo(g::FittedGenerator)

Return the [`FitInfo`](@ref) record of `g`.
"""
fitinfo(g::FittedGenerator) = g.info

dates(g::FittedGenerator) = g.info.dates
stations(g::FittedGenerator) = g.info.stations
variables(g::FittedGenerator) = g.info.variables
kinds(g::FittedGenerator) = NamedTuple{Tuple(g.info.variables)}(Tuple(g.info.kinds))
calendar(g::FittedGenerator) = g.info.calendar
sites(g::FittedGenerator) = g.info.sites
coords(g::FittedGenerator) = _coords(g.info.sites, g.info.crs)

"""
    distances(g::FittedGenerator; metric = nothing)

Return the station distance matrix stored at fit, or the one given by `metric`.
"""
function distances(g::FittedGenerator; metric = nothing)
    metric === nothing || return _distances(coords(g), metric)
    g.info.distances === nothing && throw(ArgumentError("the data of this fit had no station coordinates"))
    return g.info.distances
end

capabilities(g::FittedGenerator) = capabilities(g.spec)
likelihood_kind(g::FittedGenerator) = likelihood_kind(g.spec)
regime_source(g::FittedGenerator) = regime_source(g.spec)

"""
    nregimes(g::FittedGenerator)

Return the number of regimes of `g`, or `nothing` (the default) for a class without regimes.

Experimental: may change in minor releases.
"""
nregimes(::FittedGenerator) = nothing

# ---------------------------------------------------------------- likelihood statistics

function StatsAPI.loglikelihood(g::FittedGenerator)
    ll = g.info.loglikelihood
    ll === nothing && throw(ArgumentError("$(_classname(g.spec)) generators do not report a log-likelihood"))
    return ll
end

function StatsAPI.dof(g::FittedGenerator)
    k = g.info.dof
    k === nothing && throw(ArgumentError("$(_classname(g.spec)) generators do not report their degrees of freedom"))
    return k
end

function StatsAPI.nobs(g::FittedGenerator)
    n = g.info.nobs
    n === nothing && throw(ArgumentError("$(_classname(g.spec)) generators do not report their number of observations"))
    return n
end

function _check_ic(g::FittedGenerator, crit)
    k = likelihood_kind(g)
    k === :composite && throw(ArgumentError(
        "$crit is not valid for the composite likelihood of $(_classname(g.spec)) generators (CLIC is reserved)"))
    k === :none && throw(ArgumentError("$crit is not available: $(_classname(g.spec)) generators have no likelihood"))
    return nothing
end

"""
    bic(g::FittedGenerator; by = :total)
    aic(g::FittedGenerator; by = :total)
    aicc(g::FittedGenerator; by = :total)

Return the information criterion of `g`, in total or broken down `by` a class grouping such as `:block`.

They are refused for composite likelihoods and for classes without a likelihood.
"""
function StatsAPI.bic(g::FittedGenerator; by::Symbol = :total)
    _check_ic(g, "BIC")
    by === :total || return bic_by(g, by; criterion = :bic)
    return -2 * loglikelihood(g) + dof(g) * log(nobs(g))
end

function StatsAPI.aic(g::FittedGenerator; by::Symbol = :total)
    _check_ic(g, "AIC")
    by === :total || return bic_by(g, by; criterion = :aic)
    return -2 * loglikelihood(g) + 2 * dof(g)
end

function StatsAPI.aicc(g::FittedGenerator; by::Symbol = :total)
    _check_ic(g, "AICc")
    by === :total || return bic_by(g, by; criterion = :aicc)
    k, n = dof(g), nobs(g)
    n - k - 1 > 0 || throw(ArgumentError("AICc needs nobs > dof + 1 (nobs = $n, dof = $k)"))
    return aic(g) + 2k * (k + 1) / (n - k - 1)
end

"""
    bic_by(g::FittedGenerator, by::Symbol; criterion = :bic)

Return the `criterion` (`:bic`, `:aic` or `:aicc`) of `g` broken down by `by`; classes define it.

Experimental: may change in minor releases.
"""
bic_by(g::FittedGenerator, by::Symbol; criterion::Symbol = :bic) = throw(ArgumentError(
    "`$criterion(g; by = :$by)` is not available for $(_classname(g.spec)) generators; use `by = :total`"))

# ---------------------------------------------------------------- routing and read-outs

"""
    simulated_variables(g::FittedGenerator)

Return the canonical names of the variables that `g` simulates; the default is the fit variables.

Experimental: may change in minor releases.
"""
simulated_variables(g::FittedGenerator) = g.info.variables

"""
    component(g::FittedGenerator, var::Symbol)

Return the fitted component of `g` that owns variable `var`; one component may own several variables.

`g[var]` calls it after resolving aliases. Classes define it.

Experimental: may change in minor releases.
"""
component(g::FittedGenerator, var::Symbol) = throw(ArgumentError(
    "$(_classname(g.spec)) generators do not expose components (`g[:$var]`)"))

# original name of each simulated variable
function _simnames(g::FittedGenerator)
    vars = simulated_variables(g)
    return Symbol[(i = findfirst(==(v), g.info.variables); i === nothing ? v : g.info.names[i]) for v in vars]
end

function _simkinds(g::FittedGenerator)
    vars = simulated_variables(g)
    return Symbol[(i = findfirst(==(v), g.info.variables); i === nothing ? :custom : g.info.kinds[i]) for v in vars]
end

function Base.getindex(g::FittedGenerator, var::Union{Symbol,AbstractString})
    vars = simulated_variables(g)
    i = _varindex_or_error(vars, _simnames(g), var; what = "generator")
    return component(g, vars[i])
end

_unavailable(f, g) = throw(ArgumentError("`$f` is not available for $(_classname(spec(g))) generators"))

"""
    transitions(g::FittedGenerator)

Return the K×K×T regime transition matrices of `g`, for classes with regimes.
"""
transitions(g::FittedGenerator; kw...) = _unavailable(:transitions, g)

"""
    correlation(g::FittedGenerator, var; regime)

Return the station correlation matrix of variable `var` in `g`, for classes that model one.
"""
correlation(g::FittedGenerator, var; kw...) = _unavailable(:correlation, g)

"""
    regime_model(g::FittedGenerator)

Return the fitted regime model of `g`, for classes with regimes.
"""
regime_model(g::FittedGenerator; kw...) = _unavailable(:regime_model, g)

"""
    decode(g::FittedGenerator[, data]; method = :viterbi)

Return the most likely regime path of `data` (the fit data by default) under `g`, for classes with regimes.
"""
decode(g::FittedGenerator, args...; kw...) = _unavailable(:decode, g)

function Base.show(io::IO, g::FittedGenerator)
    print(io, "FittedGenerator{", _classname(g.spec), "}(", length(g.info.stations), " stations, ",
        join(g.info.names, ", "), ")")
end

function Base.show(io::IO, ::MIME"text/plain", g::FittedGenerator)
    info = g.info
    println(io, "FittedGenerator{", _classname(g.spec), "}: ", length(info.stations), " station",
        length(info.stations) == 1 ? "" : "s", ", variables ", join(info.names, ", "))
    println(io, "  window: ", first(info.dates), " to ", last(info.dates), ", ", info.calendar)
    print(io, "  spec: ")
    show(io, g.spec)
    info.loglikelihood === nothing ||
        print(io, "\n  loglikelihood: ", info.loglikelihood, " (", likelihood_kind(g), ")")
    info.seed === nothing || print(io, "\n  seed: ", info.seed)
end
