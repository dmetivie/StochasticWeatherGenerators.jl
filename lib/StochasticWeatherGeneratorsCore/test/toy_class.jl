# Toy generator classes written against the public API only (proof of openness).
module ToyGenerators

using Dates, Random, Statistics
using StochasticWeatherGeneratorsCore
import StochasticWeatherGeneratorsCore as SWGC

export SeasonalClimatology, CompositeToy, BrokenToy, MissingTolerantToy, ProbeToy, WeeklyToy, WeirdKindToy,
    NoFitToy, BadResolveToy, LAST_CTX

# ---------------------------------------------------------------- SeasonalClimatology

"""
Seasonal mean and standard deviation per (station, day of period), pooled over ±`window` days, with a lag-1
autocorrelation of the standardized anomalies; `K > 0` adds an iid uniform regime path; each group in
`joint` shares one innovation and one component.
"""
struct SeasonalClimatology <: SWGC.AbstractGenerator
    window::Int
    K::Int
    joint::Vector{Vector{Symbol}}
end
SeasonalClimatology(; window = 15, K = 0, joint = ()) =
    SeasonalClimatology(window, K, [collect(Symbol, g) for g in joint])

struct Component
    vars::Vector{Symbol}
    μ::Array{Float64,3}   # station × period index × variable
    σ::Array{Float64,3}
    φ::Matrix{Float64}    # station × variable
end

SWGC.capabilities(alg::SeasonalClimatology) = (regimes = alg.K > 0,)
SWGC.likelihood_kind(::Type{SeasonalClimatology}) = :exact
SWGC.declared_kinds(::Type{SeasonalClimatology}) = (WIND = :positive,)
SWGC.nregimes(g::FittedGenerator{SeasonalClimatology}) = spec(g).K > 0 ? spec(g).K : nothing
SWGC.component(g::FittedGenerator{SeasonalClimatology}, var::Symbol) =
    only(c for c in params(g).components if var in c.vars)

function SWGC.resolve(alg::SeasonalClimatology, data)
    alg.window >= 0 || throw(ArgumentError("window must be ≥ 0"))
    for g in alg.joint, v in g
        v in variables(data) || throw(ArgumentError("joint variable $v is not in the data"))
    end
    return alg
end

circdist(a, b, T) = (d = abs(a - b); min(d, T - d))
normlogpdf(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2

function SWGC.fit_generator(data, alg::SeasonalClimatology; past = nothing, seed, kw...)
    cal = SWGC.calendar(data)
    T = SWGC.period(cal)
    t = [SWGC.period_index(cal, d) for d in dates(data)]
    nd, nn = length(stations(data)), length(t)
    pools = [findall(tn -> circdist(tn, p, T) <= alg.window, t) for p in 1:T]
    vars = variables(data)
    groups = vcat(alg.joint, [[v] for v in vars if !any(g -> v in g, alg.joint)])
    comps = Component[]
    ll = 0.0
    for g in groups
        μ, σ, φ = zeros(nd, T, length(g)), ones(nd, T, length(g)), zeros(nd, length(g))
        for (k, v) in enumerate(g)
            X = Float64.(data[v])
            for s in 1:nd, p in 1:T
                x = X[s, pools[p]]
                length(x) >= 2 || continue
                μ[s, p, k] = mean(x)
                σ[s, p, k] = max(std(x), 1e-6)
            end
            for s in 1:nd
                z = [(X[s, n] - μ[s, t[n], k]) / σ[s, t[n], k] for n in 1:nn]
                φ[s, k] = nn > 1 ? clamp(sum(z[2:end] .* z[1:end-1]) / sum(abs2, z[1:end-1]), -0.95, 0.95) : 0.0
                ll += normlogpdf(z[1], 0, 1) - log(σ[s, t[1], k])
                for n in 2:nn
                    ll += normlogpdf(z[n], φ[s, k] * z[n-1], sqrt(1 - φ[s, k]^2)) - log(σ[s, t[n], k])
                end
            end
        end
        push!(comps, Component(g, μ, σ, φ))
    end
    history = (past_days = past === nothing ? 0 : length(dates(past)),
        past_last = past === nothing ? nothing : last(dates(past)), seed = seed)
    info = SWGC.FitInfo(data; loglikelihood = ll, dof = nd * length(vars) * (2T + 1) + max(alg.K - 1, 0),
        nobs = nd * nn * length(vars), history)
    return FittedGenerator(alg, (components = comps, K = alg.K), info)
end

function SWGC.simulate_member!(rng, buf, g::FittedGenerator{SeasonalClimatology}, ctx)
    p = SWGC.period_indices(ctx)
    L, nd = SWGC.nsteps(ctx), length(stations(ctx))
    gv, pst = SWGC.given(ctx), SWGC.past(ctx)
    pst isa Union{Nothing,WeatherData} || throw(ArgumentError("SeasonalClimatology takes `past` as data"))
    K = spec(g).K
    if K > 0
        z = SWGC.regimes(buf)
        haskey(gv, :regime) ? (z .= gv.regime) : (z .= rand(rng, 1:K, L))
    end
    p0 = pst === nothing ? 0 : SWGC.period_index(SWGC.calendar(ctx), last(dates(pst)))
    for c in params(g).components
        r = SWGC.stream(ctx, join(c.vars, "_"))
        outs = [buf[v] for v in c.vars]
        for s in 1:nd
            zprev = [pst === nothing ? NaN : (pst[v][s, end] - c.μ[s, p0, k]) / c.σ[s, p0, k] for (k, v) in enumerate(c.vars)]
            for l in 1:L
                ε = randn(r)
                for (k, v) in enumerate(c.vars)
                    φ = c.φ[s, k]
                    zk = isnan(zprev[k]) ? ε : φ * zprev[k] + sqrt(1 - φ^2) * ε
                    x = c.μ[s, p[l], k] + c.σ[s, p[l], k] * zk
                    if haskey(gv, v)
                        x = gv[v][s, l]
                        zk = (x - c.μ[s, p[l], k]) / c.σ[s, p[l], k]
                    end
                    outs[k][s, l] = x
                    zprev[k] = zk
                end
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------- classes for the edge cases

"Same fit as SeasonalClimatology, with a composite likelihood."
struct CompositeToy <: SWGC.AbstractGenerator end
SWGC.likelihood_kind(::CompositeToy) = :composite
function SWGC.fit_generator(data, alg::CompositeToy; kw...)
    g = SWGC.fit_generator(data, SeasonalClimatology(); kw...)
    return FittedGenerator(alg, params(g), fitinfo(g))
end

"Broken on purpose: `:nan` leaves a NaN, `:throw` throws in member 3, `:wrongtype` fits to a NamedTuple, `:nosim` cannot simulate."
struct BrokenToy{M} <: SWGC.AbstractGenerator end
BrokenToy(mode::Symbol) = BrokenToy{mode}()
SWGC.fit_generator(data, alg::BrokenToy; kw...) = FittedGenerator(alg, nothing, SWGC.FitInfo(data))
SWGC.fit_generator(data, ::BrokenToy{:wrongtype}; kw...) = (params = 1,)
function SWGC.simulate_member!(rng, buf, g::FittedGenerator{BrokenToy{:nan}}, ctx)
    foreach(v -> buf[v] .= 1.0, variables(buf))
    buf[first(variables(buf))][1, end] = NaN
end
function SWGC.simulate_member!(rng, buf, g::FittedGenerator{BrokenToy{:throw}}, ctx)
    SWGC.member(ctx) == 3 && throw(DomainError(3, "member 3 fails"))
    foreach(v -> buf[v] .= 0.0, variables(buf))
end

"Station means of the available values; it accepts gaps (`handles_missing`) and has no likelihood."
struct MissingTolerantToy <: SWGC.AbstractGenerator end
SWGC.capabilities(::Type{MissingTolerantToy}) = (handles_missing = true,)
SWGC.fit_generator(data, alg::MissingTolerantToy; kw...) = FittedGenerator(alg,
    Dict(v => [mean(skipmissing(data[v][s, :])) for s in eachindex(stations(data))] for v in variables(data)),
    SWGC.FitInfo(data))

const LAST_CTX = Ref{Any}(nothing)

"Requests TX only and writes each step's period index; it records the last context."
struct ProbeToy <: SWGC.AbstractGenerator end
SWGC.requested_variables(::ProbeToy) = [:TX]
SWGC.capabilities(::ProbeToy) = (regimes = true,)
SWGC.fit_generator(data, alg::ProbeToy; kw...) = FittedGenerator(alg, nothing, SWGC.FitInfo(data))
function SWGC.simulate_member!(rng, buf, g::FittedGenerator{ProbeToy}, ctx)
    LAST_CTX[] = ctx
    p = SWGC.period_indices(ctx)
    buf[:TX] .= p'
    SWGC.regimes(buf) .= SWGC.member(ctx)
end

struct WeeklyToy <: SWGC.AbstractGenerator end
SWGC.timestep(::Type{WeeklyToy}) = Week(1)

struct WeirdKindToy <: SWGC.AbstractGenerator end
SWGC.likelihood_kind(::WeirdKindToy) = :weird

struct NoFitToy <: SWGC.AbstractGenerator end

struct BadResolveToy <: SWGC.AbstractGenerator end
SWGC.resolve(::BadResolveToy, data) = :oops

function __init__()
    SWGC.register_type(SeasonalClimatology)
    SWGC.register_type(Component)
end

end
