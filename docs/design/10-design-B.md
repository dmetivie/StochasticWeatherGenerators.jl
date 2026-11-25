## 0. Thesis

There is one object graph, and it has three doors into it. An impact modeler types `fit(obs, RegimeHMM())`. The *same* `RegimeHMM` value, with every default resolved from the data, is what `show(fitted)` prints back as copy-pasteable Julia, and that printout is the Tier 1 spec. Each field of that spec is a block (`Rain()`, `Temperature(given = :TX)`, `Positive()`), and each block is a Tier 2 protocol implementer. Each fitted block holds the plain SPSM v5 models. Nothing in the easy layer dead-ends: a novice who edits one keyword is already working at Tier 1.

Three rules keep the novice vocabulary to six concepts:
- Defaults are resolved at `fit` time, from column names and data.
- A range anywhere (`K = 2:6`, `season = 0:2`) means "choose by BIC".
- Tables go in and tables come out, with the same column names.

The six concepts are: an observation table, `fit`, a model class, `simulate`, `DataFrame(sims)`, and `validate`, plus `save_generator`. SWG follows SciML on both axes:
- **Monorepo:** a thin `…Core` holds the verbs and the data and result types, there is one sublibrary per model class, and an umbrella re-exports them.
- **Swappable classes:** model classes are interchangeable like solvers, as in `fit(data, RegimeHMM(K = 4))` vs `fit(data, WGEN())`.

SPSM v5 becomes the domain-agnostic engine underneath. Its contract is a weighted fit, per-state log-densities, an explicit period index `t`, and multiple sequences.

## 1. Package architecture

```
StochasticWeatherGenerators.jl/                      one git repo, 7 registered packages
├── Project.toml                                     StochasticWeatherGenerators v4.0.0 (umbrella; SAME UUID as v3)
├── src/StochasticWeatherGenerators.jl               @reexport using lib/*; default solver = LBFGS()
├── lib/
│   ├── StochasticWeatherGeneratorsCore/             v1.0.0  verbs, WeatherData, calendars, results, validate
│   ├── StochasticWeatherGeneratorsBlocks/           v1.0.0  Rain/Temperature/Positive/Continuous, dynamics,
│   │                                                        copulas, `given` coupling, product engine (DAG)
│   ├── StochasticWeatherGeneratorsRegimeHMM/        v1.0.0  RegimeHMM, Regimes, TwoStage/SoftWeights
│   ├── StochasticWeatherGeneratorsWGEN/             v1.0.0  WGEN (Wilks 1998 / Srikanthan 2009)
│   ├── StochasticWeatherGeneratorsLatentGaussian/   v1.0.0  CensoredLatentGaussian (Ailliot 2009; Vaittinada Ayar 2020)
│   └── StochasticWeatherGeneratorsData/             v1.0.0  read_stations(ECA()/INRAE()/MeteoFrance()), example data
│       (later, each additive: …KNN, …GWEX, …MSTWeatherGen, …WeaGETS, …GWGEN, …AWEGEN2d)
├── docs/        one Documenter site; docs/Project.toml uses [sources] → ../lib/*
├── test/        umbrella integration + numerical equivalence (test/reference/v3/)
├── examples/    tuto_paper.jl, tuto_add_station_variable.jl, tuto_wgen.jl … rewritten
└── .github/workflows/  CI matrix GROUP ∈ {Core, Data, Blocks, RegimeHMM, WGEN, LatentGaussian, Umbrella, Docs, Downgrade}
```

| Package | Purpose | Hard deps | Exports (approx.) |
|---|---|---|---|
| **…Core** | The interface. Abstract `AbstractGenerator`, `WeatherData`, calendars, `FittedGenerator`, `WeatherSimulation`, verbs, validation statistics, save/load. Weakdeps: JLD2 (save/load), Makie/GeoMakie (plots, `map_with_stations`, `errorline*`) | Tables, Dates, Random, Statistics, StatsBase, StatsAPI, PrecompileTools | ~20 exported, ~15 `public` |
| **…Blocks** | The variables vocabulary. Marginals via SPSM, dynamics (`AR1`, `LatentAR1`, `Markov`), copulas (`MultisiteGaussian`, `SpatialExp`), `given` coupling, the block protocol, and the generic product engine (fit/simulate in DAG order given a driver) | Core, SPSM 5, Distributions, Copulas 1, LinearAlgebra, NearestCorrelationMatrix, LogExpFunctions | ~18 exported, ~15 `public` |
| **…RegimeHMM** | The flagship: a seasonal hidden regime chain × product of blocks, plus estimation modes | Core, Blocks, SPSM, HiddenMarkovModels 0.7 | 4 |
| **…WGEN** | Wilks multisite occurrence: per-site order-p Markov, latent Gaussian Ω fitted by moment matching | Core, Blocks, SPSM, SimpleNonlinearSolve | 2 |
| **…LatentGaussian** | Censored latent Gaussian rain (occurrence and amount from one field) | Core, Blocks, SPSM | 2 |
| **…Data** | Station readers, metadata, distances, example datasets as lazy Artifacts (this fixes the network download at precompile) | Core, CSV, DataFrames, Downloads, Impute | ~8 |
| **StochasticWeatherGenerators** (umbrella) | Batteries included. Re-exports all sublibraries and makes `solver = LBFGS()` the default, so Tier 0 never sees a solver | all of the above, Reexport, Optimization, OptimizationOptimJL, SimpleNonlinearSolve | ~55 total, of which a novice touches 6 |

**SPSM stays in its own repo.** The reasons:
- It is domain-agnostic (see `examples/tuto_genomics.jl`), has its own users, co-author and docs, and keeps its UUID history.
- `…Core` must not pull it in, so that a k-NN class depends on Core only. Lockstep is enforced by compat (`SmoothPeriodicStatsModels = "5"` in Blocks, RegimeHMM, WGEN and LatentGaussian) and by a coordinated tag order (§8).
- During development, `lib/*/Project.toml` points at the SPSM v5 branch with `[sources] SmoothPeriodicStatsModels = {url = "…SmoothPeriodicStatsModels.jl", rev = "v5"}`. This entry is removed before registering.
- TVP stays vendored inside SPSM for v5. The vector links go there.

**Load time.**
- Core loads in well under a second (no Distributions).
- Blocks brings in Distributions, Copulas and SPSM. The SPSM v5 load also drops the PeriodicHiddenMarkovModels dependency.
- The umbrella adds Optimization, OptimJL and ForwardDiff.
- Every sublibrary carries a `PrecompileTools.@compile_workload` that fits and simulates a 2-station × 3-year synthetic table, so the first `fit` is not a 60-second surprise.

**Registration and versioning.**
- Each `lib/X` has its own `Project.toml`, UUID and semver. It is registered with `@JuliaRegistrator register subdir=lib/X`, tagged `X-v1.0.0`, and TagBot and CompatHelper are configured with `subdirs`.
- Local development uses `[sources] X = {path = "lib/X"}`, which needs Julia ≥ 1.11 (we have 1.12). Pkg honours `[sources]` only for the active project, so registered resolution always goes through compat.
- Sublibraries start at 1.0.0 to signal a frozen interface.
- The umbrella major (4.x) moves only when the re-exported surface breaks. A new class sublibrary means a minor umbrella release.

## 2. Core interface

```julia
module StochasticWeatherGeneratorsCore
using Dates, Random, Tables, StatsAPI
import StatsAPI: fit, loglikelihood, nobs, dof, aic, bic

export WeatherData, Leap366, NoLeap365, fit, simulate, loglikelihood, aic, bic, nobs, dof,
       decode, validate, curve, spec, save_generator, load_generator,
       FittedGenerator, WeatherSimulation, members
public AbstractGenerator, AbstractCalendar, period, period_index, capabilities, resolve,
       candidates, fit_generator, simulate_member!, FitInfo, SimContext, SimBuffer

# ---- model classes ("algorithms"): unfitted, immutable specs -------------------------
abstract type AbstractGenerator end
capabilities(::Type{<:AbstractGenerator}) = (loglikelihood = false, regimes = false)

# ---- calendar: Date ↦ t ∈ 1:T -------------------------------------------------------
abstract type AbstractCalendar end
struct Leap366   <: AbstractCalendar end   # T = 366; 29 Feb is t = 60; non-leap years skip 60 (= v3 dayofyear_Leap)
struct NoLeap365 <: AbstractCalendar end   # climate-model output; a 29 Feb is an error
period(::Leap366) = 366
period(::NoLeap365) = 365
period_index(::Leap366, d::Date) = dayofyear(d) + (!isleapyear(d) && month(d) > 2)
function period_index(::NoLeap365, d::Date)
    month(d) == 2 && day(d) == 29 && throw(ArgumentError("29 February ($d) in a NoLeap365 calendar"))
    return dayofyear(d) - (isleapyear(d) && month(d) > 2)
end

# ---- the problem object --------------------------------------------------------------
struct WeatherData{T<:AbstractFloat,C<:AbstractCalendar,S,V<:NamedTuple}
    dates::StepRange{Date,Day}                 # COMPLETE daily axis; gaps are `missing`, never dropped
    stations::Vector{S}
    values::V                                  # (RR = D×N Matrix{Union{Missing,T}}, TX = …)
    coords::Union{Nothing,Matrix{Float64}}     # D×2 (lat, lon) if given
    calendar::C
    names::NamedTuple{(:date, :station),Tuple{Symbol,Symbol}}   # user's column names, reused on output
end
WeatherData(table; date = :auto, station = :auto, variables = :auto, coords = nothing,
            calendar = Leap366())
WeatherData(values::NamedTuple, dates::AbstractVector{Date}; stations = 1:D, coords = nothing,
            calendar = Leap366())                                    # arrays for experts

# ---- the fitted result: the ONLY result type of every class -------------------------
struct FittedGenerator{G<:AbstractGenerator,P} <: StatsAPI.StatisticalModel
    spec::G        # resolved: no `auto`, no ranges; `fit(data, spec(g))` refits it
    params::P      # class-owned content (for RegimeHMM: regimes + NamedTuple of fitted blocks)
    info::FitInfo  # dates, stations, variables, calendar, loglik, dof, nobs,
                   # history::Vector{Pair{Symbol,Any}} (every stage's FitHistory, kept),
                   # selection (nothing or a BIC table), provenance (versions, seed)
end
Base.getproperty(g::FittedGenerator, s::Symbol)   # g.TX, g.regimes → fitted blocks (sugar for g.params)

# ---- verbs ----------------------------------------------------------------------------
fit(data, alg::AbstractGenerator; rng = default_rng(), solver = default_solver(alg),
    reuse = nothing, verbose = true)                                   :: FittedGenerator
simulate([rng,] g::FittedGenerator, dates::AbstractVector{Date}; nsim = 1, past = nothing,
         burnin = 365, given = (;), threads = true)                    :: WeatherSimulation
Base.rand(rng::AbstractRNG, g::FittedGenerator, dates; kw...) = simulate(rng, g, dates; kw...)
loglikelihood(g, data); aic(g); bic(g); nobs(g); dof(g)                # if capabilities(G).loglikelihood
decode(g, data = training; method = :viterbi)                         # column table (date, regime[, p1…pK])
validate([rng,] g_or_gs, data; nsim = 100, stats = default_stats(g), level = 0.9) :: ValidationReport
curve(g, variable::Symbol, parameter::Symbol; station, regime = 1)    :: Vector{Float64}  # length T
spec(g) :: AbstractGenerator
save_generator(path, g); load_generator(path) :: FittedGenerator
```

**What a class implements** (all `public`, none exported). Core does everything else:

```julia
resolve(alg, data::WeatherData)     -> alg′      # fill defaults from data; validate; human error messages
candidates(alg′)                    -> iterator  # expands ranges; default (alg′,)
fit_generator(data, alg′; rng, solver, reuse, verbose) -> FittedGenerator
simulate_member!(rng, buf::SimBuffer, g::FittedGenerator, ctx::SimContext) -> buf    # ONE member
```

The generic `fit` in Core:

```julia
function StatsAPI.fit(data, alg::AbstractGenerator; rng = Random.default_rng(),
                      solver = default_solver(alg), reuse = nothing, verbose = true)
    wd = data isa WeatherData ? data : WeatherData(data)
    cands = collect(candidates(resolve(alg, wd)))
    length(cands) == 1 && return fit_generator(wd, only(cands); rng, solver, reuse, verbose)
    capabilities(typeof(alg)).loglikelihood || throw(ArgumentError(
        "$(nameof(typeof(alg))) has no likelihood, so a range cannot be chosen by BIC; give one value"))
    fits = [fit_generator(wd, c; rng, solver, reuse, verbose) for c in cands]
    return with_selection(argmin(bic, fits), fits)
end
```

The generic `simulate` in Core, which is why every class gets ensembles, Tables output, RNG handling and the calendar for free:

```julia
function simulate(rng::AbstractRNG, g::FittedGenerator, dates::AbstractVector{Date};
                  nsim::Integer = 1, past = nothing, burnin::Integer = 365, given = (;), threads = true)
    ctx   = SimContext(g, dates; past, burnin, given)   # t = period_index.(calendar, dates); checks
    seeds = rand(rng, UInt64, nsim)                     # drawn up front: result independent of nthreads
    buf   = SimBuffer(g, length(dates), nsim)
    if threads
        Threads.@threads for i in 1:nsim
            simulate_member!(Xoshiro(seeds[i]), member(buf, i), g, ctx)
        end
    else
        for i in 1:nsim
            simulate_member!(Xoshiro(seeds[i]), member(buf, i), g, ctx)
        end
    end
    return WeatherSimulation(buf, ctx)
end
```

**Tables.jl input contract**
- Any Tables.jl source is accepted: a DataFrame, `CSV.File`, a NamedTuple of vectors, or a vector of per-station DataFrames (vcat'ed).
- **Wide** format: a date column plus one column per variable. **Long** format: `variable` and `value` columns, detected automatically.
- **Date column:** eltype `Date`, or `DateTime` at midnight. Sub-daily values raise an error: "aggregate to daily first".
- **Station column:** optional (single station if absent), any eltype. Auto-detected from `station`, `STAID`, `site`, `id`, or set with `station = :col`.
- **Values:** `Real` or `missing`. `NaN` becomes `missing` with one warning. Duplicate (date, station) rows raise an error that lists the first three.
- **Variable names:** case-insensitive aliases select the presets:
  - RR/PRCP/pr → `Rain()`
  - TX/TMAX/tasmax → `Temperature()`
  - TN/TMIN/tasmin → `Temperature(given = :TX)` if TX is present, else `Temperature()`
  - QQ/rsds, ETPP/ET0/PET → `Positive()`

  The user's own names are kept. Unknown columns are ignored with an info line naming them and the fix (`add WIND = Positive()`).

**Tables.jl output contract**
- `WeatherSimulation` is a Tables.jl column table with columns `member`, `<date col>`, `<station col>`, the variables, and `regime` when the class has regimes.
- The input column names are reused, so code written for the observations runs on the simulations unchanged.
- The date, station and member columns are lazy index vectors, which costs O(1) memory.
- Experts get arrays: `sims[:TX]` is a D×N×nsim `Array`, and `sims.regime` is N×nsim.
- `members(sims)` iterates per-member tables. It is also `Tables.partitions`, so you can write files without materialising 20M rows.

**Calendar.**
- `Leap366()` is the default, so the period has one meaning everywhere (`T = 366`).
- Simulating over any `Date` range handles leap years automatically: t = 60 occurs only in leap years.
- Monthly or seasonal groupings (`by = :month`, `Monthly()`) are derived from the date in simulation and from t in bases. Both agree because the 366 calendar is the leap-year calendar.
- `n2t` disappears from the user surface.

**RNG.**
- Every stochastic verb takes an optional leading `rng`.
- Per-member streams are derived up front, so `simulate(Xoshiro(1), g, d; nsim = 300)` gives the same result with any thread count.
- `fit` takes `rng` for random initialisation and for WGEN's Monte Carlo moment matching.

**Initial conditions: one convention, `past`.**
- `past` holds observations of the days *before* `first(dates)`, given as a table or `WeatherData`. Each block reads the lags it needs: `order` days for occurrence and one day for AR.
- The regime z₀ comes from the training path if `past` lies inside it, otherwise it is drawn from a(t).
- `past = nothing` (the default) simulates `burnin` days from the initial law and discards them.
- This replaces `Y_past`, `y₁`, `z_ini`, `Y_ini` and `y_ini`.

**Conditioning.**
- `given = (regime = path,)`, or `given = (TX = table,)` for a root or parent variable, simulates everything else conditionally.
- Conditioning a parent on its child is rejected: "TX is a parent of TN; condition on TX, not TN".

**Missing data and gaps.**
- Data stay on a complete daily axis. Blocks never bridge gaps.
  - iid terms skip missing values.
  - AR and LatentAR terms drop the transition when y_{n−1} is missing; LatentAR keeps the marginal term.
  - Copula fits use pairwise-complete days.
  - The regime HMM marginalises missing sites in its emissions.
  - Runs of all-missing days split sequences (`seq_ends`).
- Imputation is never silent. The data readers offer `impute = [:RR]` explicitly.

**Validation.**
- `validate` simulates `nsim` members over the observed dates and compares statistics against a `level` envelope:
  - per variable: monthly mean, sd and quantiles, and annual maxima;
  - for RR: wet frequency, dry and wet spells, and monthly totals;
  - inter-site correlations per variable, inter-variable correlations per site, and regime frequency per month.
- `show` prints a pass/flag summary. `plot(report)` comes from the Makie extension. `validate([g1, g2, g3], obs)` compares classes side by side.

**Save/load.**
- `save_generator`/`load_generator` (a JLD2 weakdep extension) write *plain* data:
  - the spec, via `to_plain`/`from_plain`, whose default is derived from the fields so expert blocks get it free;
  - flat coefficient vectors;
  - provenance and a format version.
- The names avoid the `save`/`load` clash with CairoMakie and FileIO, which the crop tutorial loads anyway.

## 3. Three tiers, by example

### T0: impact modeler (the crop tutorial)

```julia
using StochasticWeatherGenerators, DataFrames, Dates

obs  = read_stations(INRAE(), station_urls)        # DATE, STAID, RR, TN, TX, QQ, ETPP (4 stations)
swg  = fit(obs, RegimeHMM())                       # K = 4; presets chosen from the column names
sims = simulate(swg, Date(1986):Date(2035, 12, 31); nsim = 300)   # 300 × 50 years
df   = DataFrame(sims)                             # member, DATE, STAID, RR, TN, TX, QQ, ETPP, regime
validate(swg, obs)                                 # pass/flag table; `plot(ans)` with CairoMakie
```

To reuse the paper's 37-station regimes exactly as the v3 tutorial does, change one line:

```julia
swg = fit(obs, RegimeHMM(regimes = Regimes(load_generator(example_path(:france_regimes_K4)))))
```

`show(swg)` is the bridge to T1. It is valid Julia:

```
FittedGenerator{RegimeHMM}: 4 stations × 5 variables, 1986-01-01 … 2019-12-31 (12 418 days, 0.7 % missing)
RegimeHMM(
    regimes = Regimes(K = 4, season = 1, from = :RR),          # K fixed; try K = 2:6 to choose by BIC
    RR   = Rain(occurrence = Markov(order = 1), amount = Mixture(Exponential, 2),
                copula = MultisiteGaussian(by = :regime), season = 1),
    TX   = Temperature(dynamics = AR1(), copula = MultisiteGaussian(by = :regime), season = 1),
    TN   = Temperature(given = :TX, marginal = Gamma, season = 1),   # TX − TN ~ Gamma, coupled to TX
    QQ   = Positive(marginal = Gamma, dynamics = LatentAR1(), season = 1),
    ETPP = Positive(marginal = Gamma, dynamics = LatentAR1(), season = 1),
    estimation = TwoStage(),
)
  loglik = -1.23e5   BIC = 2.51e5   fitted in 2 min 13 s
  next: simulate(swg, dates; nsim) · validate(swg, obs) · spec(swg) → edit and fit(obs, ans)
```

Before fitting, `RegimeHMM()` shows `blocks = auto (chosen from your columns at fit time)`.

The concepts a novice learns are: (1) the observation table, (2) `fit`, (3) `RegimeHMM()`, (4) `simulate(...; nsim)`, (5) `DataFrame(sims)`, and (6) `validate`. `save_generator` is the optional seventh.

Error messages name the column, station and date, and give the fix as code:

```
ERROR: TN is modelled as Temperature(given = :TX) but there is no TX column.
       Use RegimeHMM(TN = Temperature()) to model TN on its own.
ERROR: QQ < 0 at STAID 80557001 on 2003-02-11 (-0.4); Positive() needs values ≥ 0.
       Fix the data, or model it as QQ = Continuous(Normal).
WARN:  Regime 4 × summer has 6 wet days at STAID 63345002: RR amounts there are poorly identified.
       Consider Rain(season = 0) or RegimeHMM(K = 3).
```

Student variant: swapping classes, which is the SciML point.

```julia
obs = example_data(:france_eca_rain)                       # bundled Artifact, RR only
gens = [fit(obs, m) for m in (RegimeHMM(K = 4), WGEN(order = 2), CensoredLatentGaussian())]
validate(gens, obs)                                         # same statistics, side by side
```

### T1: applied statistician (same objects, customised)

```julia
using StochasticWeatherGenerators, Distributions, Dates

france = load_generator("france_K4.jld2")          # RegimeHMM fitted on 37 ECA stations (tuto_paper)
model = RegimeHMM(
    regimes = Regimes(france),                      # frozen: A(t), ẑ and γ reused, never refitted
    RR   = Rain(occurrence = Markov(order = 1),
                amount = Mixture(Exponential, Gamma),              # heterogeneous 2-component mixture
                copula = MultisiteGaussian(by = :regime)),
    TX   = Temperature(dynamics = AR1(),                           # AR on the variable (v3 form)
                       season = (μ = 2, ρ = 0, σ = 1),             # per-parameter trig degree
                       copula = MultisiteGaussian(by = (:regime, :month))),   # K × 12 matrices
    TN   = Temperature(given = :TX, marginal = Gamma, season = (α = 1, θ = 2)),  # ΔT = TX − TN
    QQ   = Positive(marginal = Gamma, dynamics = LatentAR1(), season = 0:2),     # degree chosen by BIC
    ETPP = Positive(marginal = Weibull),
    estimation = SoftWeights(),                     # posterior γ instead of hard ẑ
)
swg = fit(obs, model)

swg.TX                                              # block summary: params, copulas, history
curve(swg, :TX, :ρ; station = 49215002, regime = 2) # 366-vector
swg.TX.models[1]                                    # station 1's SPSM SeasonalAR1: the T2 door
swg.info.selection                                  # BIC table for QQ's season

# Adding stations: matching fitted pieces are reused, new stations fitted, copulas refitted on all.
swg2 = fit(vcat(obs, obs_new), spec(swg); reuse = swg)
```

To change one thing only, write `RegimeHMM(TN = Temperature(given = :TX, marginal = Weibull))`. Unlisted variables stay `auto` (`others = :ignore` to drop them). The T1 vocabulary is about 14 names: `Regimes`, `Rain`, `Temperature`, `Positive`, `Continuous`, `Markov`, `Mixture`, `Monthly`, `AR1`, `LatentAR1`, `MultisiteGaussian`, `SpatialExp`, `TwoStage`, `SoftWeights`. The keywords are `season`, `given`, `copula`, `dynamics` and `marginal`. `nothing` always means "none" (`dynamics = nothing` is iid given the regime, `copula = nothing` is independent sites).

### T2a: expert, a new variable block

The example is a regime-analog block. It resamples whole multisite vectors from observed days with the same regime and calendar window, which is useful for awkward variables such as humidity.

```julia
using StochasticWeatherGeneratorsBlocks
import StochasticWeatherGeneratorsBlocks as B

struct RegimeAnalog <: B.AbstractBlock
    window::Int
end
RegimeAnalog(; window = 15) = RegimeAnalog(window)

B.capabilities(::RegimeAnalog) = (logdensities = false, soft_weights = false)  # → SoftWeights rejected politely

function B.fit_block(b::RegimeAnalog, ctx::B.FitContext)
    y = ctx.y[ctx.name]                              # D × N, Union{Missing,Float64}
    z = B.hard_regimes(ctx)                          # argmax of the N × K weights
    pool = Dict{Tuple{Int,Int},Vector{Int}}()
    for n in axes(y, 2)
        any(ismissing, view(y, :, n)) && continue
        for dt in -b.window:b.window
            push!(get!(pool, (z[n], mod1(ctx.t[n] + dt, ctx.period)), Int[]), n)
        end
    end
    return (y = Matrix{Float64}(coalesce.(y, NaN)), pool = pool)
end

function B.simulate_block!(rng, out::B.BlockOutput, b::RegimeAnalog, θ, ctx::B.SimContext)
    v = out.name
    for n in eachindex(ctx.t)
        out.values[v][:, n] .= view(θ.y, :, rand(rng, θ.pool[(ctx.z[n], ctx.t[n])]))
    end
    return out
end
B.nparams(::RegimeAnalog, θ) = 0

swg = fit(obs, RegimeHMM(HU = RegimeAnalog(window = 10)))   # composes like any preset
```

### T2b: expert, a new model-class sublibrary (k-NN resampler, Lall & Sharma 1996)

This depends on Core only. The internal helper names are speculative.

```julia
module StochasticWeatherGeneratorsKNN
using StochasticWeatherGeneratorsCore, Random
import StochasticWeatherGeneratorsCore as Core
export KNNResampler

Base.@kwdef struct KNNResampler <: Core.AbstractGenerator
    k::Union{Int,Nothing} = nothing      # nothing → ≈ √(candidates), resolved at fit
    window::Int = 7                       # ± days around the calendar day
    features::Vector{Symbol} = Symbol[]   # empty → all variables, station-averaged anomalies
end
Core.capabilities(::Type{KNNResampler}) = (loglikelihood = false, regimes = false)

function Core.fit_generator(data::Core.WeatherData, alg::KNNResampler; kwargs...)
    feats = isempty(alg.features) ? collect(keys(data.values)) : alg.features
    X  = Core.standardized_features(data, feats)         # F × N, anomalies per calendar day, NaN if missing
    t  = Core.period_index.(Ref(data.calendar), data.dates)
    ok = [n for n in 1:size(X, 2)-1 if all(isfinite, view(X, :, n)) && all(isfinite, view(X, :, n + 1))]
    k  = something(alg.k, max(1, round(Int, sqrt(length(ok) * (2alg.window + 1) / Core.period(data.calendar)))))
    return Core.FittedGenerator(KNNResampler(k, alg.window, feats), (; X, t, ok, data), Core.FitInfo(data))
end

function Core.simulate_member!(rng, buf::Core.SimBuffer, g::Core.FittedGenerator{KNNResampler}, ctx::Core.SimContext)
    (; X, t, ok, data) = g.params
    k, w = g.spec.k, g.spec.window
    kern = cumsum([1 / i for i in 1:k]); kern ./= kern[end]
    cur = Core.initial_day(rng, g, ctx)                   # observed day matching `past`, else same t
    for n in eachindex(ctx.t)
        cands = [j for j in ok if Core.circdist(t[j], ctx.t[n], length(kern) > 0 ? 366 : 366) ≤ w]
        d  = [sum(abs2, view(X, :, cur) .- view(X, :, j)) for j in cands]
        nn = cands[partialsortperm(d, 1:min(k, length(d)))]
        cur = nn[min(searchsortedfirst(kern, rand(rng)), length(nn))] + 1   # successor of a neighbour
        Core.copy_day!(buf, n, data, cur)                 # all sites × variables: dependence preserved
    end
    return buf
end
end
```

An MSTWeatherGen-like class has the same shape. It would reuse `Regimes(path = weather_types_table)` as a driver and Blocks' marginals for the Gaussian anamorphosis. It adds a space-time covariance fit (Gneiting class, weighted composite likelihood using `data.coords`) and a latent VAR(1) simulator inside `simulate_member!`. It is not a product of blocks, and Core does not require it to be.

## 4. Flagship Regime-HMM product model

**Spec objects** (in Blocks and RegimeHMM):

```julia
struct RegimeHMM{R,B<:NamedTuple,E} <: AbstractGenerator
    regimes::R          # Regimes | FrozenRegimes | NoRegimes
    blocks::B           # variable => AbstractBlock (or Auto())
    estimation::E       # TwoStage() | SoftWeights()   (JointEM reserved, additive)
    others::Symbol      # :auto | :ignore
end
RegimeHMM(; K = nothing, regimes = Regimes(K = something(K, 4)), estimation = TwoStage(),
          others = :auto, blocks...) = RegimeHMM(regimes, NamedTuple(blocks), estimation, others)

Base.@kwdef struct Regimes{KT,S}
    K::KT = 4                    # Int, or range → BIC
    season::S = 1                # transitions' basis (Int = trig degree)
    from::Symbol = :RR           # variable whose occurrence drives the regimes
    order_by::Symbol = :wetness  # stable labels: regime 1 = wettest
end
Regimes(g::FittedGenerator) = FrozenRegimes(g)           # reuse A(t), ẑ, γ from another fit
Regimes(; path) = ObservedRegimes(path)                  # given weather types (date, regime); A(t) fitted

Rain(; occurrence = Markov(order = 1), amount = Mixture(Exponential, 2),
       copula = MultisiteGaussian(by = :regime), season = 1, wet = 0.0)
Markov(; order = 1, copula = nothing)                    # occurrence memory; copula: cross-site latent
Temperature(; given = nothing,
              marginal = given === nothing ? Normal : Gamma,
              dynamics = given === nothing ? AR1() : nothing,
              copula   = given === nothing ? MultisiteGaussian() : nothing,
              season = 1, coupling_season = 1)
Positive(; marginal = Gamma, dynamics = LatentAR1(), copula = MultisiteGaussian(), season = 1, given = nothing)
Continuous(marginal; dynamics = LatentAR1(), copula = MultisiteGaussian(), season = 1, given = nothing)
MultisiteGaussian(; by = :regime, estimator = :pairwise) # by ∈ :none | :regime | :month | (:regime, :month)
```

**Model.** Write t_n = period_index(date_n). The regime path is z, and blocks b are fitted in topological order of the DAG declared by `given`:

p(z, y) = a(z₁) · ∏ₙ A_{z_{n−1} z_n}(t_{n−1}) · ∏ₙ ∏_b p_b(yₙᵇ | zₙ, tₙ, y_{n−1}ᵇ, ε_n^{pa(b)})

- **RR.**
  - *Occurrence:* O_{s,n} = 1{RR > wet}, with P(O_{s,n} = 1 | z = k, h) = logistic(B(t)β_{s,k,h}), where h is the memory category of the last `order` days. Sites are conditionally independent given z by default; `Markov(copula = …)` switches to a latent Gaussian.
  - *Amount on the wet set W_n:* (R_s)_{s∈W_n} ~ Sklar(GaussCop(C_k[W,W]), F^R_{s,k,t}). F^R is a seasonal SPSM mixture.
- **TX (`AR1`, form A).** X = μ_{s,k}(t) + ρ_{s,k}(t)X_{n−1} + σ_{s,k}(t)ε. Here ε_{·,n} ~ N(0, C_k) and is exported as the block's **innovations**.
- **TN (`given = :TX`).** The modelled variable is Δ = TX − TN, with the transform chosen automatically: a difference because Gamma has positive support, so TN ≤ TX always. ξ = Φ⁻¹(F^Δ_{s,k,t}(Δ)) with ξ = r_{s,k}(t)·ε^{TX} + √(1−r²)·η. The coupling r has a tanh link and degree `coupling_season`.
- **QQ, ETPP (`LatentAR1`, form B).** w = Φ⁻¹(F_{s,k,t}(x)), w_n = ρ_{s,k}(t)w_{n−1} + √(1−ρ²)ε with ε ~ N(0, C_k). Positivity and the marginal are exact by construction: no clipping.

**Estimation API.** It is all the same `fit(obs, RegimeHMM(…; estimation = …))`.

1. **Stage 1: regimes.**
   - `Regimes(K, from = :RR)`: SPSM `SeasonalHMM(K, BernoulliMemory(D; order))` fitted by Baum–Welch with seasonal transitions.
     - Initialisation uses SPSM's slice initialiser (formerly `fit_mle_all_slices` + `fit_θ!`).
     - Labels are then sorted by wetness.
     - The fitted RR occurrence emission **is** the Rain occurrence of those stations, which reproduces tuto_paper.
   - `FrozenRegimes`: nothing is fitted. ẑ and γ are taken from the frozen fit on the overlapping dates, and dropped dates are reported.
2. **Stage 2: blocks given regimes.** The weights are W = one-hot(ẑ) for `TwoStage()` or γ for `SoftWeights()`. For each block in DAG order, `θ_b = fit_block(b, FitContext(v, data, W, fitted_parents))`. Every block calls SPSM's weighted M-step, so "given z" and "soft" are one code path. Within a block:
   - Marginals are fitted per site × regime by weighted seasonal MLE.
   - Then dynamics: AR1 jointly with its marginal; LatentAR1 by IFM on the scores.
   - Then copulas by IFM, using weighted pairwise-complete correlations of innovations, or of co-wet normal scores for RR, followed by `nearest_cor`.
   - Then the `given` coupling, by weighted MLE of ξ | ε^{parent}. This is the same quantity simulation uses, which fixes the v3 mismatch.
   - With `season = 0:2`, each candidate is fitted and the lowest block BIC is kept.
3. **`JointEM()` is reserved.** In the E-step, regimes would use Σ_b `logdensities!` as emissions; the M-step runs the same weighted `fit_block`s. The protocol already carries everything it needs, so adding it later is non-breaking. It is not in the launch scope.

**Simulation** (one member; Core does the ensembles):

```julia
function Core.simulate_member!(rng, buf, g::FittedGenerator{<:RegimeHMM}, ctx)
    z = haskey(ctx.given, :regime) ? ctx.given.regime :
        sample_regimes(rng, g.params.regimes, ctx)          # A(t_{n−1}); z₀ from past or burn-in
    buf.regime .= z
    for v in g.params.order                                  # DAG order: RR, TX, TN, QQ, ETPP
        fb = g.params.blocks[v]
        B.simulate_block!(rng, B.BlockOutput(buf, v), fb.spec, fb.θ, B.SimContext(ctx, z, buf))
    end
    return buf
end
```

- Rain draws occurrence with memory per site, then a latent u ~ N(0, C_k[W,W]) on the wet set, then R = F⁻¹(Φ(u)).
- TX draws multisite innovations and runs the AR.
- TN reads `buf` for TX's innovations and returns TN = TX − F_Δ⁻¹(Φ(ξ)).
- QQ and ETPP run the latent AR, then apply the quantile transform.

## 5. WGEN port and censored latent Gaussian

**WGEN.** It is a product with *no regimes* (`NoRegimes`, K = 1) whose occurrence copula is the Wilks latent Gaussian.

```julia
Base.@kwdef struct WGEN{A,S} <: AbstractGenerator
    order::Int = 1                               # Markov order p per site
    season::S = Monthly()                        # Wilks: monthly parameters (IndicatorBasis on t)
    amount::A = Mixture(Exponential, 2)          # nothing → occurrence only
    Ω = MomentMatching(nsim = 2000)              # v3 estimator; Analytic() for order 0
    others::Symbol = :auto                       # other columns get presets with K = 1
end
resolve(w::WGEN, data) = resolve(RegimeHMM(regimes = NoRegimes(), others = w.others,
    RR = Rain(occurrence = Markov(order = w.order,
              copula = MultisiteGaussian(by = :month, estimator = w.Ω)),
              amount = w.amount, season = w.season)), data) |> as_wgen
```

- **Fit.**
  - Per site × month, the transition probabilities are the closed-form counts. Blocks detects an `IndicatorBasis` and skips the optimiser, so the numbers are identical to v3 `fit_markov_chain`.
  - Ω_m is fitted per site pair by bracketing root-finding (ITP from SimpleNonlinearSolve) on the occurrence correlation, then projected with `nearest_cor`.
- **Simulate.** Each day, U ~ N(0, Ω_{month}) and O_s = U_s ≤ Φ⁻¹(p_s(t, h)). Amounts come from the shared Rain amount block.
- **Likelihood.** It has none: `capabilities(WGEN).loglikelihood = false`. `loglikelihood` errors with "compare with validate()".
- **Show.** `spec(g)` prints `WGEN(order = 2)`, and `expand(spec(g))` prints the equivalent RegimeHMM form, which is a teaching device.
- **Portability.** Because the occurrence copula is just a copula value, `RegimeHMM(RR = Rain(occurrence = Markov(copula = MultisiteGaussian(by = :regime, estimator = MomentMatching()))))` gives a *regime-dependent* Wilks model for free.

**Censored latent Gaussian (CLG).** One latent field W_{s,n} with marginal law Φ. R = 0 if Φ(W) ≤ p0_{s,k}(t); otherwise R = G⁻¹_{s,k,t}((Φ(W) − p0)/(1 − p0)). Over time W_n = ρW_{n−1} + √(1−ρ²)ε with ε ~ N(0, C_k).

```julia
Base.@kwdef struct CensoredLatentGaussian{R} <: AbstractGenerator
    regimes::R = NoRegimes()                     # Regimes(K = 3) ≈ Ailliot et al. 2009 (two-stage)
    marginal = Gamma                             # positive part G
    season = 1
    dynamics = LatentAR1()                       # the SAME type as Positive()'s default
    copula = MultisiteGaussian(by = :regime)     # or SpatialExp() with station coords
    estimator = PairwiseCensored()               # pairwise censored likelihood (bivariate normal cdf)
    others::Symbol = :auto
end
```

The CLG sublibrary adds only a `CensoredRain <: AbstractBlock`, which has the marginal `ZeroInflated(G, p0)` and a censored pairwise likelihood. It reuses the SPSM fast bivariate CDF.
- **Fit:**
  1. p0 (weighted logistic) and G on wet days, per site × regime × season, using the same SPSM calls as Rain.
  2. ρ_s from (n−1, n) pairs and C from (s, s′) same-day pairs, by maximising the pairwise censored likelihood with regime weights.
  3. `nearest_cor`.
- **Simulate:** a latent AR field, then threshold, then transform.
- **Likelihood:** none for D > 2 (orthant probabilities).

**What the three classes share**

| Piece | RegimeHMM | WGEN | CLG |
|---|---|---|---|
| Driver (`NoRegimes` / `Regimes` / `FrozenRegimes`) | Regimes | NoRegimes | either |
| Latent-Gaussian thresholding `O = U ≤ Φ⁻¹(p)` | optional (`Markov(copula=…)`) | core | core (censoring) |
| `MultisiteGaussian` type, IFM fitting, `nearest_cor` | amounts, innovations | occurrence Ω (`MomentMatching`) | field C (`PairwiseCensored`) |
| Seasonal marginals via SPSM weighted fit | all blocks | amounts | p0, G |
| `LatentAR1` | QQ, ETPP | — | the field |
| Product engine, `given` DAG, `others = :auto` | yes | yes | yes |
| Calendar, `past`, `nsim`, Tables I/O, validate | Core | Core | Core |

## 6. SPSM v5

**Engine contract.** Every model `m <: AbstractSeasonalModel` provides the methods below. `t` (the period index) is a **required** keyword everywhere, which fixes the wrong defaults of pain point #10.

```julia
m  = SeasonalAR1(period = 366, K = 4, basis = (μ = 2, ρ = 0, σ = 1))   # Int = trig degree, or AbstractBasis
f  = fit(m, y; t, z = nothing, weights = nothing, seq_ends = nothing, past = nothing,
         solver, maxiter, tol, init = :data)          # SeasonalFit: model + FitHistory + StatsAPI
m1 = fit_mle(m, y, W::AbstractMatrix; t, seq_ends, past, solver)   # weighted, MODEL ONLY → nests in EM/HMMs
LL = logdensities(m1, y; t, seq_ends, past)           # N × K per-state log densities
ε  = innovations(m1, y; t, z, seq_ends)               # standardised (AR-type models)
d  = m1(152; state = 2, past = 18.3)                  # the Distribution at t = 152
y′ = rand(rng, m1, t; z, past)                        # HMMs return (states = …, obs = …)
curve(m1, :ρ; state = 2); coef(m1)                    # T-vector; ComponentVector
```

- **Given z.** `z` is sugar for one-hot `weights`, and it is available on *every* model: AR1, mixture, LatentAR1, HMM emissions.
- **Multiple sequences.** `seq_ends` (the HMMs.jl convention) is accepted everywhere. AR models restart at sequence boundaries instead of bridging them.
- **Every HMM emission is itself a seasonal model** with `fit_mle(m, y, w)`. `SeasonalHMM(K, emission)` therefore accepts `BernoulliMemory(D; order)`, any `TimeVaryingFamily`, or a product.
- **The spatial HMM** stays as it is (`public`) until SpatialBernoulli.jl exists.

**Vector links.**
- `SoftmaxLink(K)` enters `ParamSpec` in v5.0. Transitions and mixture weights go through it, which deletes the 4 hand-written softmaxes and the ignored weight link.
- A correlation link (canonical partial correlations + tanh) is deferred to v5.1. It is additive and would enable `MultisiteGaussian(by = :regime, season = 1)` later.

**v4 exports (94) → v5**

- **Kept, exported (32):**
  - L0 parametrisation: `ParamSpec`, `TrigBasis`, `ConstantBasis`, `FourierBasis`, `PolynomialBasis`, `IndicatorBasis`, `BSplineBasis`, `CyclicBSplineBasis`, `IdentityLink`, `LogLink`, `LogitLink`, `IntervalLink`, `curve`, `TimeVaryingFamily`, `seasonal_family`.
  - Fitting and StatsAPI: `fit`, `fit_mle` (new meaning: model only), `FitHistory`, `converged`, `SeasonalFit`, `coef`, `loglikelihood`, `dof`, `nobs`, `aic`, `aicc`, `bic`.
  - Read-outs: `viterbi`, `components`, `ncomponents`, `mixtures`.
  - `parameters` is **removed** in favour of `coef` (see below), so read this as 31 plus the new names.
- **New, exported (7):** `SoftmaxLink`, `SeasonalAR1`, `SeasonalLatentAR1`, `SeasonalMixture`, `SeasonalHMM`, `BernoulliMemory`, `logdensities`.
- **Renamed:**
  - `SmoothAR1` → `SeasonalAR1`
  - `SmoothSeasonalMixture` → `SeasonalMixture`
  - `SmoothPeriodicHMM`, `SmoothARPeriodicHMM` and `SmoothsARPeriodicHMM` → `SeasonalHMM(K, emission)`. The merge is at the API only; the Bernoulli kernels are kept behind dispatch.
  - `loglikelihoods` → `logdensities`
  - keyword `n2t` → `t`
  - `Y_past`, `y₁`, `y_past` → `past`
  - `degree` / `𝐃𝐞𝐠` → `basis`
- **`public` only (≈38):**
  - parametrisation layout: `paramnames`, `ncoef`, `coefs`, `blockindex`, `blockrange`, `design_matrix`, `nbasis`, `nparams`, `ncolumns`, `column`
  - objectives and M-step plumbing: `PeriodicDesign`, `PeriodicNLL`, `PointwiseNLL`, `group_by_time`, `distributions`, `update_dist`
  - EM knobs: `ClassicEM`, `StochasticEM`
  - solver plumbing: `default_nlls_solver`, `default_adtype`, `default_ensemble`, `solve_unconstrained`, `solve_nlls`; `fit_loss_optim` is renamed `solve_objective`
  - AR internals: `ar1_spec`, `ar1_sufficient_statistics`, `ar1_nll`, `AR1`
  - layout and evaluated twins: `component_coefs`, `component_range`, `PeriodicHMMDists`, `emission_coefs`, `emission_range`, `degrees`, `ARPeriodicHMM`, `sARPeriodicHMM`, `ARPeriodicHMMSpatial`, `SmoothARPeriodicHMMSpatial`
  - HMMs.jl bridge: `SeasonalControls`
- **Removed (19):**
  - `TimeVaryingFit` (merged into `SeasonalFit`)
  - `mixture_weight_spec` (replaced by `SoftmaxLink`); `mixture_weights` (replaced by `curve(m, :weights)`)
  - `Trig2ARPeriodicHMM`, `Trig2ARPeriodicHMMSpatial`, `randARPeriodicHMM` (replaced by constructors)
  - `fit_θ`, `fit_θ!`, `fit_θᴬ!`, `fit_θᴮ!` (internal to `init = :slices`)
  - `fit_mle_all_slices` (same)
  - `fit_mle_RO` (replaced by weighted `fit_mle(BernoulliMemory…)`)
  - `μₜ`, `ρₜ`, `σₜ`, `αₜ` (replaced by `curve`)
  - `n_to_t` (`t` is now required); `params` and `parameters` (replaced by `coef`)

**Moves.**
- **SPSM → SWG Blocks:**
  - the multisite `rand(::Vector{AR1}, …, Σ)`, which is buggy and untested, becomes `MultisiteGaussian`;
  - the role of `n_to_t` goes to the Calendar.
- **SWG → SPSM:**
  - `_ar1_init`, the data-driven start, becomes `init = :data`;
  - `_fit_seasonal_gamma` becomes plain `fit(seasonal_family(Gamma, …), y; t, weights)` returning `SeasonalFit`;
  - `mix_ini` becomes the `SeasonalMixture` default init;
  - the pirated `fit_mle_RO` is deleted.
- **Stays in SWG:** `cor_groupby`-style coupling (now `given`) and all copula code.

## 7. Naming & verbs table

| New name | Meaning | Replaces (v3 SWG / v4 SPSM) |
|---|---|---|
| `WeatherData` | the problem: dates × stations × variables | per-station DataFrames + `innerjoin(df_z)`, `Y` BitMatrix |
| `Leap366()`, `period_index` | calendar, Date ↦ t | `dayofyear_Leap`, `n_to_t`, `n2t` |
| `RegimeHMM`, `WGEN`, `CensoredLatentGaussian` | model class ("algorithm") | hand glue around `ARPeriodicHMM`; `wgen` |
| `Regimes(K)`, `Regimes(fitted)` | regime chain spec, fitted / frozen | `randARPeriodicHMM`, `fit_mle_all_slices`, `fit_θ!`, `ARPeriodicHMM(a,A,B)` |
| `Rain()` | occurrence + amount + wet-set copula | `fit_mle_RO`, `fit_mle_RR`, `cov_RR`, `rand_RR`, `mix_ini`, `onefy` |
| `Temperature()` | continuous, AR1 by default | `fit_AR1`, `cov_ar1`, `rand(ar1s, n2t, z, Σ)` |
| `given = :TX` | conditional block (DAG) | `fit_TN`, `cor_groupbyTXTN`, `rand_cond`, `ϵ_TX`, `output_ϵ` |
| `Positive()` | positive variable, exact marginal | `fit_AR1` + `indicatrix` clipping |
| `marginal =`, `Mixture(...)` | seasonal marginal law | `SmoothSeasonalMixture`, `seasonal_family` |
| `season =` | basis: Int, basis, `Monthly()`, NamedTuple | `degree`, `𝐃𝐞𝐠`, `TrigBasis(T,d)`, `idx_months` |
| `AR1()`, `LatentAR1()`, `Markov(order)` | dynamics | `SmoothAR1`, `local_order`, `memory_order`, WGEN `p` |
| `MultisiteGaussian(by)`, `SpatialExp()` | cross-site dependence | `Σ²RR`, `Σ²TX`, `Ω`, `fit_Ω`, `cov_ar1` |
| `TwoStage()`, `SoftWeights()` | estimation mode | implicit plug-in ẑ |
| `fit(data, model)` | fit | `fit_mle`, `fit_mle_RO`, `fit_AR1`, `fit_TN`, `fit_wgen` |
| `FittedGenerator` | the only fitted result | 6 result types + discarded histories |
| `simulate(g, dates; nsim)` / `rand` | ensemble generation | `rand(hmm, n2t; …)`, `rand_RR`, `rand_cond`, `rand(wgen, years)` |
| `past =` | days before the first simulated day | `Y_past`, `y₁`, `z_ini`, `Y_ini`, `y_ini` |
| `given = (regime = …,)` | conditional simulation | passing `z` vectors by hand |
| `WeatherSimulation` | Tables output | hand-built `dfs_simus` |
| `decode(g, data)` | regime path / posteriors | `viterbi(hmm, Y, Y_past; n2t)` + CSV |
| `validate(g, data)` | simulation-based checks | ad hoc plotting code |
| `save_generator`/`load_generator` | persistence | `JLD.save(Dict(...))` |
| `spec(g)`, `show(g)` | the resolved spec, as code | — |
| `curve(g, var, par; station, regime)` | seasonal curve | `curve`, `μₜ…αₜ`, `θ_ΔT[j][k](t)` |
| `read_stations(INRAE(), ids)` | data readers | `collect_data_INRAE`, `collect_data_ECA`, `collect_data_MeteoFrance` |
| SPSM `logdensities`, `t =`, `fit_mle(m,y,w)` | engine verbs | `loglikelihoods`, `n2t`, tuple-returning `fit_mle` |

## 8. Migration & release plan

Each step is independently mergeable and keeps CI green.

- **S1 (SPSM)** `SoftmaxLink` in the vendored TVP. Transitions and mixture weights move onto it internally. Tests are bit-identical.
- **S2 (SPSM, additive)** `weights`/`z`/`seq_ends` on AR1, mixture and the HMM emissions. Add `logdensities`, `innovations`, `SeasonalLatentAR1` and the callable `m(t; state, past)`.
- **S3 (SPSM, breaking branch)**
  - Unify on `SeasonalFit`; `fit_mle` becomes model-only.
  - Apply the renames; `t` becomes required.
  - Make the `public`/export cuts and remove the legacy names.
  - Port the v4 test suite call by call, asserting equal numbers.
- **W1 (SWG)** Monorepo skeleton and **Core**: WeatherData, calendars, verbs, WeatherSimulation, validate. No SPSM needed yet.
- **W2** **Data**: move the readers, Artifact-based example data, fix the precompile download and the `distance_x_to_y` longitude bug.
- **W3** **Blocks** on SPSM v5 via `[sources]` rev. This includes the product engine with K = 1, plus equivalence tests of each block against v3 `fit_AR1`, `fit_mle_RR`, `cov_RR`, `cov_ar1`, `fit_TN` and `fit_mle_RO`.
- **W4** **RegimeHMM**: Regimes, frozen regimes, TwoStage/SoftWeights. The crop tutorial runs end to end.
- **W5** **WGEN** port. **W6** **LatentGaussian**.
- **W7** Umbrella, docs, migration guide, rewritten tutorials. The `.md` tutorials are generated by Literate in CI so they can never go stale again.
- **R (release order)**
  1. Tag SPSM v5.0.0.
  2. Register Core and Data. New packages wait about 3 days in General, so do this early.
  3. Register Blocks once Core is merged.
  4. Register RegimeHMM, WGEN and LatentGaussian once Blocks is merged.
  5. Release umbrella SWG v4.0.0, with the same UUID and compat on all of the above.

**Old → new (most-used calls)**

| v3 SWG / v4 SPSM | v4 SWG / v5 SPSM |
|---|---|
| `fit_mle_RO(df, order, d; solver)` + `ARPeriodicHMM(a, A, B)` | `Rain(occurrence = Markov(order = order), season = d)` inside `RegimeHMM(regimes = Regimes(france))` |
| `fit_mle_RR(df, order; solver, mix₀ = mix_ini(T))` | `Rain(amount = Mixture(Exponential, 2))` |
| `cov_RR(dfs, K)` | `Rain(copula = MultisiteGaussian(by = :regime))` |
| `fit_AR1(df, :TX, d; solver)` | `TX = Temperature(season = d)` |
| `cov_ar1(dfs, ar1s, :TX)` | `Temperature(copula = MultisiteGaussian())` |
| `fit_TN(df, 1, T)` + `cor_groupbyTXTN(df, T)` | `TN = Temperature(given = :TX)` |
| `fit_AR1(df, :QQ, d)` + `indicatrix.(rand(...))` | `QQ = Positive()` |
| `rand(hmm, n2t; Y_past, z_ini, seq = true)` | `simulate(g, dates; past)` → `sims.regime`, `sims[:RR]` |
| `rand_RR(mixs, n2t, z, y, Σ)` | inside `simulate` |
| `rand(ar1s, n2t, z, Σ; y₁, output_ϵ = true)` | inside `simulate` |
| `rand_cond(ϵ, z, θ_uni, θ_cor, n2t, T)` | inside `simulate` (`given = :TX`) |
| loops over `NSIMU` + `DataFrame(...)` assembly | `DataFrame(simulate(g, dates; nsim))` |
| `dayofyear_Leap.(dates)` | `period_index.(Leap366(), dates)` |
| `viterbi(hmm, Y, Y_past; n2t)` | `decode(g, data)` / SPSM `viterbi(m, Y; t, past)` |
| `randARPeriodicHMM` → `fit_mle_all_slices` → `fit_θ!` → `SmoothARPeriodicHMM` → `fit_mle` | `fit(obs, RegimeHMM(K = 4, others = :ignore))` / SPSM `fit(SeasonalHMM(K, BernoulliMemory(D; order)), Y; t, past, init = :slices)` |
| `fit_wgen(Y, idx_months, p)` | `fit(obs, WGEN(order = p))` |
| `rand(wgen_model, 1956:2019; Y_ini)` | `simulate(g, Date(1956):Date(2019,12,31); past)` |
| `fit_mle(SmoothAR1(d, T; K), y, z; n2t, solver)` → `(m, hist)` | `fit(SeasonalAR1(period = T, K, basis = d), y; t, z, solver)` → `SeasonalFit` |
| `fit(SeasonalFit, m, y; n2t)` | `fit(m, y; t)` |
| `SmoothSeasonalMixture(Exponential, 2, d, T)` | `SeasonalMixture(Exponential, 2; period = T, basis = d)` |
| `mixture_weights(m)`, `μₜ(t/T, θ)` | `curve(m, :weights)`, `curve(m, :μ)` |
| `loglikelihoods(m, ...)` | `logdensities(m, y; t)` |
| `TimeVaryingFit`, `fit_loss_optim` | `SeasonalFit`, `SPSM.solve_objective` (public) |
| `parameters(m)`, `params(m, t)` | `coef(m)`, `m(t)` |
| `collect_data_INRAE(path)` / `collect_data_ECA(...)` | `read_stations(INRAE(), paths)` / `read_stations(ECA(), ids)` |
| `JLD.save(file, Dict("hmm" => …))` | `save_generator(file, g)` |

**Numerical equivalence**
1. **Reference artifacts.** `test/reference/make_v3.jl` runs once in a Manifest pinned to SPSM 4.1.0 / SWG 3.0.0. It uses the tutorial data plus a synthetic CI dataset that needs no network. It stores:
   - all fitted curves (T × K × D per parameter), the Σ/Ω matrices, logliks and ẑ;
   - simulation summary statistics from nsim = 300.
2. **Level 1 (fits).** The same data, inits and solver must reproduce the curves to rtol 1e-6 (deterministic kernels or closed form: WGEN counts, AR1 sufficient statistics), or 1e-4 where an optimiser is involved. The SPSM v5 tests assert v4 numbers bitwise where the kernels are unchanged.
3. **Level 2 (simulation).** RNG draw order changes, so streams are not compared. Each statistic is compared with a Monte Carlo z-test with a Bonferroni correction.
4. **Level 3 (intentional differences).** These are documented and each is tested with a "legacy" T1 spec:
   - TX–ΔT correlation estimated on the innovation scale;
   - `rand_RR` no longer forces regime K to be independent;
   - `cov_ar1` gap and first-residual fixes;
   - default y₀ handling (`past`/burn-in);
   - `Positive()` uses LatentAR1/Gamma; the legacy spec is `Continuous(Normal; dynamics = AR1())`, whose fit must match v3 exactly.

## 9. Risks, trade-offs, open questions

1. **Argument order.** SWG uses `fit(data, model)` (SciML `solve(prob, alg)`, as the decision specifies). SPSM keeps `fit(model, y; t)` (StatsAPI/Distributions). The rule fits in one sentence, but students will notice it. The alternative is model-first everywhere. **Open.**
2. **Hidden defaults.** Auto-resolution can hide modelling choices. This is mitigated because `show` prints the resolved spec as code and `others = :ignore` exists, but default K = 4 (as in the paper) versus BIC over K is still **open**. A K range costs one HMM EM per value.
3. **Positive() quality.** A Gamma default suits ETPP poorly: there are exact zeros in winter, and Gamma has F(0) = 0. It also suits QQ poorly, since QQ is bounded by clear-sky radiation.
   - Proposal: treat values at or below data resolution as left-censored inside LatentAR1.
   - Plan a `Positive(marginal = Beta, upper = clear_sky)` option for v4.1 (speculative).
4. **LatentAR1 vs AR1 for temperature.** Keeping `AR1()` as the default preserves v3 numbers. LatentAR1 is more general, and it may become the default in v5 (a breaking change), so choosing now matters.
5. **TTFX and memory.**
   - The umbrella loads the full SciML optimisation stack. PrecompileTools workloads are mandatory.
   - 300 × 50 years × 4 stations is about 22M rows. `DataFrame(sims)` materialises roughly 1.6 GB; `members(sims)` and `sims[:TX]` avoid that.
6. **Copula estimation with missing values and wet sets.** Pairwise-complete correlations need projection (`nearest_cor`). Copulas.jl `GaussianCopula{d}` has d ≥ 2 in the type, so Blocks uses its own MvNormal path for varying wet-set sizes. Weighted pairwise estimators under SoftWeights are new code.
7. **Frozen regimes on other dates.** New-station data outside the frozen ẑ span is dropped, with a report. Filtering a frozen chain with new emissions would need stage-1 emissions for those stations. **Open.**
8. **Name clashes.**
   - `save`/`load` are avoided by design.
   - `Positive`, `Temperature`, `Markov` and `validate` might collide with other packages' exports (unverified).
   - SPSM v4 exported `AR1`, but v5 makes it `public`, so the SWG `AR1()` dynamics name is free.
9. **Package names are long** (`StochasticWeatherGeneratorsRegimeHMM`). This follows SciML convention, and experts can use `import … as`. Short `SWG*` names are an alternative but hard to search. **Open.**
10. **Release logistics.** There are seven registrations, and each new package waits about 3 days. Release order is constrained by dependencies, and there are two repos in lockstep. Pre-registering Core and Data weeks ahead removes most of the risk.
11. **Out of launch scope, but kept additive:**
    - `JointEM()`;
    - seasonal copulas `C_k(t)` (needs the correlation link);
    - per-parameter `shared = (:ρ,)` (needs a joint weighted M-step across k);
    - multivariate regimes (`Regimes(from = (:RR, :TX))`);
    - spatial occurrence `Markov(copula = SpatialExp())` with the SPSM spatial HMM;
    - wet/dry drivers (Richardson, WeaGETS, GWGEN);
    - "HMM-like" per-variable dynamics (`dynamics = LocalHMM(K)`).

    Each is a new type implementing the existing protocol, so none forces a major release.
12. **Sub-daily data.** AWE-GEN-2d-class models need sub-daily calendars. `WeatherData.dates` is `StepRange{Date,Day}` at launch. Widening it to `TimeType` later is breaking for code that reads the field directly. Proposal: document `dates(data)` as the accessor now and keep the field private.
13. **`SeasonalFit` destructuring.** Should `SeasonalFit` iterate as `(model, history)` so that `m, h = fit(...)` still reads naturally? It is a feature rather than a shim, but it blurs "one result type". **Open.**
14. **Serialization.** `to_plain` derived from fields can break when a block's fields change between versions. A format version plus a field-mapping table in the migration guide is the minimum.

### Critical Files for Implementation
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\examples\tuto_add_station_variable.jl
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\ext\SmoothPeriodicStatsModelsExt\fit_mle_TN.jl (with fit_mle_AR.jl, fit_mle_rain.jl)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\src\rain\wgen.jl
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\SmoothPeriodicStatsModels.jl (export surface, lines 51–199)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\AR\AR1.jl (with src\HMM_Dists\periodichmm.jl and src\TimeVaryingParameters\spec.jl)