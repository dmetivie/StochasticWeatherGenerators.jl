## 0. Thesis

The break should add as few concepts as it can. A user needs two nouns, `WeatherData` (a Tables.jl table backed by D×N arrays) and a model class (`RegimeHMM`, `WGEN`, `CensoredLatentGaussian`), and two verbs, `fit(data, model)` and `rand(gen, dates; nsim)`. `rand` returns another `WeatherData`, so the type that goes in is the type that comes out. Underneath, the v4 contract `fit_mle(model, data) -> (model, history)` stays the only expert verb, used by both SWG model classes and SPSM engine models.

The monorepo has four small packages (Core, RegimeHMM, WGEN, LatentGaussian) plus the existing umbrella. SPSM stays in its own repository. SPSM v5 is mostly subtraction: about 94 exports become 42 exported plus about 36 `public`, and its call shapes are made uniform (data positional, conditioning by keyword). The model types are not renamed, because migration cost matters more than cosmetics. Everything new is first added next to the old code (SPSM 4.2 additive, SWG `lib/` packages with no users yet). The break is then two short PRs merged on the same day. They are checked against frozen v3/v4 reference fixtures: exact where the RNG and the objective are unchanged, distributional elsewhere.

## 1. Package architecture

```
StochasticWeatherGenerators.jl/                  (existing repo; root = umbrella, UUID 3eadb0b9… kept)
├── Project.toml        [workspace] projects = ["lib/*", "docs", "test"]; [sources] lib paths
├── src/StochasticWeatherGenerators.jl           @reexport sublibs + data I/O + climate indices
├── ext/MakieExt, ext/GeoMakieExt                plots (unchanged)
├── lib/
│   ├── StochasticWeatherGeneratorsCore/          Project.toml, src/, test/
│   ├── StochasticWeatherGeneratorsRegimeHMM/
│   ├── StochasticWeatherGeneratorsWGEN/
│   └── StochasticWeatherGeneratorsLatentGaussian/
├── test/   umbrella integration + test/reference/v3/ (frozen fixtures, pinned env)
├── docs/   one Documenter site for all packages
└── examples/, assets/
SmoothPeriodicStatsModels.jl/                    (own repo, unchanged location; v5.0)
```

| Package | Purpose | Hard deps | Exported (public-only) |
|---|---|---|---|
| **…Core** | Data type, calendar, interface verbs, `WeatherFit`, Gaussian dependence utilities, rand plumbing (nsim, seeds, burn-in, threads) | Dates, Random, LinearAlgebra, Statistics, Tables, Distributions, StatsAPI, NearestCorrelationMatrix | **6**: `WeatherData, fit, loglikelihood, aic, bic, regimes` (~12) |
| **…RegimeHMM** | Flagship: seasonal regimes × variable blocks | Core, SmoothPeriodicStatsModels, Distributions, Dates, Random, LinearAlgebra | **6**: `RegimeHMM, Rain, AR1, CopulaAR1, Difference, curve` (5) |
| **…WGEN** | Wilks 1998 multisite occurrence, ported | Core, Distributions, Dates, Random | **1**: `WGEN` |
| **…LatentGaussian** | Censored latent Gaussian rain (Ailliot 2009 / Vaittinada Ayar 2020) | Core, Distributions, Dates, Random | **1**: `CensoredLatentGaussian` |
| **StochasticWeatherGenerators** (umbrella) | Re-export, data I/O, geo, climate indices, plot stubs, default solver | the 4 libs, Reexport, Optimization, OptimizationOptimJL, SimpleNonlinearSolve, CSV, DataFrames, DataFramesMeta, Impute, RollingFunctions, StatsBase; weak: Makie, GeoMakie, NaturalEarth | 14 re-exported + ~22 own ≈ **36** (v3: 54) |

The re-exported set is 14, not 6 + 6 + 1 + 1, because `curve` is SPSM's own function, used by RegimeHMM.

Copulas, NonlinearSolve and OrderedCollections leave SWG. All launch dependence is Gaussian and is handled directly as MvNormal on normal scores. This also removes the `GaussianCopula{d}` recompilation for every wet-set size, the d≥2 special cases, and refactorising Σ on every call.

**Why SPSM stays in its own repo.** It has its own audience (genomics, `examples/tuto_genomics.jl`, with non-periodic spline bases), its own CI and benchmarks, its own registry entry and tag history, and the vendored TVP. Moving it in would make domain-agnostic users depend on a weather repo and would complicate TagBot and registry history. Lockstep needs coordinated release days, not a shared repo. Cross-repo breakage is caught by a "Downstream" job in SPSM's CI that runs `lib/StochasticWeatherGeneratorsRegimeHMM` tests against every SPSM PR.

**The umbrella** is the T0 entry point. It is the only package with a hard dependency on the solver stack. Its `__init__` calls `StochasticWeatherGeneratorsCore.default_solver!(LBFGS())`, so T0 never writes `solver`. The sublibraries never load Optimization themselves; experts pass `solver`, as the v4 rule requires.

**Load-time budget.** These are targets, not measurements. They should be measured by a CI job with `@time_imports` (Julia 1.12, ubuntu-latest, warm cache), failing above 1.5× budget.

| Package | Budget | Dominated by |
|---|---|---|
| Core | ≤ 1.0 s | Distributions |
| WGEN, LatentGaussian | ≤ 1.2 s | Core |
| RegimeHMM | ≤ 3.5 s | SPSM (HiddenMarkovModels, ExpectationMaximization, Copulas, ComponentArrays, SciMLBase) |
| umbrella | ≤ 8 s | + Optimization/OptimJL + CSV/DataFrames |

**Registration and versioning.**
- Each `lib/X/Project.toml` has its own UUID and version, and `[sources] StochasticWeatherGeneratorsCore = {path = "../StochasticWeatherGeneratorsCore"}`. OrdinaryDiffEq does exactly this: `lib/OrdinaryDiffEqTsit5/Project.toml` has `[sources] OrdinaryDiffEqCore = {path = "../OrdinaryDiffEqCore"}`, Tsit5 is at 2.1.5 and its compat on Core is `"4.15.1"`.
- Each sublibrary is registered with `@JuliaRegistrator register subdir=lib/StochasticWeatherGeneratorsCore`. The umbrella keeps `@JuliaRegistrator register`.
- The sublibraries start at **1.0.0** and follow semver independently (SciML practice). The umbrella is 4.0.0, with compat `Core = "1"`, etc.
- TagBot runs as one workflow with a matrix over `subdir` values; tags look like `StochasticWeatherGeneratorsCore-v1.0.0`.
- New packages wait about 3 days in General, so the four libraries are registered before the release day (§8).

**CI.** SWG currently has no test workflow, only Documentation and TagBot.
- `CI.yml` runs a matrix of {Core, RegimeHMM, WGEN, LatentGaussian, umbrella} × {1.12, 1}. It adds windows for the umbrella and path filters (a change to `lib/WGEN` runs WGEN + umbrella).
- Each package's tests include Aqua (ambiguities, piracy, stale deps).
- Each package has a test that checks `names(Module)` against a list kept in the test file, so export creep fails CI.

**Docs.** One site. `docs/Project.toml` has `[sources]` for every lib. There is an API page per package, and `checkdocs = :public` makes sure every public name is documented. Tutorials stay as Literate `.jl` files rendered at build time, which fixes the stale `.md` problem.

## 2. Core interface

```julia
module StochasticWeatherGeneratorsCore
export WeatherData, fit, loglikelihood, aic, bic, regimes
public AbstractWeatherModel, WeatherFit, fit_mle, variables, dayofyear_Leap, haversine,
       default_solver, default_solver!, GaussianDependence, fit_dependence, rand_dependence!,
       group_index, normal_scores
import Distributions: fit_mle                      # same function SPSM extends
import StatsAPI: fit, loglikelihood, aic, bic, nobs, dof

abstract type AbstractWeatherModel end             # "algorithm" type, like an ODE solver

struct WeatherData{V<:NamedTuple,TT<:AbstractVector{<:TimeType}}
    dates::TT                                      # contiguous daily Date range (type left open for sub-daily later)
    sites::Vector{String}
    vars::V                                        # :RR => D×N Matrix{Union{Missing,Float64}} (obs) or D×N×S (sims)
    regimes::Union{Nothing,Matrix{Int}}            # N×S when the model has regimes
    coords::Union{Nothing,Matrix{Float64}}         # D×2 (lat, lon), optional
    colnames::NamedTuple{(:date,:site),NTuple{2,Symbol}}
end
WeatherData(table; date = :DATE, site = :STAID, vars = :all, coords = nothing)
WeatherData(tables::AbstractVector; kw...)                     # one table per station (v3 habit)
WeatherData(dates, sites, vars::NamedTuple; regimes = nothing, coords = nothing)
Base.getindex(d::WeatherData, v::Symbol)                       # D×N(×S) array
Base.merge(a::WeatherData, b::WeatherData)                     # union of sites = "adding stations"

struct WeatherFit{M<:AbstractWeatherModel,H}                   # the single fitted-result type
    model::M; history::H; nobs::Int; loglikelihood::Float64; dof::Int
end

fit(data, m::AbstractWeatherModel; solver = default_solver(), rng = Random.default_rng(), kw...) -> WeatherFit
fit_mle(m::AbstractWeatherModel, data::WeatherData; solver, rng, kw...) -> (m′, history)   # per class (v4 contract)
Base.rand([rng::AbstractRNG,] g::Union{WeatherFit,AbstractWeatherModel}, dates::AbstractVector{Date};
          nsim::Integer = 1, past = nothing, given = (;), burnin::Integer = 365) -> WeatherData
Random.rand!(rng, sim::WeatherData, m::AbstractWeatherModel; past, given) -> sim            # per class, one realisation
variables(m::AbstractWeatherModel) -> NTuple{N,Symbol}                                     # per class
loglikelihood(g::WeatherFit); aic(g); bic(g)
loglikelihood(m::AbstractWeatherModel, data::WeatherData)                                  # optional per class
regimes(g::WeatherFit) -> (DATE = Vector{Date}, REGIME = Vector{Int})                      # training path, a table
regimes(g::WeatherFit, data::WeatherData) -> Vector{Int}                                   # decode new data
```

A model class must implement three methods: `fit_mle`, `rand!` and `variables`. Everything else in Core is generic.

**`fit`** converts `data` to a `WeatherData`, calls `fit_mle`, and wraps the result. Keyword arguments other than `solver` and `rng` go to the class's iterative algorithm (for RegimeHMM, the regime EM). `fit(data, alg)` matches decision 2(b). Its argument order is the reverse of `fit_mle(model, data)`; see §9.

**Tables contract.**
- **Input.** Any Tables.jl source with a date column, a site column, and one column per variable (`DATE, STAID, RR, TX, …`), or a vector of such tables, one per station. Without a site column the data are treated as one site.
- **Variables.** Variables are the columns other than the date, site and coordinate columns. `missing` is allowed. Dates are completed to a contiguous daily range, and the gaps are filled with `missing`.
- **Units** are the user's responsibility. Defaults assume RR in mm.
- Fully long format (date, site, variable, value) is not accepted in 4.0; use `unstack`.
- **Output.** `WeatherData` is a row table with columns `SIM` (when nsim > 1), the input's date and site column names, `REGIME` (when the model has regimes), then the variables. `DataFrame(sims)` and `CSV.write` work directly. Experts read arrays with `sims[:TX]`, which is D×N×nsim.

**Calendar.** The period is T = 366 and t = `dayofyear_Leap(date)`: 1 March is always 61, and 29 February is 60 in leap years only (the v3 convention). `n2t` is computed inside `WeatherData` and passed to SPSM, which stays on integer indices. Monthly grouping uses `month(date)`. Simulation dates are any daily range, and leap days appear naturally.

**RNG and ensembles.** `rand` takes `rng` as its first positional argument, as in Random. `fit` takes `rng` as a keyword (WGEN's Monte Carlo, HMM random restarts). For `nsim`, Core draws `nsim` seeds from `rng` and simulates realisation i with `Xoshiro(seed_i)` under `Threads.@threads`, so results do not depend on the thread count. For ensembles too large for memory, call `rand(gen, dates)` in a loop.

**Initial conditions.** There is one convention, **`past`**: a `WeatherData` (or table) whose last date is `first(dates) - Day(1)`. Each block reads what it needs (occurrence memory, AR lag, regime of day 0 if present). This replaces `Y_past`, `y₁`, `Y_ini`, `tx_ini` and `z_ini`. With `past = nothing`, Core simulates `burnin` days before `dates[1]` from a neutral start (regime ~ a, dry, latent scores 0) and discards them.

**Conditioning.** `given = (z = path,)`, where `path` is a `Vector{Int}` or a table `(DATE, REGIME)`, imposes a regime path, for example one decoded from a GCM. Conditioning on observed variables is not in 4.0.

**Missing data and gaps.**
1. Conditional-likelihood terms are dropped when the current or lagged value is missing. This is done through SPSM v5's multiple sequences, so gaps are never bridged.
2. Copulas are estimated pairwise-complete, then repaired to positive definite (`nearest_cor!`).
3. Regime learning splits the record into sequences at days where any training site is missing. It warns when more than 5% of days are dropped.
4. Simulations never contain `missing`.

**Diagnostics.** Only `loglikelihood`/`aic`/`bic` on `WeatherFit`, plus `show`, which prints the T1 constructor that rebuilds the model. Observations and simulations have the same type, so the umbrella's `monthly_agg`, `longest_spell`, `pmf_spell`, `VCX3` and the Makie recipes apply to both. There is no new validation verb in 4.0.

**Save and load.** Every fitted type is closure-free: families are stored as types, never as anonymous functions, and a Serialization round-trip test runs in CI. Users call `JLD2.save_object/load_object`. For portability across versions, `CSV.write(file, regimes(gen))`. There is no versioned file format in 4.0.

## 3. Three tiers, by example

**T0, impact modeler: the crop tutorial.**

```julia
using StochasticWeatherGenerators, Dates, DataFrames
obs  = WeatherData(collect_data_INRAE.(station_path))        # DATE, STAID, RR, TN, TX, QQ, ETPP
gen  = fit(obs, RegimeHMM(K = 4))                            # blocks chosen from the column names
sims = rand(gen, Date(1986):Date(2035, 12, 31); nsim = 300)
df   = DataFrame(sims)                                       # SIM, DATE, STAID, REGIME, RR, TN, TX, QQ, ETPP
```

Defaults are chosen by column name:
- `RR` → `Rain()`: memory 1, degree 1, mixture of two exponentials, one wet-set copula per regime.
- `TX`/`TG` → `AR1()`.
- `TN` with `TX` present → `Difference(:TX, Gamma)`.
- `QQ`, `ETPP`, `FF` → `CopulaAR1(Gamma)`, so they are positive by construction.
- Any other name is an error that shows how to declare a block. Nothing is chosen silently.

To reproduce the tutorial exactly, with the regimes pre-trained on the 37 ECA stations, add one line:

```julia
using JLD2; paper = load_object(joinpath(pkgdir(StochasticWeatherGenerators), "assets", "tuto_1", "regimes_K4_d1_m1.jld2"))
gen = fit(obs, RegimeHMM(regimes = paper))                   # A(t) and ẑ(date) frozen; the 4 stations fitted given ẑ
```

**T1, applied statistician: the same model, customised.**

```julia
using StochasticWeatherGenerators, Distributions, Dates
using OptimizationOptimJL: LBFGS
model = RegimeHMM(
    regimes   = paper,                                          # frozen WeatherFit; or a (DATE, REGIME) table ⇒ chain refitted on ẑ
    variables = (
        RR   = Rain(memory = 1, degree = 1,
                    amount = MixtureModel([Exponential(1.0), Gamma(0.8, 8.0)]),   # families + start values
                    copula = :regime_month),                    # one wet-set correlation per (regime, month)
        TX   = AR1(degree = (μ = 2, ρ = 1, σ = 1)),             # classical Gaussian AR(1), per-parameter degree
        TN   = Difference(:TX, Gamma; degree = 1),              # TN = TX − Δ, Δ ~ seasonal Gamma, coupled to TX innovations
        QQ   = CopulaAR1(Gamma; degree = (α = 1, θ = 2)),       # AR(1) on normal scores of a seasonal Gamma
        ETPP = CopulaAR1(Weibull),
    ))
gen = fit(obs, model; solver = LBFGS())

gen.model.variables.TX               # show: spec + per-site/per-regime summary
curve(gen.model.variables.TX, :ρ)    # D × K × 366
bic(gen); regimes(gen)               # plug-in BIC (see §9); ẑ as a table

gen6 = fit(merge(obs, obs_new), model)                                  # adding stations, same frozen regimes
sims = rand(gen, gcm_dates; nsim = 100, given = (z = gcm_regimes,))     # regimes imposed from outside

using SmoothPeriodicStatsModels: CyclicBSplineBasis                     # expert per-parameter basis
AR1(degree = (μ = CyclicBSplineBasis(366, 8), ρ = 0, σ = 1))            # Integer = trig degree, or any basis
```

Other options:
- `copula = :regime` (default), `:regime_month` or `:none` (sites independent given regime).
- A smoothly seasonal correlation C_k(t) is **not** in 4.0 (§6, no vector links).
- AR(1) form: `AR1()` puts the AR on the variable (v3/v4 numerics). `CopulaAR1(D)` puts the AR on the latent normal scores, with any marginal.

**T2a, expert: a new variable block.**

```julia
import StochasticWeatherGeneratorsRegimeHMM as RH
import StochasticWeatherGeneratorsCore as SWGCore
using SmoothPeriodicStatsModels, Distributions, Random

struct Seasonal{D,F} <: RH.AbstractBlock              # iid given (regime, day): seasonal marginal + multisite copula
    family::Type{D}; degree::Int; fitted::F
end
Seasonal(D::Type{<:Distribution}; degree = 1) = Seasonal(D, degree, nothing)

function Distributions.fit_mle(b::Seasonal, y::AbstractMatrix, ctx::RH.BlockContext)
    D, K = size(y, 1), ctx.K
    marg = Matrix{Any}(undef, D, K); hist = similar(marg)
    for s in 1:D, k in 1:K
        idx = findall(n -> isequal(ctx.z[n], k) && !ismissing(y[s, n]), axes(y, 2))
        m0 = SmoothDistribution(b.family, 366; degree = b.degree)
        marg[s, k], hist[s, k] = fit_mle(m0, Float64.(y[s, idx]); n2t = ctx.n2t[idx], solver = ctx.solver)
    end
    S = [ismissing(y[s, n]) || ismissing(ctx.z[n]) ? missing :
         quantile(Normal(), cdf(marg[s, ctx.z[n]](ctx.n2t[n]), y[s, n])) for s in 1:D, n in axes(y, 2)]
    dep = SWGCore.fit_dependence(S, SWGCore.group_index(:regime, ctx.z, ctx.dates; K), K)
    return Seasonal(b.family, b.degree, (; marg, dep)), (; marginals = hist)
end

function Random.rand!(rng::AbstractRNG, y::AbstractMatrix, b::Seasonal, ctx::RH.BlockContext)
    (; marg, dep) = b.fitted; w = zeros(size(y, 1))
    for n in axes(y, 2)
        k, t = ctx.z[n], ctx.n2t[n]
        SWGCore.rand_dependence!(rng, w, dep, k)                 # w ~ N(0, C_k)
        for s in axes(y, 1); y[s, n] = quantile(marg[s, k](t), cdf(Normal(), w[s])); end
    end
    return y
end
RH.scores(b::Seasonal, y, ctx) = SWGCore.normal_scores(b.fitted.marg, y, ctx.z, ctx.n2t)  # lets it be a parent

RegimeHMM(variables = (RR = Rain(), FF = Seasonal(Weibull; degree = 2)))
```

**T2b, expert: a new model-class sublibrary**, here a k-NN resampler (Lall & Sharma 1996). It lives in `lib/StochasticWeatherGeneratorsKNN`, depends on Core, Dates, Random and Distributions, and is registered with `subdir=`. The helpers `standardized_features` and `circdist` are left out (sketch).

```julia
module StochasticWeatherGeneratorsKNN
using Dates, Random, Distributions
import StochasticWeatherGeneratorsCore as SWGCore
using StochasticWeatherGeneratorsCore: WeatherData
export KNN

struct KNN{O} <: SWGCore.AbstractWeatherModel
    k::Int; window::Int; vars::Vector{Symbol}; obs::O           # "fitting" stores the record
end
KNN(; k = 5, window = 7) = KNN(k, window, Symbol[], nothing)
SWGCore.variables(m::KNN) = Tuple(m.vars)
Distributions.fit_mle(m::KNN, data::WeatherData; kw...) = (KNN(m.k, m.window, collect(keys(data.vars)), data), nothing)

function Random.rand!(rng::AbstractRNG, sim::WeatherData, m::KNN; past = nothing, given = (;))
    X   = standardized_features(m.obs)                          # (V·D) × N, complete days only
    doy = SWGCore.dayofyear_Leap.(m.obs.dates)
    cur = rand(rng, findall(==(SWGCore.dayofyear_Leap(first(sim.dates) - Day(1))), doy))
    w   = (1 ./ (1:m.k)); w ./= sum(w)
    for (n, d) in enumerate(sim.dates)
        t    = SWGCore.dayofyear_Leap(d)
        cand = [i for i in 1:length(doy)-1 if circdist(doy[i+1], t) ≤ m.window]
        dist = [sum(abs2, view(X, :, i) .- view(X, :, cur)) for i in cand]
        cur  = cand[partialsortperm(dist, 1:m.k)[rand(rng, Categorical(w))]] + 1    # successor of a neighbour
        for v in m.vars; sim[v][:, n] .= m.obs[v][:, cur]; end
    end
    return sim
end
end
```

`fit(obs, KNN(k = 7))` and `rand(gen, dates; nsim = 300)` then work unchanged, which is the "swappable like solvers" property.

## 4. Flagship Regime-HMM product model

**Spec objects.** Each block holds its spec and, once fitted, its parameters, following the v4 "model holds coefficients" idea. `fitted` is `nothing` before `fit`.

```julia
struct RegimeHMM{V,R,C,P} <: AbstractWeatherModel
    K::Int; degree::Int; memory::Int       # transitions' seasonal degree; occurrence memory used to learn regimes
    regimes::R                             # nothing | WeatherFit{<:RegimeHMM} (freeze A(t)+ẑ) | table (DATE, REGIME)
    variables::V                           # nothing (defaults by column name) | NamedTuple of blocks
    chain::C                               # fitted SPSM.SmoothARPeriodicHMM (a, A(t); B = regime-defining occurrence)
    path::P                                # fitted (dates, z)
    sites::Vector{String}
end
RegimeHMM(; K = 4, degree = 1, memory = 1, regimes = nothing, variables = nothing)

Rain(; memory = 1, degree = 1, amount = MixtureModel([Exponential(1.0), Exponential(5.0)]),
       copula = :regime, threshold = 0.0)                       # amount = nothing ⇒ occurrence only
AR1(; degree = 1, copula = :regime)                             # x = μ + ρ x₋₁ + σ ε, ε ~ N(0, C_g)
CopulaAR1(marginal = Normal; degree = 1, copula = :regime)      # W = Φ⁻¹F(x); W = μ + ρ W₋₁ + σ ε
Difference(parent::Symbol, marginal = Gamma; sign = -1, degree = 1, copula = :regime)  # x = parent + sign·Δ

abstract type AbstractBlock end                                  # public
Base.@kwdef struct BlockContext{Z,P,S,R}                         # public
    z::Z; K::Int; n2t::Vector{Int}; dates::Vector{Date}
    parents::P = (;)                                             # parent => (values = D×N, scores = D×N)
    past::Union{Nothing,AbstractMatrix} = nothing
    solver::S = nothing; rng::R = Random.default_rng()
end
# block contract: fit_mle(b, y, ctx) -> (b′, hist); rand!(rng, y, b, ctx); optional parents(b) = (), scores(b, y, ctx)
```

**Composition.**
- `variables` is a NamedTuple, and the declared order is kept.
- `parents(b)` defines a DAG. It is checked at construction: parents must exist and there must be no cycles. Blocks are processed in topological order, with ties kept in declared order, so RNG consumption is deterministic.
- **TN|TX.** `Difference` fits Δ = TX − TN with a seasonal `SmoothDistribution(Gamma)` per site × regime. Its dependence is **one** joint Gaussian over [ε^TX; ξ^Δ] (2D sites) per `copula` group, where ε^TX are TX's standardized innovations (`scores`) and ξ = Φ⁻¹F_Δ(Δ). In simulation, ξ | ε ~ N(C₂₁C₁₁⁻¹ε, C₂₂ − C₂₁C₁₁⁻¹C₁₂). This replaces `fit_TN`, `cor_groupbyTXTN`, `rand_cond` and the `ϵ_TX` side channel. TN < TX always holds.
- **Positive QQ/ETPP.** `CopulaAR1(Gamma)` is positive by construction; there is no clipping.

**Estimation (two-stage plug-in, the only mode in 4.0).**

```julia
function SWGCore.fit_mle(m::RegimeHMM, data::WeatherData; solver, rng, kw...)
    n2t    = SWGCore.dayofyear_Leap.(data.dates)
    blocks = something(m.variables, default_blocks(data))
    chain, path, hz = fit_regimes(m, data, n2t; solver, rng, kw...)   # frozen | SPSM EM on RO + viterbi
    z = align(path, data.dates)                                          # Vector{Union{Missing,Int}}
    fitted, hists = (;), (; regimes = hz)
    for v in topological_order(blocks)
        ctx = BlockContext(; z, K = m.K, n2t, dates = data.dates,
                           parents = parent_scores(fitted, data, blocks[v]), solver, rng)
        b, h = fit_mle(blocks[v], data[v], ctx)
        fitted = merge(fitted, NamedTuple{(v,)}((b,))); hists = merge(hists, NamedTuple{(v,)}((h,)))
    end
    return RegimeHMM(m.K, m.degree, m.memory, m.regimes, fitted, chain, path, data.sites), hists
end
```

The regime fit works as follows:
1. **Learned (default).** Build `SmoothARPeriodicHMM(K, D, 366; memory, degree)` on the RO of all sites (RO = RR > threshold) and fit it with `fit_mle(m0, RO_seqs; n2t, solver, kw...)`. The slice start is automatic. Then decode ẑ with `viterbi`.
2. **Frozen `WeatherFit`.** Take its chain and its ẑ joined by date. Dates outside the path give an error that names the date range.
3. **Table of (DATE, REGIME).** The chain is fitted on ẑ (seasonal multinomial logit).

Each block fit, given ẑ:
- **Rain.** The occurrence part is `fit_mle(SmoothARPeriodicHMM(K, D, 366; memory, degree), RO; z = ẑ, fixed = (:A,), n2t)`, which replaces `fit_mle_RO`. When `Rain` has the regime spec and the regimes were learned on these sites, the EM's B is reused and nothing is refitted, so the paper is reproduced exactly. The amount part is one `SmoothSeasonalMixture` per (s, k) on wet days, with v3's `fit_mle_RR` defaults (maxiter 5000, tol 2e-4, robust). The wet-set correlation per group is pairwise Spearman → Pearson on co-wet days, then `nearest_cor!` (v3 `cor_RR`).
- **AR1.** One `SmoothAR1(d, 366; K)` per site, fitted with `fit_mle(…; z = ẑ, n2t)` on gap-split sequences. The start is data-driven (moved from SWG `_ar1_init`). The innovation correlation is Pearson on standardized residuals per group (v3 `cov_ar1`, with the gap fix).
- **CopulaAR1.** IFM: a `SmoothDistribution` per (s, k), then the scores W, then a `SmoothAR1` on W per site, then the innovation correlation per group.

The `history` is a NamedTuple of SPSM `FitHistory`s, one per component. The block contract is already weighted underneath: one-hot ẑ is the special case. In 4.x, `ctx.weights` (N×K) and `RegimeHMM(estimation = :posterior)` can be added without breaking anything. Joint EM would further need `loglikelihoods(b, y, ctx) -> N×K` per block (§9).

**Simulation algorithm, one realisation.**
1. **z.** Either `given.z`, or z₁ ~ a (or A(t₀)[z₀,:] when `past` has regimes), then zₙ ~ Categorical(A(t_{n−1})[z_{n−1},:]). This is the same loop as SPSM v4 `rand`.
2. **Blocks in topological order.**
   - **Rain:** RO from `rand(rng, occ, n2t; z, past)` (v4 draw order). On the wet set W_n: w ~ N(0, C_g[W,W]), via a Cholesky of the submatrix cached by (g, mask) when D ≤ 12; then RR_s = F⁻¹_{s,k,t}(Φ(w_s)). The v3 hack that made regime K independent is removed.
   - **AR1:** ε_n ~ N(0, C_g) per day, then the recursion.
   - **CopulaAR1:** the same recursion on W, then x = F⁻¹(Φ(W)).
   - **Difference:** recompute the parent's ε from its simulated values (exact for AR1), then ξ | ε, then x = parent + sign·F⁻¹(Φ(ξ)).
3. Core handles burn-in, nsim and seeds.

## 5. WGEN port and censored latent Gaussian

**WGEN (Wilks 1998; Srikanthan 2009): a straight port.**

```julia
struct WGEN{F} <: AbstractWeatherModel
    order::Int; threshold::Float64; nmc::Int                # nmc = MC length of the Ω fit (v3: 2000)
    fitted::F           # (p = D × 12 × 2^order P(wet | site, month, last `order` days), Ω = GaussianDependence(:month))
end
WGEN(; order = 4, threshold = 0.0, nmc = 2000)
SWGCore.variables(::WGEN) = (:RO,)
```

- **Fit.** Transition counts per (site, month) give `p`. This is v3 `fit_markov_chain` with arrays instead of OrderedDicts. v3's quirk of concatenating each month's days across years is kept for equivalence and documented.
- **Ω.** Each pair (s, s′) and month gets a bracketed bisection on ω ∈ (−0.999, 0.999), matching the observed lag-0 correlation of occurrences through an nmc-day Monte Carlo simulation. Each evaluation of ω reuses the same seed (common random numbers), which makes the objective deterministic. This is an intentional improvement over v3, which advanced the RNG on every evaluation. The result is repaired with `nearest_cor!`. NonlinearSolve is dropped.
- **rand!.** For each month block, `U = rand(rng, MvNormal(zeros(D), Ω_m), ndays)`, which is v3's draw order, then wet ⇔ U[s] ≤ Φ⁻¹(p[s, m, h]). `past` supplies the first `order` days. If the data also have TX etc., `fit` reports that WGEN models RR occurrence only.

**CensoredLatentGaussian.** One latent field gives both occurrence and amount.

```julia
struct CensoredLatentGaussian{D,F} <: AbstractWeatherModel
    amount::Type{D}; spatial::Symbol; temporal::Bool; threshold::Float64; fitted::F
end
CensoredLatentGaussian(; amount = Gamma, spatial = :empirical, temporal = true, threshold = 0.0)  # spatial = :exponential needs coords
SWGCore.variables(::CensoredLatentGaussian) = (:RR,)
```

**Model.** For month m:
- W_n = ρ_m W_{n−1} + √(1−ρ_m²) ε_n, with ε_n ~ N(0, C_m), so W_{s,n} ~ N(0,1).
- R_{s,n} = 0 if W ≤ c_{s,m} = Φ⁻¹(p0_{s,m}).
- Otherwise R = F⁻¹_{s,m}((Φ(W) − p0)/(1 − p0)).

**Fit (IFM).**
1. p0 is the dry frequency; F_{s,m} = `Distributions.fit_mle(Gamma, wet amounts)`.
2. Normal scores on wet days; dry days are censored below c.
3. C_m[s,s′] maximises the pairwise censored likelihood: φ₂ when both are wet, φ·Φ((c′−rw)/√(1−r²)) when one is wet, Φ₂(c,c′;r) when both are dry. This is golden-section on r. Alternatively, R_m of exp(−h/R) uses Core's `haversine`.
4. ρ_m uses the same likelihood on (W_n, W_{n−1}), pooled over sites (separable).

The bivariate normal CDF is a ported copy of SPSM's `fast_bivariate_cdf.jl` as a Core internal. This duplication is accepted.

**rand!.** AR field, then the anamorphosis.

**What the three classes share.**
- `GaussianDependence(by ∈ {:none, :regime, :month, :regime_month})`: fit (Pearson or Spearman, pairwise-complete, PD repair), subset sampling, and conditional sampling `rand_dependence!(…; given)`.
- Thresholding a latent at Φ⁻¹(p): WGEN, CLG, and SPSM's spatial HMM.
- The PIT/anamorphosis `normal_scores`: Rain amount, CopulaAR1, Difference, CLG.
- The calendar and the rand plumbing.

Seasonality differs at launch: smooth (SPSM bases) in the flagship, monthly in WGEN and CLG. Monthly is the IndicatorBasis special case, so smooth WGEN/CLG can come later without breaking anything.

## 6. SPSM v5

**Engine contract.** Data are positional; all conditioning is by keyword.

| Capability | v5.0 signature | Models |
|---|---|---|
| fit, model in/out | `fit_mle(m, y; n2t, solver, …) -> (m′, FitHistory)` | all |
| given z | `fit_mle(m, y; z::AbstractVector{<:Integer})` | SmoothAR1 (v4, keyword now), SmoothDistribution (new), SmoothARPeriodicHMM / sAR / SmoothPeriodicHMM (new) |
| freeze blocks | `fit_mle(hmm, Y; fixed = (:A,))` | the HMMs (new; replaces `fit_mle_RO`) |
| weights | `fit_mle(m, y; weights)` | SmoothSeasonalMixture (v4); SmoothDistribution (new); SmoothAR1 **5.1** |
| per-state log-densities | `loglikelihoods(m, y; n2t, past) -> N×K` | HMMs (v4); SmoothAR1, SmoothDistribution (new) |
| multiple sequences | `y::AbstractVector{<:AbstractVecOrMat}` | HMMs (v4); SmoothAR1, SmoothDistribution, mixture (new: no bridging) |
| simulate | `rand(rng, m, n2t; z, past, init) -> (; z, Y)` or values | all; `init` = law/state of z₁ (HiddenMarkovModels' meaning); `seq` removed |
| decode | `viterbi(m, Y; past, n2t)` | HMMs |
| start | `SmoothARPeriodicHMM(K, D, T; memory, degree)` + `fit_mle(…; init = :slices)` (default) | HMMs (replaces `randARPeriodicHMM` → slices → `fit_θ!` → rebuild) |
| model selection | `fit(SeasonalFit, m, y; …)` | unchanged |

Returning `(; z, Y)` still destructures as `z, Y = rand(…)`.

**Vector links: no, not in 5.0.** Correlations stay piecewise-constant by group, in SWG Core. Deduplicating the four softmax copies is internal refactoring only. The vendored TVP is unchanged (D13 deferred).

**Exports.**
- **Kept exported (42):**
  - TrigBasis, ConstantBasis, FourierBasis, PolynomialBasis, IndicatorBasis, BSplineBasis, CyclicBSplineBasis, ParamSpec
  - IdentityLink, LogLink, LogitLink, IntervalLink, curve
  - TimeVaryingFamily, seasonal_family
  - SmoothAR1, **SmoothDistribution** (new), SmoothSeasonalMixture, SmoothPeriodicHMM, SmoothARPeriodicHMM, SmoothsARPeriodicHMM, SmoothARPeriodicHMMSpatial
  - fit_mle, fit, SeasonalFit, FitHistory, converged, parameters, nparams, loglikelihoods, viterbi
  - coef, loglikelihood, dof, nobs, aic, aicc, bic
  - mixtures, components, ncomponents, n_to_t
- **`public` only (~36):**
  - paramnames, design_matrix, ncoef, coefs, blockindex, blockrange, nbasis
  - PeriodicDesign, PeriodicNLL, PointwiseNLL, group_by_time, distributions
  - ClassicEM, StochasticEM, ncolumns, column
  - default_nlls_solver, default_adtype, default_ensemble, solve_unconstrained, solve_nlls
  - ar1_spec, ar1_sufficient_statistics, ar1_nll
  - AR1, ARPeriodicHMM, sARPeriodicHMM, ARPeriodicHMMSpatial, PeriodicHMMDists
  - SeasonalControls, emission_coefs, emission_range, component_coefs, component_range, degrees, mixture_weights
- **Removed (17 names + 1 method):**
  - TimeVaryingFit → SmoothDistribution
  - mixture_weight_spec, fit_loss_optim
  - Trig2ARPeriodicHMM, Trig2ARPeriodicHMMSpatial, randARPeriodicHMM
  - fit_θ, fit_θ!, fit_θᴬ!, fit_θᴮ!, fit_mle_all_slices
  - fit_mle_RO
  - μₜ, ρₜ, σₜ, αₜ
  - params(m, t) → m(t)
  - the multisite `rand(::AbstractArray{<:AR1}, n2t, z, Σ)` method
- **Renamed: no type.** Model names stay as in v4 (v4 is two days old). The evaluated twins are no longer exported, which frees `AR1` for the SWG block. Keywords become uniform: `Y_past`/`y₁` → `past`, `z_ini` → `init`, positional z/weights → `z=`/`weights=`.
- **Copulas compat:** `"1"`.

**Moves.**
- **SPSM → SWG Core/RegimeHMM:** multisite AR(1) simulation with Σ (now GaussianDependence plus the AR1 block, with a fit and a test); the `fit_mle_RO` use case (now the Rain block via `fixed`).
- **SWG → SPSM:** `_ar1_init`, which becomes SmoothAR1's default start; the averaged-NLL seasonal-family fit `_fit_seasonal_gamma`, which becomes `SmoothDistribution`.

## 7. Naming & verbs table

| Concept | New name | Meaning | Replaces (SWG v3 / SPSM v4) |
|---|---|---|---|
| Observations / simulations | `WeatherData` | Tables.jl table over D×N(×S) arrays | per-station DataFrames, innerjoin with df_z, Y/BitMatrix, dicts_simu |
| Model class | `RegimeHMM`, `WGEN`, `CensoredLatentGaussian` | swappable algorithm type | hand-wired pipelines; `wgen`, `fit_wgen` |
| Fit (facade) | `fit(data, model)` → `WeatherFit` | one result type | 6 return shapes |
| Fit (engine) | `fit_mle(model, data)` → (model, history) | v4 contract, everywhere | same |
| Simulate | `rand([rng,] gen, dates; nsim)` | returns `WeatherData` | `rand(hmm…; seq)`, `rand_RR`, `rand(ar1s…)`, `rand_cond`, `simulate_markov_gaussian` |
| Initial condition | `past` | the days before the first simulated day | `Y_past`, `y₁`, `Y_ini`, `tx_ini`, `z_ini` |
| Conditioning | `given = (z = …,)` | imposed regime path | passing `zs` to every sampler |
| Decoding | `regimes(gen[, data])` | ẑ as a table / vector | `viterbi(hmm, Y, Y_past; n2t)` + CSV |
| Rain block | `Rain` | occurrence with memory × amount on the wet set | `fit_mle_RO`, `fit_mle_RR`, `mix_ini`, `cov_RR`/`cor_RR`, `rand_RR`, `onefy` |
| AR block | `AR1` | Gaussian AR(1) per site × regime + innovation copula | `fit_AR1`, `cov_ar1`, `rand(ar1s, n2t, z, Σ)` |
| Latent AR | `CopulaAR1(D)` | AR(1) on normal scores, any marginal | AR1 + `indicatrix` clipping |
| Conditional block | `Difference(:TX, Gamma)` | x = parent ± Δ, joint Gaussian with parent innovations | `fit_TN`, `cor_groupbyTXTN`, `rand_cond`, `ϵ_TX` |
| Dependence | `copula = :regime \| :regime_month \| :none` | grouping of correlation matrices | Σ²RR, Σ²TX… per regime |
| Seasonal degree | `degree` (Int or NamedTuple of Int/basis) | per-parameter seasonality | 𝐃𝐞𝐠, `degree`, `d` |
| Memory | `memory` (days) | occurrence lag | `local_order`, `memory_order = 2^order` |
| Read-out | `curve(block, :ρ)` | D×K×366 | `curve`, μₜ/ρₜ/σₜ/αₜ |
| Adding stations | `merge(obs, obs_new)` | site union | manual joins |
| Calendar | `dayofyear_Leap` (public, Core) | t ∈ 1:366 | same, exported |
| Seasonal marginal (engine) | `SmoothDistribution` | K-state seasonal law | `TimeVaryingFit`, `fit_loss_optim` |
| Freeze (engine) | `fixed = (:A,)` | skip M-step blocks | `fit_mle_RO` |
| HMM start (engine) | `SmoothARPeriodicHMM(K, D, T; memory, degree)` | slices init inside `fit_mle` | `randARPeriodicHMM`, `fit_mle_all_slices`, `fit_θ!` |

## 8. Migration & release plan

**Ordered steps.** Each is one or more mergeable PRs, and the tutorials keep running throughout.

| # | Repo | Step | Why it is safe |
|---|---|---|---|
| S0 | SWG | Add CI.yml. Remove the network download at precompile. Render tutorials via Literate. Freeze **v3 reference fixtures**: `test/reference/make_v3.jl` runs in a pinned env (SWG 3.0.0 + SPSM 4.1.0, Manifest committed) on synthetic data simulated from the tutorial's fitted model (no licensing issue: INRAE data are not open), 30 years × 4 sites + 10 sites for regimes. | no API change |
| P1 | SPSM | **4.2, additive.** Keyword forms; `SmoothDistribution`; size constructors + `init = :slices`; `fixed`; sequences for AR1/mixture/SmoothDistribution; `loglikelihoods` for AR1/SmoothDistribution; `seq = true` returns a NamedTuple. Downstream CI job. | old forms untouched |
| S1 | SWG | `lib/…Core` + tests. Root `[workspace]` lists it. | umbrella untouched |
| S2 | SWG | `lib/…WGEN` + equivalence tests. | same |
| S3a–c | SWG | `lib/…RegimeHMM`: (a) regimes + Rain, (b) AR1 + CopulaAR1, (c) Difference, against SPSM ≥ 4.2. If P1 is not registered yet, the workspace uses `[sources] SmoothPeriodicStatsModels = {url, rev}`. | one manifest; SPSM 4.2 satisfies both the umbrella's "4.1" and the lib's "4.2, 5" |
| S4 | SWG | `lib/…LatentGaussian` + parameter-recovery tests. | same |
| S5 | SWG | New tutorials in `examples/v4/` built under "Preview (v4)". The spatial-HMM tutorial becomes an SPSM-level expert tutorial. | old tutorials still built |
| R1 | General | Register Core, WGEN, LatentGaussian 1.0.0, then RegimeHMM 1.0.0 (SPSM compat `"4.2, 5"`). Allow about 3 days for new packages. | no user depends on them |
| P2 | SPSM | **5.0** breaking PR: removals, export pruning, CHANGELOG, migration page. Register. | SWG 3.0 caps SPSM at "4.1" ⇒ not pulled in |
| S6 | SWG | **4.0** cutover, same day: umbrella depends on the libs + Reexport; delete `src/rain`, `src/AR1`, `src/temperature`, `ext/SmoothPeriodicStatsModelsExt`; swap tutorials; migration page; compat SPSM "5". Register. | the tutorials were already validated in S5 |
| R2 | General | RegimeHMM 1.0.1 with SPSM compat "5" only, once the 4.x lane is dropped from CI. | patch |

**Old → new (most-used calls).**

| v3 / v4 | v4 / v5 |
|---|---|
| `collect_data_INRAE.(p)` + `innerjoin(df, df_z)` | `WeatherData(collect_data_INRAE.(p))`; regimes via `RegimeHMM(regimes = …)` |
| `@transform!(df, :RO = onefy.(:RR))` | automatic (`Rain(threshold = 0.0)`) |
| `fit_mle_RO(df, order, deg; solver)` + `ARPeriodicHMM(a, hmm.A, B)` | `Rain(memory = order, degree = deg)` in `fit(obs, RegimeHMM(regimes = paper))` |
| `fit_mle_RR.(dfs, order; mix₀ = mix_ini(T))` | `Rain()` (default amount) |
| `cov_RR(dfs, K)` / `cor_RR` | `Rain(copula = :regime)` (default) |
| `fit_AR1.(dfs, :TX, deg; solver)` + `cov_ar1(dfs, ar1s, :TX)` | `TX = AR1(degree = deg)` |
| `fit_TN.(dfs, 1, T)` + `cor_groupbyTXTN.(dfs, T)` | `TN = Difference(:TX, Gamma; degree = 1)` |
| QQ/ETPP `fit_AR1` + `indicatrix` | `CopulaAR1(Gamma)` (or `AR1()` and clip yourself = v3) |
| `rand(hmm, n2t; Y_past, z_ini, seq = true)` | `rand(gen, dates; past)`; `sims.regimes` |
| `rand_RR(mixs, n2t, z, y, Σ)` | included in `rand(gen, dates)` |
| `rand(ar1s, n2t, z, Σ; y₁, output_ϵ)` | included |
| `rand_cond(ϵ_TX, z, θ_ΔT, θ_cor, n2t, T)` | included |
| `fit_wgen(Y, idx_months, p)` | `fit(obs, WGEN(order = p))` |
| `rand(wgen_model, 1956:2019; Y_ini)` | `rand(gen, Date(1956):Date(2019,12,31); past)` |
| `fit_markov_chain`, `fit_Ω`, `simulate_markov_gaussian` | internal to WGEN |
| `viterbi(hmm, Y, Y_past; n2t)` (SWG use) | `regimes(gen)` / `regimes(gen, obs)` |
| result DataFrame loops | `DataFrame(rand(gen, dates; nsim))` |
| `longuest_spell` | `longest_spell` |
| `whole_year`, `onefy`, `zerofy`, `vec_triu`, `my_color` | removed |
| SPSM `fit_mle(SmoothAR1(d,T;K), y, z; n2t)` | `fit_mle(SmoothAR1(d,T;K), y; z, n2t)` |
| SPSM `fit_mle(mix, y, w; n2t)` | `fit_mle(mix, y; weights = w, n2t)` |
| SPSM `fit_mle(hmm, Y, Y_past; n2t)` | `fit_mle(hmm, Y; past = Y_past, n2t)` |
| `randARPeriodicHMM` → `fit_mle_all_slices` → `fit_θ!` → `SmoothARPeriodicHMM(a, θᴬ, θᴮ, T)` | `fit_mle(SmoothARPeriodicHMM(K, D, T; memory, degree), Y; past, n2t)` |
| SPSM `fit_mle_RO(Y, Y_past, z, n2t, d, K, T)` | `fit_mle(SmoothARPeriodicHMM(K, D, T; memory, degree), Y; past, z, fixed = (:A,), n2t)` |
| SPSM `rand(hmm, n2t; Y_past, z_ini, seq)` | `rand(hmm, n2t; past, init = z_ini)` → `(; z, Y)` |
| SPSM `rand(ar1, n2t, z; y₁)` | `rand(ar1, n2t; z, past = y₀)` (the value *before*) |
| `TimeVaryingFit` / TVP `fit` / `fit_loss_optim` | `fit_mle(SmoothDistribution(fam, T; K), y; z, n2t)` |
| `ARPeriodicHMM(m)`, `AR1(m)`, μₜ… | `SmoothPeriodicStatsModels.ARPeriodicHMM(m)` (public), `m(t)`, `curve(m, :μ)` |

**Numerical equivalence against v3/v4 fixtures.**

| Component | Test | Tolerance |
|---|---|---|
| `dayofyear_Leap`, WGEN transition probabilities | exact | ==, since both are counting |
| RO coefficients given ẑ; regime HMM on 10 sites (same slice options, seed) | same objective, solver and start | rtol 1e-6 |
| (z, RO) path, single realisation, same seed and `init` | same draw order | exact |
| RR mixture per (s, k); AR1 per site | same SPSM call | rtol 1e-6 |
| ΔT Gamma per k | v3 used averaged NLL in a closure; SmoothDistribution uses the same normalisation | rtol 1e-4 |
| `cor_RR`; `cov_ar1` → correlation (on the gap-free fixture) | same estimator | 1e-10 |
| WGEN path given p, Ω and seed | same draw order | exact |
| WGEN Ω | common random numbers vs v3 MC | atol 0.03 |
| Full simulated climatology, 200 sims | monthly mean/sd, wet frequency, spell PMFs, cross-site and cross-variable correlation | within 4 Monte Carlo standard errors |

**Intentional differences**, listed in the migration guide:
- TN coupling: v3 fitted a raw correlation but simulated on the innovation scale; v4 estimates the coupling on the innovation scale.
- The regime-K independence hack in `rand_RR` is removed.
- QQ/ETPP are positive through CopulaAR1.
- `cov_ar1` handles gaps and the first residual correctly.
- Regime-indexed fits no longer depend on `groupby` order.
- Each of these gets a property test instead of an equivalence test.

## 9. Risks, trade-offs, open questions

**Deliberately not in the first release (all can be added later without a break):**
1. Soft-γ and joint-EM estimation. The weighted contract is reserved and AR1 weights come in 5.1; `RegimeHMM(estimation = …)` is added only when implemented.
2. Vector links and smoothly seasonal correlation C_k(t).
3. Non-Gaussian copulas (a future Copulas.jl extension on Core).
4. Regime-dependent CLG / `CensoredRain` block; smooth-seasonal WGEN/CLG; WGEN amounts.
5. Merging SPSM's four HMM types; renaming SPSM types; moving SPSM into the monorepo.
6. Covariates in transitions (NHMM), except through `given`.
7. A versioned on-disk format.
8. A validation verb.
9. A spatial (latent-Gaussian) regime emission inside `RegimeHMM`; it stays an SPSM-level tutorial.
10. Fully long-format input; sub-daily time; AR(p); truncated AR; GWEX/MSTWeatherGen/GWGEN/WeaGETS/AWE-GEN-2d. The 3-method model contract and the `TT` type parameter of `WeatherData` leave room for these.
11. A formula or operator DSL.
12. Partial refit (`reuse`) when adding stations.
13. Deprecation shims.

**Risks and trade-offs.**
- **Argument order.** `fit(data, model)` (decision 2b, SciML `solve` style) is the reverse of `fit_mle(model, data)` and of StatsAPI's `fit(Type, data)`. This is a teaching cost. It is mitigated because T0/T1 never see `fit_mle`. Open question: also accept `fit(model, data)`? I recommend not; one order only.
- **Plug-in likelihood.** `loglikelihood(gen)` for RegimeHMM is the HMM's marginal log-likelihood plus the block log-likelihoods conditional on ẑ. It is only comparable across models that share a regime path. Should `bic(gen)` throw instead for frozen regimes?
- **Defaults by column name** are convenient but brittle: ECA `RR` is in 0.1 mm, `TG` vs `TM`, and so on. Mitigation: unknown names fail loudly, and `show` prints the chosen blocks.
- **CopulaAR1.** Fitting a free `SmoothAR1` on the scores makes marginal invariance approximate. Exact invariance would need tied coefficients (σ = √(1−ρ²), μ = 0), which is SPSM engine work for 5.x.
- **Cost of Difference's joint copula.** It has (2D)(2D−1)/2 parameters per group. That is fine for D = 4; at D = 37 with `:regime_month` it is poorly determined. Mitigation: default to `:regime`, plus a warning when the number of days per group is less than 5× the number of parameters.
- **Wet-set Cholesky** per day at D = 37: about 23k days × 300 sims, roughly a minute if each Cholesky costs about 10 µs (unverified). Acceptable. A mask cache only applies when D ≤ 12.
- **Missing occurrences in regime training** are handled by sequence splitting, which wastes data when gaps are staggered across many stations. SPSM-level marginalisation is the real fix (5.x).
- **JLD2 persistence** breaks when struct layouts change. `regimes(gen)` to CSV is the stable path for frozen regimes.
- **Registration timing.** New packages wait about 3 days in General; hence R1 comes before the release day. Sublibrary names are long; the `SWG*` prefix was rejected for discoverability (open).
- **Versions.** Sublibraries at 1.0 while the umbrella is at 4.0 may confuse users (open; SciML precedent says independent).
- **Workspace manifest.** The single workspace manifest forces one SPSM version across the umbrella and the libraries during the transition. P1 being additive is what makes that possible; any breaking change slipped into 4.x would stall S3.
- **Umbrella load time.** The hard dependency on the Optimization stack is the price of a solver-free T0. Experts who want fast loads use the sublibraries directly.
- **Lockstep risk.** P2 and S6 must land together. The Downstream CI job and R1's `"4.2, 5"` compat let RegimeHMM work on both sides of the cut.

### Critical Files for Implementation
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\examples\tuto_add_station_variable.jl
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\ext\SmoothPeriodicStatsModelsExt\fit_mle_TN.jl (and fit_mle_rain.jl, fit_mle_AR.jl; src\rain\generation.jl, correlations.jl, wgen.jl)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\SmoothPeriodicStatsModels.jl (export surface)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\fit\smooth_hmm.jl (with src\fit\add_sites.jl, src\HHMM_Bernoulli\periodichmm.jl)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\AR\AR1.jl (with src\fit\ar1.jl, src\fitted.jl, src\TimeVaryingParameters\family.jl)