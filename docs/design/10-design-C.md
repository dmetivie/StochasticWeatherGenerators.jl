## 0. Thesis

The three launch models (the Regime-HMM product, WGEN and the censored latent Gaussian) are built from the same five primitives. So are the later ones (GWEX, Richardson-type, MSTWeatherGen-like):

- **(P1)** A seasonal parameter θ(t) = g⁻¹(B(t)β). The link g can be scalar or vector (softmax, correlation). The parameter is replicated over discrete indices: regime k, site s, memory category h, month, parent state.
- **(P2)** A conditional marginal law G_θ(y | own past), taken from Distributions.jl.
- **(P3)** A latent standard-Gaussian layer W with a structured correlation and optional latent AR(1). Observations are read off as **Y = G⁻¹(Φ(W))**. This one map covers three cases:
  - thresholding when G is Bernoulli (WGEN, SPSM's spatial HMM, regime-model occurrence);
  - a PIT transform when G is continuous (rain amounts, QQ);
  - censoring when G has an atom at 0 (Ailliot).
- **(P4)** A seasonal Markov chain Z that indexes every parameter.
- **(P5)** A DAG of blocks, coupled through their latent innovations (TN|TX).

Estimation is a strategy object over one engine contract. Every block exposes a *weighted* `fit!` and an N×K matrix of per-state log-densities. "Given ẑ" is then one-hot weights, "soft" is posterior weights γ, and joint EM sums the blocks' log-densities.

SPSM v5 becomes that engine: P1, P2, P4, weighted fitting and the seasonal HMM. SWG v4 owns P3, P5, data and calendar, and the model classes. Each model class either *lowers* to the primitives or implements the four core verbs directly.

## 1. Package architecture

```
StochasticWeatherGenerators.jl/                   (one repo, SciML-style monorepo)
├── Project.toml                 StochasticWeatherGenerators 4.0.0 (umbrella)
├── src/StochasticWeatherGenerators.jl   @reexport using SWGCore, SWGBlocks, SWGRegimeHMM,
│                                        SWGWGEN, SWGCensored, SWGData; using OptimizationOptimJL
├── lib/
│   ├── SWGCore/        interface, WeatherData, calendar, simulation container, diagnostics
│   │   └── ext/        SWGCoreMakieExt, SWGCoreGeoMakieExt, SWGCoreJLD2Ext
│   ├── SWGBlocks/      P3 + P5: Dependence, blocks, ProductModel, estimation strategies
│   │   └── ext/        SWGBlocksOptimJLExt (sets the default solver)
│   ├── SWGRegimeHMM/   RegimeHMM (flagship preset over SWGBlocks)
│   ├── SWGWGEN/        WGEN (Wilks 1998 / Srikanthan 2009) + WilksMC estimator
│   ├── SWGCensored/    CensoredGaussian (Ailliot 2009; Vaittinada Ayar 2020)
│   └── SWGData/        ECA / Météo-France / INRAE readers, cleaning, imputation
├── docs/   one Documenter site, organised by tier, plus the migration guide
├── test/   cross-library integration tests + golden v3/v4 fixtures
└── .github/workflows/  per-lib CI with path filters; "downstream of SWGCore" job; TagBot per subdir
SmoothPeriodicStatsModels.jl/    separate repo, v5.0.0: the seasonal statistics engine
```

| Package | Purpose | Hard deps | ~exports |
|---|---|---|---|
| SWGCore | Abstract types, verbs, `WeatherData` (Tables.jl input), `Calendar`, `WeatherSimulation` (Tables.jl output), `FittedGenerator`, validation statistics (spells, monthly aggregates), save/load protocol | Dates, Random, Statistics, LinearAlgebra, Tables, StatsAPI, PrecompileTools | 30 |
| SWGBlocks | `Seasonal` vocabulary, `Regimes`, blocks (`Occurrence`, `Amount`, `Rain`, `AR1`, `Marginal`, `Difference`), `Dependence` and its structures and estimators, `LatentAR`, `ProductModel` (DAG), `TwoStage` / `JointEM`; re-exports SPSM bases and links | SWGCore, SPSM 5, Distributions, Copulas 1, MvNormalCDF, NearestCorrelationMatrix, ComponentArrays, Optimization, SimpleNonlinearSolve | 40 |
| SWGRegimeHMM | `RegimeHMM`; default block per variable name | SWGBlocks | 2 |
| SWGWGEN | `WGEN`, `WilksMC` | SWGBlocks, NonlinearSolve (only for WilksMC) | 2 |
| SWGCensored | `CensoredGaussian`, `CensoredRain`, `PowerCensoredNormal` | SWGBlocks, SpecialFunctions | 3 |
| SWGData | `collect_data_*`, `clean_data`, geography helpers; returns plain tables | SWGCore, CSV, DataFrames, Impute, Downloads | 18 |
| Umbrella | Batteries included: re-exports everything and loads OptimizationOptimJL so `LBFGS()` is the default solver | all of the above + Reexport | ≈95, tiered in the docs; T0 uses about 8 |

**Where SPSM lives: in its own repository, not in the monorepo.** Four reasons:
- It is domain-agnostic (tuto_genomics.jl fits a `SmoothARPeriodicHMM` with spline bases to gene sequences).
- Its name, UUID and issue tracker already exist.
- The vector links and covariates belong next to TimeVaryingParameters and should go upstream there (D13).
- SciML keeps DiffEqBase outside OrdinaryDiffEq for the same layering reason.

Lockstep is enforced by CI rather than co-location. During development, SWGBlocks' Project.toml carries `[sources] SmoothPeriodicStatsModels = {url = "...", rev = "main"}` (Julia ≥ 1.11; the registry ignores `[sources]`). The release order is SPSM 5.0 first, then the SWG libraries the same day (§8).

**Load time.**
- A model-class author's minimum is `using SWGCore`, which pulls no Distributions, no SciML and no DataFrames. A k-NN resampler needs nothing else.
- `using SWGRegimeHMM` pays for Distributions + HiddenMarkovModels + ComponentArrays + Copulas + Optimization (several seconds; *not measured*).
- The umbrella adds CSV/DataFrames/Optim for T0 convenience.
- The network download that today runs at precompile time (extracting.jl:176-178) moves to a runtime call in SWGData.
- Each library ships a PrecompileTools workload: a tiny synthetic fit and simulate.

**Registration and versioning.**
- Each `lib/X` has its own Project.toml (name, uuid, version, [compat]) and test/ folder.
- Registration: `@JuliaRegistrator register subdir=lib/SWGCore` on the release commit (the SciML pattern).
- TagBot runs with `subdir:` and produces tags like `SWGCore-v1.0.0`.
- Versions follow independent SemVer. Sublibraries start at 1.0.0, the umbrella at 4.0.0, and the umbrella's `[compat]` pins the set.
- Locally, a root `[workspace] projects = ["lib/SWGCore", …, "docs", "test"]` (Julia 1.12) resolves one manifest, and each lib's `[sources]` points at `../SWGCore`.
- A breaking change in SWGCore is a coordinated bump of every library. This is the reason SWGCore stays tiny.

## 2. Core interface

### 2.1 Types (SWGCore)

```julia
abstract type AbstractWeatherGenerator end   # a model class + options, unfitted (≈ SciML algorithm)
abstract type AbstractFittedGenerator end
abstract type Calendar end
struct DayOfYear366 <: Calendar end          # T = 366; Feb 29 ↦ 60; non-leap Mar 1 ↦ 61 (= v3 dayofyear_Leap)
struct NoLeap365 <: Calendar end             # T = 365; Feb 29 ↦ 59 (merged with Feb 28)
abstract type DecodeMethod end
struct Viterbi <: DecodeMethod end
struct Posterior <: DecodeMethod end

struct Site
    id::String; lat::Float64; lon::Float64; alt::Float64; name::String  # NaN / "" when unknown
end

struct WeatherData{NT<:NamedTuple, C<:Calendar}
    dates::Vector{Date}       # contiguous daily grid; absent days are rows of `missing`
    sites::Vector{Site}
    values::NT                # (RR = D×N Matrix{Union{Missing,Float64}}, TX = …)
    calendar::C
end

struct FittedGenerator{A<:AbstractWeatherGenerator, M, I} <: AbstractFittedGenerator
    alg::A        # the spec as written; `show` prints it as pasteable Tier-1 code
    model::M      # the lowered, fitted ProductModel (or a class-specific object)
    info::I       # dates, sites, calendar, decoded path, per-block histories, versions
end

struct WeatherSimulation{NT<:NamedTuple, Z}
    dates::Vector{Date}; sites::Vector{Site}; calendar::Calendar
    values::NT    # (RR = D×N×nsim Array{Float64,3}, …)
    z::Z          # N×nsim Matrix{Int}, or nothing for classes without regimes
end
```

### 2.2 Verbs

```julia
WeatherData(table; date = :DATE, site = :STAID, variables = :auto, layout = :auto,
            sites = nothing, calendar = DayOfYear366())               # any Tables.jl table
WeatherData(values::NamedTuple, dates::AbstractVector{Date}; sites, calendar = DayOfYear366())
Base.getindex(d::WeatherData, v::Symbol)        # D×N view
variables(d), sites(d), dates(d), periods(d)::Vector{Int}, period(cal)::Int
select(d::WeatherData; sites = :, variables = :, dates = :)::WeatherData

StatsAPI.fit(data::WeatherData, alg::AbstractWeatherGenerator;
             solver = default_solver(), rng::AbstractRNG = Random.default_rng(),
             verbose::Bool = false)::AbstractFittedGenerator
StatsAPI.fit(table, alg::AbstractWeatherGenerator; kw...) = fit(WeatherData(table), alg; kw...)

simulate([rng::AbstractRNG,] fg::AbstractFittedGenerator, dates::AbstractVector{Date};
         nsim::Int = 1, init = :stationary, given::NamedTuple = (;),
         latent::Bool = false, threaded::Bool = true)::WeatherSimulation
simulate([rng,] fg, years::AbstractUnitRange{<:Integer}; kw...)   # 1 Jan first → 31 Dec last
Base.rand([rng,] fg, dates; init, given)::WeatherData             # one realization
Base.getindex(s::WeatherSimulation, i::Integer)::WeatherData      # realization i, re-fittable

StatsAPI.loglikelihood(fg[, data]); nobs(fg); dof(fg); aic(fg); bic(fg)
likelihood_kind(fg)::Symbol            # :exact | :qmc | :composite | :none (trait)
decode(fg[, data]; method::DecodeMethod = Viterbi())  # Tables.jl (DATE, z)
posterior(fg[, data])::Matrix{Float64}                # N×K
validate([rng,] fg, data = training(fg); nsim = 100, statistics = default_statistics(fg))
write_generator(path, fg); read_generator(path)       # JLD2 extension, versioned schema

# path-2 hooks for model-class authors (§3 T2b)
simulate!(buf::SimulationBuffer, rng::AbstractRNG, fg::AbstractFittedGenerator, ctx::SimContext)
```

`fit(data, alg)` follows the `solve(prob, alg)` order. It is a `StatsAPI.fit` method that dispatches on `WeatherData`, so there is no ambiguity with type-first `fit(::Type, x)` methods.

### 2.3 Block protocol (SWGBlocks), path 1

```julia
abstract type AbstractBlock end           # immutable spec of one variable block
abstract type AbstractFittedBlock end
struct StateScope end                     # density depends on z_n only
struct EdgeScope end                      # depends on (z_{n-1}, z_n)  (latent AR, §4.4)

outputs(b::AbstractBlock)::Tuple          # variables written, e.g. (:RR,)
inputs(b::AbstractBlock)::Tuple = ()      # DAG parents, e.g. (:TX,)
memory(b::AbstractBlock)::Int = 0         # past days needed
scope(b::AbstractBlock) = StateScope()
StatsAPI.fit(b::AbstractBlock, d::BlockData, w::AbstractStateWeights; solver, rng)::AbstractFittedBlock
loglikelihoods!(LL::AbstractMatrix, fb::AbstractFittedBlock, d::BlockData)   # N×K, optional
simulate!(out::NamedTuple, fb::AbstractFittedBlock, p::SimPath, rng::AbstractRNG)
lower(alg::AbstractWeatherGenerator, data::WeatherData)::ProductModel   # path-1 classes
```

- `BlockData` holds the block's D×N matrices (with missing), `t`, the parents' values and innovations, sequence boundaries, `Site`s (distances) and the calendar.
- `SimPath` holds the z path, `t`, dates, the parents' simulated values and latent innovations, and `init`.
- `AbstractStateWeights` comes from SPSM (§6).

### 2.4 Tables.jl contract

**Input.**
- Any Tables.jl table, in one of two layouts. `layout = :auto` detects which:
  - wide: date, site, and one column per variable;
  - long: date, site, variable, value.
- The site column is optional (single site).
- Values are `Union{Missing,Real}`, and dates are `Date` (DateTime is truncated).
- An optional `sites =` table (id, LAT, LON, ALT, NAME) provides the coordinates needed by `ExpKernel`. A block that needs distances errors if they are absent.
- **Arrays underneath:** `WeatherData((RR = R, TX = X), dates; sites)` with D×N matrices.

**Output.**
- `WeatherSimulation` is a Tables.jl table in wide rows, with columns `sim, <date col>, <site col>, vars...` named as in the input. `DataFrame(sims)` works without DataFrames being a dependency, and `long(sims)` gives the long layout.
- `sims.values.TX` is the D×N×nsim array for experts.
- `decode` and `validate` also return tables.

### 2.5 Calendar

- `t` is always computed from dates by the calendar and is never inferred from a length. This fixes the silent `n_to_t` default (pain point #10).
- In SWG, T = 366 everywhere (`DayOfYear366`). Monthly models use `Monthly()`, an `IndicatorBasis` over the 366-day index. WGEN's monthly parameters are therefore *a basis choice*, not a different time axis.
- `:season` means DJF/MAM/JJA/SON on the same index.
- Simulated dates include Feb 29 in leap years. Day 60 is fitted from about a quarter of the data, which smooth bases handle.

### 2.6 RNG

- Every stochastic verb takes an optional leading `rng`.
- `simulate` draws one `UInt64` seed per realization sequentially from `rng`, then runs the realizations threaded with `Xoshiro(seed_i)`. Results are therefore identical for any thread count.
- Randomised fits (random restarts, QMC orthant probabilities, `WilksMC`) take `rng =`.

### 2.7 Initial conditions and conditioning (one convention)

`init` always describes **the days before the first simulated date** (z₀ and each block's `memory(b)` past days). It replaces `y₁`, `Y_past`, `z_ini` and `Y_ini`.

- `init = :stationary` (default): z₀ is drawn from the empirical regime frequency at t₀; each block's past is drawn from its own stationary or marginal law.
- `init = :data`: the actual days before the start date, taken from the training data. It errors if they were not observed.
- `init = (z = 2, RR = D×m matrix, TX = D×1 matrix)`: partial NamedTuples are allowed, and missing entries fall back to `:stationary`.

`given = (z = zpath,)` conditions on a regime path. `given = (RR = obs,)` conditions children on observed parents. Conditioning on a child to simulate its parent is rejected at lowering, using the DAG. `latent = true` also returns the innovations.

### 2.8 Missing data and gaps

The rule is to marginalise where that is exact, and otherwise split.
- Independent margins and Gaussian copulas on observed sub-vectors marginalise exactly (the correlation matrix is subset).
- Blocks with memory treat a missing day as a sequence break. The next run takes its first `memory` days as its `past`, and those days do not enter the likelihood. This fixes the silent bridging of gaps in the AR(1) (pain point #11).
- The regime HMM gives a missing site a log-density of 0.
- There is no imputation inside fitting; `SWGData.impute` is opt-in.
- `fit` reports per-site, per-variable coverage in `fg.info`.

## 3. Three tiers, by example

**T0: impact modeller (crop tutorial: 4 stations × RR, TN, TX, QQ, ETPP).**

```julia
using StochasticWeatherGenerators, DataFrames, Random
obs  = reduce(vcat, collect_data_INRAE.(station_files))     # DATE, STAID, RR, TN, TX, QQ, ETPP
regs = Regimes(read_generator("assets/tuto_1/regimes_K4_d1_m1.swg"); frozen = true)
fg   = fit(obs, RegimeHMM(regimes = regs))
sims = simulate(Xoshiro(50000), fg, 1986:2035; nsim = 300)
df   = DataFrame(sims)                                      # sim, DATE, STAID, RR, TN, TX, QQ, ETPP
```

- Without pre-trained regimes, write `fit(obs, RegimeHMM(K = 4))`. The regimes are then learnt from the stations' own rain occurrence.
- The defaults are chosen by variable name (`default_block(::Val{:TX})`, extensible):
  - RR → `Rain()`, i.e. occurrence with memory 1, an exponential mixture for amounts, and a Gaussian copula per regime;
  - TX → `AR1()`;
  - TN → `Difference(:TX, Gamma)`;
  - QQ and ETPP → `Marginal(Gamma)` (positive, latent AR(1));
  - any other variable → `AR1()`.
- `validate(fg; nsim = 100)` followed by `plot(…)` reproduces the tutorial's figures.
- `show(fg)` prints the full Tier-1 spec that was used.

**T1: applied statistician (same model, customised).** An integer means a trig degree; `basis => link` is the override.

```julia
using StochasticWeatherGenerators, Distributions
gen = RegimeHMM(
    regimes    = Regimes(fg; frozen = true),               # reuse fitted regimes (path + chain)
    estimation = TwoStage(decode = Posterior()),            # γ-weighted M-steps instead of ẑ
    RR = Rain(
        occurrence = Occurrence(memory = 2, p = Seasonal(2)),
        amount = Amount(Exponential => (θ = 2,), Gamma => (α = 0, θ = 2); weights = 1,
                        dependence = Dependence(Free(); by = :regime))),
    TX = AR1(μ = 3, ρ = 1, σ = TrigBasis(366, 2) => LogLink();
             dependence = Dependence(Free(); by = (:regime, :season))),   # C_{k,season}
    TN = Difference(:TX, Gamma; α = 1, θ = 2, r = 1),     # ΔT = TX−TN; corr r_k(t) with TX innovation
    QQ = Marginal(Gamma; α = 1, θ = 2, dynamics = LatentAR(ρ = 1),
                  dependence = Dependence(Free(); by = Seasonal(1))),     # smooth C_k(t)
    ETPP = Marginal((μ, σ) -> truncated(Normal(μ, σ); lower = 0); μ = 2, σ = 1),
)
fg2 = fit(vcat(obs, new_stations), gen)     # adding stations = refit under frozen regimes
```

Links are inferred from the parameter's support: the Distributions parameter table for types, and a name convention for closures (μ identity; σ/θ/α/λ log; ρ/r in (−1, 1); p logit). *(The convention is a proposal.)*

**T2a: expert writes a new block** (an AR(2), the others.md TODO).

```julia
using SWGBlocks, SmoothPeriodicStatsModels, Distributions, Random
import SWGBlocks: outputs, memory, simulate!, rand_innovations, innovations, fit_dependence, sitedata
import StatsAPI: fit

Base.@kwdef struct AR2 <: AbstractBlock
    var::Symbol = :TX
    μ = 1; φ1 = 1; φ2 = 0; σ = 1
    dependence = Dependence(Free(); by = :regime)
end
outputs(b::AR2) = (b.var,)
memory(::AR2) = 2

# y_n | x_n = (y_{n-1}, y_{n-2}): an SPSM v5 `Conditional` family (lags as covariates, §6)
ar2_family(b, T) = TimeVaryingFamily(
    Conditional((μ, φ1, φ2, σ) -> x -> Normal(μ + φ1 * x[1] + φ2 * x[2], σ)),
    ParamSpec(μ = basis(b.μ, T) => IdentityLink(), φ1 = basis(b.φ1, T) => IntervalLink(-1, 1),
              φ2 = basis(b.φ2, T) => IntervalLink(-1, 1), σ = basis(b.σ, T) => LogLink()))

struct FittedAR2{M,C} <: AbstractFittedBlock
    var::Symbol; margins::Vector{M}; dependence::C
end

function fit(b::AR2, d::BlockData, w::AbstractStateWeights; solver, rng = Random.default_rng())
    fam = StateIndexed(ar2_family(b, period(d)), nstates(w))
    margins = map(eachindex(sites(d))) do s
        sd = sitedata(d, b.var, s; lags = 2)        # SeasonalData; gaps split sequences
        fit!(initialize(fam, sd, w), sd, w; solver)
    end
    ε = innovations(margins, d)                     # D×N standardised residuals, missing kept
    return FittedAR2(b.var, margins, fit_dependence(b.dependence, ε, w, d))
end

function simulate!(out::NamedTuple, fb::FittedAR2, p::SimPath, rng::AbstractRNG)
    Y = out[fb.var]; ε = rand_innovations(rng, fb.dependence, p)   # D×N, ~ N(0, C_{z_n}(t_n))
    past = p.init[fb.var]                                          # D×2: days -1, 0
    for s in axes(Y, 1)
        y2, y1 = past[s, 1], past[s, 2]
        for n in eachindex(p.z)
            law = fb.margins[s](p.t[n]; state = p.z[n])((y1, y2))
            Y[s, n] = quantile(law, cdf(Normal(), ε[s, n]))   # any continuous law, coupling-ready
            y2, y1 = y1, Y[s, n]
        end
    end
    return out
end
```

`loglikelihoods!` is optional. Without it, the block works under `TwoStage(Viterbi())` but not under `Posterior()` or `JointEM()`. A clear error says so at lowering.

**T2b: expert writes a new model class** (path 2, k-NN resampler, Lall & Sharma 1996), in a new `lib/SWGKNN`.

```julia
module SWGKNN
using SWGCore, Random, Statistics
import SWGCore: fit, simulate!, variables, sites
export KNN
Base.@kwdef struct KNN <: AbstractWeatherGenerator
    k::Int = 10; window::Int = 7; features::Vector{Symbol} = [:RR, :TX]
end
struct FittedKNN <: AbstractFittedGenerator
    alg::KNN; data::WeatherData; Z::Matrix{Float64}         # standardised features, F×N
end
function fit(data::WeatherData, alg::KNN; kw...)
    Z = reduce(vcat, [mean(data[v]; dims = 1) for v in alg.features])   # assumes no missing (sketch)
    return FittedKNN(alg, data, (Z .- mean(Z; dims = 2)) ./ std(Z; dims = 2))
end
variables(fg::FittedKNN) = variables(fg.data)
sites(fg::FittedKNN) = sites(fg.data)
function simulate!(buf, rng::AbstractRNG, fg::FittedKNN, ctx)
    cur = rand(rng, calendar_window(fg.data, ctx.t[1], fg.alg.window))   # SWGCore helper
    for n in eachindex(ctx.t)
        copy_day!(buf, n, fg.data, cur)                                    # all vars, all sites
        n == lastindex(ctx.t) && break
        pool = filter(<(size(fg.Z, 2)), calendar_window(fg.data, ctx.t[n], fg.alg.window))
        d = [sum(abs2, view(fg.Z, :, j) .- view(fg.Z, :, cur)) for j in pool]
        nn = pool[partialsortperm(d, 1:fg.alg.k)]
        p = cumsum(1 ./ (1:fg.alg.k)); cur = nn[searchsortedfirst(p, rand(rng) * p[end])] + 1
    end
    return buf
end
end
```

**T2c: an MSTWeatherGen-like class via path 1** *(speculative: `JointField`, `ObservedRegimes` and `SpaceTimeMatern` are post-launch types)*.

```julia
struct MSTLike <: AbstractWeatherGenerator; K::Int; vars::Vector{Symbol}; end
SWGBlocks.lower(a::MSTLike, data) = ProductModel(Regimes(a.K; emission = ObservedRegimes(:WT)),
    (; field = JointField(a.vars; margins = (RR = CensoredRain(), TX = Marginal(Normal)),
                         dependence = Dependence(SpaceTimeMatern(); by = :regime),
                         dynamics = LatentAR(ρ = 0))))
```

## 4. Flagship Regime-HMM product model

### 4.1 Joint law

Write t_n = t(date_n), and let pa(b) be the DAG parents of block b.

$$p(z,y)=a(z_1)\prod_{n\ge2}A_{z_{n-1}z_n}(t_{n-1})\;\prod_n\prod_b p_b\big(y^b_n\mid z_n[,z_{n-1}],t_n,y^b_{n-m_b:n-1},\varepsilon^{\mathrm{pa}(b)}_n\big)$$

with A_{k·}(t) = softmax(B(t)β_k) (`SoftmaxLink`, reference category K, A stored at the period being left). The factors, per block:

- **Rain** (hurdle, two independent latent fields given z):
  - Occurrence: O_{s,n} = 1{W^O_{s,n} ≤ Φ⁻¹(p_{s,k,t,h})}, with W^O ~ N(0, C^O_k). The default `Independent()` gives Π_s Bern(p).
  - Amounts on the wet set 𝒲_n: u_s = F^R_{s,k,t}(r_s), and p = c_{C^R_k[𝒲,𝒲]}(u_𝒲) Π_{s∈𝒲} f^R_{s,k,t}(r_s). The Gaussian copula restricted to 𝒲 is the exact marginal.
- **AR1 (form A, v3):**
  - ε_{s,n} = (x_{s,n} − μ − ρ x_{s,n−1})/σ, with all three seasonal and indexed by (s, k).
  - p = φ_{C_k(t)}(ε_n) Π_s σ⁻¹. Its scope is z_n only.
- **Marginal + LatentAR (form B):**
  - w_{s,n} = Φ⁻¹(F_{s,z_n,t_n}(y_{s,n})) and w_n = ρ w_{n−1} + √(1−ρ²) ε_n, with ε_n ~ N(0, C_k(t)).
  - Because ρ² + (1−ρ²) = 1, **Y_{s,n} | z_{1:N} ~ F_{s,z_n,t_n} exactly**, even across regime switches. This marginal invariance is what makes positive QQ/ETPP coherent.
  - The density needs w_{n−1}, which depends on z_{n−1}, so the block is `EdgeScope`.
- **Difference(:TX, G)** (TN):
  - δ = x^TX − y^TN, v_s = Φ⁻¹(G_{s,k,t}(δ_s)), with v_s = r ε^TX_s + √(1−r²) η_s and η ~ N(0, I) (option: `Dependence`).
  - p = Π_s g(δ_s) · φ((v_s − r ε_s)/√(1−r²)) / (√(1−r²) φ(v_s)).
  - TN < TX holds by construction. The block inherits the parent's scope.

### 4.2 Spec objects (SWGRegimeHMM / SWGBlocks)

```julia
struct RegimeHMM{R,B<:NamedTuple,E} <: AbstractWeatherGenerator
    regimes::R; blocks::B; estimation::E
end
RegimeHMM(; K::Int = 4, regimes = Regimes(K), estimation = TwoStage(), blocks...) =
    RegimeHMM(regimes, NamedTuple(blocks), estimation)          # empty blocks ⇒ defaults by name

Regimes(K::Int; transitions = 1, on::Symbol = :RR, emission = Occurrence(memory = 1),
        sites = :all, fit = EM(maxiter = 10_000, tol = 1e-3, init = SliceInit()),
        order = :wettest_first)
Regimes(fg::AbstractFittedGenerator; frozen::Bool = true)       # chain + decoded path
Regimes(path, transitions::AbstractArray{<:Real,3})            # table (DATE, z) + K×K×T

Rain(; occurrence = Occurrence(), amount = Amount(), threshold = 0.0)
Occurrence(; memory = 1, p = 1, dependence = Independent())
Amount(components::Pair...; weights = 1, dependence = Dependence(Free(); by = :regime,
       estimator = RankPairwise(:spearman)))              # default = 2-exponential mixture
AR1(; μ = 1, ρ = 1, σ = 1, dependence = Dependence(Free(); by = :regime, estimator = NormalScores()))
Marginal(family; dynamics = LatentAR(ρ = 1), dependence = Dependence(Free(); by = :regime), params...)
Difference(parent::Symbol, family = Gamma; r = 1, params...)
Dependence(structure = Free(); by = :regime, estimator = default_estimator(structure))
#  structure: Independent() | Free() | ExpKernel(; range = 1)
#  by:        :regime | (:regime, :month) | (:regime, :season) | Seasonal(d) (CorrelationLink)
#  estimator: NormalScores() | RankPairwise(:spearman) | PairwiseLikelihood() | WilksMC()
Seasonal(d::Integer); Seasonal(b::AbstractBasis); Constant() = Seasonal(0); Monthly()
TwoStage(; decode::DecodeMethod = Viterbi()); JointEM(; maxiter = 200, tol = 1e-4)
```

### 4.3 Composition rules

- `lower(::RegimeHMM, data)` resolves the defaults, fills T from the calendar, checks that the DAG is acyclic and that inputs exist, and checks scope against the estimation mode. It returns `ProductModel(regimes, blocks_in_topological_order, estimation)`.
- Parents are read only through `inputs`. Coupling uses the parent's **standardised innovation** ε (form A) or latent innovation (form B), never raw values, so `Difference` works with either TX form.
- Rain's occurrence is internal to the `Rain` block: O = RR > threshold.
- If `Regimes(on = :RR)` is not frozen, Rain's `occurrence` is the regime HMM's emission and is fitted in stage 1, not refitted.

### 4.4 Estimation, and what each mode maximises

Write y^R for the regime-defining data (`on`, `sites`) and y^b for every other block.

| Mode | Stage 1 | Stage 2 (block b) | What is maximised |
|---|---|---|---|
| `TwoStage(Viterbi())` (default, = v3) | θ̂_Z, θ̂_R = argmax log Σ_z p(z;θ_Z) p(y^R\|z;θ_R) by EM (local); ẑ = argmax_z p(z\|y^R) | argmax Σ_n log p_b(y^b_n \| ẑ_n[, ẑ_{n−1}], past) | A classification (complete-data) likelihood given a plug-in path. Biased toward over-separated regimes; consistent only as the decoding error → 0. Works for every scope. |
| `TwoStage(Posterior())` | same; γ_n(k) and ξ_n(k,l) from HMMs.jl `forward_backward!` storage (`transition_marginals = true`, verified in 0.7.1) | argmax Σ_n Σ_k γ_nk log p_b(y_n\|k) (edge scope: Σ ξ_n(k,l) log p_b(y_n\|k,l)) | The Jensen lower bound E_{z\|y^R} log p(y^b\|z) of the **cut-model** likelihood log E_{z\|y^R} p(y^b\|z). Block b never informs the regimes. |
| `JointEM()` | none | EM on log Σ_z p(z) Π_b p_b(y^b\|z): the E-step uses LL = Σ_b LL_b, the M-step is each block's weighted `fit!` | The full MLE (local) **only if** every M-step maximises its Q_b. Copula blocks fitted by IFM make it generalised-EM-like, without guaranteed monotonicity, so SPSM's own EM loop keeps the best iterate. Requires `StateScope`, or K² edge emissions (post-launch). |

- **Inside a block with dependence**, estimation is IFM:
  1. Margins maximise Σ_n Σ_k w_nk Σ_s log f_s, a composite likelihood that ignores cross-site dependence.
  2. The dependence term is then estimated with the margins plugged in, by one of:
     - `NormalScores`: weighted MLE of a Gaussian on normal scores, i.e. Copulas' `fit(GaussianCopula, U; weights)`;
     - `PairwiseLikelihood`: bivariate normal CDFs, for binary or censored data;
     - `RankPairwise`: v3 Spearman→Pearson with `nearest_cor`.
- **Form B** follows the same pattern:
  - its margins ignore the temporal dependence;
  - ρ_s(t) is the weighted Gaussian regression of w_n on w_{n−1};
  - C comes from ε̂ = (w_n − ρw_{n−1})/√(1−ρ²).
  - If ρ varies across sites, the stationary same-day correlation is C_ss'√((1−ρ_s²)(1−ρ_s'²))/(1−ρ_sρ_s'), not C. The default therefore shares ρ across sites.
- **TN|TX:** r_{s,k}(t) is a weighted MLE of a standard bivariate normal on (ε^TX, v), with an `IntervalLink(-1,1)` seasonal parameter in SPSM. This deliberately replaces v3's least-squares fit of raw cor(TX, TX−TN) (fit_mle_TN.jl:75-88), which estimated a different quantity from the one simulated.
- **`loglikelihood(fg)`** evaluates the *joint* likelihood at the estimates by a forward pass over Σ_b LL_b. It is exact for continuous and independent blocks and a QMC estimate for latent-threshold blocks with dependence (`likelihood_kind`). Under IFM, BIC from it is a comparison device, not a maximised-likelihood BIC.

### 4.5 Simulation

1. Compute t from the dates and resolve `init` (§2.7).
2. Draw the z path from (a, A(t)), or take `given.z`. Regimes are exogenous to the variables (an autonomous chain), so z is always drawn first.
3. Run each block in topological order with `simulate!(out_b, fb, SimPath(z, t, parents, latents, init_b), rng_b)`:
   - Rain: W^O ~ N(0, C^O_k), thresholded with the memory h updated day by day; then amounts on 𝒲_n via ε ~ N(0, C^R_k[𝒲,𝒲]) and r_s = F⁻¹(Φ(ε_s)). One Cholesky factor per (k, 𝒲) is cached by bitmask when D ≤ 16.
   - AR1: ε ~ N(0, C_k(t)), x = μ + ρx₋₁ + σε, with ε kept for children.
   - Difference: v = rε^TX + √(1−r²)η, δ = G⁻¹(Φ(v)), TN = TX − δ.
   - Marginal: w = ρw₋₁ + √(1−ρ²)ε, y = F⁻¹(Φ(w)).
4. Repeat for nsim realizations, threaded, with per-realization seeds, into preallocated D×N×nsim arrays.

## 5. WGEN port and censored latent Gaussian

**WGEN (SWGWGEN).**

```julia
Base.@kwdef struct WGEN{A} <: AbstractWeatherGenerator
    order::Int = 1
    p = Monthly()                                   # v3: monthly transition probabilities
    dependence = Dependence(Free(); by = :month, estimator = PairwiseLikelihood())
    amount::A = nothing                             # optional Amount → GWEX-like
end
SWGBlocks.lower(a::WGEN, data) = ProductModel(Regimes(1),
    (; RR = Rain(occurrence = Occurrence(memory = a.order, p = a.p, dependence = a.dependence),
                 amount = a.amount)), TwoStage())
```

- **Model:** O_{s,n} = 1{W_{s,n} ≤ Φ⁻¹(p_s(t_n, h_{s,n}))}, with W_n iid N(0, Ω_{m(n)}). This is exactly the flagship `Occurrence` block with K = 1, a `Monthly` basis and `Free` monthly correlation.
- **Fit (IFM):**
  - Margins: weighted Bernoulli MLE per (s, month, h). With an indicator basis and a logit link this is the count ratio of v3 `fit_markov_chain`, in closed form with no solver.
  - Empty cells give p = 0, as in v3, with a warning.
  - Ω_m, default: pairwise likelihood Σ_n log P(o_s, o_s' | h_s, h_s'; ω) = log Φ₂(c_s, c_s'; ω)-type terms. This is the *same* kernel as the spatial HMM's `pairwise_loglikelihood_memory` (SPSM likelihoods.jl:178), with a free ω in place of exp(−d/R). It is followed by `nearest_cor`.
  - `WilksMC()` keeps v3's estimator (Monte-Carlo correlation matching by root-finding), which is why NonlinearSolve stays confined to this library.
- **Simulate:** the shared `Occurrence` simulator. v3 restarts each month with `Y_ini` carried over (wgen.jl:145-152), which is identical in distribution to continuous simulation with month-indexed parameters.
- **Likelihood:** an orthant probability (QMC), so `likelihood_kind = :qmc`.

**Censored latent Gaussian (SWGCensored).**

```julia
Base.@kwdef struct CensoredGaussian{R} <: AbstractWeatherGenerator
    regimes::R = Regimes(1)
    rain = CensoredRain(PowerCensoredNormal; μ = 2, σ = 1, β = 0,
                        dependence = Dependence(ExpKernel(; range = 1); by = :regime,
                                                estimator = PairwiseLikelihood()),
                        dynamics = LatentAR(ρ = 1))
end
SWGBlocks.lower(a::CensoredGaussian, data) = ProductModel(a.regimes, (; RR = a.rain), TwoStage())
```

- **Model:**
  - W is a latent standard field: w_n = ρw_{n−1} + √(1−ρ²)ε_n, with ε ~ N(0, C_k(t)) and C_ss' = exp(−d_ss'/R_k(t)).
  - R_{s,n} = G⁻¹(Φ(W_{s,n})), where G is the law of max(0, μ+σX)^β (`PowerCensoredNormal`, with an atom Φ(−μ/σ) at 0).
  - Equivalently, R = max(0, μ+σW)^β, so the site is **dry iff W ≤ −μ/σ**. That is the same thresholding as WGEN, with amounts produced by the same W.
  - Ailliot 2009 is `Regimes(K; on = :RR, emission = :self)` with ρ = 0. Its emission is the censored vector, so it needs N×K log-densities from QMC over the dry sites conditional on the wet ones (MvNormalCDF, as in the spatial HMM). This is feasible for D ≲ 20 and is flagged experimental.
- **Fit (IFM):**
  - Margins: censored MLE per (s, k), Σ_n [1{r=0} log G(0) + 1{r>0} log g(r)], through SPSM `fit!` with a seasonal family. Distributions' `censored` handles the atom.
  - Dependence: pairwise composite likelihood over same-day pairs within `maxdist` (for R) and lag-1 pairs (for ρ):
    - both wet: bivariate normal density with Jacobians;
    - one wet: φ(w) Φ((c − r w)/√(1−r²));
    - both dry: Φ₂(c, c'; r), using SPSM's `norm_cdf_2d_vfast` kernel.
  - Temporal pairs are edge-scoped and use ẑ.
  - *The estimator follows Bessac/Ailliot practice; the exact choices should be checked against Vaittinada Ayar 2020.*

**What the three share.**

| Primitive | Flagship | WGEN | Censored |
|---|---|---|---|
| Seasonal parameters (SPSM ParamSpec) | trig | `Monthly` indicator | trig |
| Y = G⁻¹(Φ(W)) | threshold (RO), PIT (RR, QQ) | threshold | censor |
| Correlation structure | Independent / Free | Free per month | ExpKernel per regime |
| Latent AR | `Marginal` | none (memory is in the margin) | `LatentAR` |
| Pairwise bivariate-normal estimator | spatial RO option | Ω | R, ρ |
| Regimes | K = 4 | K = 1 | K = 1 or K |
| Driver, calendar, Tables I/O, `init`, `validate` | shared | shared | shared |

## 6. SPSM v5: the engine contract

```julia
SeasonalData(y::AbstractVecOrMat; t::AbstractVector{<:Integer}, T::Integer,
             seq_ends = [length(t)], past = nothing, lags::Int = 0, x = nothing, split_missing = true)
SeasonalData(ys::AbstractVector{<:AbstractVecOrMat}; t::AbstractVector{<:AbstractVector}, T, pasts = nothing)

abstract type AbstractStateWeights end
struct Unweighted <: AbstractStateWeights end
struct ObsWeights{V} <: AbstractStateWeights; w::V; end                   # N
struct StatePath <: AbstractStateWeights; z::Vector{Int}; K::Int; end      # given z = one-hot
struct StateWeights{M} <: AbstractStateWeights; γ::M; end                 # N×K
struct PairWeights{A} <: AbstractStateWeights; ξ::A; end                  # N×K×K (edge scope)

# every seasonal model m (coefficients: ComponentVector, parameter-major, state axis last)
nstates(m)::Int; period(m)::Int; coef(m)::ComponentVector
(m)(t::Integer; state::Integer = 1)       # Distribution, or x -> Distribution if Conditional
curve(m, name::Symbol, ts = 1:period(m); state = :)
loglikelihoods!(LL::AbstractMatrix, m, d::SeasonalData)   # LL[n,k] = log p(y_n | Z_n=k, t_n, x_n)
loglikelihoods(m, d); loglikelihood(m, d, w = Unweighted())
fit!(m, d::SeasonalData, w::AbstractStateWeights = Unweighted(); solver, adtype, solve_kwargs) -> m
#   maximises Q(θ) = (Σ_nk w_nk)⁻¹ Σ_n Σ_k w_nk log p(y_n | k, t_n, x_n; θ)  (average, not sum)
fit(m, d, w = Unweighted(); kw...)::SeasonalFit          # copy + history + loglik + nobs + dof
initialize(m, d, w = Unweighted(); rng) -> m             # data-driven start (slices for HMMs)
rand([rng,] m, t::AbstractVector{<:Integer}; state = 1, past = nothing)

SeasonalTransitions(K::Int, basis::AbstractBasis)        # A_k·(t) = softmax(B(t)β_k), ref K
SeasonalHMM(trans::SeasonalTransitions, emission; init = fill(1 / K, K))
fit!(hmm::SeasonalHMM, d; solver, maxiter = 1000, tol = 1e-4) -> hmm   # own EM loop, best iterate
posteriors(hmm, d; pairs = false) -> (γ = N×K, ξ = N×K×K | nothing, logL)
viterbi(hmm, d)::Vector{Int}; rand([rng,] hmm, t; z_past = nothing, past = nothing) -> (z, y)
```

- **Emission protocol.** Any object with `nstates == K`, `loglikelihoods!`, `fit!(·, d, ::StateWeights)` and `rand(·; state = z)` can be an emission. SWG's spatial `Occurrence` and `CensoredRain` therefore run through SPSM's EM unchanged. That reuse is the test of the protocol.
- **HMMs.jl bridge.** `obs_distributions(hmm, control)` returns K `CachedLogDensity(LL, n, k)` objects whose `logdensityof` reads a precomputed LL. Only the documented API is used, and the `obs_logdensities!` override goes away.
- **Building blocks:**
  - `StateIndexed(fam, K; shared = ())` gives any `TimeVaryingFamily` a state axis. Shared parameters are solved jointly across k.
  - `Conditional(f)`: lags or parent values become covariates x_n. This removes the hand-written AR NLL (pain point #1) without changing TVP's core.
  - `SeasonalAR1` is `StateIndexed(Conditional(Normal))` plus the existing sufficient-statistics M-step.
  - `BernoulliMemory(D; memory, p)` is the merged emission of the ARPeriodicHMM / sARPeriodicHMM family, with per-site order and a closed-form indicator-basis M-step.
  - `Product(blocks...)` replaces BlockProduct.

**Vector links: yes.**
- `SoftmaxLink(n)`: n−1 predictors → simplex. It is used by transitions and mixture weights and removes the four softmax copies.
- `CorrelationLink(d)`: d(d−1)/2 predictors → tanh canonical partial correlations → correlation matrix. It is used for smooth C_k(t) in SWG and flagged experimental for D > 10.
- ParamSpec stores nbasis × insize(link) coefficients per parameter. These land in the vendored TVP and go upstream later.

**v4 exports kept (exported):** ParamSpec, TrigBasis, ConstantBasis, FourierBasis, PolynomialBasis, IndicatorBasis, BSplineBasis, CyclicBSplineBasis, IdentityLink, LogLink, LogitLink, IntervalLink, curve, TimeVaryingFamily, SeasonalFit, fit, coef, loglikelihood, loglikelihoods, viterbi, converged, dof, nobs, aic, aicc, bic, components, ncomponents.

**New exports:** SoftmaxLink, CorrelationLink, Conditional, StateIndexed, BernoulliMemory, Product, SeasonalTransitions, SeasonalHMM, SeasonalData, Unweighted, ObsWeights, StatePath, StateWeights, PairWeights, fit!, initialize, posteriors, history, nstates. That is about 45 exports in total.

**`public` only:** paramnames, ncoef, coefs (renamed `coefblock`), blockindex, blockrange, design_matrix, nbasis, PeriodicDesign, PeriodicNLL, PointwiseNLL, group_by_time, update_dist (renamed `mstep!`), ClassicEM, StochasticEM, SeasonalControls, ar1_spec, ar1_sufficient_statistics, ar1_nll, component_coefs, component_range, emission_coefs, emission_range, degrees, solve_unconstrained, solve_nlls, default_adtype, default_ensemble, default_nlls_solver, plus `evaluate(m)`, which returns the evaluated twins (AR1, ARPeriodicHMM, PeriodicHMMDists).

**Renamed:**

| v4 | v5 |
|---|---|
| SmoothAR1 | SeasonalAR1 |
| SmoothSeasonalMixture | SeasonalMixture |
| SmoothPeriodicHMM, SmoothARPeriodicHMM, SmoothsARPeriodicHMM | SeasonalHMM with `Product` / `BernoulliMemory` emissions |
| seasonal_family(D, b) | TimeVaryingFamily(D, b) |
| mixture_weights | curve(m, :weights) |
| mixtures, AR1(m), ARPeriodicHMM(m) | evaluate(m) |
| fit_mle_all_slices | initialize |
| parameters, params(m, t) | coef, m(t) |
| FitHistory | history(f) |

**Removed:**
- `fit_mle` tuple form: `fit_mle(m, data, w)` now returns only the model, so it nests in ExpectationMaximization.jl.
- TimeVaryingFit (→ SeasonalFit).
- mixture_weight_spec (→ SoftmaxLink).
- Trig2ARPeriodicHMM, Trig2ARPeriodicHMMSpatial, randARPeriodicHMM.
- fit_θ, fit_θ!, fit_θᴬ!, fit_θᴮ!.
- fit_mle_RO (→ `fit!(BernoulliMemory…, StatePath(z, K))`).
- μₜ, ρₜ, σₜ, αₜ.
- fit_loss_optim.
- n_to_t (t is explicit; the PeriodicHiddenMarkovModels dependency goes).

**SPSM → SWGBlocks:**
- SpatialBernoulli, ARPeriodicHMMSpatial / SmoothARPeriodicHMMSpatial, the pairwise and QMC likelihoods. They become `Occurrence(dependence = Dependence(ExpKernel()))`, or Caroline's planned SpatialBernoulli package.
- The multisite `rand(ARs, n2t, z, Σ)`.
- SPSM drops Copulas, MvNormalCDF and SpecialFunctions.

**SWG → SPSM** (as generic idioms, not new names):
- `_fit_seasonal_gamma` becomes `fit!(StateIndexed(TimeVaryingFamily(Gamma, b), K), d, StatePath(z, K))`.
- `_ar1_init` and `mix_ini` become `initialize`.
- `fit_markov_chain` counts become the `BernoulliMemory` closed form.
- The seasonal correlation fit in `cor_groupby` becomes a `TimeVaryingFamily` over a standard bivariate normal with `IntervalLink(-1,1)`.

## 7. Naming and verbs

| Concept | New name | Replaces (v3 SWG / v4 SPSM) |
|---|---|---|
| Model class (algorithm) | `RegimeHMM`, `WGEN`, `CensoredGaussian` (`<: AbstractWeatherGenerator`) | hand-wired fit_mle_RO/RR/AR1/TN pipelines; `wgen` |
| Data / problem | `WeatherData` | vector of station DataFrames + n2t |
| Fit | `fit(data, alg)` | fit_mle_RO, fit_mle_RR, fit_AR1, fit_TN, cov_RR, cov_ar1, cor_groupbyTXTN, fit_wgen, fit_Ω |
| Fitted result | `FittedGenerator` | arrays and tuples, discarded histories |
| Simulate | `simulate(fg, dates; nsim)`; `rand(fg, dates)` | rand(hmm…), rand_RR, rand(ar1s…), rand_cond, simulate_markov_gaussian, rand(wgen…) |
| Initial state | `init =` (days before start) | y₁, Y_past, z_ini, Y_ini |
| Conditioning | `given = (z = …,)` | positional z in rand |
| Decoding | `decode(fg, data)`; `posterior` | viterbi(hmm, Y, Y_past; n2t) |
| Regimes | `Regimes(K; …)`, `Regimes(fg; frozen)` | randARPeriodicHMM + slices + SmoothARPeriodicHMM + fit_mle |
| Seasonal parameter | `Seasonal(d)`, `Constant()`, `Monthly()`, integer shorthand | 𝐃𝐞𝐠, degree, TrigBasis(T, d) |
| Occurrence | `Occurrence(memory = m)` | local_order, memory_order, ARPeriodicHMM emission |
| Rain amounts | `Amount(…)`, `Rain(…)` | mix_ini, fit_mle_RR, rand_RR |
| Observed-scale AR | `AR1(…)` | fit_AR1, SmoothAR1, AR1 |
| Positive / any-margin dynamics | `Marginal(F; dynamics = LatentAR())` | `indicatrix` clipping |
| TN given TX | `Difference(:TX, Gamma)` | fit_TN, cor_groupbyTXTN, rand_cond |
| Multisite dependence | `Dependence(Free(); by = …)` | cov_RR, cor_RR, cov_ar1, Σ arrays, fit_Ω |
| Spatial kernel | `ExpKernel(; range)` | SpatialBernoulli range R |
| Estimation mode | `TwoStage(decode = …)`, `JointEM()` | implicit workflow |
| Calendar | `DayOfYear366()` | dayofyear_Leap, n_to_t |
| Diagnostics | `validate` | hand-written plotting code |
| Save / load | `write_generator`, `read_generator` | ad-hoc JLD |
| SPSM data | `SeasonalData(y; t, T, past)` | positional (y, n2t, Y_past) |
| Weights | `StatePath`, `StateWeights`, `ObsWeights` | positional z / w / Y_past (pain point #6) |
| SPSM fit | `fit!` (model only), `fit` → `SeasonalFit` | fit_mle → (m, hist), TimeVaryingFit |
| Evaluated form | `evaluate(m)`, `m(t; state)` | AR1(m), ARPeriodicHMM(m), mixtures |
| Start values | `initialize` | fit_mle_all_slices, _ar1_init, mix_ini |
| Per-state log-densities | `loglikelihoods` (N×K) | unchanged name, single arity |

## 8. Migration and release plan

**Ordered steps.** Each step is mergeable on its own into `main` (v5 / v4 development); `release-4.x` and `release-3.x` branches take patches.

1. **Golden fixtures** (both repos, run on v4/v3): small deterministic scenarios (3 sites, 6 years, K = 2) covering coefficients, log-likelihoods, Viterbi paths and noise-injected simulations. Save them as CSV/JLD2 under `test/fixtures`.
2. **SPSM:** vector links in the vendored TVP, plus `Conditional`.
3. **SPSM:** `SeasonalData`, the weights types, `fit!` / `loglikelihoods!` for `StateIndexed`, `SeasonalAR1` (sequences, gaps) and `SeasonalMixture`; equivalence tests (L1).
4. **SPSM:** `SeasonalHMM`, `SeasonalTransitions`, `BernoulliMemory`, `Product`; own EM loop; `CachedLogDensity`; equivalence against SmoothARPeriodicHMM, sARPeriodicHMM and SmoothPeriodicHMM.
5. **SWG:** monorepo skeleton (`lib/SWGCore`, workspace, `[sources]`, CI matrix), with the Makie extensions moved. The umbrella still runs v3 code.
6. **SWG:** `lib/SWGData` (moved from src/data), with the precompile-time download removed.
7. **SWG:** `lib/SWGBlocks`: Dependence and its estimators, the blocks, the spatial `Occurrence` port, ProductModel, TwoStage/JointEM; L1 and L2 tests against v3 functions.
8. **SWG:** `lib/SWGRegimeHMM`. Port tuto_paper, tuto_paper_HMMspatial and tuto_add_station_variable, and re-save `assets/tuto_1` regimes as `.swg` (v3 JLD files hold v4 types).
9. **SWG:** `lib/SWGWGEN` and `lib/SWGCensored`.
10. **SPSM removal commit:** v4 names out, `public` declarations, docs and migration table.
11. **Same-day release:** SPSM 5.0.0 → SWGCore → SWGBlocks → model libraries and SWGData → umbrella 4.0.0. Then docs and the migration guide.

**Old → new, most-used calls.**

| v3 / v4 | v4 / v5 |
|---|---|
| `collect_data_INRAE.(paths)` | unchanged (SWGData); then `WeatherData(reduce(vcat, dfs))` |
| `dayofyear_Leap.(dates)` | implicit; `periods(data)` |
| `fit_mle_RO(df, order, deg; solver)` | `Rain(occurrence = Occurrence(memory = order, p = deg))` under frozen `Regimes` |
| `ARPeriodicHMM(a, hmm.A, B)` | `RegimeHMM(regimes = Regimes(fg0))` |
| `fit_mle_RR(df, order; mix₀ = mix_ini(T))` | `Amount()` (default) |
| `cov_RR(dfs, K)` | `Dependence(Free(); by = :regime, estimator = RankPairwise(:spearman))` |
| `fit_AR1(df, :TX, deg; solver)` | `TX = AR1(μ = deg, ρ = deg, σ = deg)` |
| `cov_ar1(dfs, ar1s, :TX)` | `AR1(; dependence = Dependence(Free(); by = :regime))` |
| `fit_TN(df, 1, T)` + `cor_groupbyTXTN(df, T)` | `TN = Difference(:TX, Gamma; r = 1)` |
| `rand(hmm, n2t; Y_past, z_ini, seq = true)` | `simulate(fg, dates; init = (z = z₀, RR = …))` |
| `rand_RR(mixs, n2t, z, y, Σ)` | inside `simulate`; `given = (z = z,)` |
| `rand(ar1s, n2t, z, Σ; y₁, output_ϵ = true)` | `simulate(…; latent = true)` |
| `rand_cond(ϵ, z, θ_ΔT, θ_cor, n2t, T)` | automatic (DAG) |
| `indicatrix.(rand(ar1sQQ, …))` | `QQ = Marginal(Gamma)` |
| `fit_wgen(Y, idx_months, p)` | `fit(data, WGEN(order = p))` |
| `rand(wgen, years; Y_ini)` | `simulate(fg, years; init = (RR = Y_ini,))` |
| `fit_markov_chain`, `fit_Ω` | internal to WGEN; `WGEN(dependence = Dependence(Free(); by = :month, estimator = WilksMC()))` |
| `viterbi(hmm, Y, Y_past; n2t)` | `decode(fg, data)` |
| `fit_mle(m0, y; n2t, solver)` | `fit(m0, SeasonalData(y; t = n2t, T); solver)` |
| `fit_mle(SmoothAR1(d, T; K, init), y, z; n2t)` | `fit!(SeasonalAR1(d, T; K, init), SeasonalData(y; t, T), StatePath(z, K); solver)` |
| `fit_mle(mix0, y, w)` | `fit!(mix0, data, ObsWeights(w))` |
| `randARPeriodicHMM` + `fit_mle_all_slices` + `fit_θ` + `SmoothARPeriodicHMM(…)` | `initialize(SeasonalHMM(SeasonalTransitions(K, b), BernoulliMemory(D; memory, p = b)), data)` |
| `fit_mle(hmm, Y, Y_past; n2t, maxiter, tol)` | `fit(hmm, SeasonalData(Y; t = n2t, T, past = Y_past); solver, maxiter, tol)` |
| `ARPeriodicHMM(m)`, `AR1(m)`, `mixtures(m)` | `evaluate(m)` / `m(t; state = k)` |
| `μₜ(t/T, θ)` | `curve(m, :μ, t)` |
| `n_to_t(N, T)` | explicit `t` (an error if omitted) |
| `loglikelihoods(hmm, Y, Y_past; n2t)` | `loglikelihoods(emission, SeasonalData(…))` |

**Numerical-equivalence strategy.**
- **L1, parameters (deterministic):** same data, same starting coefficients, same solver. Coefficients agree to rtol 1e-6 and log-likelihoods to 1e-8. Coverage:
  - every SPSM model;
  - the given-ẑ fits fit_mle_RO, fit_mle_RR, fit_AR1, the fit_TN margins, cov_ar1, cov_RR (RankPairwise) and the fit_markov_chain counts.
- **L2, pathwise (deterministic given noise):** an unexported `noise =` hook in `SimPath` accepts pre-drawn ε and U. Outputs must equal v4 `rand(ARs, ϵ, n2t, z, Σ; y₁)`, `rand(AR, ϵ, n2t, z)` and `rand_cond(ϵ, …)` at equal r, and the RO thresholding.
- **L3, distributional:** tutorial-scale ensembles. Compare about 200 summary statistics (monthly quantiles, spell pmfs, inter-site and inter-variable correlations) against stored v3 values, with Bonferroni-style MC tolerance.
- **Documented intentional deviations:** none of these is expected to match v3/v4.
  - TN's r is now estimated on the innovation scale.
  - QQ/ETPP use positive margins instead of clipping.
  - The regime-K independence hack in `rand_RR` is removed (generation.jl:31).
  - The default `init` changes.
  - WGEN's default Ω estimator changes.
  - RNG streams change.
  - SWG's `groupby(:z)`-ordered indexing (fit_mle_TN.jl:28) is replaced by explicit k.

## 9. Risks, trade-offs, open questions

1. **Package names and maintenance.** `SWG*` acronyms may draw comments in General's name review; the fallback is `StochasticWeatherGenerators*` (long). Six libraries plus SPSM is a lot of compat and CI surface for one maintainer. Keep SWGCore tiny and let only SWGBlocks import SPSM.
2. **Edge scope.** Form-B blocks, and children coupled to them, depend on (z_{n−1}, z_n). Plug-in works for them, soft mode needs ξ (available), and joint EM needs K²-state forward–backward (post-launch). If this is too subtle for students, the default for TX stays form A.
3. **Non-exact M-steps** (IFM copulas, QMC orthant log-likelihoods) can make EM non-monotone. HMMs.jl's `baum_welch` throws on a decrease, hence SPSM's own loop with best-iterate tracking and reported `likelihood_kind`.
4. **Copula costs.** Copulas.jl puts the dimension in the type and needs d ≥ 2, so wet-set subsets use SWGBlocks' own Cholesky cache. With D = 37 and many distinct subsets the cost is unmeasured, and caching by bitmask is bounded by D.
5. **`CorrelationLink`** seasonal C_k(t) has O(D²·nbasis) coefficients per regime, so it is limited to small D. `by = (:regime, :season)` is the scalable alternative.
6. **Serialization.** JLD2 of Julia structs is fragile across versions. The proposal is to store a schema version, the alg spec and coefficient arrays, and re-lower on load, guaranteed within a major version.
7. **Default solver via an extension `__init__` that sets a `Ref`.** This avoids method overwriting during precompilation, but behaviour depends on what is loaded. Closed-form M-steps (Bernoulli indicator, AR1 sufficient statistics) need no solver.
8. **Breaking every v4 caller at once** (no shims), two days after v4.0.0. The unknown is the number of non-SWG users, the genomics tutorial at least.
9. **Open questions:**
   - Should the default QQ margin be Gamma, or a Beta bounded by clear-sky radiation?
   - Should T0 regimes be learnt from only 4 stations, or should a reference network be required (warn)?
   - Should `Shared` parameters across regimes ship at launch (SPSM `StateIndexed(…; shared)`) or later?
   - Does the spatial latent Bernoulli move to SWGBlocks or to Caroline's SpatialBernoulli package?
   - Should parent-state conditioning (`AR1(given = :RR)`, Richardson-type) be a launch feature or v4.1?
   - Is the pairwise estimator for the censored model faithful to Vaittinada Ayar 2020 (to verify)?

### Critical Files for Implementation
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\TimeVaryingParameters\spec.jl (with family.jl and fit.jl in the same folder): vector links, `Conditional`, weighted average NLL
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\hmm_interface.jl (with src\HMM_Dists\periodichmm.jl and src\fit\smooth_hmm.jl): the generic SeasonalHMM, the emission protocol, CachedLogDensity
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\HHMM_Bernoulli_spatial\likelihoods.jl (with SpatialBernoulli.jl and src\fit\spatial.jl): the shared thresholded-latent-Gaussian and pairwise kernels moving to SWGBlocks
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\examples\tuto_add_station_variable.jl (with ext\SmoothPeriodicStatsModelsExt\fit_mle_TN.jl and src\rain\generation.jl): the flagship workflow to reproduce and the golden fixtures
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\src\rain\wgen.jl: the WGEN port and its equivalence target