## 0. Thesis

**Base and grafts.** The base is **Design B**, the novice-first progressive-disclosure design. Across the three judges it has the best average rank (B 1/3/2, the other three designs 2/2/4, 4/1/3 and 3/4/1). The other designs contribute:
- from C, the statistical spine: block scope traits, the table of what each estimation mode maximises, a typed dependence object, multi-output blocks, and the noise-injection equivalence test;
- from D, the delivery plan: an additive SPSM 4.2 bridge, pre-registration, test CI first, a small package count, a closure-free spec, and the WGEN legacy counting path;
- from A, the reproducibility devices: `seed`, `pretrained(...)`, plain-array persistence, `Tables.partitions`, the years shortcut, and lazy `OneHot` weights.

Every fatal flaw the judges raised is fixed. Each fix is listed in §10.

**The idea.** There is one object graph, entered through three doors:
- An impact modeller writes `fit(obs, RegimeHMM(...))`.
- `show(fitted)` prints the resolved spec back as valid Julia. That printout is the Tier-1 spec.
- Each field of the spec is a block value (`Rain()`, `Temperature()`, `Positive()`, `Difference(:TX, Gamma)`), and each block implements the Tier-2 block protocol.
- Each fitted block holds ordinary SPSM models, which is the Tier-2 door into the engine.

SWG follows SciML on both axes:
- A monorepo with a thin Core, a shared Blocks library, one sublibrary per model class, and an umbrella package.
- Model classes are swappable like solvers: `fit(data, RegimeHMM())`, `fit(data, WGEN())` and `fit(data, CensoredLatentGaussian())` all return the same `FittedGenerator` and support the same verbs.

**One primitive behind all three launch classes.** All three classes, and the later ones, are built from **Y = G⁻¹(Φ(W))**, where W is a latent standard-Gaussian field with a structured correlation and an optional latent AR(1). The same map covers three cases:
- thresholding, when G is Bernoulli: WGEN, and the occurrence latent copula;
- a PIT transform, when G is continuous: rain amounts, `Positive`;
- censoring, when G has an atom at 0: the censored latent Gaussian (CLG), and `Positive` with exact zeros.

This is taught in the docs and is reflected in the code. The censored-latent machinery is shared by `Positive()` and by CLG.

**SPSM v5.** SPSM becomes the domain-agnostic seasonal engine. The release is mostly subtraction, plus a small, stable contract:
- `fit_mle(m, y[, w]; n2t, past, seq_ends, solver)` returns the model only, so it nests in EM and HMMs;
- `fit(SeasonalFit, m, y[, w]; …)` returns the model together with its history and StatsAPI methods;
- the third positional argument is always weights (`OneHot(z, K)` means "given z");
- `n2t` is a required keyword, and `loglikelihoods` returns an N×K matrix for every model.

## 1. Package architecture

```
StochasticWeatherGenerators.jl/                    one repo; root = umbrella (SAME UUID 3eadb0b9…, v4.0.0)
├── Project.toml      [workspace] projects = ["lib/StochasticWeatherGeneratorsCore", …explicit list…, "docs", "test"]
│                     [sources] each lib = {path = "lib/…"}
├── src/StochasticWeatherGenerators.jl   explicit `using X: a, b` + export list (no blanket @reexport),
│                                        data readers, climate indices, pretrained(), default solver
├── ext/  MakieExt (plot recipes incl. validate), GeoMakieExt (map_with_stations)
├── lib/
│   ├── StochasticWeatherGeneratorsCore/            v1.0.0  verbs, WeatherData, calendar, FittedGenerator,
│   │                                                       WeatherSimulation, simulate driver, validate, save/load
│   ├── StochasticWeatherGeneratorsBlocks/          v1.0.0  blocks, MultisiteGaussian, latent/censored kernels,
│   │                                                       product engine (DAG), Regimes, NoRegimes, TwoStage
│   ├── StochasticWeatherGeneratorsRegimeHMM/       v1.0.0  RegimeHMM (flagship class)
│   ├── StochasticWeatherGeneratorsWGEN/            v1.0.0  WGEN
│   └── StochasticWeatherGeneratorsCensoredGaussian/ v1.0.0 CensoredLatentGaussian
│       later, each additive at 0.x: …KNN, …GWEX, …MSTWeatherGen, …WeaGETS, …GWGEN, …AWEGEN2d
├── test/      umbrella integration + test/reference/ (golden fixtures, pinned env)
├── docs/      one Documenter site (Literate-rendered tutorials), DocumenterInterLinks → SPSM
└── .github/workflows/  CI.yml (GROUP matrix + path filters), Documentation.yml, TagBot.yml (subdirs), CompatHelper (subdirs), Downgrade
SmoothPeriodicStatsModels.jl/    own repo (v4.2 additive bridge → v5.0)
```

| Package | Purpose | Hard deps | Exports |
|---|---|---|---|
| **Core** | The interface: `AbstractGenerator`, traits, `WeatherData` (Tables in), `Leap366`/`NoLeap365`, `FittedGenerator`, `WeatherSimulation` (Tables out), generic `fit`/`simulate`/`validate`/`decode`, JSON save/load | Dates, Random, Statistics, LinearAlgebra, Tables, StatsAPI, JSON, PrecompileTools | ~22, plus ~15 `public` |
| **Blocks** | `Rain`, `Occurrence`, `Temperature`, `Continuous`, `Positive`, `Difference`, `AR1`, `LatentAR1`, `Mixture`, `MultisiteGaussian` and its estimators, `Regimes`/`NoRegimes`, `TwoStage`; the block protocol; the product engine; latent/censored kernels | Core, SPSM, Distributions, NearestCorrelationMatrix, LogExpFunctions | ~24, plus ~15 `public` |
| **RegimeHMM** | `RegimeHMM` and default blocks by variable kind | Core, Blocks, SPSM | 1 |
| **WGEN** | `WGEN` (Wilks 1998 / Srikanthan 2009) | Core, Blocks | 1 |
| **CensoredGaussian** | `CensoredLatentGaussian`, `PowerTransform` | Core, Blocks | 2 |
| **umbrella** | Re-exports; `read_stations(INRAE()/ECA()/MeteoFrance(), …)`, `pretrained`, `example_data`, climate indices, default solver `LBFGS()` | the 5 libs, CSV, DataFrames, Impute, Downloads, Optimization, OptimizationOptimJL | ~60, of which T0 uses 7 |

**Package count.** There are five libraries plus the umbrella, which follows the maintainer judge's cap. The data readers live in the umbrella rather than in a separate Data package: they are not a model class, and the umbrella already carries CSV and DataFrames. Copulas, NonlinearSolve and OrderedCollections leave SWG altogether:
- all launch dependence is Gaussian and is sampled through cached Cholesky factors;
- the root-find becomes a bisection with common random numbers.

**SPSM stays in its own repo.** Reasons:
- It is domain-agnostic (`examples/tuto_genomics.jl`, and the `HMM_Spatial_codepaper` scripts extend `SmoothPeriodicStatsModels.loglikelihoods`).
- It has a co-author and its own registry history.
- OrdinaryDiffEq pulled DiffEqBase in only because it was tightly coupled to solver internals. Here the tightly coupled glue is **Blocks**, and Blocks is inside the monorepo.
- The coupling between the two repos is the roughly 12-function contract in §6. SPSM's CI gets a `Downstream` job that runs the Blocks and RegimeHMM tests.

**Load time.** These are targets. CI measures them with `@time_imports`, Julia 1.12, warm cache, and fails above 1.5× the budget.

| Package | Budget |
|---|---|
| Core | ≤ 0.8 s |
| Blocks | ≤ 3.5 s (SPSM dominates) |
| WGEN / CLG | about the same as Blocks |
| umbrella | ≤ 8 s |

- The network download that runs at precompile today (`src/data/extracting.jl:176`, `df_api_url = CSV.read(Base.download(...))`) becomes a lazy call at run time.
- Every library has a PrecompileTools workload: a 2-site × 3-year synthetic fit and simulate.

**Registration and versioning.**
- Each `lib/X` has its own UUID, `Project.toml`, `test/` and semver. It is registered with `@JuliaRegistrator register subdir=lib/X`. TagBot and CompatHelper run with `subdirs`, producing tags like `StochasticWeatherGeneratorsCore-v1.0.0`.
- Launch libraries start at 1.0.0. Future classes start at 0.x and join the umbrella in a minor release.
- The umbrella keeps the v3 UUID.
- `[sources]` is used for development only; the registry ignores it.
- The workspace lists the libraries explicitly. Glob support in `projects` is unverified, and OrdinaryDiffEq itself uses only `[sources]`. A single manifest works only because SPSM `main` stays additive (4.2) until the cutover (§8).

## 2. Core interface

```julia
module StochasticWeatherGeneratorsCore
using Dates, Random, Statistics, Tables, StatsAPI
import StatsAPI: fit, loglikelihood, nobs, dof, aic, bic
export WeatherData, Leap366, NoLeap365, fit, simulate, decode, validate, curve, spec,
       loglikelihood, aic, bic, nobs, dof, likelihood_kind, save_generator, load_generator,
       FittedGenerator, WeatherSimulation, members, dates, stations, variables
public AbstractGenerator, AbstractCalendar, period, period_index, capabilities, resolve,
       fit_generator, simulate_member!, FitInfo, SimContext, SimBuffer, default_solver, set_default_solver!

abstract type AbstractGenerator end                       # unfitted model class ("algorithm")
capabilities(::Type{<:AbstractGenerator}) = (loglikelihood = false, regimes = false)
likelihood_kind(::Type{<:AbstractGenerator}) = :none      # :exact | :plugin | :composite | :qmc | :none
timestep(::Type{<:AbstractGenerator}) = Day(1)            # reserved for sub-daily classes
```

### 2.1 Calendar

```julia
struct Leap366  <: AbstractCalendar end   # T = 366; == v3 dayofyear_Leap (SWG src/utilities.jl:15)
struct NoLeap365 <: AbstractCalendar end  # climate-model output; a 29 Feb raises an error
period_index(::Leap366, d::Date) = dayofyear(d) + (!isleapyear(d) && month(d) > 2)
```

- Core owns the map Date → t. SPSM only ever receives an integer `n2t`.
- Monthly groupings come from `month(date)` in simulation and from `Monthly()`, an `IndicatorBasis` on the 366-day index, in the bases. The two agree.

### 2.2 Problem object and Tables input

```julia
struct WeatherData{T,C<:AbstractCalendar,S,V<:NamedTuple}   # fields private; use accessors
    dates::StepRange{Date,Day}          # COMPLETE daily axis; gaps are `missing`
    stations::Vector{S}
    values::V                           # (RR = D×N Matrix{Union{Missing,T}}, …)
    coords::Union{Nothing,Matrix{Float64}}
    calendar::C
    names::NamedTuple{(:date,:station),NTuple{2,Symbol}}    # reused on output
end
WeatherData(table; date = :auto, station = :auto, variables = :auto, kinds = (;),
            coords = nothing, calendar = Leap366())
WeatherData(values::NamedTuple, dates; stations, coords, calendar)          # arrays for experts
dates(d); stations(d); variables(d); d[:TX]; merge(d1, d2)                  # merge = add stations
```

**Input rules (B, plus judges' fixes):**
- Any Tables.jl source is accepted: a DataFrame, `CSV.File`, a NamedTuple of vectors, or a vector of per-station tables.
- **Wide or long layout** is auto-detected. Long format has `variable`/`value` columns.
- **Date column:** `Date`, or `DateTime` at midnight. Sub-daily values raise "aggregate to daily first".
- **Station column:** optional (single site if absent). Auto-detected from `station`, `STAID`, `site` or `id`.
- **Values:** `NaN` becomes `missing` with one warning. Duplicate (date, station) rows raise an error listing the first three.
- **Variable names** are matched case-insensitively against aliases that select a *kind*:

  | Names | Kind |
  |---|---|
  | RR / PRCP / pr | rain |
  | TX / TMAX / tasmax | temperature |
  | TN / TMIN / tasmin | lower temperature |
  | QQ / rsds, ETPP / ET0 / PET, FG | positive |

  `kinds = (WIND = :positive,)` declares any other column.
- **Unknown columns are never modelled silently.** They are dropped with one info line that names them and the fix, for example `CO2 ignored (unknown kind); declare kinds = (CO2 = :continuous,) to model it`. This handles the constant `CO2 = 330.0` that `collect_data_INRAE` emits (SWG `src/data/extracting.jl:139`). The new `read_stations` returns weather columns only.

### 2.3 Result and verbs

```julia
struct FittedGenerator{G<:AbstractGenerator,P} <: StatsAPI.StatisticalModel
    spec::G        # resolved: no `auto`, no ranges; `fit(data, spec(g))` refits it
    params::P      # class-owned
    info::FitInfo  # dates used/dropped, stations, kinds, calendar, loglik, likelihood_kind, dof, nobs,
                   # history::Vector{Pair{Symbol,Any}} (every SPSM FitHistory), selection table, provenance
end
g[:TX]                            # fitted block (indexing, not getproperty: no field shadowing)

fit(data, alg::AbstractGenerator; rng = default_rng(), seed = nothing,
    solver = default_solver(), verbose = true)                       :: FittedGenerator
simulate([rng,] g, dates::AbstractVector{Date}; nsim = 1, seed = nothing, past = nothing,
         burnin = 365, given = (;), threads = true, eltype = Float64) :: WeatherSimulation
simulate([rng,] g, years::AbstractUnitRange{<:Integer}; kw...)      # 1 Jan first → 31 Dec last
Base.rand([rng,] g, dates; kw...)                                   # one member (alias)
decode(g, data = training; method = :viterbi)                        # table (date, regime)
validate([rng,] g_or_gs, data; nsim = 100, seed = nothing, stats = default_stats(g), level = 0.9)
curve(g, var::Symbol, par::Symbol; station, regime = 1)              # length-T vector
spec(g); loglikelihood(g); aic(g); bic(g; by = :total); dof(g); nobs(g); likelihood_kind(g)
save_generator(path, g); load_generator(path); 
```

**What a class implements.** All of these are `public`; Core supplies everything else.

```julia
resolve(alg, data::WeatherData) -> alg′                  # fill defaults, check, human errors
fit_generator(data, alg′; rng, solver, verbose) -> FittedGenerator
simulate_member!(rng, buf::SimBuffer, g::FittedGenerator, ctx::SimContext) -> buf
capabilities(::Type{MyClass}); likelihood_kind(::Type{MyClass})       # traits
```

Core's `fit` performs: Tables → `WeatherData` → `resolve` → `fit_generator`. Model selection over ranges is **not** a generic Core feature. Each class decides what may be selected and by which criterion (§4.4). This removes B's invalid `argmin(bic, fits)` over TwoStage plug-in fits.

**Solver.** `default_solver()` reads a `Ref` that the umbrella's `__init__` sets to `LBFGS()`. If the Ref is empty, the error names the fix: "pass `solver = LBFGS()` after `using OptimizationOptimJL`, or load StochasticWeatherGenerators". Closed-form steps never need a solver: indicator-basis counts, AR(1) sufficient statistics, and correlations.

### 2.4 Simulation, RNG, initial conditions

```julia
function simulate(rng::AbstractRNG, g::FittedGenerator, dates; nsim = 1, seed = nothing, kw...)
    base = something(seed, rand(rng, UInt64))
    ctx  = SimContext(g, dates; kw...)                 # n2t from calendar, past/burnin/given checks
    buf  = SimBuffer(g, length(dates), nsim; kw...)
    tforeach(1:nsim) do i                              # Threads.@threads when threads = true
        simulate_member!(Xoshiro(hash(i, base)), member(buf, i), g, ctx)
    end
    return WeatherSimulation(buf, ctx, base)
end
```

**Seeding.** Member i draws from `Xoshiro(hash(i, seed))`. Its output therefore does not depend on the number of threads or on `nsim`. A `seed` keyword is enough for T0, so no `using Random` is needed.

**Initial conditions: one convention.**
- `past` is a table or `WeatherData` of days ending at `first(dates) - Day(1)`. Each block reads `memory(b)` days from it. The regime z₀ comes from `past` if it carries a `regime` column, otherwise it is drawn from the chain.
- `past = nothing` runs `burnin` days from a neutral start and discards them.
- `past` replaces `Y_past`, `y₁`, `z_ini`, `Y_ini` and `tx_ini`.
- No `init` verb or keyword exists anywhere, which removes A's triple overload. The streaming entry point is reserved as `stepper(g, dates)` (4.x).

**Conditioning.**
- `given = (regime = path,)` imposes a regime path, for example GCM weather types.
- `given = (TX = table,)` conditions children on a parent.
- Conditioning on a child to simulate its parent is rejected using the DAG: "TX is a parent of TN; condition on TX".

**Simulation is block-major,** as in v3: the whole path of one block, then the next block. This keeps the noise-injection equivalence tests possible (§8). Day-major streaming arrives with `stepper`.

### 2.5 Output contract

- `WeatherSimulation` is a lazy Tables.jl column table. Its columns are `member`, the input's date and station column names, the variables, and `regime` when the class has regimes.
- `sims[:TX]` is a D×N×nsim array, and `sims.regime` is N×nsim.
- `Tables.partitions(sims)` yields one partition per (member, station), i.e. one crop-model file each.
- `members(sims)` iterates per-member `WeatherData`, which can be refitted (recovery exercises).
- `DataFrame(sims)` materialises everything: 300 × 4 × 18 262 rows ≈ 22M rows. The T0 tutorial shows the partition loop next to it.

### 2.6 Missing data

Data stay on a complete daily axis. Nothing is imputed implicitly; `read_stations(...; impute = [:RR])` is opt-in.

- **iid terms** skip missing values.
- **Dynamic terms** at (s, n) require the `memory(b)` previous values at s. A missing day invalidates the memory category or lag for the next m days. Those terms are **dropped**, which is a conditional likelihood and is documented as one. Gaps are never bridged; this fixes `cov_ar1` and `fit_mle_AR.jl:1`.
- **Copulas** use pairwise-complete days, then `nearest_cor`.
- **Stage-1 regimes** at launch split sequences at days where any regime-defining station is missing, with a warning when more than 5% of days are lost. SPSM HMM emissions do not take masks yet; masked emissions are SPSM 5.1 (additive).
- **Simulations** never contain `missing`.

### 2.7 Save, load, pretrained

- `save_generator` writes **JSON of plain data**:
  - a format tag;
  - package versions;
  - `repr(spec(g))`;
  - flat coefficient vectors per fitted SPSM model, with their constructor arguments;
  - correlation matrices;
  - regime path, posterior and dates.

  `load_generator` rebuilds the spec by evaluating that text, then injects the coefficients. No JLD2 of structs is used: that format is what strands `assets/tuto_1/*.jld` today.
- `pretrained()` lists the bundled artifacts. `pretrained(:france_eca37_K4_d1_m1)` returns the 37-ECA-station regime generator from `tuto_paper`, converted in Phase 0 from `assets/tuto_1/hmm_fit_K_4_d_1_m_1.jld` and `z_hat_K_4_d_1_m_1.csv`.

## 3. Three tiers, by example

### T0 — impact modeller (crop tutorial)

```julia
using StochasticWeatherGenerators, DataFrames
obs  = read_stations(INRAE(), station_files)          # DATE, STAID, RR, TN, TX, QQ, ETPP (1986–2022)
swg  = fit(obs, RegimeHMM(regimes = Regimes(pretrained(:france_eca37_K4_d1_m1))))
sims = simulate(swg, 1986:2035; nsim = 300, seed = 50000)
validate(swg, obs)                                     # pass/flag table; plot(ans) with CairoMakie
df   = DataFrame(sims)                                 # or: for p in Tables.partitions(sims) CSV.write(…, p) end
```

`fit` prints info lines and never fails silently:

```
[ Info: Regimes cover 1956-01-01 … 2019-12-31: fitting on the overlap 1986-01-01 … 2019-12-31
        (1 095 days of observations after 2019 not used).
[ Info: ETPP has exact zeros (223, 961, 834, 1391 days by station): modelled as censored at 0
        (Positive(zeros = :auto) → seasonal P(ETPP = 0)).
```

The overlap rule reproduces the v3 `innerjoin(df, df_z; on = :DATE)` exactly (`tuto_add_station_variable.jl:164`). Without the pretrained regimes, `fit(obs, RegimeHMM(K = 4))` learns them from the data. With fewer than 8 regime-defining stations it warns: "regimes learned from 4 stations are weakly identified; consider Regimes(pretrained(...)) or Regimes(data = larger_network)".

`show(swg)` prints valid Julia and is the bridge to T1:

```
FittedGenerator{RegimeHMM}: 4 stations × 5 variables, 1986-01-01 … 2019-12-31 (0.7 % missing)
RegimeHMM(
    regimes = Regimes(pretrained(:france_eca37_K4_d1_m1)),          # frozen: K = 4, transitions season = 1
    RR   = Rain(occurrence = Occurrence(memory = 1), amount = Mixture(Exponential, 2), season = 1,
                copula = MultisiteGaussian(by = :regime, estimator = PairwiseRank())),
    TX   = Temperature(dynamics = AR1(), season = 1, copula = MultisiteGaussian(by = :regime)),
    TN   = Difference(:TX, Gamma; season = 1, coupling_season = 1, copula = MultisiteGaussian(by = :regime)),
    QQ   = Positive(Gamma; zeros = :constant, dynamics = LatentAR1(season = 1), season = 1),
    ETPP = Positive(Gamma; zeros = :seasonal, dynamics = LatentAR1(season = 1), season = 1),
    estimation = TwoStage(),
)
  likelihood: plug-in (regime path fixed) · loglik = … · BIC comparable only across models sharing these regimes
  next: simulate(swg, dates; nsim, seed) · validate(swg, obs) · spec(swg) → edit → fit(obs, ans)
```

Before fitting, `RegimeHMM()` shows `blocks = auto (chosen from your columns at fit time)`.

**Error messages** name the column, the station and the date, and give the fix as code:

```
ERROR: TN is modelled as Difference(:TX, Gamma) but there is no TX column.
       Use RegimeHMM(TN = Temperature()) to model TN on its own.
ERROR: QQ < 0 at STAID 80557001 on 2003-02-11 (-0.4); Positive() needs values ≥ 0.
       Fix the data, or declare QQ = Continuous(Normal).
WARN:  Regime 4 × JJA has 6 wet days at STAID 63345002: RR amounts poorly identified.
       Consider Rain(season = 0) or fewer regimes.
ERROR: WGEN has no likelihood, so K or season cannot be chosen by a range; give one value
       and compare with validate([g1, g2], obs).
```

**Student variant** (swap classes like solvers):

```julia
obs  = example_data(:france_eca_rain)                    # bundled artifact, RR only
gens = [fit(obs, m; seed = 1) for m in (RegimeHMM(K = 4), WGEN(order = 2), CensoredLatentGaussian())]
validate(gens, obs)                                      # same statistics, side by side
```

The concepts a novice needs are: the observation table, `fit`, a model class, `simulate`, `validate` and `DataFrame`. `Regimes(pretrained(...))` is a one-line copy.

### T1 — applied statistician

```julia
using StochasticWeatherGenerators, Distributions
paper = load_generator("france37_K4.swg.json")
model = RegimeHMM(
    regimes = Regimes(paper),                                    # frozen chain + path + posterior
    RR   = Rain(occurrence = Occurrence(memory = 2),
                amount = Mixture(Exponential, Gamma),            # heterogeneous mixture
                copula = MultisiteGaussian(by = (:regime, :month), estimator = PairwiseRank())),
    TX   = Temperature(dynamics = AR1(), season = (μ = 2, ρ = 0, σ = 1)),   # per-parameter trig degree
    TN   = Difference(:TX, Gamma; season = (α = 1, θ = 2)),     # Δ = TX − TN ~ seasonal Gamma; TN < TX
    QQ   = Positive(Gamma; dynamics = LatentAR1(season = 1), season = 0:2), # season chosen by block BIC
    ETPP = Positive(Weibull),
)
swg = fit(obs, model)
bic(swg; by = :block)                         # valid: every block shares the same regime path
curve(swg, :TX, :ρ; station = 49215002, regime = 2)
swg[:TX].models[1]                            # station 1's SPSM SmoothAR1: the T2 door
swg2 = fit(merge(obs, obs_new), spec(swg))    # add stations under the same frozen regimes
alt  = RegimeHMM(spec(swg); TN = Difference(:TX, LogNormal))  # copy-with-override constructor
```

- **Partial specification.** `RegimeHMM(TN = Difference(:TX, Weibull))` leaves every unlisted variable on `auto`; `others = :ignore` drops them instead.
- **`nothing` means "none".** `dynamics = nothing` is iid given the regime; `copula = nothing` is independent sites.
- **Integers in `season =` are degrees; values never appear in a spec.** Distributions-native start values are optional: `Mixture(Exponential(1.0), Gamma(0.8, 8.0))`.
- **Retraining regimes on another network:** `Regimes(K = 4, data = eca37)`.
- **Choosing K:** `Regimes(K = 2:6)` selects K by the **Stage-1 regime-HMM BIC** (paper practice) and stores the table in `swg.info.selection`.

**T1 vocabulary (16 names).** `Regimes`, `NoRegimes`, `Rain`, `Occurrence`, `Mixture`, `Temperature`, `Continuous`, `Positive`, `Difference`, `AR1`, `LatentAR1`, `MultisiteGaussian`, `PairwiseRank`, `Monthly`, `TwoStage`, `pretrained`. Keywords: `season`, `copula`, `dynamics`, `zeros`, `memory`.

### T2a — expert: a new block

Here the new block is a regime analog that resamples whole multisite vectors, which suits awkward variables such as humidity.

```julia
import StochasticWeatherGeneratorsBlocks as B

struct RegimeAnalog <: B.AbstractBlock; window::Int; end
RegimeAnalog(; window = 15) = RegimeAnalog(window)
B.memory(::RegimeAnalog) = 0
B.scope(::RegimeAnalog) = B.StateScope()
B.capabilities(::RegimeAnalog) = (logdensities = false,)   # → only TwoStage(); 4.x modes rejected politely

function B.fit_block(b::RegimeAnalog, ctx::B.FitContext)
    y, z = ctx.y[ctx.name], B.hard_regimes(ctx)              # D×N, Vector{Int}
    pool = Dict{Tuple{Int,Int},Vector{Int}}()
    for n in axes(y, 2)
        any(ismissing, view(y, :, n)) && continue
        for dt in -b.window:b.window
            push!(get!(pool, (z[n], mod1(ctx.n2t[n] + dt, ctx.period)), Int[]), n)
        end
    end
    return (y = Matrix{Float64}(coalesce.(y, NaN)), pool = pool)
end
function B.simulate_block!(rng, out::B.BlockOutput, b::RegimeAnalog, θ, ctx::B.SimContext)
    for n in eachindex(ctx.n2t)
        out.values[:, n] .= view(θ.y, :, rand(rng, θ.pool[(ctx.z[n], ctx.n2t[n])]))
    end
    return out
end
B.nparams(::RegimeAnalog, θ) = 0

fit(obs, RegimeHMM(kinds = (HU = :custom,), HU = RegimeAnalog(window = 10)))
```

**Full block protocol** (Blocks, all `public`):

| Method | Default | Notes |
|---|---|---|
| `outputs(b)` | `(name,)` | Multi-output blocks are allowed: an MST joint field, or a Richardson (TX, TN, QQ) VAR |
| `inputs(b)` | `()` | DAG parents |
| `memory(b)` | `0` | |
| `scope(b)` | `StateScope()` | `EdgeScope()` for latent AR and its children |
| `fit_block(b, ctx::FitContext)` | — | `ctx.W` is N×K weights, `OneHot(ẑ, K)` at launch |
| `simulate_block!(rng, out, b, θ, ctx)` | — | Must also write `out.innovations` when children exist |
| `innovations(b, θ, y, ctx)` | — | |
| `nparams(b, θ)` | — | |
| `logdensities!(LL, b, θ, ctx)` | optional | Needed later for Posterior and JointEM |

### T2b — expert: a new model class (k-NN, Lall & Sharma 1996), depending on Core only

```julia
module StochasticWeatherGeneratorsKNN
using StochasticWeatherGeneratorsCore, Random, Statistics
import StochasticWeatherGeneratorsCore as SWGCore
export KNNResampler

Base.@kwdef struct KNNResampler <: SWGCore.AbstractGenerator
    k::Union{Int,Nothing} = nothing; window::Int = 7
end
function SWGCore.fit_generator(data, alg::KNNResampler; kw...)
    X, ok = SWGCore.standardized_features(data)            # F×N anomalies; complete consecutive days
    t = SWGCore.period_index.(Ref(data.calendar), SWGCore.dates(data))
    k = something(alg.k, max(1, round(Int, sqrt(length(ok) * (2alg.window + 1) / 366))))
    SWGCore.FittedGenerator(KNNResampler(k, alg.window), (; X, t, ok, data), SWGCore.FitInfo(data))
end
function SWGCore.simulate_member!(rng, buf, g::SWGCore.FittedGenerator{KNNResampler}, ctx)
    (; X, t, ok, data) = g.params; k, w = g.spec.k, g.spec.window
    kern = cumsum(1 ./ (1:k)); kern ./= kern[end]
    cur = SWGCore.initial_day(rng, g, ctx)
    for n in eachindex(ctx.n2t)
        cands = [j for j in ok if SWGCore.circdist(t[j], ctx.n2t[n], 366) ≤ w]
        d  = [sum(abs2, view(X, :, cur) .- view(X, :, j)) for j in cands]
        nn = cands[partialsortperm(d, 1:min(k, length(d)))]
        cur = nn[min(searchsortedfirst(kern, rand(rng)), length(nn))] + 1
        SWGCore.copy_day!(buf, n, data, cur)
    end
    return buf
end
end
```

With nothing more than this, `simulate`, `nsim`/`seed`, Tables output, `validate` and the plots all work.

## 4. Flagship Regime-HMM product model

### 4.1 Joint law

Notation: t_n = period_index(date_n), and pa(b) are the DAG parents of block b.

p(z, y) = a(z₁) · Πₙ A_{z_{n−1}z_n}(t_{n−1}) · Πₙ Π_b p_b(y_nᵇ | z_n [, z_{n−1}], t_n, y^b_{n−m_b:n−1}, ε_n^{pa(b)})

SPSM's leaving-period convention is used: A is stored at the period being left, A_{k·}(t) = softmax(B(t)β_k). The blocks:

- **Rain.**
  - *Occurrence:* O_{s,n} = 1{RR > threshold}, with P(O = 1 | z = k, h) = logistic(B(t)β_{s,k,h}). Sites are independent given z by default. `Occurrence(copula = MultisiteGaussian(...))` thresholds a latent field instead: O_s = 1{U_s ≤ Φ⁻¹(p)}, with U ~ N(0, C^O).
  - *Amounts on the wet set 𝒲:* u = F^R(r), and p = c_{C_k[𝒲,𝒲]}(u) Π f^R. The Gaussian copula restricted to 𝒲 is its exact marginal.
- **Temperature, form A (AR on the variable, v3).** x = μ + ρx₋₁ + σε, with every coefficient seasonal and indexed by (s, k), and ε_n ~ N(0, C_k). The density depends on z_n only (`StateScope`).
- **Positive, form B (censored latent AR).** p0_{s,k}(t) = P(Y = 0) and G_{s,k,t} is the law of positive values.
  - w_n = ρ_k(t) w_{n−1} + √(1−ρ²) ε_n, with ε_n ~ N(0, C_k).
  - Y = 0 if w ≤ Φ⁻¹(p0); otherwise Y = G⁻¹((Φ(w) − p0)/(1 − p0)).
  - μ = 0 and σ = √(1−ρ²) are **tied**, so Y_n | z_{1:N} ~ the fitted hurdle law exactly, across regime switches, with no clipping.
  - `zeros = :auto` resolves per site to `:none`, `:constant` or `:seasonal` (seasonal when at least 0.5% of days are zero). With `:none` the block is the uncensored LatentAR1.
  - The density needs w_{n−1}, which is defined under z_{n−1}, so the block is `EdgeScope`.
- **Difference(:TX, G)** (TN):
  - δ = x^TX − y^TN, and v = Φ⁻¹(G_{s,k,t}(δ)), with v = r ε^TX + √(1−r²) η and η ~ N(0, C^η_k).
  - Density: Π_s g(δ_s) φ((v_s − r ε_s)/√(1−r²)) / (√(1−r²) φ(v_s)), times the η copula term.
  - TN < TX always. The transform is **named**, never inferred from the support of the marginal.
  - Coupling goes through the parent's *standardised or latent innovation*, so it works whichever form TX uses. The block inherits the parent's scope.

**Same-day correlation.** With form B and site-varying ρ, the stationary same-day correlation is C_ss′√((1−ρ_s²)(1−ρ_s′²))/(1−ρ_sρ_s′), not C. `LatentAR1` therefore **shares ρ across sites by default** (`shared = :sites`), which makes it equal to C. The formula is documented for `shared = :none`.

### 4.2 Spec objects

```julia
struct RegimeHMM{R,B<:NamedTuple,E} <: AbstractGenerator
    regimes::R; blocks::B; estimation::E; kinds::NamedTuple; others::Symbol
end
RegimeHMM(; K = nothing, regimes = Regimes(K = something(K, 4)), estimation = TwoStage(),
          kinds = (;), others = :auto, blocks...)
RegimeHMM(base::RegimeHMM; overrides...)                     # copy with override

Base.@kwdef struct Regimes{KT,S,D}                           # in Blocks (shared with CLG)
    K::KT = 4; season::S = 1; from::Symbol = :RR; memory::Int = 1
    stations = :all; data::D = nothing; order_by::Symbol = :wetness
end
Regimes(g::FittedGenerator) -> FrozenRegimes                 # chain, path, posterior by Date
Regimes(hmm, path_table) -> FrozenRegimes                    # any externally fitted SPSM HMM (e.g. spatial)
Regimes(; path::Table) -> ObservedRegimes                    # given weather types; A(t) fitted on labels
NoRegimes()                                                  # K = 1

Rain(; occurrence = Occurrence(), amount = Mixture(Exponential, 2), season = 1, threshold = 0.0,
       copula = MultisiteGaussian(by = :regime, estimator = PairwiseRank()))
Occurrence(; memory = 1, season = nothing, copula = nothing, counting = :calendar)
Temperature(; dynamics = AR1(), marginal = Normal, season = 1, copula = MultisiteGaussian())
Continuous(marginal; dynamics = LatentAR1(), season = 1, copula = MultisiteGaussian())
Positive(marginal = Gamma; zeros = :auto, dynamics = LatentAR1(), season = 1, copula = MultisiteGaussian())
Difference(parent::Symbol, marginal = Gamma; sign = -1, season = 1, coupling_season = 1,
           copula = MultisiteGaussian())
AR1(); LatentAR1(; season = 1, shared = :sites)
MultisiteGaussian(; by = :regime, structure = Unstructured(), estimator = Auto())
#   by:        :none | :regime | :month | (:regime, :month) | :season        (Seasonal(d) reserved, CorrelationLink)
#   structure: Unstructured() | ExponentialKernel(; range) (needs coords; falls back to Unstructured with a warning)
#   estimator: Auto() | PairwiseRank() | NormalScores() | PairwiseCensored() | MomentMatching(; nsim) | PairwiseLikelihood()
TwoStage()                                                   # only mode at launch
```

Defaults by kind:

| Kind | Default block |
|---|---|
| rain | `Rain()` |
| temperature | `Temperature()` |
| lower temperature (with TX present) | `Difference(:TX, Gamma)` |
| lower temperature (no TX) | `Temperature()` |
| positive | `Positive()` |
| continuous | `Continuous(Normal)` |

The `MultisiteGaussian` object owns the draw path, `rand_innovations!(rng, ε, dep, k, t, mask)`. A future t-copula or space-time structure is therefore a new `structure` type and does not change the block contract.

### 4.3 Composition rules

- `resolve` fills defaults, sets T from the calendar, and checks four things: the DAG is acyclic, every input exists, scope is compatible with the estimation mode, and `capabilities` cover the mode.
- Blocks run in topological order. Ties keep the declared order, so RNG consumption is deterministic.
- If the regimes are learned (not frozen) from these stations' RR, Rain's occurrence **is** the Stage-1 emission. It is reused and not refitted, which reproduces `tuto_paper`.

### 4.4 Estimation, and what each mode maximises

| Mode | Stage 1 | Stage 2 (block b) | What is maximised | Status |
|---|---|---|---|---|
| `TwoStage()` = v3 | EM for log Σ_z p(z) p(y^R \| z) (local), then Viterbi ẑ | weighted M-step with W = `OneHot(ẑ, K)`; edge blocks use the (ẑ_{n−1}, ẑ_n) pair | A **classification likelihood** given a plug-in path. Biased toward over-separated regimes; consistent as decoding error → 0. Works for every scope. | 4.0 |
| `TwoStage(decode = Posterior())` | same, plus γ and ξ (HMMs `transition_marginals = true`, verified in 0.7.1 `forward_backward.jl:9`) | StateScope: Σγ log p_b. EdgeScope: Σξ(k,l) log p_b(·\|k,l) | The Jensen bound of the **cut-model** likelihood; blocks never inform the regimes | 4.1 (additive) |
| `JointEM()` | none | E-step on Σ_b LL_b; M-step is each block's weighted fit | The full MLE only if every M-step maximises Q_b. IFM copulas make it **generalised EM**: own loop with best-iterate tracking, **no monotonicity assertion** (HMMs' `loglikelihood_increasing = false` exists) | 4.2 (StateScope or K² edge pass) |

**Inside a block.** The procedure is IFM:
1. Fit the margins per site × regime by weighted seasonal MLE in SPSM.
2. Fit the dynamics:
   - AR1 jointly with its margin (SPSM sufficient statistics);
   - LatentAR1 by weighted regression of w_n on w_{n−1} with tied variance (SPSM `SmoothLatentAR1`);
   - censored sites by the pairwise censored likelihood on lag-1 pairs.
3. Fit the copula:
   - continuous innovations: pairwise-complete Pearson, which is the Gaussian MLE on complete data;
   - rain amounts: pairwise Spearman → Pearson on co-wet days (v3 `cor_RR`, `src/rain/correlations.jl:102`);
   - censored data: pairwise censored likelihood.

   Then `nearest_cor` **always**. This replaces v3's "regime K independent" hack (`src/rain/generation.jl:31`), whose own comment says it was a workaround for non-PD matrices.
4. Fit the coupling r_{s,k}(t) by weighted MLE of v | ε ~ N(rε, 1−r²). This is the quantity used in simulation, which fixes the mismatch between `fit_mle_TN.jl:75-88` and `:103`.

**Likelihood and BIC.**
- `loglikelihood(g)` is the Stage-1 HMM log-likelihood plus the block log-likelihoods given ẑ. `likelihood_kind = :plugin`.
- `bic(g; by = :block)` is valid for comparing blocks under a shared path.
- `bic(g)` totals are labelled in `show` as "comparable only across models sharing these regimes".
- K is never selected on totals: a K range uses the Stage-1 BIC.
- A `season = 0:2` range inside a block is selected by block BIC under the shared path, which is valid.

### 4.5 Simulation (one member; Core runs the ensemble)

```julia
function SWGCore.simulate_member!(rng, buf, g::FittedGenerator{<:RegimeHMM}, ctx)
    z = haskey(ctx.given, :regime) ? ctx.given.regime : B.sample_regimes(rng, g.params.regimes, ctx)
    buf.regime .= z
    for name in g.params.order                               # topological, block-major
        fb = g.params.blocks[name]
        B.simulate_block!(rng, B.BlockOutput(buf, name), fb.spec, fb.θ, B.SimContext(ctx, z, buf))
    end
    return buf
end
```

- **Rain:** draws occurrence with memory, then ε ~ N(0, C_k[𝒲,𝒲]) on the wet set, then r = F⁻¹(Φ(ε)). Cholesky factors are cached per (k, mask) when D ≤ 16; otherwise the factor is computed per day.
- **AR1:** runs the recursion and keeps ε for its children.
- **Difference:** reads the parent's ε.
- **Positive:** runs the latent AR, then censors and transforms.

## 5. WGEN port and censored latent Gaussian

### WGEN: a NoRegimes product on the shared Blocks engine

```julia
Base.@kwdef struct WGEN{A,S,E} <: AbstractGenerator
    order::Int = 1
    season::S = Monthly()
    amount::A = nothing                                    # Mixture(Exponential, 2) → Wilks amount layer
    estimator::E = MomentMatching(nsim = 2000)            # Ω; PairwiseLikelihood() also available
    counting::Symbol = :calendar                           # :v3 = legacy month-slice counts
    others::Symbol = :ignore
end
SWGCore.capabilities(::Type{<:WGEN}) = (loglikelihood = false, regimes = false)
SWGCore.resolve(w::WGEN, data) = as_wgen(SWGCore.resolve(RegimeHMM(regimes = NoRegimes(), others = w.others,
    RR = Rain(occurrence = Occurrence(memory = w.order, season = w.season, counting = w.counting,
              copula = MultisiteGaussian(by = :month, estimator = w.estimator)),
              amount = w.amount, season = w.season)), data))
```

**Margins.** With a `Monthly()` basis, the occurrence block uses a **closed-form probability table**, with no logit and no optimiser, so cells that are exactly 0 or 1 are reproduced. There are two counting rules:
- **`:calendar`** (default): each day n is counted in cell (s, month(n), h_n), where h_n is built from the actual preceding `order` days. This is consistent with v3's own simulator, which carries `Y_ini` across months (`wgen.jl:145-152`).
- **`:v3`**: reproduces `fit_markov_chain(Y[j, idx], p)` (`wgen.jl:110`) over each month's days concatenated across years, counting from i = 1 to length − p (`wgen.jl:91-95`). This includes the cross-year jumps and skips month-boundary transitions. It is kept only for the equivalence gate and documented as legacy.

Empty cells give p = 0 with a warning, as in v3.

**Ω_m.**
- Per pair and month, `MomentMatching` runs a bisection on ω ∈ (−0.999, 0.999), matching the observed lag-0 occurrence correlation by an `nsim`-day simulation that uses **common random numbers**. The objective is then deterministic and monotone. v3 advanced the RNG inside the root-find (`wgen.jl:179`). The result goes through `nearest_cor`. NonlinearSolve is dropped.
- `PairwiseLikelihood()` maximises Σ log Φ₂-quadrant probabilities. It reuses the binary-pair subset of the censored kernel built on SPSM's `norm_cdf_2d_vfast` (`HHMM_Bernoulli_spatial/fast_bivariate_cdf.jl:19`), the same kernel as `pairwise_loglikelihood_memory` (`likelihoods.jl:178`) with a free ω.

**Simulation and likelihood.** Each day, U ~ N(0, Ω_{month}) and O_s = 1{U_s ≤ Φ⁻¹(p_s(month, h))}. The noise hook can inject v3's per-month U blocks (`wgen.jl:122`). `likelihood_kind = :none`, so `loglikelihood` errors with "compare with validate".

**Teaching device.** `expand(spec(g))` prints the equivalent `RegimeHMM`. A regime-dependent Wilks model comes free: `RegimeHMM(RR = Rain(occurrence = Occurrence(copula = MultisiteGaussian(by = :regime, estimator = MomentMatching()))))`.

### CensoredLatentGaussian: Ailliot 2009, Vaittinada Ayar 2020

```julia
Base.@kwdef struct CensoredLatentGaussian{R,M,T,D,C} <: AbstractGenerator
    regimes::R = NoRegimes()                    # Regimes(K = 3), Regimes(g), Regimes(path = weather_types)
    marginal::M = Mixture(Exponential, 2)       # positive part G
    transform::T = :anamorphosis                # or PowerTransform()  (R = a·max(0, W − c)^β)
    season = 1
    dynamics::D = LatentAR1(season = 1)         # nothing ⇒ temporal independence given regimes (Ailliot)
    copula::C = MultisiteGaussian(by = :regime, estimator = PairwiseCensored())   # or ExponentialKernel
    threshold::Float64 = 0.0
    others::Symbol = :ignore                    # :auto ⇒ flagship presets for TX, … (shared Blocks)
end
SWGCore.likelihood_kind(::Type{<:CensoredLatentGaussian}) = :composite
```

**It is the `Positive` block applied to rain.** The model is `RR = CensoredRain(...)`, an internal alias of the same censored-latent block type, with `zeros = :seasonal`:
- W_n = ρW_{n−1} + √(1−ρ²)ε, with ε ~ N(0, C_k(t));
- R = 0 iff W ≤ Φ⁻¹(p0_{s,k}(t)); otherwise R = G⁻¹((Φ(W) − p0)/(1 − p0)).

Occurrence and amount come from one field.

**Fit (IFM).**
1. p0 by weighted seasonal logistic; G on wet days. Both use SPSM `SmoothDistribution` with weights.
2. ρ by the pairwise censored likelihood on lag-1 pairs:
   - both wet: φ₂ with Jacobians;
   - one wet: φ·Φ((c′ − rw)/√(1−r²));
   - both dry: Φ₂.
3. C by the same likelihood on same-day pairs, then `nearest_cor`.
4. Regimes: the shared Stage 1 when learned; frozen; or observed weather patterns.

**Likelihood.** `loglikelihood` returns the composite log-likelihood, labelled as such. BIC is undefined (CLIC is reserved).

**Scope against the literature.** Ailliot's joint HMM, which uses the censored vector as its emission and needs QMC orthant probabilities, is **experimental** in 4.x. The launch version is labelled "two-stage approximation of Ailliot 2009".

### What the three classes share

| Piece | RegimeHMM | WGEN | CLG |
|---|---|---|---|
| Driver | Regimes / frozen / observed | NoRegimes | any |
| Y = G⁻¹(Φ(W)) | occurrence copula (opt.), amounts, Positive, Difference | threshold | censor |
| `MultisiteGaussian` + `nearest_cor` | PairwiseRank, NormalScores, PairwiseCensored | MomentMatching / PairwiseLikelihood | PairwiseCensored |
| Pairwise bivariate-normal kernel (SPSM `norm_cdf_2d_vfast`) | Positive with zeros | Ω option | ρ, C |
| `LatentAR1` / censored latent AR | QQ, ETPP | — | the field |
| SPSM weighted margins | all | amounts | p0, G |
| Calendar, `past`, `seed`, Tables I/O, validate, JSON | Core | Core | Core |

Golden test: `WGEN(order = 1, copula → independent, counting = :calendar)` and `RegimeHMM(regimes = NoRegimes(), RR = Rain(occurrence = Occurrence(season = Monthly())))` must give **identical** probability tables.

## 6. SPSM v5

### 6.1 Engine contract (5.0; everything marked "4.2" is first added in 4.2)

```julia
fit_mle(m, y[, w]; n2t, past = nothing, seq_ends = nothing, solver, kw...) -> m′      # MODEL ONLY (Distributions contract)
fit(SeasonalFit, m, y[, w]; kw...) -> SeasonalFit                                    # model + FitHistory + StatsAPI (unchanged name)
#   w (3rd positional, ALWAYS weights): N-vector | N×K matrix | OneHot(z, K) (lazy Bool matrix)   [4.2]
loglikelihoods(m, y; n2t, past, seq_ends) -> N×K          # every model incl. SmoothAR1, mixture, SmoothDistribution [4.2]
rand([rng,] m, n2t; z = nothing, past = nothing)          # one order; HMMs return (; z, Y)          [4.2 keyword form]
viterbi(m, Y; n2t, past, seq_ends)
curve(m, name, ts = 1:T; state = :); m(t); m(t; state = k)
SmoothDistribution(family, T; K = 1, season = 1)          # K-state seasonal law (replaces TimeVaryingFit)   [4.2]
SmoothLatentAR1(T; K = 1, season = 1)                     # y_n | x_n ~ N(ρ(t) x_n, 1 − ρ(t)²); x = lag(y) by default,
                                                          # or keyword x = covariate (TN coupling)       [4.2]
SmoothARPeriodicHMM(K, D, T; memory, season)              # size constructors; fit_mle(...; init = :slices) [4.2]
fit_mle(hmm, Y, W; fixed = (:A,), past, n2t)              # complete-data M-step given weights (replaces fit_mle_RO) [4.2]
```

**Contract rules:**
- `n2t` is **required** in 5.0. All 69 `n_to_t(...)` default sites are removed (counted by grep over `src/`, e.g. `fit/mixture.jl:22`, `fit/ar1.jl:28`, `HHMM_Bernoulli_spatial/likelihoods.jl:178`). `n_to_t` stays exported as an explicit helper.
- `past` replaces positional `Y_past`, `y₁`'s "first day" meaning (now the day *before*), and `z_ini` (now `z` of day 0 inside `past`).
- Every M-step minimises the **average** weighted NLL.
- The AR(1) sufficient-statistics path (`fit/ar1.jl:74-76`) is kept, and `SmoothLatentAR1` gets the same T-grouped statistics (Σwx², Σwxy, Σwy²).
- Vectors of sequences and `seq_ends` work for AR1, the mixture and SmoothDistribution. Missing values split sequences.

### 6.2 Vector links, covariates, TVP

- **Deferred, additive:** `SoftmaxLink` in 5.1 (deduplicates the 4 softmax copies and fixes the ignored weight link); `CorrelationLink` (CPC/onion, experimental for D > 10) in 5.2; general `Conditional` covariate families in 5.1. The fast `PeriodicNLL` T-grouping is kept as a specialisation.
- **5.0 scope:** one covariate, the `x` keyword on `SmoothLatentAR1`. That covers both form-B dynamics and the TN coupling.
- **TVP:** stays vendored. The upstream move is revisited in 5.x.

### 6.3 Exports: v4 (94) → v5

- **Exported (≈43):**
  - bases: `TrigBasis ConstantBasis FourierBasis PolynomialBasis IndicatorBasis BSplineBasis CyclicBSplineBasis ParamSpec`;
  - links: `IdentityLink LogLink LogitLink IntervalLink`;
  - `curve TimeVaryingFamily seasonal_family`;
  - models: `SmoothAR1 SmoothLatentAR1 SmoothDistribution SmoothSeasonalMixture SmoothPeriodicHMM SmoothARPeriodicHMM SmoothsARPeriodicHMM SmoothARPeriodicHMMSpatial`;
  - `OneHot fit_mle fit SeasonalFit FitHistory converged loglikelihoods viterbi`;
  - StatsAPI: `coef loglikelihood dof nobs aic aicc bic`;
  - `components ncomponents mixtures n_to_t`.
- **`public` only:**
  - layout: `paramnames design_matrix ncoef coefs blockindex blockrange nbasis ncolumns column`;
  - objectives: `PeriodicDesign PeriodicNLL PointwiseNLL group_by_time distributions update_dist`;
  - EM knobs: `ClassicEM StochasticEM`;
  - solver plumbing: `default_nlls_solver default_adtype default_ensemble solve_unconstrained solve_nlls solve_objective` (the old `fit_loss_optim`);
  - AR internals: `ar1_spec ar1_sufficient_statistics ar1_nll`;
  - evaluated twins: `AR1 ARPeriodicHMM sARPeriodicHMM ARPeriodicHMMSpatial PeriodicHMMDists`;
  - HMM plumbing: `SeasonalControls emission_coefs emission_range component_coefs component_range degrees`;
  - slice initialiser: `fit_mle_all_slices`;
  - shared kernels: `norm_cdf_2d_vfast pairwise_loglikelihood_memory`.
- **Removed:**
  - `TimeVaryingFit` (→ `SmoothDistribution`/`SeasonalFit`), `mixture_weight_spec`, `mixture_weights` (→ `curve(m, :weights)`);
  - `Trig2ARPeriodicHMM`, `Trig2ARPeriodicHMMSpatial`, `randARPeriodicHMM`;
  - `fit_θ fit_θ! fit_θᴬ! fit_θᴮ!`, `fit_mle_RO`;
  - `μₜ ρₜ σₜ αₜ`, `params` (→ `m(t)`), `parameters` and `nparams` (→ `coef`, `dof`);
  - the multisite `rand(::AbstractArray{<:AR1}, n2t, z, Σ; y₁)` (`AR/AR1.jl:123-156`: untested, ignores Σ), which moves to Blocks.
- **No type renames in 5.0.** v4 is 3 days old, and `tuto_genomics.jl` uses `SmoothARPeriodicHMM` and `fit(SeasonalFit, …)`, both of which are unchanged.

### 6.4 Moves

| From → to | Item |
|---|---|
| SWG → SPSM | `_ar1_init` (`fit_mle_AR.jl:56`) → `SmoothAR1` default start |
| SWG → SPSM | `mix_ini` (`fit_mle_rain.jl:71`) → `SmoothSeasonalMixture` default start |
| SWG → SPSM | `_fit_seasonal_gamma` (`fit_mle_TN.jl:34-43`) → `SmoothDistribution` |
| SPSM → SWG Blocks | multisite AR simulation, as part of `MultisiteGaussian` + `AR1` |
| SWG ext → Blocks | `cor_groupby*`, which becomes `SmoothLatentAR1(x = ε)` |
| stays in SPSM | spatial HMM (unregistered SpatialBernoulli.jl; the codepaper depends on it) |
| stays in SPSM | the `obs_logdensities!` override in 5.0; the documented `CachedLogDensity` bridge is 5.1 and an upstream issue is opened |

## 7. Naming and verbs

| Concept | New name | Replaces (SWG v3 / SPSM v4) |
|---|---|---|
| problem | `WeatherData` | per-station DataFrames + `innerjoin(df_z)` + n2t |
| calendar | `Leap366`, `period_index` | `dayofyear_Leap`, `n_to_t` |
| model class | `RegimeHMM`, `WGEN`, `CensoredLatentGaussian` | hand glue; `wgen`, `fit_wgen` |
| regimes | `Regimes(K)`, `Regimes(g)`, `Regimes(path = …)`, `NoRegimes` | `randARPeriodicHMM` → slices → `fit_θ!` → `SmoothARPeriodicHMM`; `ARPeriodicHMM(a, A, B)` |
| pretrained | `pretrained(:name)`, `pretrained()` | `JLD.load` + z_hat CSV |
| rain block | `Rain`, `Occurrence(memory)`, `Mixture` | `fit_mle_RO`, `fit_mle_RR`, `cov_RR`, `rand_RR`, `mix_ini`, `onefy` |
| AR on variable | `Temperature()` / `AR1()` | `fit_AR1`, `cov_ar1`, `rand(ar1s, n2t, z, Σ)` |
| positive variable | `Positive(Gamma)`, `LatentAR1()` | `fit_AR1` + `indicatrix` clipping |
| TN given TX | `Difference(:TX, Gamma)` | `fit_TN`, `cor_groupbyTXTN`, `rand_cond`, `ϵ_TX`, `output_ϵ` |
| dependence | `MultisiteGaussian(by, structure, estimator)` | `Σ²RR`, `Σ²TX`, `Ω`, `fit_Ω`, `cor2cov` |
| seasonality | `season =` (Int, NamedTuple, basis, `Monthly()`, range) | `𝐃𝐞𝐠`, `degree`, `TrigBasis(T,d)`, `idx_months` |
| memory | `memory =` (days) | `local_order`, `memory_order`, WGEN `p` |
| estimation | `TwoStage()` | implicit plug-in |
| fit | `fit(data, model)` → `FittedGenerator` | 6 result shapes |
| simulate | `simulate(g, dates; nsim, seed)`, `rand` | `rand(hmm…; seq)`, `rand_RR`, `rand(ar1s…)`, `rand_cond`, `simulate_markov_gaussian` |
| initial state | `past =`, `burnin =` | `Y_past`, `y₁`, `z_ini`, `Y_ini`, `tx_ini` |
| conditioning | `given = (regime = …,)` | z passed to every sampler |
| output | `WeatherSimulation`, `Tables.partitions`, `members` | `dfs_simus`, `dicts_simu` |
| decode | `decode(g, data)` | `viterbi(hmm, Y, Y_past; n2t)` + CSV |
| check | `validate(g or [g…], data)` | ad hoc plotting, `monthly_agg` loops |
| persistence | `save_generator`/`load_generator` (JSON) | `JLD.save(Dict(...))` |
| read-out | `curve(g, var, par; station, regime)`, `g[:TX]` | `μₜ…`, `θ_ΔT[j][k](t)` |
| data | `read_stations(INRAE()/ECA()/MeteoFrance(), …)` | `collect_data_*` |
| SPSM fit | `fit_mle(m, y, w)` → m; `fit(SeasonalFit, …)` | tuple `fit_mle`; `TimeVaryingFit` |
| SPSM given z | `OneHot(z, K)` | positional z / weights / Y_past |
| SPSM seasonal law | `SmoothDistribution` | `seasonal_family` + `PeriodicNLL` + `fit_loss_optim` |

**Argument-order rule, in one sentence:** in SWG the data come first, as in SciML `solve(prob, alg)`; in SPSM the model comes first, as in Distributions `fit_mle(d, x)`. Each package accepts one order only. The Aqua ambiguity between `fit(::Type{<:Distribution}, x)` and `fit(data, ::AbstractGenerator)` is closed in Blocks with an erroring disambiguation method.

## 8. Migration and release plan

### 8.1 Ordered steps (detail in §12)

1. **P0 (SWG):** add test CI, golden fixtures, and conversion of the assets.
2. **P1 (SPSM 4.2, additive):** register it. `main` stays non-breaking.
3. **P2–P6 (SWG):** build `lib/` against SPSM 4.2 inside the single workspace. The umbrella root stays v3 and its compat `SmoothPeriodicStatsModels = "4.1"` resolves to 4.2. The libraries use only API present in both 4.2 and 5.0: weights, keyword `past`, required `n2t` passed explicitly, `fit(SeasonalFit, …)`, `loglikelihoods`, keyword `rand`.
4. **P7:** v4 tutorials built under "Preview (v4)", and the migration guide drafted.
5. **P8:** pre-register Core → Blocks → the three classes at 1.0.0, with SPSM compat `"4.2"`, at least 2 weeks before the release. Each new package waits about 3 days, and dependencies must be registered first.
6. **P9 (release day):**
   - SPSM 5.0.0 breaking PR merged and registered.
   - Patch releases of Blocks, RegimeHMM, WGEN and CLG widen compat to `"4.2, 5"` after their CI passes against 5.0. These merge in minutes.
   - The umbrella 4.0.0 cutover deletes `src/rain`, `src/AR1`, `src/temperature` and `ext/SmoothPeriodicStatsModelsExt`, swaps the tutorials, and sets SPSM compat `"5"`.
   - One announcement for both packages.
7. **P10:** drop `"4.2"` from compat in the next patches.

### 8.2 Old → new (most-used calls)

| v3 SWG / v4 SPSM | v4 SWG / v5 SPSM |
|---|---|
| `collect_data_INRAE.(p; impute_missing)` + `innerjoin(df, df_z)` | `read_stations(INRAE(), p; impute = [:RR, :TX])` + `Regimes(pretrained(...))` |
| `fit_mle_RO(df, order, d; solver)` + `ARPeriodicHMM(a, hmm.A, B)` | `Rain(occurrence = Occurrence(memory = order), season = d)` under frozen `Regimes` |
| `fit_mle_RR(df, order; mix₀ = mix_ini(T))` | `Rain(amount = Mixture(Exponential, 2))` (default) |
| `cov_RR(dfs, K)` | `Rain(copula = MultisiteGaussian(by = :regime, estimator = PairwiseRank()))` (default) |
| `fit_AR1.(dfs, :TX, d)` + `cov_ar1(dfs, ar1s, :TX)` | `TX = Temperature(season = d)` |
| `fit_TN.(dfs, 1, T)` + `cor_groupbyTXTN.(dfs, T)` | `TN = Difference(:TX, Gamma; season = 1)` |
| `fit_AR1(df, :QQ, d)` + `indicatrix.(rand(…))` | `QQ = Positive(Gamma)` (legacy: `Continuous(Normal; dynamics = AR1())`) |
| NSIMU loops: `rand(hmm…)`, `rand_RR`, `rand(ar1s…; y₁, output_ϵ)`, `rand_cond` | `simulate(g, dates; nsim, seed, past)` |
| `Random.seed!(50000)` | `seed = 50000` |
| `dfs_simus` assembly | `DataFrame(sims)` / `Tables.partitions(sims)` |
| `fit_wgen(Y, idx_months, p)` / `rand(wgen, years; Y_ini)` | `fit(obs, WGEN(order = p))` / `simulate(g, years; past)` |
| `viterbi(hmm, Y, Y_past; n2t)` (SWG) | `decode(g, data)` |
| `longuest_spell`, `monthly_agg`, `VCX3` | `validate(g, obs)` (functions kept, renamed `longest_spell`) |
| SPSM `fit_mle(SmoothAR1(d,T;K), y, z; n2t)` → `(m, h)` | `fit_mle(SmoothAR1(d,T;K), y, OneHot(z,K); n2t)` → `m`; `fit(SeasonalFit, …)` for `h` |
| SPSM `fit_mle(mix, y, w; n2t)` → `(m, h)` | `fit_mle(mix, y, w; n2t)` → `m` |
| SPSM `fit_mle(hmm, Y, Y_past; n2t)` | `fit_mle(hmm, Y; past = Y_past, n2t)` |
| `randARPeriodicHMM` → `fit_mle_all_slices` → `fit_θ!` → `SmoothARPeriodicHMM(a,θᴬ,θᴮ,T)` | `fit_mle(SmoothARPeriodicHMM(K, D, T; memory, season), Y; past, n2t, init = :slices)` |
| `fit_mle_RO(Y, Y_past, z, n2t, d, K, T)` | `fit_mle(SmoothARPeriodicHMM(K, D, T; memory, season = d), Y, OneHot(z, K); past, n2t, fixed = (:A,))` |
| `rand(hmm, n2t; Y_past, z_ini, seq)` / `rand(ar1, n2t, z; y₁)` | `rand(hmm, n2t; past)` → `(; z, Y)` / `rand(ar1, n2t; z, past = y₀)` |
| `TimeVaryingFit` / `fit_loss_optim` | `SmoothDistribution` / `SPSM.solve_objective` (public) |
| `μₜ(t/T, θ)`, `mixture_weights(m)`, `parameters(m)`, `params(m, t)` | `curve(m, :μ)`, `curve(m, :weights)`, `coef(m)`, `m(t)` |

### 8.3 Numerical equivalence

- **Fixtures (P0).** `test/reference/make_v3.jl` runs in a committed Manifest pinned to SWG 3.0.0 + SPSM 4.1.0. It has two inputs:
  - (a) tiny deterministic fixtures: 3 sites × 6 years, K = 2;
  - (b) a tutorial-scale synthetic dataset simulated from the tutorial's fitted model. No INRAE licensing issue, and no network.

  It stores coefficients, logliks, ẑ, Σ/Ω, and ensemble statistics as CSV/JSON.
- **L1, parameters.** Same data, starting values and solver. Tolerances:
  - rtol 1e-8 for closed-form fits (AR1 sufficient statistics, counts);
  - rtol 1e-6 for the regime HMM and RO given ẑ;
  - rtol 1e-4 where an optimiser runs (mixtures, ΔT Gamma);
  - `==` for **WGEN `counting = :v3`** and `dayofyear_Leap`;
  - atol 0.03 for WGEN Ω (CRN vs v3 Monte Carlo).
- **L2, pathwise.** An unexported `noise =` hook in `SimContext` injects pre-drawn ε and U. Outputs must equal:
  - v4 `rand(ARs, ϵ, n2t, z, Σ; y₁)` (`AR1.jl:147-156`) under the `y₁` → `past` mapping;
  - `rand_cond(ϵ, …)` at equal r;
  - RO thresholding;
  - `simulate_markov_gaussian` given per-month U (`wgen.jl:120-133`).

  Full-run bitwise equality is impossible because v3 draws variable-major across all members from one global RNG (`tuto_add_station_variable.jl:323-342`). It is not attempted.
- **L3, distributional.** 300 members. About 200 statistics are compared: monthly mean, sd and quantiles, wet frequency, spell pmfs, inter-site and inter-variable correlations. Bonferroni-corrected z < 4.
- **Legacy is tested through specs, not code paths, wherever possible.** Example: `QQ = Continuous(Normal; dynamics = AR1(), copula = MultisiteGaussian(estimator = NormalScores()))` must match v3's fit. The one legacy code path is WGEN `:v3` counting.
- **Intentional differences,** listed in the migration guide, each with a property test:
  - TN's r on the innovation scale;
  - TN's η copula;
  - Positive is censored or uncensored latent AR instead of clipping;
  - the regime-K independence hack is removed (PD projection instead);
  - `cov_ar1` gap and first-residual fixes;
  - explicit k instead of `groupby(:z)` order (`fit_mle_TN.jl:28`);
  - `past` and burn-in;
  - WGEN calendar-correct counts and CRN Ω;
  - RNG streams.

## 9. Risks, trade-offs, open questions

1. **Launch bandwidth.** About 2k lines of code in SWG and 8k in SPSM, essentially one maintainer. Mitigations:
   - 5 libraries, not 7;
   - Posterior and JointEM, vector links, the streaming integrator, partial refit (`reuse`) and the spatial regime class are all deferred;
   - every deferral is additive because the weighted contract, the scope trait and the reserved names exist now.
2. **The censored latent AR is new code.** The pairwise censored kernel serves Positive, CLG and the WGEN option. If it slips, the `zeros = :auto` fallback is a **loud error** naming `Continuous(Normal; dynamics = AR1())`. A silent Gamma default is never acceptable.
3. **Weak logistic cells.** A seasonal logistic for p0 with very few zeros risks separation. `zeros = :auto` uses a constant p0 below 0.5% zeros (QQ has 3–11 zero days), and `show` prints the choice.
4. **Plug-in likelihood semantics** must be clear in `show` and in the docs. BIC selection is restricted to Stage-1 K and to blocks under a shared path.
5. **Single-manifest discipline.** Any breaking change slipped into SPSM 4.x stalls P2–P6. The SPSM `Downstream` CI job enforces this.
6. **Registration timing.** Tiered pre-registration with compat only on released versions. Long package names follow SciML convention; General discourages acronyms.
7. **`obs_logdensities!`** stays undocumented upstream in 5.0. The 5.1 `CachedLogDensity` bridge removes it.
8. **Cost of wet-set Cholesky at D = 37** is unmeasured. The mask cache is used only when D ≤ 16. Benchmark in P4.
9. **CLG fidelity.** The exact Vaittinada Ayar 2020 parametrisation is unverified, and Ailliot's joint HMM is experimental.
10. **Memory.** 22M rows. Mitigated by `eltype = Float32`, lazy Tables columns and `Tables.partitions`.
11. **Reserved now for growth:**
    - `timestep` and the private `dates` field (accessor only), for AWE-GEN-2d;
    - `ObservedRegimes`, for MSTWeatherGen weather types;
    - multi-`outputs` blocks, for Richardson VAR and MST joint fields;
    - `structure` types, for space-time Matérn;
    - `LatentVAR1`, for GWEX;
    - an EGPD family;
    - `stepper`, `TwoStage(decode = Posterior())`, `JointEM`, `Seasonal(d)` correlation, covariates in `WeatherData` and `simulate`.

## 10. Decisions taken

| # | Decision | One-line reason |
|---|---|---|
| D1 | The product model lives in SWG `…Blocks`. Generic engine pieces (`SmoothDistribution`, `SmoothLatentAR1`, weights, N×K LL) go to SPSM | Author decision 1; Blocks is the "DiffEqBase" glue inside the monorepo |
| D2 | `TwoStage()` (Viterbi plug-in) is the only mode in 4.0. Posterior (with ξ for EdgeScope) in 4.1; JointEM (GEM, best-iterate) in 4.2 | Matches the paper; avoids the ill-defined soft-γ-on-edge flaw judges found in A and B |
| D3 | Keep the specialised SPSM HMMs; no merge in 5.0 | Bandwidth; the codepaper and genomics users depend on them; dedup internally later |
| D4 | TX keeps form A (v3 parity). Positive and CLG use form B with **tied** μ = 0, σ = √(1−ρ²), censored at 0; ρ shared across sites; general covariates in 5.1 | Exact marginal and positivity; handles ETPP exact zeros; v3 numbers preserved for TX |
| D5 | Vector links deferred: SoftmaxLink 5.1, CorrelationLink 5.2. Correlations piecewise by `by` group in 4.0 | Not needed for launch; additive later |
| D6 | Evaluated twins `public`, not exported | `curve`, `m(t)` and `rand` cover users; experts keep access |
| D7 | Tables at SWG (wide or long, aliases); arrays and `n2t` in SPSM; no DataFrames in any library | Decision 7; SPSM stays domain-agnostic |
| D8 | Keyword constructors with partial specification and a copy-with-override constructor | Discoverable; DSL sugar can come later |
| D9 | SPSM: `fit_mle` → model; `fit(SeasonalFit, …)` → result. SWG: `fit` → `FittedGenerator` (one type for all classes) | Distributions/EM nesting (constraint 2) plus one user result |
| D10 | Keep SPSM `Smooth*` names; `season =` for degrees, `memory =` for lags; `past` for initial state; `n2t` stays the keyword name in SPSM | Least churn three days after v4; unambiguous for students |
| D11 | SPSM solver stays mandatory. The SWG umbrella sets `LBFGS()` through a Ref in its `__init__`; libraries error with the fix | Solver-free T0 without hard Optimization dependencies in libraries |
| D12 | Clean SPSM 5.0 break with a migration table, preceded by an **additive** 4.2 (no deprecations, no shims) | Decision 5; one manifest during development |
| D13 | TVP stays vendored in 5.0 | Changes land fastest in one place; upstream later |
| D14 | Rain amounts: PairwiseRank (v3). Continuous innovations: pairwise Pearson. Censored: PairwiseCensored. Occurrence: MomentMatching with CRN. Always `nearest_cor`; Copulas.jl leaves SWG | v3 continuity plus correct handling of varying wet sets and missing data |
| N1 | Monorepo with Core, Blocks, RegimeHMM, WGEN, CensoredGaussian plus umbrella (readers in the umbrella); explicit workspace list | Maintainer cap; Core free of Distributions |
| N2 | SPSM in its own repo with a Downstream CI job | Domain-agnostic users and own history |
| N3 | Verbs: `fit`, `simulate`/`rand`, `decode`, `validate`, `curve`, `spec`, `save_generator`/`load_generator`; no `init`/`solve` verbs | Small, consistent vocabulary; no overloaded `init` |
| N4 | RNG: `seed` keyword or leading `rng`; member i uses `Xoshiro(hash(i, seed))` | Reproducible T0, independent of thread count |
| N5 | Unknown columns ignored with an info line; readers return weather columns only | `CO2 = 330.0` (`extracting.jl:139`) is never modelled |
| N6 | Frozen regimes: fit on the overlap, report dropped dates | Reproduces the v3 `innerjoin`; INRAE runs to 2022 but z_hat ends 2019 |
| N7 | K selection only by Stage-1 BIC; block `season` ranges by block BIC under a shared path | Valid criteria only |
| N8 | Persistence as JSON plain arrays plus spec text; `pretrained()` artifacts | Survives refactors; `.jld` assets rot |
| N9 | Block-major simulation at launch; day-major `stepper` later | Allows the pathwise equivalence tests |
| N10 | Block scope trait (`StateScope`/`EdgeScope`) and `capabilities`, checked at `resolve` | Makes 4.1/4.2 modes statistically safe and additive |
| N11 | WGEN: calendar-correct counting by default; `counting = :v3` legacy for the golden test | Fixes the `wgen.jl:110` bias while keeping exact equivalence testable |
| N12 | Launch libraries at 1.0.0; future classes at 0.x; umbrella keeps its UUID | Per-package maturity signal (SciML practice) |
| N13 | Spatial HMM stays in SPSM; usable via `Regimes(hmm, path)` | SpatialBernoulli.jl unregistered; codepaper depends on it |
| N14 | Long `StochasticWeatherGenerators*` sublibrary names | General naming guidance; SciML precedent |

## 11. Decisions left for the author

1. **Default Positive marginal for QQ.**
   - Options: (a) `Gamma` hurdle; (b) `Beta` bounded by clear-sky radiation with `upper = clear_sky(lat, t)`; (c) `Weibull`.
   - **Recommendation:** (a) at launch, and (b) as an additive option in 4.1. It needs a radiation formula the author should choose.
2. **WGEN default Ω estimator.**
   - Options: (a) `MomentMatching` with CRN, faithful to Wilks 1998; (b) `PairwiseLikelihood`, deterministic, faster and principled.
   - **Recommendation:** (a) for 4.0, so papers and teaching match Wilks; revisit after benchmarking.
3. **CLG parametrisation.**
   - Options: anamorphosis vs power transform as the default; ρ seasonal or constant.
   - **Recommendation:** anamorphosis with seasonal ρ. The author must confirm it against Vaittinada Ayar 2020 and Ailliot 2009 before freezing the 1.0 spec.
4. **TN η copula default.**
   - Options: (a) `MultisiteGaussian(by = :regime)` on the coupling residuals (new); (b) independent across sites (v3).
   - **Recommendation:** (a). v3's independence looks accidental, but it changes the TN spatial statistics.
5. **Should `fit(data)` without a model class exist?**
   - **Recommendation:** no. Always name the class; it is one word and keeps the solver analogy explicit.
6. **Spatial regime emission in RegimeHMM.**
   - Options: (a) 4.1 as `Regimes(emission = :spatial)`; (b) only via `Regimes(hmm, path)` from SPSM.
   - **Recommendation:** (b) at launch, (a) in 4.1. `tuto_paper_HMMspatial` becomes an SPSM-level expert tutorial plus `Regimes(hmm, path)`.
7. **Renaming SPSM `Smooth*` types.**
   - Options: keep, or rename in a later major release.
   - **Recommendation:** keep in 5.0. If renamed later, do it once, together with the HMM merge.
8. **Default K behaviour without pretrained regimes.**
   - Options: (a) K = 4 with a warning below 8 stations; (b) require `K` explicitly.
   - **Recommendation:** (a).

## 12. Implementation phases

Path roots: **SPSM** = `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels`, **SWG** = `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators`.

**P0. Safety net (SWG and SPSM, no API change)**
- **Goal:** CI, golden fixtures, and stable assets before any refactor.
- **Create:**
  - `SWG\.github\workflows\CI.yml` (GROUP matrix, Aqua, export-list test, `@time_imports` budget);
  - `SWG\test\reference\make_v3.jl` + `Manifest.toml` pinned to SWG 3.0.0 / SPSM 4.1.0;
  - `SWG\test\reference\fixtures\*.csv|json`;
  - `SWG\assets\pretrained\france_eca37_K4_d1_m1.swg.json`.
- **Modify:** `SWG\src\data\extracting.jl` (lazy MeteoFrance download, replacing the `df_api_url` at :176).
- **Reuse:** `SWG\examples\tuto_add_station_variable.jl`, `tuto_paper.jl` and `src\rain\wgen.jl` (`fit_wgen`, `fit_markov_chain`, `simulate_markov_gaussian`) as fixture generators; `SWG\assets\tuto_1\hmm_fit_K_4_d_1_m_1.jld` + `z_hat_K_4_d_1_m_1.csv`.
- **Exit:** CI green on v3; fixtures committed; the pretrained JSON reloads into v4-independent arrays.

**P1. SPSM 4.2 (additive), registered**
- **Goal:** the stable subset the SWG libraries need.
- **Modify:**
  - `SPSM\src\fit\ar1.jl` (weights matrix and `OneHot` method next to the z-vector one at :66; vectors of sequences; `past`);
  - `SPSM\src\AR\AR1.jl` (keyword `rand(…; z, past)`; N×K `loglikelihoods`);
  - `SPSM\src\fit\mixture.jl` (N×K weights; `loglikelihoods`; sequences);
  - `SPSM\src\fit\smooth_hmm.jl` (size constructors, `init = :slices`, weighted complete-data path, `fixed = (:A,)`, keyword `past`);
  - `SPSM\src\fit\add_sites.jl` (route through `update_B!`);
  - `SPSM\src\SmoothPeriodicStatsModels.jl` (new exports).
- **Create:** `SPSM\src\fit\distribution.jl` (`SmoothDistribution`), `SPSM\src\AR\latent_ar1.jl` (`SmoothLatentAR1` with the `x` covariate and T-grouped sufficient statistics), `SPSM\src\onehot.jl`.
- **Reuse:**
  - `update_B!` and `fit_mle_RO` logic (`fit\add_sites.jl:13-36`);
  - `ar1_sufficient_statistics` (`fit\ar1.jl:74-76`);
  - `_fit_seasonal_gamma` pattern (`SWG\ext\SmoothPeriodicStatsModelsExt\fit_mle_TN.jl:34-43`);
  - `_ar1_init` (`fit_mle_AR.jl:56`);
  - `mix_ini` (`fit_mle_rain.jl:71`);
  - `fit_mle_all_slices` (`SPSM\src\HHMM_Bernoulli\mle_slice.jl:16`).
- **Exit:**
  - all v4 tests unchanged;
  - new tests: OneHot ≡ z-vector AR1 (bitwise); `fit_mle(hmm, Y, OneHot; fixed = (:A,))` ≡ `fit_mle_RO`; `SmoothDistribution(Gamma)` ≡ SWG `fit_TN` margins (rtol 1e-6); `SmoothLatentAR1` recovers ρ(t) on simulated data;
  - Downstream job added.

**P2. SWG Core**
- **Goal:** verbs, `WeatherData`, calendar, simulate driver, validate, JSON I/O, with a toy class in the tests.
- **Create:** `SWG\lib\StochasticWeatherGeneratorsCore\{Project.toml, src\*.jl, test\runtests.jl}`; add it to the root `[workspace]`.
- **Reuse:**
  - `dayofyear_Leap` (`SWG\src\utilities.jl:15`);
  - `longuest_spell`, `monthly_agg`, `VCX3` (`SWG\src\climate_indices.jl:5,85,38`) as validation statistics.
- **Exit:**
  - Tables wide/long/alias/unknown-column tests, including the CO2 info line;
  - seed independence across thread counts and `nsim`;
  - JSON round-trip;
  - Core loads in ≤ 0.8 s.

**P3. SWG Blocks**
- **Goal:** blocks, `MultisiteGaussian`, censored latent kernel, DAG engine, `Regimes`/`NoRegimes`, `TwoStage`.
- **Create:** `SWG\lib\StochasticWeatherGeneratorsBlocks\src\{protocol, dependence, latent, rain, continuous, positive, difference, regimes, engine, show}.jl`.
- **Reuse:**
  - `Σ_Spearman2Pearson`, `cor_RR` (`SWG\src\rain\correlations.jl:34,102`);
  - `cov_ar1` (`SWG\src\AR1\fit_mle.jl:6`), with the gap fix;
  - `rand_RR` structure (`SWG\src\rain\generation.jl`), without the :31 hack;
  - `rand_cond` (`fit_mle_TN.jl`);
  - SPSM `norm_cdf_2d_vfast` (`HHMM_Bernoulli_spatial\fast_bivariate_cdf.jl:19`);
  - NearestCorrelationMatrix.
- **Exit:**
  - L1 per block against fixtures (`fit_mle_RO`/RR/AR1/`fit_TN` margins/`cov_RR`/`cov_ar1`);
  - L2 noise-hook equality against v4 `rand(ARs, ϵ, …)` and `rand_cond`;
  - censored latent AR recovers (p0, G, ρ, C) on simulated data;
  - scope/mode rejection test.

**P4. RegimeHMM + crop tutorial preview**
- **Goal:** the flagship end to end.
- **Create:** `SWG\lib\StochasticWeatherGeneratorsRegimeHMM\src\*.jl`; `SWG\examples\v4\tuto_add_station_variable.jl`, `tuto_paper.jl`.
- **Add to the v3 umbrella** (additive): `read_stations`, `pretrained`, `example_data` in `SWG\src\data\`; fix the longitude bug in `distance_x_to_y` (`SWG\src\data\stations.jl:53`).
- **Reuse:** SPSM `SmoothARPeriodicHMM` fit + `viterbi`; the P0 pretrained artifact.
- **Exit:**
  - T0 runs in ≤ 6 lines;
  - frozen-overlap report matches the v3 `innerjoin`;
  - L1 regime HMM rtol 1e-6, identical ẑ;
  - L3 statistics within 4 MC SE, apart from the listed intentional differences;
  - benchmark of wet-set Cholesky at D = 37.

**P5. WGEN**
- **Create:** `SWG\lib\StochasticWeatherGeneratorsWGEN\src\*.jl`.
- **Reuse:** `fit_markov_chain` semantics (`wgen.jl:84-110`) for `counting = :v3`; `simulate_markov_gaussian` (`wgen.jl:120-155`) as the L2 reference; `fit_Ω` logic (`wgen.jl:173-212`) rewritten with CRN bisection.
- **Exit:**
  - `:v3` probability tables `==` v3;
  - Ω atol 0.03;
  - pathwise equality given U;
  - the WGEN ≡ NoRegimes RegimeHMM probability-table test.

**P6. CensoredGaussian**
- **Create:** `SWG\lib\StochasticWeatherGeneratorsCensoredGaussian\src\*.jl`.
- **Reuse:** the Blocks censored kernel and `Regimes`; SPSM `SmoothDistribution`.
- **Exit:**
  - simulate → refit recovery: 3 sites, NoRegimes and K = 2;
  - `likelihood_kind = :composite`;
  - `bic` errors with a message;
  - the student side-by-side `validate` runs.

**P7. Docs and migration**
- **Modify:** `SWG\docs\make.jl` (Literate already at :3 and :44; add the Preview section and per-package API pages with `checkdocs = :public`), `SWG\docs\src\models\rain.md` and `others.md` (TODOs mapped to reserved names), SPSM docs plus `SPSM\examples\tuto_genomics.jl`.
- **Create:** the migration guide with the §8.2 table and intentional differences, in both repos.
- **Exit:** docs build in CI; every tutorial runs.

**P8. Pre-registration**
- **Goal:** Core, then Blocks, then the classes, all at 1.0.0 with SPSM compat `"4.2"`, at least 2 weeks before release.
- **Exit:** all merged in General.

**P9. Release day**
- **SPSM 5.0:**
  - `SPSM\src\SmoothPeriodicStatsModels.jl`: exports and `public` (lines 51-199);
  - delete `μₜ`… (`src\utilities.jl:34-74`), `Trig2*`, `randARPeriodicHMM`, `fit_θ*`, `fit_mle_RO`, `TimeVaryingFit`, `mixture_weight_spec`, and the multisite `rand` (`AR\AR1.jl:123-156`);
  - `fit_mle` returns the model only;
  - `n2t` required at all 69 sites;
  - CHANGELOG (fix the 4.0.1/4.1.0 mismatch).
- **SWG:**
  - library patches widen compat to `"4.2, 5"`;
  - umbrella 4.0: rewrite `SWG\Project.toml` and `SWG\src\StochasticWeatherGenerators.jl`; delete `src\rain`, `src\AR1`, `src\temperature`, `ext\SmoothPeriodicStatsModelsExt`; move the v4 examples into place.
- **Exit:** both registered; Downstream CI green; one announcement.

**P10. Follow-ups**
- Drop the 4.2 compat lane.
- 4.1: `TwoStage(decode = Posterior())` with ξ, `reuse`, spatial regime emission, `stepper`.
- SPSM 5.1: SoftmaxLink, `Conditional`, masked HMM emissions, `CachedLogDensity`.

### Critical Files for Implementation
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\examples\tuto_add_station_variable.jl (flagship workflow and golden-fixture source)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\src\rain\wgen.jl (WGEN port; month-slice counting at :110 and RNG inside the root-find at :179)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\ext\SmoothPeriodicStatsModelsExt\fit_mle_TN.jl (TN|TX coupling to re-estimate; `_fit_seasonal_gamma` → `SmoothDistribution`)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\fit\ar1.jl, plus src\AR\AR1.jl, src\fit\add_sites.jl and src\fit\smooth_hmm.jl (weights, `OneHot`, `fixed`, sequences, `SmoothLatentAR1`)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\SmoothPeriodicStatsModels.jl (export surface at lines 51-199 for the 5.0 prune)