## 0. Thesis

SWG v4 should be built like DifferentialEquations.jl, applied to daily weather. Observed weather enters as a `WeatherProblem`: any Tables.jl source becomes a regular daily grid of sites × variables with a calendar. A model class plays the role of the algorithm: `RegimeHMM()`, `WGEN()` or `CensoredLatentGaussian()`, each in its own `lib/` sublibrary. `fit(prob, alg)` returns one result type, `WeatherGenerator{Alg}`, which carries `alg`, `prob`, `retcode` and `stats` the way an `ODESolution` does. Simulation is a second problem, `SimulationProblem(gen, dates; init)`. It is solved by `simulate`/`solve` with SciML‑v7 ensemble semantics (`nsim`, `seed`, a per-trajectory `ctx.rng`, `output_func`), or stepped one day at a time with `init`/`step!` so that a crop model can drive it.

In OrdinaryDiffEq an algorithm supplies only `alg_cache` and `perform_step!`, and it inherits the integrator, saving and ensembles. Here a model class supplies only `fit`, `initialize_state` and `advance!`, and it inherits ensembles, streaming, Tables output, validation, plots and save/load. That is what lets GWEX, MSTWeatherGen, k‑NN or AWE‑GEN‑2d be added later as new sublibraries without any change to the core.

The three launch classes are thin assemblies over one shared sublibrary, `…Blocks`. It holds the regimes, the variable blocks (all driven by a latent Gaussian) and a multisite copula layer. `…Blocks` sits on SPSM v5. SPSM v5 stays in its own repo as a domain-agnostic engine with a narrow contract: weighted fits, N×K log-densities, a one-step `conditional`, forward–backward on a log-density matrix, vector links, and covariate families.

Deliberate deviations from SciML:
- the "algorithm" changes the statistical model, not just the numerics;
- estimation is spelled `fit` (StatsAPI), not `solve`;
- outputs are Tables-first;
- the core types do not subtype SciMLBase's problem/solution hierarchy. From SciMLBase we reuse only `ReturnCode`, `remake`, the integrator verbs and the ensemble-algorithm singletons, plus the CommonSolve verbs.

## 1. Package architecture

```
StochasticWeatherGenerators.jl/                  one git repo; the root is the umbrella (as OrdinaryDiffEq.jl)
├── Project.toml      name="StochasticWeatherGenerators" version="4.0.0"
│                     [sources]   StochasticWeatherGeneratorsCore = {path="lib/StochasticWeatherGeneratorsCore"} …
│                     [workspace] projects = ["lib/*", "docs", "test"]   (one shared Manifest)
├── src/StochasticWeatherGenerators.jl          explicit `using X: a, b` + `export` lists, no blanket @reexport
├── lib/
│   ├── StochasticWeatherGeneratorsCore/        types, traits, problems, calendars, Tables I/O, driver, ensembles, validation
│   │   └── ext/ MakieExt (plot recipes), OptimJLExt (default optimizer hook)
│   ├── StochasticWeatherGeneratorsBlocks/      Regimes, Rain/Occurrence/Variable blocks, Multisite layer, block protocol, DAG
│   ├── StochasticWeatherGeneratorsRegimeHMM/   flagship class + estimation modes + default blocks
│   ├── StochasticWeatherGeneratorsWGEN/        Wilks 1998 / Srikanthan 2009 port
│   ├── StochasticWeatherGeneratorsCensoredGaussian/   Ailliot 2009 / Vaittinada Ayar 2020
│   └── StochasticWeatherGeneratorsData/        ECA / INRAE / Météo-France readers, station metadata, pretrained artifacts
│       └── ext/ GeoMakieExt (map_with_stations)
├── docs/   one Documenter site (DocumenterInterLinks → SPSM docs)
└── test/   cross-package integration + numerical-equivalence suite (each lib/X/test is self-contained)
```

| Package (short name) | Purpose | Hard deps | Exports |
|---|---|---|---|
| **Core** | `AbstractWeatherModel`, traits, `WeatherProblem`, `SimulationProblem`, calendars, `fit`/`simulate`/`solve`/`init`/`step!`, ensemble driver, `WeatherGenerator`, `WeatherSimulation`, `WeatherIntegrator`, validation statistics, `write_generator`/`read_generator` | Dates, Random, Tables, StatsAPI, CommonSolve, SciMLBase, JSON, PrecompileTools; weak: Makie, OptimizationOptimJL | ~40 |
| **Blocks** | `Regimes`, `freeze`, `ObservedRegimes`, `Rain`, `Occurrence`, `Variable`, `Below`, `LatentCorrelation`, `Multisite` + estimators, latent thresholding, block protocol | Core, SPSM 5, Copulas 1, Distributions, NearestCorrelationMatrix, SpatialBernoulli (C. Cognot's package) | ~22, plus about 8 SPSM declarations re-exported |
| **RegimeHMM** | `RegimeHMM`, `TwoStage`, `JointEM`, `Viterbi`, `Posterior`, default block per variable kind | Core, Blocks | ~7 |
| **WGEN** | `WGEN`, `Monthly`, `MomentMatching` | Core, Blocks, BracketingNonlinearSolve | ~3 |
| **CensoredGaussian** | `CensoredLatentGaussian`, `Anamorphosis`, `PowerTransform`, `PairwiseCensored` | Core, Blocks | ~4 |
| **Data** | `read_stations`, `ECA`, `INRAE`, `MeteoFrance`, `station_metadata`, `distance_matrix`, `pretrained` | Core, CSV, Downloads, Artifacts; Impute only on explicit request | ~10 |
| **umbrella** | curated re-export of all of the above; `LBFGS()` default optimizer | all of the above + OptimizationOptimJL | ~75 |

**Where SPSM lives: in its own repo.** SciML keeps SciMLBase, NonlinearSolve and Optimization as separate repos because they serve many downstream users. It moved DiffEqBase *into* the OrdinaryDiffEq monorepo because DiffEqBase was "tightly coupled" to solver internals (SciML v7 post, 2026‑07‑28). SPSM is the first kind:
- it is domain-agnostic (`examples/tuto_genomics.jl`, the HMM_Spatial_codepaper environment);
- it has a co-author and its own docs.

The tightly coupled glue that does exist (multisite composition, regimes over blocks) becomes **Blocks**, which plays the role of DiffEqBase and lives inside the monorepo. The coupling between the two repos shrinks to the engine contract in §6, which has about 10 functions. SPSM CI runs a "Downstream" job on the SWG Blocks and RegimeHMM tests, which is SciML's own pattern.

**Umbrella.** It follows OrdinaryDiffEq v7, which says it loads its packages with "no blanket @reexport" and keeps explicit export lists. The umbrella contains only the launch classes. A future class (GWEX, KNN…) ships first as its own sublibrary at 0.x and joins the umbrella in a minor release once mature. Adding exports is non-breaking under ColPrac, but name clashes are possible (§9).

**Load time.** Figures are estimates, not measured.
- Core avoids SPSM, Copulas and DataFrames, so it should load in under 1 s.
- WGEN alone needs Core + Blocks + SPSM.
- The umbrella adds the Optim stack and CSV, a few seconds in total.

Measures:
- no DataFrames anywhere; only docs and tutorials use it;
- network access only at call time, which fixes the download at precompile time in `src/data/extracting.jl:176-178`;
- a PrecompileTools workload in every sublibrary on a synthetic 2-site × 3-year problem;
- Makie and GeoMakie through extensions.

**Registration and versioning.**
- Each `lib/X` has its own UUID, version and tests. It is registered with `@JuliaRegistrator register subdir=lib/X`, and TagBot uses `subdirs` so tags look like `StochasticWeatherGeneratorsCore-v1.0.0`.
- `[sources]` paths are for development only; the registry ignores them.
- Launch sublibraries start at **1.0.0**. Future classes start at **0.x**, so maturity is signalled per package, which is a real advantage of the monorepo.
- The umbrella `[compat]` pins sublibrary majors.
- CI runs a `GROUP=Core|Blocks|RegimeHMM|WGEN|CensoredGaussian|Data|Umbrella` matrix, as SciML does.
- Registration order: SPSM 5.0 → Core → Data → Blocks → the three classes → umbrella 4.0.

## 2. Core interface

### 2.1 SciML ↔ SWG map

| SciML | SWG v4 | Deviation |
|---|---|---|
| `ODEProblem(f,u0,tspan,p)` | `WeatherProblem(table; date, site, variables, sites, calendar, given)` | holds data, not equations |
| algorithm `Tsit5()` | model class `RegimeHMM()`, `WGEN()`, … `<: AbstractWeatherModel` | **semantic**: swapping the class changes the model |
| nested `nlsolve=`, `autodiff=` | `estimation = TwoStage()`, `solver = LBFGS()` | — |
| `solve(prob, alg)` | `fit(prob, alg)` | named `fit` (StatsAPI) |
| `solve(prob)` + `DefaultODEAlgorithm` | `fit(prob)` → `RegimeHMM()` resolved from the columns | the resolved spec is printed by `show` |
| `ODESolution` (`.prob .alg .retcode .stats`) | `WeatherGenerator` (same fields + `.params`) | — |
| `ReturnCode`, `successful_retcode` | reused from SciMLBase | — |
| `EnsembleProblem` + `trajectories`, v7 `seed`/`rng`/`ctx` | `solve(SimulationProblem; nsim, seed, rng, ensemble, output_func)` | keywords instead of a wrapper; `nsim` not `trajectories` |
| `EnsembleSolution` (`sim.u[j]`) | `WeatherSimulation` (`sim.u[j]`, `sim[:TX]`) | Tables-first |
| `init`/`step!`/`reinit!`/`set_u!`/`tuples` | same verbs on `WeatherIntegrator` | the step is a `Period` (default `Day(1)`) |
| `remake` | `SciMLBase.remake` extended for problems, model classes, generators | — |
| `alg_cache` / `perform_step!` | `initialize_state` / `advance!` | — |
| callbacks | reserved (not in 4.0) | — |

### 2.2 Types, traits, hooks

```julia
abstract type AbstractWeatherModel end          # ≈ AbstractSciMLAlgorithm
abstract type AbstractWeatherProblem end
abstract type AbstractCalendar end
abstract type AbstractInit end

# traits: a new class overrides only what it supports
has_likelihood(::Type{<:AbstractWeatherModel})     = false
has_regimes(::Type{<:AbstractWeatherModel})        = false
supports_streaming(::Type{<:AbstractWeatherModel}) = true
timestep(::AbstractWeatherModel)                   = Day(1)     # reserved for sub-daily classes

# the hooks a model class implements
fit(prob::WeatherProblem, alg::MyModel; rng, solver, ensemble, kwargs...) :: WeatherGenerator      # required
initialize_state(gen::WeatherGenerator{<:MyModel}, sp::SimulationProblem, rng)                        # required
advance!(today::StepBuffer, state, gen::WeatherGenerator{<:MyModel}, step::StepContext, rng)        # required
simulate_trajectory!(out, gen::WeatherGenerator{<:MyModel}, sp, rng)   # optional, for non-Markov classes
loglikelihood(gen, prob; by = :total); decode(gen, prob, method)       # optional
to_dict(params) ; from_dict(::Type{MyModel}, d)                        # optional, for write/read_generator

struct StepContext; n::Int; time::Date; t::Int; tprev::Int; end   # `Step`, not `Day`, so sub-daily needs no rename later
```

### 2.3 Problem and Tables input contract

```julia
struct WeatherProblem{V<:NamedTuple,C<:AbstractCalendar,G<:NamedTuple,S} <: AbstractWeatherProblem
    data::V                    # (RR = Matrix{Union{Missing,Float64}} D×N, TX = …)
    dates::StepRange{Date,Day} # always regular: missing days are inserted as `missing`
    sites::S                   # SiteTable: id (+ LAT, LON, ALT, NAME)
    calendar::C
    given::G                   # observed side information, e.g. (regime = Vector{Union{Missing,Int}},)
end
WeatherProblem(table; date = :DATE, site = :STAID, variables = :auto,   # wide: one row per (date, site)
               variable = nothing, value = nothing,                     # long: (date, site, variable, value)
               sites = nothing, calendar = LeapCalendar(), given = (;), kinds = (;))
WeatherProblem(tables::AbstractVector; kw...)                           # one table per station
fit(table, alg; kw...)   # splits out the problem keywords (date, site, variables, …), passes the rest to fit
```

- A missing `site` column means a single site.
- `variables = :auto` takes every numeric non-key column.
- `kinds` maps unknown names to a kind, for example `(WIND = Positive(),)`.
- Accessors: `dates(prob)`, `sites(prob)`, `variables(prob)`, `prob[:TX]`, `period_index(prob)` (that is, n2t).
- Arrays are used underneath: experts may build `WeatherProblem((TX = M,), dates; sites)` directly.

### 2.4 Calendar

```julia
struct LeapCalendar   <: AbstractCalendar end   # T = 366; 1 March is always index 61 (= v3 dayofyear_Leap)
struct NoLeapCalendar <: AbstractCalendar end   # T = 365; 29 Feb shares 28 Feb's index
period_length(::LeapCalendar) = 366
period_index(::LeapCalendar, d::Date) = dayofyear(d) + (!isleapyear(d) && month(d) > 2)
```

- Core owns Date → t. SPSM only ever sees an integer `n2t`, never Dates.
- `CustomCalendar(f, T)` covers 360-day climate-model calendars; a CFTime extension can come later.

### 2.5 fit and the result

```julia
fit(prob::WeatherProblem, alg::AbstractWeatherModel; rng = default_rng(), seed = nothing,
    solver = default_solver(alg), ensemble = EnsembleThreads(), warmstart = nothing,
    keep_data = true, verbose = false) -> WeatherGenerator
fit(prob) = fit(prob, default_model(prob))

struct WeatherGenerator{A<:AbstractWeatherModel,P,Pr,St}
    alg::A          # resolved spec (Auto blocks filled in)
    params::P       # class-specific fitted content
    prob::Pr        # training problem, or nothing if keep_data = false
    retcode::ReturnCode.T
    stats::St       # FitStats(loglikelihood, dof, nobs, elapsed, history::NamedTuple of SPSM FitHistory)
end
gen[:TX]                    # fitted block (curve, bic, dof, show)
successful_retcode(gen)
```

`default_solver(alg)` looks up `Base.get_extension(Core, :OptimJLExt)`. If OptimizationOptimJL is loaded, which the umbrella always does, it returns `LBFGS()`. Otherwise it throws a message that names the fix. This mirrors `DefaultODEAlgorithm` while sublibraries keep v4's rule of an explicit solver.

### 2.6 Simulation, ensembles, RNG, initial conditions

```julia
struct SimulationProblem{G<:WeatherGenerator,I,V} <: AbstractWeatherProblem
    gen::G; dates::Vector{Date}; init::I; given::V
end
SimulationProblem(gen, dates; init = Burnin(365), given = (;))
SimulationProblem(gen, years::AbstractUnitRange{<:Integer}; kw...) =
    SimulationProblem(gen, collect(Date(first(years)):Day(1):Date(last(years), 12, 31)); kw...)

solve(sp::SimulationProblem; nsim = 1, seed = nothing, rng = default_rng(), ensemble = EnsembleThreads(),
      output_func = nothing, save_latent = false, save_regimes = true, eltype = Float64) -> WeatherSimulation
simulate(gen, dates; init = Burnin(365), given = (;), kw...) = solve(SimulationProblem(gen, dates; init, given); kw...)
Base.rand(rng::AbstractRNG, gen::WeatherGenerator, dates) = solve(SimulationProblem(gen, dates); rng).u[1]
```

- **RNG.** This follows the SciML v7 keywords.
  - If `seed === nothing`, then `seed = rand(rng, UInt64)`.
  - Trajectory `i` uses `Xoshiro(hash(i, seed))`, so results do not depend on the thread count.
  - `output_func(traj, ctx) -> (value, rerun)` receives `ctx.i`, `ctx.rng` and `ctx.seed`. It can stream each trajectory into a crop model without storing it.
  - `fit` takes the same `rng`/`seed` keywords. They matter for stochastic estimators such as WGEN's moment matching and slice initialisation.
- **Initial conditions.** There is one keyword, `init`, and it always means **the state before `dates[1]`**, generalising `Y_past`.
  - `Burnin(365)` (default): start from the fitted chain and blocks and discard one year.
  - `Date(1999,12,31)`: read the last p days ending on that date from `gen.prob`.
  - `(regime = 2, RR = Bool[…], TX = […])`: explicit values. Each entry is a D-vector, or D×p for memory p.
  - `given = (regime = path,)` conditions the simulation on a regime path, for example weather types from a climate model.

### 2.7 Streaming integrator (crop coupling)

```julia
integ = init(SimulationProblem(gen, Date(2030):Day(1):Date(2031,12,31)); seed = 1)
integ.t; integ.u; integ.regime     # last completed date; (RR = Vector(D), TX = …); Int or nothing
step!(integ); step!(integ, Day(7))
for (u, t) in tuples(integ) crop_step!(crop, u.RR[1], u.TX[1], t) end
reinit!(integ; init = (regime = 2,), seed = 2); terminate!(integ)
set_u!(integ, (RR = observed_rr,))  # [speculative] overwrite today; later days condition on it
```

### 2.8 Output contract

```julia
struct WeatherSimulation{G,A<:NamedTuple,Z}
    gen::G; dates::Vector{Date}; sites::SiteTable
    data::A          # (RR = Array{T,3} D×N×nsim, …)
    regimes::Z       # Matrix{Int} N×nsim or nothing
    seed::UInt64; elapsed::Float64
end
sim[:TX]                # D×N×nsim, no copy
sim.u[i]                # WeatherTrajectory of views, Tables-compatible, as SciML v7 `sim.u[j]`
DataFrame(sim)          # columns: simulation, DATE, STAID, RR, TN, TX, QQ, ETPP[, regime]
Tables.partitions(sim)  # one lazy partition per (simulation, site): one STICS file each
latent(sim, :TX)        # when save_latent = true (replaces output_ϵ)
```

### 2.9 Likelihood, decoding, validation, save/load, remake

```julia
loglikelihood(gen); loglikelihood(gen, prob; by = :total | :block)   # StatsAPI; errors if !has_likelihood
aic(gen); bic(gen; by = :total | :block); dof(gen); nobs(gen); coef(gen)
decode(gen, prob = gen.prob, Viterbi())    -> RegimePath           # Tables: DATE, regime
decode(gen, prob, Posterior())             -> PosteriorRegimes     # N×K + Tables: DATE, p1..pK
validate(sim, prob; statistics = default_statistics(prob)) -> ValidationReport  # Tables + plot recipe
#   statistics: MonthlyMean(), MonthlyTotal(), MonthlyQuantile(q), SpellLength(:dry), SiteCorrelation(),
#               VariableCorrelation(), AnnualMaximum(), VCX3()
summarize(sim; quantiles = (0.05, 0.5, 0.95))   # EnsembleSummary analogue, for band plots
write_generator(path, gen); read_generator(path)  # JSON: format tag, package versions, spec text, plain arrays
remake(prob; sites, variables, dates); remake(alg; kw...); remake(gen; blocks...); remake(sp; dates, init)
```

`save`/`load` are not used because CairoMakie exports `save`, and the crop tutorial uses it. Files are written as JSON of plain arrays, not JLD2 of Julia types, so they survive refactors.

### 2.10 Missing data and gaps

- The grid is regular, and `missing` is kept. Nothing is imputed implicitly; Data readers accept `impute = :interpolate` only when asked.
- Marginal M-steps use the observed entries.
- A dynamics term at (s, n) requires both n and n−1 observed at s. This is a conditional likelihood: gaps are never bridged, which fixes `cov_ar1` and the TODO at `fit_mle_AR.jl:1`.
- In the regime E-step, a missing emission has log-density 0. This is exact when sites are independent given the regime; copula blocks marginalise to the observed sub-vector. The chain therefore runs through gaps with no sequence cutting. `seq_ends` appear only when problems from separate periods are concatenated.
- Copula estimation uses pairwise-complete data.
- Simulation never produces `missing`.

## 3. Three tiers, by example

### T0 — impact modeler (crop tutorial)

```julia
using StochasticWeatherGenerators, DataFrames
data = read_stations(INRAE(), [49215002, 80557001, 40272002, 63345002])   # DATE, STAID, RR, TN, TX, QQ, ETPP
gen  = fit(data, RegimeHMM(regimes = pretrained(:France37_K4_d1_m1)))    # the paper's regimes, frozen
sim  = simulate(gen, 1986:2035; nsim = 300, seed = 50000)
df   = DataFrame(sim)                 # 300 × 4 × 18 262 rows
validate(sim, data)                   # monthly quantiles, spells, correlations vs observations
```

Without pretrained regimes, write `fit(data, RegimeHMM(K = 4))` instead. `show(gen)` prints the resolved spec, which T1 users can copy and edit:

```
WeatherGenerator{RegimeHMM}  4 sites × 5 variables, 1956-01-01 … 2019-12-31   retcode: Success
  regimes = freeze(France37_K4_d1_m1)            K = 4, transitions Periodic(degree = 1)
  RR   = Rain(Occurrence(memory = 1, degree = 1), PeriodicMixture(Exponential => 1, Exponential => 1),
              Multisite(GaussianCopula; by = :regime, estimator = PairwiseRank()))
  TX   = Variable(PeriodicAR1(μ = 1, ρ = 1, σ = 1); spatial = Multisite(GaussianCopula))
  TN   = Variable(Periodic(Gamma; α = 1, θ = 1); transform = Below(:TX), coupling = LatentCorrelation(:TX; degree = 1))
  QQ   = Variable(LatentAR1(Periodic(Gamma; α = 1, θ = 1); ρ = 1); spatial = Multisite(GaussianCopula))
  ETPP = Variable(LatentAR1(Periodic(Gamma; α = 1, θ = 1); ρ = 1); spatial = Multisite(GaussianCopula))
  estimation = TwoStage(decode = Viterbi())   solver = LBFGS()   loglik = …   BIC = …
```

### T1 — applied statistician

```julia
using StochasticWeatherGenerators, Distributions
using OptimizationOptimJL: LBFGS

paper = read_generator("france37_K4_d1_m1.swg.json")     # fitted on the 37 ECA stations
prob  = WeatherProblem(data; sites = station_meta)         # STAID, LAT, LON, ALT

alg = RegimeHMM(
    regimes = freeze(paper),                               # chain A(t) + stored path/posterior by Date
    blocks = (
        RR = Rain(occurrence = Occurrence(memory = 1, degree = 1),
                  amount  = PeriodicMixture(Exponential => 1, Gamma => (α = 0, θ = 2)),  # per-component, per-parameter degree
                  spatial = Multisite(GaussianCopula; by = :regime, estimator = PairwiseRank())),
        TX = Variable(PeriodicAR1(μ = 2, ρ = 1, σ = 1);                  # AR on the variable (form A)
                      spatial = Multisite(GaussianCopula; degree = 1)),   # C_k(t): regime × season
        TN = Variable(Periodic(Gamma; α = 1, θ = 2);                      # law of ΔT = TX − TN
                      transform = Below(:TX), coupling = LatentCorrelation(:TX; degree = 1)),
        QQ   = Variable(LatentAR1(Periodic(Gamma; α = 1, θ = 1); ρ = 1)),   # latent AR (form B): positive
        ETPP = Variable(LatentAR1(Periodic(Weibull; α = 1, θ = 1); ρ = 0)),
    ),
    estimation = TwoStage(decode = Posterior()),            # soft γ weights instead of one-hot ẑ
    solver = LBFGS())

gen = fit(prob, alg)
bic(gen; by = :block)                                       # (RR = …, TX = …, TN = …, QQ = …, ETPP = …)
curve(gen[:TX], :ρ; site = 2, regime = 3)                   # 366-vector

alt = remake(alg; blocks = merge(alg.blocks, (TN = Variable(Periodic(LogNormal; μ = 1, σ = 1);
                     transform = Below(:TX), coupling = LatentCorrelation(:TX)),)))
bic(fit(prob, alt)[:TN]) < bic(gen[:TN])                    # compare one block, everything else identical

# Retrain the regimes on another network instead of freezing them
eca = WeatherProblem(read_stations(ECA(), staid37); variables = [:RR])
alg2 = remake(alg; regimes = Regimes(4; degree = 1, memory = 1, data = eca))

# Add a station: same spec, warm start
prob5 = WeatherProblem(vcat(data, read_stations(INRAE(), [21231001])); sites = meta5)
gen5  = fit(prob5, alg; warmstart = gen)
```

### T2 — expert: a new block and a new model class

**Block protocol** (Blocks, all `public`):

```julia
abstract type AbstractBlock end; abstract type AbstractFittedBlock end
parents(b::AbstractBlock) = ()                                              # DAG edges
fit_block(b, y::BlockData, W::AbstractMatrix{<:Real}, ctx::FitContext)     # W: N×K (one-hot or γ) -> fitted
scores(f, y::BlockData, ctx)          # D×N normal scores / innovations fed to the Multisite layer
init_state(f, past, ctx)              # from the `init` convention
draw!(out, state, f, k::Int, t::Int, ε::AbstractVector, parents::NamedTuple, rng)   # one step; ε ~ N(0, C_k(t))
logdensities!(LL::AbstractMatrix, f, y, ctx)   # optional: N×K; enables JointEM and decoding
```

**New block: QQ bounded by clear-sky radiation.** This addresses the `others.md` TODO on positive variables.

```julia
import StochasticWeatherGeneratorsBlocks as B
using SmoothPeriodicStatsModels: LatentAR1, Periodic

struct ClearSkyFraction{L} <: B.AbstractBlock; law::L; end      # QQ = Qmax_s(t) · Y, Y ∈ (0,1)
struct FittedClearSky{F,Q} <: B.AbstractFittedBlock; inner::F; qmax::Q; end

function B.fit_block(b::ClearSkyFraction, y::B.BlockData, W, ctx)
    qmax  = [clear_sky(lat, t) for lat in ctx.sites.LAT, t in 1:ctx.T]   # user formula, D×T
    inner = B.fit_block(B.Variable(b.law), B.BlockData(y.values ./ qmax[:, ctx.n2t], y.mask), W, ctx)
    return FittedClearSky(inner, qmax)
end
B.scores(f::FittedClearSky, y, ctx) = B.scores(f.inner, B.BlockData(y.values ./ f.qmax[:, ctx.n2t], y.mask), ctx)
B.init_state(f::FittedClearSky, past, ctx) = B.init_state(f.inner, past ./ f.qmax[:, ctx.tprev], ctx)
function B.draw!(out, st, f::FittedClearSky, k, t, ε, parents, rng)
    B.draw!(out, st, f.inner, k, t, ε, parents, rng)
    out .*= view(f.qmax, :, t)
end

alg = RegimeHMM(K = 4, blocks = (QQ = ClearSkyFraction(LatentAR1(Periodic(Beta; α = 1, β = 1); ρ = 1)),))
```

A truncated AR(1) needs no new type in v5, because SPSM covariate families cover it: `Variable(Periodic((μ, ρ, σ, x) -> truncated(Normal(μ + ρ*x, σ); lower = 0); μ = 1, ρ = 1, σ = 1, covariate = Lag(1)))`.

**New model class: `lib/StochasticWeatherGeneratorsKNN`.** This is a sketch; the adjacency checks are omitted.

```julia
module StochasticWeatherGeneratorsKNN
using StochasticWeatherGeneratorsCore, Dates, Random, Statistics
import StochasticWeatherGeneratorsCore as C
export KNNResampler

Base.@kwdef struct KNNResampler <: C.AbstractWeatherModel
    k::Int = 10
    window::Int = 15                           # ± period indices
end
struct KNNParams{A}; data::A; feature::Matrix{Float64}; n2t::Vector{Int}; end
mutable struct KNNState; j::Int; end

function C.fit(prob::C.WeatherProblem, alg::KNNResampler; kwargs...)
    keep = C.complete_days(prob)
    data = map(x -> Float64.(x[:, keep]), C.data(prob))
    f = reduce(vcat, (mean(x; dims = 1) for x in values(data)))
    f = (f .- mean(f; dims = 2)) ./ std(f; dims = 2)
    C.WeatherGenerator(alg, KNNParams(data, f, C.period_index(prob)[keep]), prob; retcode = C.ReturnCode.Success)
end
C.initialize_state(gen::C.WeatherGenerator{KNNResampler}, sp, rng) =
    KNNState(rand(rng, findall(==(C.period_index(gen, sp.dates[1] - Day(1))), gen.params.n2t)))
function C.advance!(today, st::KNNState, gen::C.WeatherGenerator{KNNResampler}, s::C.StepContext, rng)
    p, T, a = gen.params, C.period_length(gen), gen.alg
    near(t) = min(abs(t - s.tprev), T - abs(t - s.tprev)) <= a.window
    cand = [j for j in 1:size(p.feature, 2)-1 if near(p.n2t[j])]
    d = [sum(abs2, view(p.feature, :, j) .- view(p.feature, :, st.j)) for j in cand]
    nn = cand[partialsortperm(d, 1:a.k)]
    w = cumsum(1 ./ (1:a.k)); st.j = nn[findfirst(>=(rand(rng) * w[end]), w)] + 1
    foreach(v -> today[v] .= view(p.data[v], :, st.j), keys(today))
end
end
```

With only this code, `simulate`, ensembles, `init`/`step!`, Tables output, `validate` and plots all work.

## 4. Flagship Regime-HMM product model

### 4.1 Spec objects

```julia
struct RegimeHMM{R<:AbstractRegimes,B<:NamedTuple,E<:AbstractEstimation,S} <: AbstractWeatherModel
    regimes::R; blocks::B; estimation::E; solver::S
end
RegimeHMM(; K = 4, degree = 1, memory = 1, regimes = Regimes(K; degree, memory),
          blocks = (;), estimation = TwoStage(), solver = nothing)

Regimes(K; degree = 1, from = :RR, memory = 1, spatial = Independent(), data = nothing, init = SliceEM())
freeze(gen_or_regimes)      # FrozenRegimes: chain + path + posterior by Date; Stage 1 skipped
ObservedRegimes(table)      # given labels (e.g. weather types); chain A(t) fitted on the labels
Rain(; occurrence = Occurrence(), amount = PeriodicMixture(Exponential => 1, Exponential => 1),
       spatial = Multisite(GaussianCopula; estimator = PairwiseRank()), threshold = 0.0)
Occurrence(; memory = 1, degree = 1, spatial = Independent())   # or Multisite(ExponentialKernel) = Cognot spatial HMM
Variable(law; spatial = Multisite(GaussianCopula), transform = Identity(), coupling = nothing)
Multisite(family; by = :regime, degree = 0, estimator = IFM())  # estimators: IFM, PairwiseRank, MomentMatching, PairwiseCensored
```

Data columns without a block get `default_block(kind)`, where the kind comes from the name or `kinds`:

| Kind | Default block | Equal to v3? |
|---|---|---|
| RR | `Rain()` | yes |
| TX/TG | `Variable(PeriodicAR1(μ=1, ρ=1, σ=1))` | yes |
| TN with TX present | `Below(:TX)` Gamma + `LatentCorrelation(:TX; degree=1)` | no: bug-fixed r scale (§8.3) |
| Positive (QQ, ETPP, FG) | `LatentAR1(Periodic(Gamma))` | no: replaces clipping |

An unknown kind raises an error that suggests `blocks = (X = Variable(…),)`.

### 4.2 How variables compose

Given z_n = k and t = t_n, blocks are processed in topological order of the DAG built from `transform` and `coupling` parents. A cycle is an error at construction. Each block b receives ε^b ~ N_D(0, C^b_k(t)) from its Multisite layer.

- **RR.** O_s = 1{U_s < p_{s,k,t}(h_s)}, with h the memory category. With a latent occurrence layer, the uniform is Φ(ε^occ_s). On the wet set, R_s = F⁻¹_{s,k,t}(Φ(ε^RR_s)). The full vector is drawn and the wet entries are kept. This is exact, because the marginal of N(0,C) on a subset W is N(0, C[W,W]). It removes `rand_RR`'s daily SklarDist rebuild and the hard-coded independence of regime K.
- **Form A** (`PeriodicAR1`): X_s = μ + ρ X_{s,n−1} + σ ε_s. This is the v3 behaviour.
- **Form B** (`LatentAR1`): W_s = ρ W_{s,n−1} + √(1−ρ²) ε_s, then X_s = F⁻¹_{s,k,t}(Φ(W_s)). Because W_n ~ N(0,1) whatever the regime path, the marginal is exactly F_{s,k,t} on every day. A positive F gives positive QQ and ETPP with no clipping.
- **TN|TX:** ξ_s = r_{s,k}(t) ε^TX_s + √(1−r²) ε^TN_s, then Δ_s = F⁻¹_Γ(Φ(ξ_s)), then TN = TX − Δ. So TN < TX always. r is estimated on the same pair (TX innovation, normal score of ΔT) that the simulation uses, which fixes the mismatch at `fit_mle_TN.jl:75-80` vs `:103`.
- Any block can declare `coupling = LatentCorrelation(:P)`. A future `LocalRegimes(K₂)` dynamics type, a nested per-block chain, is reserved as an `AbstractDynamics` subtype.

### 4.3 Estimation API and procedure

```julia
TwoStage(; decode = Viterbi())     # v3 behaviour: blocks fitted given ẑ (one-hot weights)
TwoStage(; decode = Posterior())   # blocks weighted by γ
JointEM(; maxiter = 100, tol = 1e-4, start = TwoStage())
```

**TwoStage:**
1. Resolve the Auto blocks.
2. Regimes:
   - if frozen, join the path or posterior by date;
   - otherwise run `initialize(…; method = SliceEM())` (the old `fit_mle_all_slices`), then fit `fit_mle(PeriodicHMM(K, BernoulliMemory(memory; degree)), Y_from; past, n2t, solver)` on the `from` occurrence of `data` (default: the problem itself);
   - then decode.
3. Set `W = OneHot(ẑ, K)` or γ.
4. For each block and site, threaded: `fit_mle(law, y_s, W; n2t, past, x, solver)` in SPSM.
5. Pass `scores` to Multisite: per-regime weighted correlation. Use IFM through Copulas' `fit(GaussianCopula, U; weights)`, or PairwiseRank (Spearman → 2 sin(πρ/6), then `nearest_cor`). With `degree > 0`, fit a weighted Gaussian likelihood through SPSM's `CorrelationLink`.
6. Fit couplings as the covariate family N(r x, √(1−r²)).
7. Assemble the `FitStats` with per-block histories. `retcode` is the worst across stages.

**JointEM:**
- LL = Σ_b `logdensities!` (N×K). SPSM's `forward_backward(chain, LL; n2t)` gives γ and ξ-stats.
- The chain is refitted on ξ and every block is refitted with W = γ.
- The log-likelihood is checked for monotonicity.
- Blocks without `logdensities!` are listed in the error message.
- Rain log-densities cache Cholesky factors per (k, wet-set bitmask).

### 4.4 Simulation algorithm

```
precompute: L[b,k] = cholesky(C^b_k) (lazily per t if degree > 0); topo order; buffers per thread
state ← initialize_state(gen, sp, rng)            # from `init`: z, occurrence history h, W/X of the day before
for n in 1:N
    t, tprev = n2t[n], n2t[n-1]
    z ← sample(A(tprev)[z, :])  (or given.regime[n])
    for b in topo
        ε  = L[b,z] * randn(rng, D_b)
        draw!(today[b], state[b], fitted[b], z, t, ε, (parents' ε and values), rng)
    end
    copy today → out[:, n, i]
end
```

The cost is O(Σ_b D_b²) per day. Trajectories are threaded through `ensemble`, each with its own `Xoshiro(hash(i, seed))`.

## 5. WGEN port and censored latent Gaussian

**WGEN**

```julia
Base.@kwdef struct WGEN{S,M,A} <: AbstractWeatherModel
    order::Int = 1
    seasonality::S = Monthly()                    # IndicatorBasis on months; Smooth(d) = trig degree d
    spatial::M = Multisite(GaussianCopula; estimator = MomentMatching(nsim = 2000))
    amount::A = nothing                           # Rain amount + Multisite → Wilks 1998 amount layer
end
```

- **Fit.**
  1. Per site, `fit_mle(BernoulliMemory(order; basis = Monthly()), y_s; past, n2t)`. With an indicator basis and no regimes, this MLE *is* v3's transition-count estimator, which allows an exact equivalence test.
  2. Ω_m pair by pair, by moment matching: an ITP root-find (BracketingNonlinearSolve, lighter than NonlinearSolve) on the simulated-vs-observed occurrence correlation, as `fit_Ω` does today, then `nearest_cor`.
  3. Optional amounts.
- **Simulate.** `advance!` draws ε ~ N(0, Ω_{m(t)}) and sets O_s = 1{ε_s ≤ Φ⁻¹(p_s(t, h_s))}. The integer-coded history h replaces the `OrderedDict` state, which addresses the "generation is slow" TODO.
- **Traits.** `has_likelihood = false` (moment matching), `has_regimes = false`.

**CensoredLatentGaussian**

```julia
Base.@kwdef struct CensoredLatentGaussian{R,T,S,D,O} <: AbstractWeatherModel
    regimes::R = nothing                            # Regimes(K) for Ailliot 2009 (HMM); nothing for a single field
    transform::T = Anamorphosis(PeriodicMixture(Exponential => 1, Exponential => 1))   # or PowerTransform()
    spatial::S = Multisite(GaussianCopula; estimator = PairwiseCensored())             # or ExponentialKernel
    dynamics::D = LatentAR1Dynamics(ρ = 1)
    other::O = (;)                                  # TX = Variable(…), …: the flagship blocks are reused as-is
end
```

- **Model.** W_n = ρ(t) W_{n−1} + √(1−ρ²) ε_n, with ε_n ~ N(0, C_k(t)). Then R_s = 0 if W_s ≤ c_{s,k}(t); otherwise R_s = F⁻¹((Φ(W_s) − Φ(c))/(1 − Φ(c))) with an anamorphosis, or a(W_s − c)^β for the power form. The exact Ailliot parametrisation is still to be checked against the paper. Occurrence and amount come from one field.
- **Fit.**
  1. Margins: c through a periodic probit for P(R > 0), and F on wet days, both weighted.
  2. ρ: pairwise censored likelihood on lag-1 pairs.
  3. C: pairwise censored likelihood on same-day pairs. Dry–dry pairs use Φ₂, mixed pairs φ·Φ(conditional), wet–wet pairs φ₂. Reuse `norm_cdf_2d_vfast` (SPSM `HHMM_Bernoulli_spatial/fast_bivariate_cdf.jl`, moved to Blocks).
  4. Regimes through the flagship's Stage 1.
- **Likelihood.** `loglikelihood` returns the composite (pairwise) value, flagged by `stats.likelihood_kind = :pairwise`. BIC is undefined; CLIC can come later.

**What the three classes share**

| Component (Blocks or SPSM) | RegimeHMM | WGEN | CLG |
|---|---|---|---|
| Memory coding h, `BernoulliMemory` | ✓ | ✓ | — |
| Latent threshold 1{ε ≤ Φ⁻¹(p)} | latent occurrence | ✓ | ✓ (W > c) |
| `Multisite` + estimators, `nearest_cor` | IFM, PairwiseRank | MomentMatching | PairwiseCensored |
| Quantile mapping F⁻¹(Φ(·)) | RR, form B, TN | amounts | anamorphosis |
| Latent AR(1) | `LatentAR1` | — | ✓ |
| Periodic laws (Trig/Indicator bases) | ✓ | Monthly | ✓ |
| `Regimes` | ✓ | — | optional |
| Driver, ensembles, integrator, I/O | ✓ | ✓ | ✓ |

Test: `WGEN(order = 1, spatial = Independent())` and `RegimeHMM(K = 1, …Monthly…)` must give identical occurrence probabilities.

## 6. SPSM v5

### 6.1 Engine contract

```julia
# m ∈ {PeriodicAR1, LatentAR1, PeriodicMixture, Periodic family, BernoulliMemory, PeriodicHMM, PeriodicMarkovChain}
fit_mle(m, y, w = nothing; n2t, past = nothing, x = nothing, seq_ends = nothing, solver, kw...) -> (m̂, FitHistory)
fit(PeriodicFit, m, y, w = nothing; kw...) -> PeriodicFit         # StatsAPI tier: aic/bic/dof
fit_mle!(m, y, w; kw...) -> m                                       # model-only refit, nestable in EM/HMM
loglikelihoods(m, y; n2t, past, x, seq_ends) -> Matrix N×K          # K = nstates(m); missing ⇒ 0
conditional(m, k, t; past = nothing, x = nothing) -> Distribution   # one-step law (streaming)
rand([rng], m, n2t; states = nothing, past = nothing, x = nothing)  # one argument order everywhere
curve(m, name, ts = 1:T); m(t); m(k, t)
forward_backward(chain, LL; n2t, seq_ends) -> (γ, ξstats, loglik)   # product models plug their own LL in
viterbi(chain, LL; n2t, seq_ends); fit_mle(chain, ξstats; n2t)
```

- The third positional argument is **always weights**: `nothing`, an N-vector, an N×K matrix, or `OneHot(z, K)`, a lazy Bool matrix. "Given z" is therefore one-hot weights, and Stage 2, soft γ and JointEM share one code path.
- `past` is the single initial-condition convention.
- `x` holds covariates: `Lag(1)`, a parent variable, or a categorical such as wet/dry.
- Every M-step minimises the average weighted NLL (constraint 6).
- The chain-level functions remove the dependence on the undocumented `HMMs.obs_logdensities!` for product models.

### 6.2 Vector links and covariates: yes

- `ParamSpec` gains vector-valued parameters.
  - `SoftmaxLink(K)`, with reference K, replaces the 4 softmax copies and `mixture_weight_spec`.
  - `CorrelationLink(d)` maps d(d−1)/2 canonical partial correlations tanh(B(t)β) through an onion/LKJ Cholesky to a correlation matrix that is positive definite for every t. It enables C_k(t).
- **Covariate families**, a closure with covariate argument `x`, unlock any AR noise law, truncated AR, AR(p), regression TN|TX and Richardson-style wet/dry conditioning. The fast sufficient-statistics AR1 path is kept as a specialised `update_dist`.
- Links are inferred from parameter support (or name for closures): σ, α, θ → log; ρ, r → (−1,1); p → logit. A pair `σ = 1 => LogLink()` overrides.
- TVP stays vendored for 5.0, since vector links and covariates are needed fast. Moving to the registered TimeVaryingParameters can be revisited in 5.x.

### 6.3 v4 exports (94) → v5

- **Kept exported:**
  - bases: `TrigBasis ConstantBasis FourierBasis PolynomialBasis IndicatorBasis BSplineBasis CyclicBSplineBasis`;
  - links: `IdentityLink LogLink LogitLink IntervalLink`;
  - `ParamSpec TimeVaryingFamily curve fit_mle fit loglikelihoods viterbi FitHistory converged components ncomponents`;
  - StatsAPI: `coef loglikelihood dof nobs aic aicc bic`.
- **New exports:** `Periodic SoftmaxLink CorrelationLink PeriodicMarkovChain BernoulliMemory SpatialBernoulliMemory LatentAR1 OneHot Lag conditional forward_backward`. That gives about 47 exports in total.
- **`public` only:** `paramnames design_matrix ncoef blockindex blockrange nbasis coefs PeriodicDesign PeriodicNLL PointwiseNLL group_by_time distributions update_dist ClassicEM StochasticEM ncolumns column parameters default_nlls_solver default_adtype default_ensemble solve_unconstrained solve_nlls ar1_spec ar1_sufficient_statistics ar1_nll component_coefs component_range mixtures PeriodicHMMDists emission_coefs emission_range degrees ARPeriodicHMM sARPeriodicHMM ARPeriodicHMMSpatial AR1 SeasonalControls fit_objective`.
- **Removed:**
  - `TimeVaryingFit`, merged into PeriodicFit;
  - `mixture_weight_spec`;
  - `Trig2ARPeriodicHMM`, `Trig2ARPeriodicHMMSpatial`, `randARPeriodicHMM`;
  - `fit_θ fit_θ! fit_θᴬ! fit_θᴮ! fit_mle_RO μₜ ρₜ σₜ αₜ params nparams`;
  - `fit_loss_optim`, which becomes `public fit_objective`.
- **Renamed:**
  - `SmoothAR1→PeriodicAR1`, `SmoothSeasonalMixture→PeriodicMixture`;
  - `SmoothPeriodicHMM`, `SmoothARPeriodicHMM`, `SmoothsARPeriodicHMM` and `SmoothARPeriodicHMMSpatial` all become `PeriodicHMM(K, emission)`, with emission `Periodic(…)`, `BernoulliMemory(order or orders)` or `SpatialBernoulliMemory(order; distances)`;
  - `seasonal_family→Periodic`, `SeasonalFit→PeriodicFit`;
  - `mixture_weights→curve(m, :weights)`;
  - `fit_mle_all_slices→initialize(m0, Y; method = SliceEM())`, which returns a smooth model;
  - `n_to_t→period_index(N, T)`.

"Periodic" is chosen over "Seasonal" because SPSM is domain-agnostic: genomic positions are not seasons.

### 6.4 Moves

- **SPSM → SWG Blocks:**
  - the multisite `rand(::Array{AR1}, n2t, z, Σ; y₁)` (`AR1.jl:123-156`, untested, ignores Σ);
  - the Copulas dependency;
  - `norm_cdf_2d_vfast`.
- **SpatialBernoulli and MvNormalCDF** go to Cognot's SpatialBernoulli.jl, which both packages use.
- **SWG → SPSM:**
  - `_ar1_init`, as PeriodicAR1's default start;
  - `mix_ini`, as PeriodicMixture's default start;
  - `_fit_seasonal_gamma` and `cor_groupby`, which become plain `fit_mle(Periodic(…), y, w)` (with a covariate family for r).
- **SWG Blocks keeps** `Σ_Spearman2Pearson`/`Kendall`, with an upstream PR to Copulas.jl as an option.
- **SPSM drops** PeriodicHiddenMarkovModels, which it uses only for `n_to_t`.

## 7. Naming and verbs

| Concept | New name | Replaces (v3/v4) |
|---|---|---|
| observed data | `WeatherProblem` | vector of DataFrames + innerjoin(:z) + n2t |
| model class | `RegimeHMM`, `WGEN`, `CensoredLatentGaussian` | hand assembly; `wgen` struct |
| estimation | `fit(prob, alg)` | fit_mle_RO/RR, fit_AR1, fit_TN, cov_RR, cov_ar1, cor_groupbyTXTN, fit_wgen, fit_Ω |
| fitted generator | `WeatherGenerator` | (hmm, mix_allE, Σ²RR, ar1sTX, Σ²TX, θ_ΔT, θ_cor, …) |
| simulation | `simulate` / `solve(SimulationProblem)` / `rand` | rand(hmm…), rand_RR, rand(ar1s…), rand_cond, simulate_markov_gaussian |
| ensemble size | `nsim` | NSIMU loops |
| output | `WeatherSimulation`, `sim[:TX]`, `DataFrame(sim)` | zs/ys/rs/txs arrays + Dict + DataFrames |
| streaming | `init`, `step!`, `WeatherIntegrator` | — |
| initial state | `init` (state before day 1) | Y_past, y₁, z_ini, Y_ini, tx_ini |
| RNG | `seed`, `rng` keywords | `Random.seed!` |
| calendar | `LeapCalendar`, `period_index` | dayofyear_Leap, n_to_t |
| regimes | `Regimes`, `freeze`, `ObservedRegimes` | randARPeriodicHMM … SmoothARPeriodicHMM, hmm_fit_full.A |
| decoding | `decode(gen, prob, Viterbi())` | viterbi |
| periodic law | `Periodic(D; θ = degree)` | seasonal_family, TimeVaryingFamily sugar |
| AR forms | `PeriodicAR1`, `LatentAR1` | SmoothAR1/AR1; `indicatrix` |
| mixture | `PeriodicMixture` | SmoothSeasonalMixture, mix_ini |
| occurrence | `Occurrence` / `BernoulliMemory` | fit_mle_RO, B arrays |
| rain block | `Rain` | fit_mle_RO + fit_mle_RR + cov_RR + rand_RR |
| variable block | `Variable` | fit_AR1 + cov_ar1 |
| TN given TX | `Below(:TX)`, `LatentCorrelation` | fit_TN, cor_groupbyTXTN, rand_cond, txs − ΔTs |
| spatial dependence | `Multisite(GaussianCopula; by, degree, estimator)` | Σ² matrices, cor2cov, fit_Ω |
| estimation mode | `TwoStage`, `JointEM` | implicit |
| read-out | `curve(x, name; site, regime)` | μₜ…, mixture_weights |
| diagnostics | `gen.retcode`, `gen.stats`, `bic(gen; by)` | (m, history) tuples, TimeVaryingFit |
| validation | `validate`, statistic types | monthly_agg, longuest_spell, VCX3 |
| persistence | `write_generator`, `read_generator` | JLD.save + ẑ CSV |
| data | `read_stations(ECA()/INRAE()/MeteoFrance(), ids)` | collect_data_* |
| weights / given z (SPSM) | `fit_mle(m, y, OneHot(z, K))` | 3rd positional = z / weights / Y_past |

## 8. Migration and release plan

### 8.1 Ordered steps (each PR mergeable alone)

0. **Golden files** (before anything else). Add `test/reference/generate.jl` with a committed Manifest pinned to SPSM 4.1 / SWG 3.0. It runs the crop, paper and WGEN pipelines on bundled data and saves parameters, log-likelihoods, ẑ and simulation statistics as CSV/Arrow. It also converts the `assets/tuto_1` JLD into the `pretrained` artifact.
1. SPSM: vector-valued `ParamSpec` + `SoftmaxLink`, dedup of the softmaxes. Internal only.
2. SPSM: covariate families. Additive.
3. SPSM: weights everywhere (`OneHot`), N×K `loglikelihoods` for AR1, `seq_ends`/`missing` for AR1 and mixtures. Additive.
4. SPSM: `PeriodicMarkovChain`, `forward_backward`/`viterbi` on LL, `conditional`, `LatentAR1`, `CorrelationLink`.
5. SPSM: unified `rand` order, renames, export cut, `public`, the `PeriodicHMM(K, emission)` front end, removals. **Breaking**; main becomes 5.0.0‑DEV.
6. SPSM: docs, tutorials (genomics included), migration page.
7. SWG: `lib/…Core` with a test-only toy class. Root stays v3, untouched.
8. SWG: `lib/…Data`, with the precompile-download fix and `pretrained`.
9. SWG: `lib/…Blocks`, depending on SPSM through `[sources] {url=…, rev="main"}`.
10. SWG: `lib/…RegimeHMM`, TwoStage plus the equivalence suite; then JointEM.
11. SWG: `lib/…WGEN` + equivalence.
12. SWG: `lib/…CensoredGaussian`, with simulate→refit recovery tests.
13. SWG: the root becomes the umbrella 4.0.0‑DEV, old `src/` is deleted, the three tutorials are rewritten, and the migration guide is added.
14. Release day: register SPSM 5.0.0, drop the url `[sources]`, register the sublibraries in dependency order, register umbrella 4.0.0, and announce both releases with one migration guide.

### 8.2 Old → new (most-used calls)

| v3 / v4 | v4 / v5 |
|---|---|
| `randARPeriodicHMM(K,T,D,m; ξ, ref_station)` | `Regimes(K; memory=m)` · `PeriodicHMM(K, BernoulliMemory(m; degree=d))` |
| `fit_mle_all_slices(…)` + `fit_θ!(…)` + `SmoothARPeriodicHMM(a,θᴬ,θᴮ,T)` | `initialize(m0, Y; past=Y_past, n2t, method=SliceEM())` |
| `fit_mle(hmm, Y, Y_past; solver, n2t)` | `fit_mle(m0, Y; past=Y_past, n2t, solver)` |
| `ARPeriodicHMM(hmm_fit_smooth)` | not needed: `m(t)`, `curve(m, :A)` |
| `viterbi(hmm, Y, Y_past; n2t)` | `viterbi(m, Y; past, n2t)` · `decode(gen, prob)` |
| `rand(hmm, n2t; Y_past, z_ini, seq=true)` | `rand(rng, m, n2t; past, states)` · `simulate(gen, dates)` |
| `fit_mle_RO(df, m, d)` + `ARPeriodicHMM(a, A, B)` | `Rain(occurrence=Occurrence(memory=m, degree=d))` · `fit_mle(BernoulliMemory(m; degree=d), Y, OneHot(z,K); past, n2t)` |
| `fit_mle_RR(df, m; mix₀=mix_ini(T))` | `Rain(amount=PeriodicMixture(Exponential=>1, Exponential=>1))` |
| `cov_RR(dfs, K)` | `Multisite(GaussianCopula; estimator=PairwiseRank())` |
| `fit_AR1(df, :TX, d)` | `Variable(PeriodicAR1(μ=d, ρ=d, σ=d))` |
| `cov_ar1(dfs, ar1s, :TX)` | `spatial = Multisite(GaussianCopula)` |
| `fit_TN(df,1,T)` + `cor_groupbyTXTN(df,T)` | `Variable(Periodic(Gamma; α=1,θ=1); transform=Below(:TX), coupling=LatentCorrelation(:TX; degree=1))` |
| `rand_RR(…)`, `rand(ar1s,…; y₁)`, `rand_cond(…)` | `simulate(gen, dates; init)` |
| `output_ϵ = true` | `save_latent = true`; `latent(sim, :TX)` |
| `indicatrix.(rand(ar1sQQ, …))` | `Variable(LatentAR1(Periodic(Gamma)))` |
| `dayofyear_Leap.(d)` / `n_to_t(N,T)` | `period_index(LeapCalendar(), d)` / `period_index(N, T)` |
| `fit_wgen(Y, idx_months, p)` | `fit(prob, WGEN(order=p))` |
| `rand(wgen, years; Y_ini)` | `simulate(gen, years; init)` |
| `collect_data_ECA(…)` / `collect_data_INRAE(…)` | `read_stations(ECA(), ids)` / `read_stations(INRAE(), ids)` |
| `SmoothAR1(d,T; K, init)` + `fit_mle(m, y, z; n2t)` | `fit_mle(PeriodicAR1(μ=d,ρ=d,σ=d), y, OneHot(z,K); n2t, past, solver)` |
| `SmoothSeasonalMixture(Exponential,2,d,T)` | `PeriodicMixture(Exponential=>d, Exponential=>d)` |
| `seasonal_family(Gamma, TrigBasis(T,d))` + `PeriodicNLL` + `fit_loss_optim` | `fit_mle(Periodic(Gamma; α=d, θ=d), y, w; n2t, solver)` |
| `fit(SeasonalFit, m, y)` / `TimeVaryingFit` | `fit(PeriodicFit, m, y, w)` |
| `μₜ(t/T, θ)`, `mixture_weights(m)` | `curve(m, :μ)`, `curve(m, :weights)` |
| `SmoothARPeriodicHMMSpatial(…)` | `PeriodicHMM(K, SpatialBernoulliMemory(m; distances))` · `Regimes(K; spatial=Multisite(ExponentialKernel))` |
| `JLD.save(…)` + ẑ CSV | `write_generator` / `read_generator` |
| `longuest_spell`, `monthly_agg`, `VCX3` | `validate(sim, prob; statistics=(SpellLength(:dry), MonthlyTotal(), VCX3()))` |

### 8.3 Numerical-equivalence strategy

The checks compare against the golden files from step 0, using the same initial values and solver.

- **E1:** regime HMM θᴬ and θᴮ to rtol 1e‑6, log-likelihood equal, ẑ identical.
- **E2:** occurrence given ẑ equal, since it is the same M-step.
- **E3:** RR mixtures to rtol 1e‑5.
- **E4:** AR1 TX, QQ and ETPP to rtol 1e‑8.
- **E5:** WGEN transition probabilities exact; Ω exact when the same RNG stream is used.

Intentional differences are kept in a test-only `Legacy` preset, so that equivalence is asserted *with* the old behaviour and each fix is then measured separately:
- the gap bridging in `cov_ar1`;
- the `groupby` order in `fit_TN`;
- the raw-scale r;
- regime-K independence in `rand_RR`;
- the `y₁` convention;
- clipping;
- TN spatial independence.

Simulations cannot match bitwise, because the day-major draw order differs from v3's variable-major order. Two checks replace that:
- **(a)** per-block `draw!` fed the same ε and z reproduces the v3 formulas exactly;
- **(b)** a distributional check: 1000 × 10-year runs of both versions, comparing monthly means, quantiles, spell distributions and inter-site/inter-variable correlations, with |z| < 4 on the Monte-Carlo error.

## 9. Risks, trade-offs, open questions

1. **Alg = model semantics.** SciML users expect algorithms to converge to the same answer. Mitigations: traits, `show` of the resolved spec, and errors such as "WGEN has no likelihood; use validate".
2. **Seven packages plus SPSM, released in lockstep.** This brings registration ordering, compat cascades and a CI matrix. Mitigations: `[workspace]`, a release script, and SciML-style test groups. Fallback: fold Blocks into RegimeHMM if WGEN and CLG ended up not reusing it. They do reuse it, so Blocks stays separate.
3. **SPSM in a separate repo** means cross-repo PRs during the rewrite. Mitigations: url `[sources]` and SPSM's Downstream CI. The alternative is `lib/SmoothPeriodicStatsModels` inside the monorepo. That would give atomic PRs but cost SPSM its independent identity and its non-weather users.
4. **SciMLBase dependency.** It adds load time, and v3 just changed ensemble internals and `AbstractVectorOfArray` semantics. We reuse only stable names and own the ensemble loop. Open question: should Core instead be CommonSolve-only with its own `ReturnCode`? That would load faster, but `remake` would clash with DifferentialEquations users. A related open question: should `SimulationProblem <: AbstractSciMLProblem`, so that SciMLBase's own `EnsembleProblem` works? *(Unverified.)*
5. **JointEM cost** with copula blocks and 37-site wet subsets (factorisation cache). Ship TwoStage as the default and mark JointEM experimental in 4.0.
6. **Results change** because of fixes and better defaults (positive variables, r scale, TN spatial). The changes must be listed in the release notes, with the `Legacy` presets in tests.
7. **Memory.** 300 × 18 262 × 4 × 5 values is about 0.9 GB in Float64. Mitigations: `eltype = Float32`, lazy Tables views, and `output_func` streaming.
8. **Copulas.jl** puts the dimension in the type and requires d ≥ 2. Blocks uses Copulas only for estimation; hot loops use Cholesky and `quantile` directly.
9. **Naming.**
   - The long sublibrary names (`SWGCore` would be shorter, but General discourages acronyms).
   - `Periodic` vs `Seasonal`.
   - `PeriodicHMM` collides with PeriodicHiddenMarkovModels.
   - `nsim` vs SciML's `trajectories`.
   - `fit(::Any, ::AbstractWeatherModel)` vs Distributions' `fit(::Type{<:Distribution}, ::Any)` is a method ambiguity; add a narrow method and run Aqua.
10. **Defaults.** The umbrella's default `LBFGS()` hides the solver choice, so `show` prints it. Whether `fit(prob)` without a model class should exist at all is open.
11. **Room to grow without breaks** (reserve before 4.0):
    - `StepContext`/`timestep`, and gridded `SiteTable`, for AWE‑GEN‑2d;
    - `ObservedRegimes`/`ClusteredRegimes` for MSTWeatherGen weather types;
    - `covariates` keywords on `WeatherProblem` and `simulate` for non-stationary (climate-change) runs;
    - `LatentVAR1` for GWEX's MAR(1);
    - an EGPD `Periodic` family;
    - callbacks.

    Each of these is additive, provided the names are reserved now.
12. **Unverified items carried over** from the synthesis brief: the actual `groupby` order, how fast a Bernoulli `update_dist` is, and Copulas' weighted fit on degenerate regimes.

### Critical Files for Implementation
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\examples\tuto_add_station_variable.jl
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\TimeVaryingParameters\spec.jl (with family.jl, fit.jl)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\AR\AR1.jl
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\HMM_Dists\smooth.jl (with src\fit\smooth_hmm.jl, src\hmm_interface.jl)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\src\rain\wgen.jl (with ext\SmoothPeriodicStatsModelsExt\fit_mle_TN.jl, src\rain\generation.jl)

Sources: [OrdinaryDiffEq.jl lib/](https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib), [OrdinaryDiffEqTsit5 Project.toml](https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/lib/OrdinaryDiffEqTsit5/Project.toml), [OrdinaryDiffEq umbrella module](https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/src/OrdinaryDiffEq.jl), [SciML v7 post](https://sciml.ai/news/2026/07/28/OrdinaryDiffEqv7/index.html), [DiffEq v8 / ODE v7 migration guide](https://docs.sciml.ai/DiffEqDocs/dev/migration/ordinarydiffeq_v7/).