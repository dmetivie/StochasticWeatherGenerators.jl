# SWG 2.0 + SPSM 5.0: a generic architecture for weather generators, launched with today's models

## Context

The multisite multivariable generator in StochasticWeatherGenerators (SWG) is assembled by hand in
`examples/tuto_add_station_variable.jl`. It uses about ten ad-hoc functions (`fit_mle_RO`, `fit_mle_RR`, `cov_RR`,
`fit_AR1`, `cov_ar1`, `fit_TN`, `cor_groupbyTXTN`, `rand_RR`, `rand_cond`, `rand(ar1s, n2t, z, Σ²; y₁)`). These return 6
different types, and the regime path, `n2t`, the initial values and the `ϵ_TX` side channel are all threaded by hand.

SmoothPeriodicStatsModels (SPSM) 4.0 unified `fit_mle(model, data...) -> (model, history)`, but several problems remain:
- It exports 94 names, about 40 of which are internal or legacy.
- The third positional argument of `fit_mle` means z, weights or `Y_past` depending on the model.
- `rand` takes `(n2t, z)` for AR(1) but `(z, n2t)` for HMMs.
- Only `SmoothAR1` can be fitted given a regime path z.
- Four HMM families duplicate code.

**Versions.** General has SPSM up to **4.1.0** and SWG only **1.3.1**. SWG `master` is labelled **1.4.0** and is called
**"legacy SWG"** below. The new architecture is **SWG 2.0.0**.

**Goal of this release: structure and genericity, not better models.**
- The release rebuilds the packages so that impact modelers (tables in, simulations out, in about 5 lines), applied
  statisticians (declare seasonal, regime-dependent blocks without internals) and students (a small, consistent
  vocabulary) are served.
- It **ships today's models**: the RegimeHMM multivariable model of the tutorials and WGEN. They are re-expressed in the
  new architecture, with code defects fixed and modelling choices kept.
- Improved and new models (censored latent AR for positive variables, generic zero atoms, the censored latent
  Gaussian, the codepaper spatial model, MSTWeatherGen, …) are **designed in, not built**:
  - every one of them has a named extension point reserved now (§4.2);
  - they arrive later as **additive** 2.x releases (§4.3).

**How this plan was built.**
- **Exploration:** 7 agents mapped the code, docs, upstream packages and prior art.
- **Design:** 4 independent designs, scored by 3 judges, merged by a synthesizer, then attacked by 3 critics (65 issues, all blockers and major issues resolved).
- **Follow-up research:** 3 agents covered monorepo practice, the codepaper model and MSTWeatherGen.
- **Implementation planning:** 10 agents wrote verified implementation plans; their findings are at the end.

Full documents: `SWG/docs/design/` on branch `evolution`.

### Decisions made by the author
1. **The composite multivariable model lives in SWG.** SPSM stays a domain-agnostic seasonal engine.
2. **SWG copies DifferentialEquations.jl on both axes:**
   - a monorepo of sublibraries in `lib/` plus an umbrella;
   - model classes swappable like solvers (`fit(data, RegimeHMM())` vs `fit(data, WGEN())`).
3. **Long term, SWG hosts many classes:** the censored latent Gaussian, the codepaper model, MSTWeatherGen, GWEX, GWGEN, WeaGETS, AWE-GEN-2d, k-NN.
4. **Launch scope:** today's RegimeHMM multivariable model and a port of WGEN. Every other model is deferred but must be addable without a breaking change.
5. **Faithful port:** code defects are fixed (§4.1); modelling choices are kept exactly. Every fix is listed in the migration guide.
6. **Lockstep clean break:** SWG 2.0.0 (sequential after 1.x) and SPSM 5.0.0, with a migration guide and no deprecation shims.
7. **Users:** impact modelers, applied statisticians and students.
8. **Data:**
   - Tables.jl tables in at the high level, arrays underneath.
   - Data are complete at launch (impute explicitly with Impute.jl, as today); multi-sequence fits for long gaps come later.
9. **Names stay consistent with SPSM:**
   - the `Smooth*` type names are kept;
   - the shared keywords are `K`, `season`, `memory`, `past` and `n2t`;
   - blocks mirror SPSM types (`AR1()` ↔ `SmoothAR1`, `Mixture` ↔ `SmoothSeasonalMixture`, a marginal ↔ `SmoothDistribution`).
10. **`validate` is deferred to 2.1.** The tutorials keep their current plotting code.
11. **Dependence is abstract.** Gaussian dependence is the only launch implementation; other copulas and space-time structures use the same slot later.
12. **Defaults never change silently.** Improved models arrive as new blocks, keywords or values, opt-in through `preset = :improved` in 2.x. A default switches only in a future major release.

---

## 1. Architecture

```text
StochasticWeatherGenerators.jl/            one repo; work happens on branch `evolution` (root = 2.0 umbrella, same UUID);
├── Project.toml                           `master` keeps legacy 1.x for hotfixes until the cutover
├── src/StochasticWeatherGenerators.jl     explicit `using X: a, b` + exports; data readers, climate indices, pretrained()
├── ext/  MakieExt (plot recipes), GeoMakieExt (station maps)
├── lib/
│   ├── StochasticWeatherGeneratorsCore/       interface: verbs, WeatherData, calendars, FittedGenerator,
│   │                                          WeatherSimulation, ensemble driver, persistence
│   ├── StochasticWeatherGeneratorsBlocks/     block protocol, ProductGenerator engine (DAG), Regimes, TwoStage,
│   │                                          today's blocks (Rain, Variable/AR1, Difference), MultisiteGaussian,
│   │                                          the reserved extension points (§4.2)
│   ├── StochasticWeatherGeneratorsRegimeHMM/  RegimeHMM: a preset over ProductGenerator = today's tutorial model
│   └── StochasticWeatherGeneratorsWGEN/       WGEN: lowers to ProductGenerator with NoRegimes
│        deferred (§4.3), additive, starting at 0.x: …CensoredGaussian, …SpaceTime, …SpatialHMM, …MSTWeatherGen, …KNN, …
├── test/reference/   legacy golden fixtures (own env pinned to legacy SWG + SPSM 4.1, committed Manifest)
└── docs/             one Documenter site (Literate tutorials, DocumenterInterLinks → SPSM); docs/design/ (this plan & research)
SmoothPeriodicStatsModels.jl/   own repo, branch `evolution`: 4.2.0 additive bridge → 5.0.0 prune
```

| Package | Hard deps | Exports |
|---|---|---|
| Core | Dates, Random, Statistics, Tables, StatsAPI, JSON (1.x), PrecompileTools. Weak: OptimizationOptimJL, through an extension that defines `default_solver()` | ~22, plus a `public` class-author API |
| Blocks | Core, SPSM, Distributions, NearestCorrelationMatrix | ~16, plus a `public` block protocol |
| RegimeHMM / WGEN | Core and Blocks only (WGEN never depends on RegimeHMM) | 1 each |
| umbrella 2.0.0 | the 4 libraries, plus Optimization, OptimizationOptimJL, SimpleNonlinearSolve, ForwardDiff, CSV, Downloads, Impute | about 50. It also re-exports `Normal, Gamma, Exponential, Weibull, LogNormal`, so printed specs run as-is |

**Monorepo rules** (verified against OrdinaryDiffEq, NonlinearSolve, Optimization, Lux, CUDA and Makie):
- **No root `[workspace]` listing `lib/*`.** Only repos versioned in lockstep (CUDA.jl, GPUArrays.jl) do that. Independently versioned monorepos use per-library `[sources]`.
- **Lux layout per library:**
  - `[sources]` pointing to its siblings;
  - `[workspace] projects = ["test"]`, with its own gitignored manifest;
  - `test/Project.toml` whose `[sources]` points to `..` and to its siblings;
  - a `LICENSE` copy, which General requires.
- **The root workspace stays `["test", "docs"]`.** `lib/*` and `test/reference` are never members.
- **Development link to SPSM.** Until SPSM 4.2 is registered, Blocks' `[sources]` points to the sibling checkout `../../../SmoothPeriodicStatsModels`. This local-only path is replaced by registered compat before any push.
- **Versions:** the launch libraries are 1.0.0, and the T2 authoring protocol is labelled "experimental: may change in minor releases". Sibling lower bounds are bumped whenever a library starts using new sibling API.

### 1.1 Operations
| Task | How | Cost |
|---|---|---|
| Tests | `CI.yml`: a matrix of 5 projects (4 libraries + the umbrella) × Julia `['1.12','1']`, using `julia-buildpkg`/`julia-runtest` with `project:`. A job skips if its project does not exist yet | 10 jobs |
| Docs | One site. `docs/Project.toml` `[sources]` → `..` and `../lib/*`; `checkdocs = :public`; `deploydocs` unchanged | `stable` moves on umbrella `v*` tags only |
| Tags | `TagBot.yml`: a root job plus a `subdir` matrix. Tags look like `StochasticWeatherGeneratorsCore-v1.0.0` | none |
| Bumps | Dependabot `julia` with `directories: ["/", "/docs", "/test", "/lib/*", "/lib/*/test"]` | none |
| First registration | SPSM 4.2 → `@JuliaRegistrator register branch=evolution subdir=lib/…Core` → Blocks → RegimeHMM + WGEN → on release day SPSM 5.0, a Blocks compat patch and the umbrella | about 7 comments; 3 days per new package |
| One-library release | Bump `lib/X/Project.toml`, then comment `@JuliaRegistrator register subdir=lib/X` | 1 comment, about 15 minutes |

---

## 2. Public API

### Tier 0: impact modeler (the crop tutorial)
```julia
using StochasticWeatherGenerators, DataFrames
obs  = read_stations(INRAE(), station_files; impute = [:RR, :TX])            # DATE, STAID, RR, TN, TX, QQ, ETPP
swg  = fit(obs, RegimeHMM(regimes = Regimes(pretrained(:france10_K4_d1_m1))))  # frozen paper regimes (10 stations)
sims = simulate(swg, 1986:2035; nsim = 300, seed = 50000)
df   = DataFrame(sims)                                    # or: for p in Tables.partitions(sims) … end
```
- `fit` prints info lines, for example the overlap of the frozen regimes with the data, which reproduces the legacy `innerjoin`.
- `show(swg)` prints the resolved spec as runnable Julia; that printout is the Tier-1 entry point.
- Errors name the column, station and date, and give the fix as code.

Students swap classes like solvers:
```julia
gens = [fit(obs, m; seed = 1) for m in (RegimeHMM(K = 4), WGEN(order = 2))]
```

### Tier 1: applied statistician (today's model, customised)
```julia
model = RegimeHMM(
    regimes = Regimes(load_generator("paper.swg.json")),
    RR   = Rain(occurrence = Occurrence(memory = 2), amount = Mixture(Exponential, Gamma),
                copula = MultisiteGaussian(by = :regime, estimator = PairwiseRank())),
    TX   = Temperature(season = (μ = 2, ρ = 0, σ = 1)),        # = Variable(Normal; dynamics = AR1()), per-parameter degree
    TN   = Difference(:TX, Gamma; season = 1),                  # TN = TX − Δ, Δ seasonal Gamma, coupled to TX's noise
    QQ   = Variable(Normal; dynamics = AR1(), clip = 0.0),      # today's model for positive variables
    ETPP = Variable(Normal; dynamics = AR1(), clip = 0.0, copula = nothing))
swg  = fit(obs, model; window = Date(1986):Date(2019, 12, 31))
curve(swg, :TX, :ρ; station = 49215002, regime = 2);  transitions(swg);  correlation(swg, :RR; regime = 1)
swg2 = fit(merge(obs, obs_new), spec(swg))                      # add stations under the same frozen regimes
```

**Launch vocabulary:** `Regimes, NoRegimes, Rain, Occurrence, Mixture, Variable, Temperature, AR1, Difference, MultisiteGaussian, PairwiseRank, Monthly, TwoStage, pretrained`.

**Keywords:** `season`, `memory`, `copula`, `dynamics`, `clip`.
- `nothing` means "none".
- An integer `season` is a trig degree, as in SPSM.
- `K = 2:6` selects K by the Stage-1 BIC.

**The slots are generic; launch implements today's combinations.** A slot value not yet implemented (for example `Variable(Gamma; dynamics = LatentAR1())`) raises "not available in this version", never a silent fallback.

### Tier 2: experts
- **New block:** subtype `Blocks.AbstractBlock` and implement:
  - `fit_block(b, ctx::FitContext)` (`ctx.W` is N×K weights, `OneHot(ẑ, K)` at launch);
  - `simulate_block!(rng, out, b, θ, ctx)`;
  - `innovations`, `nparams`;
  - traits `outputs`, `inputs`, `memory`, `scope` (`StateScope` | `EdgeScope`), `likelihood_kind`, `capabilities`;
  - optional `logdensities!`;
  - `to_plain` / `from_plain` with `register_type`.
- **New model class**, depending on Core only: subtype `AbstractGenerator` and implement `resolve`, `fit_generator`, `simulate_member!` and the traits. The class then inherits ensembles, seeding, Tables output, plots and persistence.
- **Fitted blocks hold plain SPSM models:** `swg[:TX]` gives each station's `SmoothAR1`.

### Verbs
| Verb | Signature | Replaces |
|---|---|---|
| `fit` | `fit(data, alg; window, seed, solver, verbose) -> FittedGenerator` | six result shapes |
| `simulate` / `rand` | `simulate([rng,] g, dates\|years; nsim, seed, members = a:b, past, burnin, given, sink, threads, eltype) -> WeatherSimulation` | `rand(hmm…)`, `rand_RR`, `rand(ar1s…)`, `rand_cond`, `simulate_markov_gaussian` |
| `decode` | `decode(g[, data]; method = :viterbi)` | `viterbi(...)` + CSV |
| read-outs | `curve(g, var, par; station, regime, history)`, `transitions(g)` (K×K×T), `correlation(g, var; regime)`, `regime_model(g)`, `g[:TX]`, `spec(g)` | `μₜ…`, `θ_ΔT[j][k](t)` |
| stats | `loglikelihood`, `aic`, `bic(g; by = :block)`, `dof`, `nobs`, `likelihood_kind` | — |
| persistence | `save_generator` / `load_generator` | `JLD.save(Dict(...))` |
| data | `read_stations(INRAE() \| ECA() \| MeteoFrance(), …)`, `example_data`, `pretrained` | `collect_data_*` |
| `validate` | 2.1, name reserved | ad hoc plot loops |

Rules:
- **Argument order:** SWG puts the data first (like SciML `solve(prob, alg)`); SPSM puts the model first (like `fit_mle(d, x)`).
- **`curve`** is SPSM's function; Blocks adds methods for fitted generators. Core does not export SPSM's names.

---

## 3. Core semantics

- **WeatherData input.**
  - Any Tables.jl source, wide or long, with the date and station columns auto-detected.
  - Aliases (`TMAX`, `tasmax`, …) map to canonical names internally, and the original names are restored on output.
  - The class hooks `requested_variables` and `declared_kinds` run **before** unknown columns are dropped.
  - Unknown columns are dropped with one info line, for example the constant `CO2 = 330.0` from `collect_data_INRAE`.
  - `NaN` becomes `missing`; duplicate (date, station) rows raise an error; `data[a:b]`; `merge`.
  - **Site metadata** (LAT, LON, ALT, CRS kind) is kept: `coords(data)`, `distances(data; metric)` (one documented haversine, no rounding), and the distance matrix is persisted. A 3-D array constructor serves gridded data.
- **Calendar.** T = 366 internally. `Leap366` equals legacy `dayofyear_Leap`; `NoLeap365` skips index 60; `Day360` exists. `simulate` takes consecutive steps.
- **Complete data.**
  - The default window is the range common to all stations.
  - A `missing` value inside the window raises an error naming the station, variable and dates, with both fixes (`impute = …` or a narrower `window`).
- **Fit window.** Days before the window only serve as `past`, and the regime chain starts on its first day.
- **Initial conditions: one keyword, `past`.** It is a table ending the day before the start, `Date(…)` (pulled from the training data), or omitted (then `burnin`). Short lags are drawn from the fitted conditional. With `given`, `past` is required.
- **Conditioning.**
  - `given = (regime = path,)` imposes the regime path.
  - A non-regime `given` requires `given.regime` in 2.0.
  - Conditioning a parent on its child is rejected using the DAG.
- **RNG.**
  - A SplitMix64 derivation per (seed, member, block id), plus fit streams `(seed, :fit, block id)`.
  - Block ids are stable string hashes, not `hash(::Symbol)`.
  - Output is independent of the thread count and of `nsim`; `members = a:b` reproduces slices of a full run.
  - The derivation version is stamped into outputs; Int and UInt seeds both work.
  - Guarantee: identical output given the same Manifest and Julia minor version.
- **Scale.** `sink` consumes `Tables.partitions` by (member, station) or (member, time chunk); `eltype = Float32` is available.
- **Persistence.**
  - A structured JSON tree (type tag, format version, every field) built with `to_plain`/`from_plain` and a `register_type` registry. **Never `eval`.**
  - Frozen regimes are stored as data; NaN and Inf are written as tagged strings.
  - Evaluated curves are stored and re-checked on load; an unknown type raises "load package X first".
- **Likelihood kind** is a per-block trait (`:exact`, `:conditional`, `:composite`, `:mc_noisy`, `:none`). The generator's kind is the weakest of its blocks. `aic`/`bic` refuse composite blocks.
- **Solver.** Core's OptimizationOptimJL extension defines `default_solver() = LBFGS()`. Without it, the error lists the packages to load.

---

## 4. Launch models: today's models, re-expressed

### 4.1 `RegimeHMM` = the tutorial model (Blocks `ProductGenerator` + preset)

**Default block per variable kind** (each reproduces the legacy code):

| Kind | Default block | Legacy code it replaces |
|---|---|---|
| rain | `Rain()`: occurrence logistic in (site, regime, memory category), independent across sites given z; amounts `Mixture(Exponential, 2)` per (site, regime); copula `MultisiteGaussian(by = :regime, estimator = PairwiseRank(), independent = (K,))` on the wet set | `fit_mle_RO`, `fit_mle_RR`, `mix_ini`, `cov_RR`, `rand_RR` |
| temperature | `Temperature()` = `Variable(Normal; dynamics = AR1())`, a seasonal AR(1) per (site, regime) with a copula on residuals per regime | `fit_AR1`, `cov_ar1`, `rand(ar1s, n2t, z, Σ; y₁)` |
| lower temperature (TX present) | `Difference(:TX, Gamma)`: Δ = TX − TN, a seasonal Gamma per (site, regime); coupling r_k(t) on the raw cor(TX, Δ) scale as today; sites independent given TX's noise | `fit_TN`, `cor_groupbyTXTN`, `rand_cond`, `ϵ_TX` |
| positive (QQ, ETPP) | `Variable(Normal; dynamics = AR1(), clip = 0.0)` | `fit_AR1` + `indicatrix` clipping |
| other | `Variable(Normal; dynamics = AR1())` | — |

**Modelling choices kept exactly:**
- clipping at 0 for positive variables;
- the last regime independent in the rain copula (`independent = (K,)`), plus the legacy `force_PosDef` projection;
- the raw-scale TX–Δ coupling;
- independence across sites for Δ given TX's noise;
- a copula constant over the season, per regime;
- the Viterbi plug-in estimation (`TwoStage()`).

**Code defects fixed** (each listed in the migration guide with a test):
- `cov_ar1` bridging gaps, and its first residual;
- regime labels depending on `groupby(:z)` order (`fit_mle_TN.jl:28, :109`);
- the AR(1) sampler's default `y₁` reading σ[1,1] whatever z₁ is;
- the `rand(::AR1{<:AbstractMatrix}, N)` MethodError;
- WGEN counting across year boundaries (§5);
- WGEN Ω's root-find advancing the RNG, which made the objective noisy (now common random numbers);
- the longitude bug in `distance_x_to_y`.

**Regimes.**
```julia
Regimes(; K = 4, season = 1, init = SliceEM(window, n_random, dirichlet) | <SPSM model>,
        em = (maxiter, tol, robust), order_by = :wetness)
Regimes(g::FittedGenerator)   # frozen
Regimes(hmm, path)            # external SPSM HMM, including SmoothARPeriodicHMMSpatial (tuto_paper_HMMspatial)
NoRegimes()
```
- The emission memory and season are read only from the RR `Occurrence`.
- `Occurrence(from = :regimes)` reuses the frozen emission when it covers exactly these stations; otherwise the occurrence is refitted given ẑ, as `fit_mle_RO` does today.
- Frozen regimes are never relabelled; ẑ must equal `z_hat_K_4_d_1_m_1.csv` exactly.

**Estimation.**
- `TwoStage()`: EM on occurrence, then Viterbi ẑ, then each block by weighted IFM given `OneHot(ẑ)`.
- `likelihood_kind = :plugin`. `bic(g; by = :block)` is valid; a K range uses the Stage-1 BIC.
- Simulation is block-major in DAG order. Wet-set sampling draws over all D sites from one read-only cached Cholesky factor per regime, then keeps the wet subset; this is exact and thread-safe.

### 4.2 Extension points reserved in Core and Blocks 1.0
These are signatures and slots only; launch uses only today's cases. They make every model in §4.3 additive.
1. **Site metadata** in `WeatherData` and `FittedGenerator` (§3).
2. **Multi-output blocks everywhere.** `g[:var]`, `curve` and `correlation` route to the block that owns the variable; a joint block can be a DAG parent.
3. **Blocks expose derived quantities.** For example the occurrence block exposes p^dry[s, n]. `inputs` can name a (block, quantity) pair. The parts of a variable can be addressed separately (`RR_occurrence`).
4. **Dependence protocol.**
   - `AbstractDependence` over an index set: sites, or (variable × site).
   - A `structure::AbstractStructure` slot with `cov!(C, s, θ, coords; lag, vars)` from day one.
   - Constrained sampling `draw!(rng, x, dep, θ, ctx; lower, upper)`.
   - A `memory(dep)` trait.
   - `by` accepts `:pooled`, `:regime` and `(:regime, :season_node)`.
5. **A dependence-estimator protocol** separate from the structure: pair design, soft weights, composite kind.
6. **Pluggable regime source and chain estimator.** `Regimes` variants are subtypes of an abstract source; there is a chain-estimator slot; `criterion = BIC() | ICL()`; `decode(...; rng)`.
7. **Typed `season`:** an Int, `Monthly()`, `Seasons(...)` or any SPSM basis; costly objects are evaluated at season nodes.
8. **Block-owned latent state:** `latent_past(b, past)`, for latent lags with L ≥ 1.
9. **A factor cache** keyed by (block, regime, season node): read-only, lazy, thread-safe, memory-accounted.
10. **Fit RNG streams** in `FitContext`.
11. **Marginal access** on fitted marginal blocks: `cdf` and `quantile` at (k, s, t), and `pit(...; ties)`.
12. **Persistence registry tags** for structures, dependence types and families; compact typed arrays.
13. **Sink partitions** by time chunk; a NetCDF/Zarr sink slot.
14. **Capabilities:** `sequential_only`, `approximate_conditional`, `handles_missing`, `:mc_noisy`.
15. **Proof of openness:** a CI test with a toy block **and** a toy class defined outside the repo, against the `public` protocol.

### 4.3 Deferred models (additive 2.x releases)

| Deferred item | Launch behaviour (today's model) | How it plugs in |
|---|---|---|
| A generic atom at zero, for any variable: `Hurdle(G; zeros = TwoPart(Occurrence) \| Censored())` | only rain has an occurrence/amount split | a new block type; part addressing (2, 3); constrained sampling (4) |
| Positive variables as a censored latent AR(1) with an exact marginal, plus a zero policy (`:atom`, `:suspect`) | AR(1) + clip | `Variable` `dynamics = LatentAR1()` (SPSM `SmoothLatentAR1`, 5.x); `latent_past` (8) |
| TN coupling on the innovation scale, plus a site copula on Δ | raw-scale r, sites independent | new `coupling` and `copula` keyword values on `Difference` |
| A copula without the "last regime independent" rule | `independent = (K,)` | an existing keyword value |
| Non-Gaussian copulas (t, Archimedean, extreme-value, vines) through Copulas.jl | Gaussian only | a new `AbstractDependence` subtype (4) |
| Soft-weight estimation (`TwoStage(decode = Posterior())`) and `JointEM` | Viterbi plug-in | the estimation slot plus the scope trait |
| Censored latent Gaussian class (Ailliot 2009; Vaittinada Ayar 2020) | — | `…CensoredGaussian`, built on Hurdle `Censored` |
| Codepaper model `SpatialRegimeHMM`: spatial regime HMM, EGPD margins, conditional space-time field | spatial HMM usable today through `Regimes(hmm, path)` | `…SpaceTime` + `…SpatialHMM`; SPSM EGPD/`Spliced` (5.x) through `SmoothDistribution` (2, 3, 4, 5, 6, 8, 11) |
| A seasonal, improved MSTWeatherGen | — | `…MSTWeatherGen` on `…SpaceTime` (2, 4, 6, 7, 9) |
| `validate`, multi-sequence fits, FFBS for `given`, the `stepper` streaming API | — | Core verbs; SPSM `seq_ends` (5.1) |

Full designs: `docs/design/01-unified-design.md` (improved blocks), `05-research-codepaper.md` and `06-research-mstweathergen.md`.

---

## 5. WGEN port

`WGEN(; order = 1, season = Monthly(), amount = nothing, copula = MultisiteGaussian(by = :month, estimator = MomentMatching(nsim = 2000)))`
- It lowers to `ProductGenerator(NoRegimes(), RR = Rain(...))`, and `expand(spec(g))` prints that form.
- **Probability tables** are closed-form under `Monthly()`.
  - Counting follows the calendar (the cross-year fix).
  - An internal `counting = :legacy` switch reproduces `fit_markov_chain` (`src/rain/wgen.jl:84-110`) exactly, for the equivalence test.
- **Ω:** the same moment-matching estimator as today (`wgen.jl:173-212`). It now uses common random numbers, so the objective is deterministic and solved by bisection; then `nearest_cor`. NonlinearSolve leaves SWG.
- **Output:** with `amount = nothing` it is `RO::Bool`. `likelihood_kind = :none`.

---

## 6. SPSM: 4.2 (additive) then 5.0 (prune)

### Engine contract
Added in 4.2, it becomes the only form in 5.0; only these names are `public` for Blocks.
```julia
fit_mle(m, y[, w]; n2t, past, seq_ends, solver, kw...) -> m′    # MODEL ONLY in 5.0; new types follow this already in 4.2
fit(SeasonalFit, m, y[, w]; kw...) -> SeasonalFit
loglikelihoods(m, y; n2t, past) -> N×K                         # the column axis is ALWAYS the state axis
rand([rng,] m, n2t; z = nothing, past = nothing)                # HMMs return (; z, Y) on the keyword path
viterbi(m, Y; n2t, past);  curve(m, name; state);  m(t);  coef(m)   # coef(m) = live ComponentArray (new for models)
initialize(m0, Y; past, n2t, method = SliceEM(window, n_random, dirichlet)) -> smooth model
SmoothDistribution(family, T; K = 1, season = 1)   # any seasonal_family; per-parameter season; fit keywords adtype, penalty
SmoothARPeriodicHMM(K, D, T; memory, season)       # size constructors (memory required)
fit_mle(hmm, Y, W; fixed = (:A,), past, n2t)       # weighted M-step given state weights (replaces fit_mle_RO, exact match)
```

Contract rules:
- **`OneHot(z, K)` and typed weights.** Weights are `AbstractVector{<:AbstractFloat}`, `AbstractMatrix{<:Real}` or `OneHot`. In 5.0, an integer third argument throws "pass OneHot(z, K)".
  - The commit that exports `OneHot` also adds the weighted HMM method. Otherwise the v4 `(Y::Bool, Y_past::Bool)` method silently catches a `OneHot`.
- **`past` is uniform:** `(; Y = D×m [, z = k₀])`.
  - `past.z` is used by `rand` and `viterbi`.
  - HMM fits raise a clear error when it is given, because HiddenMarkovModels 0.7.1 has no per-sequence initial-state hook.
- **`n2t`** becomes a required keyword in 5.0.
- **Objectives.** 4.2 keeps the summed objectives of the existing models, bit-identical to v4. 5.0 switches them to the average; the new types use the average from the start.
- **Two extensions:** one for Optimization alone (AR1, mixture, TVP, `SmoothDistribution`) and one for SimpleNonlinearSolve alone (`fit_θ`, `solve_nlls`, `default_nlls_solver`). ForwardDiff becomes a direct dependency. `fit_mle` without an extension gives a helpful error.
- **`adtype`:** `NoAutoDiff()` is mapped to `SciMLBase.NoAD()`, the only form Optimization 5.6 accepts.
- **`curve`** gains `:A` (K×K×T) and `:p` (K×D×H×T) for the Bernoulli HMMs.
- **The spatial HMM takes an `rng`** in its likelihood, fit and `viterbi`, passed through to `mvnormcdf`.
- **`SliceEM`, not `Slices`:** Julia 1.13 `Base` exports `Slices`.
- **`SmoothDistribution` reproduces SWG's `_fit_seasonal_gamma` exactly.** It serves the Δ Gamma of today's model, and it is the open family protocol for later EGPD.

### 5.0 prune (exports 94 → about 40)
- **Exported:**
  - bases and links; `ParamSpec`, `curve`, `TimeVaryingFamily`, `seasonal_family`;
  - the `Smooth*` models and `SmoothDistribution`;
  - `OneHot`, `fit_mle`, `fit`, `SeasonalFit`, `FitHistory`, `converged`, `loglikelihoods`, `viterbi`, `initialize`, `SliceEM`;
  - the StatsAPI verbs, `components`, `ncomponents`, `mixtures`, `n_to_t`.
- **`public` only:** layout helpers, NLL objects, the EM knobs, solver plumbing (`solve_objective`), the evaluated twins, `SeasonalControls`, shared kernels.
- **Removed:**
  - `TimeVaryingFit`, `mixture_weight_spec`, `mixture_weights`;
  - `Trig2ARPeriodicHMM*`, `randARPeriodicHMM`;
  - `fit_θ*`, `fit_mle_RO`, `fit_mle_all_slices`;
  - `μₜ ρₜ σₜ αₜ`, `params`, `parameters`, `nparams`;
  - the multisite `rand(::Array{AR1}, …)`, which moves to Blocks.

**Moves:**
- SWG → SPSM: `_ar1_init`, `mix_ini`, `_fit_seasonal_gamma` (→ `SmoothDistribution`).
- SPSM → Blocks: multisite AR simulation.
- The spatial HMM stays in SPSM.

**Deferred to 5.x (additive):** `SmoothLatentAR1`, EGPD/`Spliced`, `SoftmaxLink`, `Conditional` covariates, `CorrelationLink`, `seq_ends` for non-HMM models, and the `CachedLogDensity` bridge.

---

## 7. Release mechanics, migration, equivalence

1. **SPSM 4.2.0** is registered first. SWG libraries declare compat `"4.2"`. SPSM CI gets a `Downstream` job that runs the SWG libraries.
2. **SWG `evolution`** holds the libraries and the 2.0 umbrella; `master` stays on 1.x (1.4.0 can be registered as a last legacy release).
3. **Cutover branch.** The libraries' compat becomes `"4.2, 5"`, tested against the SPSM 5.0 branch through a temporary `[sources]`.
4. **Pre-registration** in the order Core → Blocks → RegimeHMM + WGEN.
5. **Release day.** Register SPSM 5.0.0, then a Blocks compat patch, then merge `evolution` into `master` (deleting `src/rain`, `src/AR1`, `src/temperature` and `ext/SmoothPeriodicStatsModelsExt`), then SWG 2.0.0 with release notes that say "breaking".
6. **Migration guide.**
   - Generated mechanically from `names(M)` for SWG 1.3.1, SWG 1.4.0 and SPSM 4.1; a CI test checks that every old export has a row.
   - It includes the rows for `fit_mle(m, Ys, Y_pasts)` → `fit_mle(m, Ys; past = Y_pasts)`, `rand(…; seq, z_ini)` and `viterbi(m, Ys, Y_pasts)`.
   - It has a "fixed defects" section (§4.1).
7. **Equivalence.** The **defaults must reproduce legacy** against fixtures from `test/reference/make_legacy.jl`. That environment pins legacy SWG at `bb4d8bf` through `{url = "../..", rev = <sha>}` plus SPSM 4.1.0, uses synthetic data, and needs no network.
   - **L1, parameters:** rtol 1e-8 (closed form), 1e-6 (regime HMM, occurrence given ẑ), 1e-4 (optimizer fits). WGEN `:legacy` tables use `==`. Ω is compared with a high-nsim legacy reference in Monte Carlo SE units. ẑ labels must match exactly.
   - **L2, pathwise:** a `noise =` hook must reproduce legacy samplers given pre-drawn noise.
   - **L3, distributional:** about 200 statistics over 300 members.
   - **Allowed differences:** the fixed defects (§4.1), `past`/burn-in, and the RNG streams.
8. **Crop-tutorial yields.** The legacy STICS weather ensemble does not exist anywhere, so the yield sections are labelled as computed on the legacy ensemble. Nothing is fabricated.

---

## 8. Implementation phases (all local commits on `evolution`; no push, no registration)

- **P0 — Safety net (SWG).**
  - Delete the stray `docs/design/-X`.
  - Add CI (§1.1), TagBot subdir matrix, Dependabot subdirs.
  - Create `test/reference/` (env, committed Manifest, `make_legacy.jl`, fixtures, README).
  - Convert `assets/tuto_1/*.jld` + `z_hat` CSV into `assets/pretrained/france10_K4_d1_m1.json`, with a loader check to 1e-12 against the `.jld` (which is already in the SPSM-4 sign convention).
  - Write `docs/design/07-p0-notes.md`. (The Météo-France load-time download is already disabled.)
- **P1 — SPSM 4.2 (additive)**, following the agent plans listed below:
  - `src/onehot.jl`;
  - typed weights, `past`, N×K `loglikelihoods`, keyword `rand` (`src/fit/ar1.jl`, `src/AR/AR1.jl`, `src/fit/mixture.jl`, `src/fit/smooth_hmm.jl`);
  - the extension split (`ext/`, `src/fit_stubs.jl`);
  - `src/fit/distribution.jl` (`SmoothDistribution`);
  - size constructors, `initialize`/`SliceEM`, the weighted HMM M-step (reuse `update_B!`, `src/fit/add_sites.jl:13-36`), `curve :A/:p`, `coef(m)`, the spatial `rng`;
  - CHANGELOG/version 4.2.0, CITATION.

  The full SPSM suite must pass unchanged.
- **P2 — Core**, following its agent plan: `lib/StochasticWeatherGeneratorsCore` with §2–§3, its §4.2 slots, the climate indices on plain arrays, and a toy class.
- **P3 — Blocks.** The protocol; `ProductGenerator`; `Regimes`/`NoRegimes`/`TwoStage`; `Rain`/`Occurrence`/`Mixture`; `Variable`/`AR1`/`Temperature` with `clip`; `Difference`; `MultisiteGaussian` with `PairwiseRank`, Pearson on residuals and `independent`; `show`; the registry; the §4.2 slots. It reuses:
  - `cor_RR` and `Σ_Spearman2Pearson` (`src/rain/correlations.jl`);
  - `cov_ar1`, with the defects fixed (`src/AR1/fit_mle.jl`);
  - `rand_RR` (`src/rain/generation.jl`);
  - `rand_cond` and `cor_groupbyTXTN` (`ext/SmoothPeriodicStatsModelsExt/fit_mle_TN.jl`).

  Exit: L1/L2 per block, and the out-of-repo toy block/class test.
- **P4 — RegimeHMM and the umbrella.** The preset; `read_stations`, `pretrained` and `example_data` in the umbrella; new versions of `tuto_add_station_variable`, `tuto_paper` and `tuto_paper_HMMspatial` (spatial via `Regimes(hmm, path)`). Exit: L1/L2/L3 against the fixtures, T0 in ≤ 6 lines, and a time/memory budget at 37 sites × 50 years × 300 members.
- **P5 — WGEN.** Exit: `:legacy` tables `==`; Ω within Monte Carlo SE; pathwise equality given U.
- **P6 — Docs and migration guide** (both repos): the per-package API pages; `docs/src/models/*.md` mapping each TODO to its §4.3 extension point; the SPSM docs and its 5 tutorials.
- **P7 — SPSM 5.0 prune** on its own branch (`v5`), with the cutover branch tests.
- **P8–P9 — Registration and release day.** These are outward-facing, so they wait for the author.

---

## Verification

- **Per library:** `julia --project=lib/X -t 4 -e "using Pkg; Pkg.test()"`. SPSM: `julia --project=. -e "using Pkg; Pkg.test()"` (full suite unchanged in 4.2). Legacy fixtures: `julia --project=test/reference test/reference/make_legacy.jl` (a second run leaves the files unchanged).
- **Edge cases:**
  - missing values in the window → guided error; stations with different ranges → common window;
  - 29 February under `Leap366`/`NoLeap365`; one-site wet days and wet sets of varying size;
  - non-PD estimates with and without `independent`; |ρ| → 1; a regime with zero weight for a block;
  - TN ≥ TX in the input;
  - integer weights to SPSM; `OneHot` not caught by the v4 `Y_past` method;
  - `past` shorter than the memory; `given` without `regime`; frozen regimes not covering the data;
  - seeds independent of thread count and of `members` slicing; Int vs UInt seeds;
  - JSON round-trips (frozen regimes, coordinates, a user block, an unknown type);
  - the out-of-repo toy block and class;
  - name resolution with `using StochasticWeatherGenerators, Distributions, SmoothPeriodicStatsModels, DataFrames`;
  - an unimplemented slot value → "not available in this version".
- **End to end:**
  - the crop tutorial and `tuto_paper` run on the new API and match legacy within the L1–L3 tolerances, apart from the listed fixed defects;
  - the docs build.

---

## Implementation status and findings (2026-11-25)

- **Stage A ran in plan mode,** so the agents wrote verified plans instead of code. Those plans are in `C:\Users\metivier\.claude\plans\`:

  | Track | Plan files |
  |---|---|
  | SPSM 4.2 | `…-agent-a3ff36a6be09d9a58.md` (part 1), `…-agent-a05b58e43f7fb869b.md` (part 2), `…-agent-a601e15215d791af4.md` (merged order) |
  | SWG P0 | `…-agent-a97bc2cbcac3035c9.md`, `…-agent-ac57610420afbca4e.md` |
  | SWG Core | `…-agent-ac19a77095bcdfaef.md` (supersedes `…-a52ce2d103dc2dd0a.md`) |

  Their findings are folded into §2–§8 above.
- **Scope cut:** where those plans include work for **deferred** models, it is skipped. This covers `SmoothLatentAR1`, the `Hurdle`/`Positive` blocks and the improved-default variants.
- **Already on disk:** the SPSM `evolution` branch, and the SWG commit `b6ca90c` with the design documents. Both were created while plan mode was still on; both are easy to undo if unwanted.
- **Next:** rerun Stage A fresh (not resumed), then Stage B (P3–P6) and P7.

## Open decisions (recommended default in **bold**)
1. **Default K without pretrained regimes.**
   - **K = 4, with a warning below 8 stations.**
2. **Should SWG 1.4.0 be registered** as a last legacy release before 2.0.0?
   - **Yes,** if users need the current fixes.
