# SWG 2.0 + SPSM 5.0: one interface for every weather generator, one engine for every seasonal block

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
- The multisite copula AR(1) is a simulator only. It ignores Σ (`src/AR/AR1.jl:147-156`) and is untested.
- Four HMM families duplicate code.

**Versions.** General has SPSM up to **4.1.0** and SWG only **1.3.1**. SWG `master` is now labelled **1.4.0**; the earlier 3.0.0 label was a mistake. This plan calls the code on `master` **"legacy SWG"**. The new architecture is **SWG 2.0.0**.

**Goal.** A coordinated clean-break release that serves three audiences:
- **impact modelers:** tables in, simulations out, in about 5 lines;
- **applied statisticians:** declare seasonal, regime-dependent marginals, copulas and dynamics without internals;
- **students:** a small, consistent vocabulary and readable printing.

It must also let experts add new blocks and whole new model classes without breaking changes.

**How this plan was built.** It comes from two read-only multi-agent workflows:
- **Exploration:** 7 agents mapped the code, docs, upstream packages and prior art.
- **Design:** 4 independent designs (SciML-faithful, beginner-first, statistical composition, smallest change). Three judges (beginners, experts, maintainers) scored them. A synthesizer merged them, and three adversarial critics attacked the result, raising 65 issues, 2 of them blockers. All blockers and major issues are resolved below.

### Decisions made by the author
1. The composite multivariable model lives in **SWG**.
2. SWG copies **DifferentialEquations.jl on both axes**:
   - a monorepo of sublibraries in `lib/` plus an umbrella package;
   - model classes swappable like solvers (`fit(data, RegimeHMM())` vs `fit(data, WGEN())`).
3. Long term, SWG hosts many generator classes: MSTWeatherGen, GWEX, GWGEN, WeaGETS, AWE-GEN-2d, k-NN, and others.
4. **Launch scope:** the flagship Regime-HMM product model, a port of WGEN, and the censored latent Gaussian.
5. **Lockstep clean break:** **SWG 2.0.0** (sequential after 1.3.1, the only registered version) and **SPSM 5.0.0**, a migration guide, no deprecation shims.
6. **Users:** impact modelers, applied statisticians and students.
7. **Data:** Tables.jl tables in at the high level, arrays underneath.
8. **Every model variant is selectable.**
   - The legacy behaviours and the improved models are all first-class, named choices (§4.1).
   - The improved models are the defaults.
   - `preset = :legacy` reproduces the legacy tutorial model in one keyword.
9. **Names stay consistent with SPSM.**
   - The `Smooth*` type names are kept.
   - SWG blocks mirror SPSM types (`AR1` ↔ `SmoothAR1`, `LatentAR1` ↔ `SmoothLatentAR1`, `Mixture` ↔ `SmoothSeasonalMixture`, a marginal ↔ `SmoothDistribution`).
   - Both packages share the keyword names `K`, `season`, `memory`, `past` and `n2t`.
10. **`validate` is deferred to 2.1.** The tutorials keep their current plotting code.
11. **Complete data at launch.**
    - 2.0 assumes gap-free daily data in the fit window; imputation is done explicitly with Impute.jl, as today.
    - Multi-sequence fits (sum of log-likelihoods over sequences) for long gaps come later and are additive.
12. **An atom at zero is generic.** Rain is one example of a law that is continuous with an atom at 0. Any variable can use `Hurdle(G; zeros = TwoPart(…) | Censored())` (§4).
13. **Dependence is abstract.** Gaussian dependence is the only launch implementation, because it is the most common. Other copulas and space-time structures come in through the same slot.
14. **Planned classes after launch, designed in now:**
    - the HMM_Spatial_codepaper model: a spatial regime HMM for occurrence, EGPD margins, and a conditional space-time field for amounts;
    - a seasonal and improved MSTWeatherGen.

    Core and Blocks 1.0 reserve the slots both need (§4.2).

---

## 1. Architecture

```
StochasticWeatherGenerators.jl/            one repo. Development happens on a long-lived `next` branch whose root
├── Project.toml                           is already the 2.0 umbrella (same UUID 3eadb0b9…); `master` keeps the
├── src/StochasticWeatherGenerators.jl     legacy code for hotfixes until the cutover. Explicit `using X: a, b` + exports.
├── ext/  MakieExt (plot recipes), GeoMakieExt (station maps)
├── lib/
│   ├── StochasticWeatherGeneratorsCore/              interface: verbs, WeatherData, calendars, FittedGenerator,
│   │                                                 WeatherSimulation, ensemble driver, persistence
│   ├── StochasticWeatherGeneratorsBlocks/            blocks, MultisiteGaussian, latent/censored kernels, Regimes,
│   │                                                 ProductGenerator engine (DAG), TwoStage, kind→default-block table
│   ├── StochasticWeatherGeneratorsRegimeHMM/         RegimeHMM (a thin preset over ProductGenerator)
│   ├── StochasticWeatherGeneratorsWGEN/              WGEN (lowers to ProductGenerator with NoRegimes)
│   └── StochasticWeatherGeneratorsCensoredGaussian/  CensoredLatentGaussian
│        planned (§4.2), additive, starting at 0.x: …SpaceTime (shared space-time machinery), …SpatialHMM
│        (codepaper model), …MSTWeatherGen; later: …KNN, …GWEX, …WeaGETS, …GWGEN, …AWEGEN2d
├── test/reference/   legacy golden fixtures (pinned by git tree, outside any workspace)
└── docs/             one Documenter site, Literate tutorials, DocumenterInterLinks → SPSM
SmoothPeriodicStatsModels.jl/   stays in its own repo (domain-agnostic, co-authored): 4.2 additive bridge → 5.0
```

| Package | Hard deps | Exports |
|---|---|---|
| Core (verbs, `WeatherData`, calendars, results, ensemble driver, persistence) | Dates, Random, Statistics, LinearAlgebra, Tables, StatsAPI, JSON, PrecompileTools. Weak: OptimizationOptimJL, through an extension that defines `default_solver()` | ~22 exported, plus a `public` class-author API |
| Blocks | Core, SPSM, Distributions, NearestCorrelationMatrix, LogExpFunctions | ~24, plus a `public` block protocol |
| RegimeHMM / WGEN / CensoredGaussian | Core and Blocks only. None of them depends on another class library | 1–2 each |
| umbrella (2.0.0) | the 5 libraries, plus Optimization, OptimizationOptimJL, SimpleNonlinearSolve, ForwardDiff, CSV, Downloads | about 60. It also re-exports `Normal, Gamma, Exponential, Weibull, LogNormal` so that printed specs run as-is |

Rules for the monorepo:
- **No root `[workspace]` listing `lib/*`.** This is what independently versioned monorepos do:
  - SciML: OrdinaryDiffEq.jl, NonlinearSolve.jl and Optimization.jl have per-library `[sources]` and no `[workspace]`.
  - Lux.jl: per-library `[sources]`, plus a library-local `[workspace] projects = ["test"]`.
  - Root workspaces that list `lib/*` appear only in **lockstep**-versioned repos (CUDA.jl, GPUArrays.jl). One shared manifest would intersect the compat of every library, the umbrella and the docs, and would block a per-library Downgrade lane.
- **Follow the Lux layout.** Each `lib/X` has:
  - `[sources]` pointing to its siblings (`../StochasticWeatherGeneratorsCore`);
  - `[workspace] projects = ["test"]`, with its own gitignored manifest;
  - `test/Project.toml`, whose `[sources]` points to `..` and to its siblings;
  - a `LICENSE` copy, which General requires for subdir packages.

  OrdinaryDiffEqTsit5 instead uses `[extras]`/`[targets]`, which Pkg now calls legacy. With `julia = "1.12"`, `[sources]` is honoured natively, so no dev scripts are needed.
- **The root workspace is `["test", "docs"]`.** It already exists on `master` (`Project.toml:6-7`) and is valid on `next` from day one. `lib/*` and `test/reference` are never members, so the legacy fixture environment, pinned to SPSM 4.1, never constrains the libraries.
- **Dependence is an abstract type, `AbstractDependence`.**
  - `MultisiteGaussian` is the only launch implementation, because Gaussian dependence is the most common and is sampled exactly through Cholesky factors.
  - Non-Gaussian dependence (t, Archimedean or extreme-value copulas, vines) is reserved as `MultisiteCopula(family)`, backed by Copulas.jl through a package extension. It is additive.
  - `Copulas` therefore leaves the launch dependencies but not the design. `NonlinearSolve` leaves too: the WGEN Ω root-find becomes a bisection with common random numbers.
- Each library is registered with `@JuliaRegistrator register branch=next subdir=lib/X`; TagBot and CompatHelper run with `subdirs`.
- The launch libraries are 1.0.0. The T2 protocol (block and class authoring) is labelled **experimental: may change in minor releases** in its docstrings and docs. Users see accessors only, never struct fields.
- Sibling lower bounds are bumped whenever a library starts using new sibling API.

### 1.1 Operations: test, docs, tags, registration
All of this is verified against OrdinaryDiffEq, Lux, Makie, the Pkg docs, the General README, the Registrator README and the TagBot README.

| Task | How it works | Cost |
|---|---|---|
| **Tests** | `CI.yml` runs a matrix of 6 projects (5 libraries + the umbrella) × Julia `['1.12', '1']`, using `julia-actions/julia-buildpkg` and `julia-runtest` with `project: lib/X`. Coverage is restricted to `lib/X/src`. Optional: SciML's reusable `sublibrary-project-tests.yml`, which tests only the libraries affected by a change; each library then needs a `test_groups.toml` with versions `["1.12","1"]`. | 12 jobs. No dev scripts |
| **Docs** | One Documenter site in the root `docs/`. `docs/Project.toml` has `[sources]` pointing to `..` and every `../lib/X`. `make.jl` lists every module, sets `checkdocs = :public` and uses DocumenterInterLinks to the SPSM docs. `deploydocs` is unchanged. | `dev` redeploys on every push to master. `stable` moves only on umbrella `v*` tags, so a library-only release does not refresh `stable` |
| **Tags** | `TagBot.yml` has one job for the root and one matrix job over `subdir: lib/X`. Library tags are `StochasticWeatherGeneratorsCore-v1.0.0`; the umbrella tag stays `v2.0.0`. Uses the existing `DOCUMENTER_KEY`. | none |
| **Dependency bumps** | `CompatHelper.main(; subdirs = ["", "docs", "test", "lib/…"])`, or Dependabot `package-ecosystem: julia` with `directories: ["/", "/docs", "/test", "/lib/*", "/lib/*/test"]` (SciML's current choice). | none |
| **First registration** | About 9 comments in total:<br>1. SPSM 4.2.<br>2. `@JuliaRegistrator register branch=next subdir=lib/StochasticWeatherGeneratorsCore` (new package: 3 days).<br>3. Blocks.<br>4. The 3 classes, in parallel.<br>5. Release day: SPSM 5.0, a Blocks patch widening its SPSM compat, and the umbrella `@JuliaRegistrator register` on `master` with release notes that mention "breaking". | About 9 days if run strictly in sequence. It can be shortened by opening every PR on day 0, since the 3-day timer starts when the PR opens. |
| **Later release of one library** | Bump `lib/X/Project.toml`, merge, and comment `@JuliaRegistrator register subdir=lib/X`. Optional: Lux's `Register.yml` posts that comment automatically on a version bump. | 1 comment, about 15 minutes |

---

## 2. Public API

### Tier 0: impact modeler (the crop tutorial in 5 lines)
```julia
using StochasticWeatherGenerators, DataFrames
obs  = read_stations(INRAE(), station_files; impute = [:RR, :TX])         # DATE, STAID, RR, TN, TX, QQ, ETPP
swg  = fit(obs, RegimeHMM(regimes = Regimes(pretrained(:france10_K4_d1_m1))))  # frozen paper regimes
sims = simulate(swg, 1986:2035; nsim = 300, seed = 50000)
df   = DataFrame(sims)                              # or: for p in Tables.partitions(sims) … end
```

What `fit` does on its own:
- It prints info lines, for example the overlap of frozen regimes with the data, which reproduces the legacy `innerjoin`, or the zero policy chosen per variable.
- `show(swg)` prints the resolved spec as runnable Julia, every variant choice included. That printout is the Tier-1 entry point. Frozen regimes are printed as a descriptive comment, not as fake code.
- Errors name the column, the station and the date, and give the fix as code.
- `fit(obs, RegimeHMM(preset = :legacy, regimes = …))` gives the legacy tutorial model exactly.

Students swap classes like solvers:
```julia
gens = [fit(obs, m; seed = 1) for m in (RegimeHMM(K = 4), WGEN(order = 2), CensoredLatentGaussian())]
sims = [simulate(g, 1986:2035; nsim = 100, seed = 1) for g in gens]   # compared with the existing climate-index helpers
```
A `validate` verb (a side-by-side statistics report) is **deferred to 2.1**. The name is reserved, and the tutorials keep their current plotting code.

### Tier 1: applied statistician
```julia
model = RegimeHMM(
    regimes = Regimes(load_generator("paper.swg.json")),
    RR   = Rain(occurrence = Occurrence(memory = 2), amount = Mixture(Exponential, Gamma),
                copula = MultisiteGaussian(by = (:regime, :month), estimator = PairwiseRank())),
    TX   = Temperature(dynamics = AR1(), season = (μ = 2, ρ = 0, σ = 1)),     # per-parameter trig degree
    TN   = Difference(:TX, Gamma; season = (α = 1, θ = 2)),                 # TN = TX − Δ, so TN < TX always
    QQ   = Positive(Gamma; dynamics = LatentAR1()),                         # censored latent AR, no clipping
    ETPP = Unbounded(Normal; dynamics = AR1(), clip = 0.0))                 # legacy variant, chosen explicitly
swg  = fit(obs, model; window = Date(1986):Date(2019, 12, 31))
curve(swg, :TX, :ρ; station = 49215002, regime = 2);  transitions(swg);  correlation(swg, :RR; regime = 1)
swg2 = fit(merge(obs, obs_new), spec(swg))               # add stations under the same frozen regimes
alt  = RegimeHMM(spec(swg); TN = Difference(:TX, LogNormal))   # copy with an override
```

Vocabulary:
- **Spec names:** `Regimes, NoRegimes, Rain, Occurrence, Mixture, Temperature, Unbounded, Positive, Hurdle, TwoPart, Censored, Difference, AR1, LatentAR1, MultisiteGaussian, PairwiseRank, Monthly, TwoStage, pretrained`.
  - `Unbounded` replaces `Continuous`, which clashes with `Distributions.Continuous`.
- **Keywords:** `season`, `memory`, `copula`, `dynamics`, `zeros`, `pool`, `preset`.
- **Keyword rules:**
  - `nothing` means "none".
  - An integer in `season =` is a trig degree, as in SPSM.
  - A range (`K = 2:6`, `season = 0:2`) means "select by a valid criterion", otherwise it is an error.

### Tier 2: experts
- **New block:** subtype `Blocks.AbstractBlock` and implement:
  - `fit_block(b, ctx::FitContext)` (`ctx.W` is N×K weights; `OneHot(ẑ, K)` at launch);
  - `simulate_block!(rng, out, b, θ, ctx)`;
  - `innovations`, `nparams`;
  - traits `outputs`, `inputs` (DAG parents), `memory`, `scope` (`StateScope` | `EdgeScope`), `likelihood_kind` (`:exact` | `:conditional` | `:composite` | `:none`), `capabilities`;
  - optional `logdensities!`, needed later for Posterior and JointEM;
  - `to_plain` / `from_plain` with `register_block`.
- **New model class**, depending on Core only: subtype `AbstractGenerator` and implement `resolve(alg, data)`, `fit_generator`, `simulate_member!` and the traits. The class then inherits ensembles, seeding, Tables output, plots and persistence, and `validate` once it ships in 2.1. This is what keeps GWEX, k-NN and MSTWeatherGen additive.
- **Fitted blocks hold plain SPSM models** (`swg[:TX].models[s]` is a `SmoothAR1`), which is the door into the engine.

### Verbs (one meaning each)
| Verb | Signature | Replaces |
|---|---|---|
| `fit` | `fit(data, alg; window, seed, solver, verbose) -> FittedGenerator` | the six result shapes |
| `simulate` / `rand` | `simulate([rng,] g, dates\|years; nsim, seed, members = a:b, past, burnin, given, sink, threads, eltype) -> WeatherSimulation` | `rand(hmm…)`, `rand_RR`, `rand(ar1s…)`, `rand_cond`, `simulate_markov_gaussian` |
| `decode` | `decode(g[, data]; method = :viterbi)` | `viterbi(...)` + CSV |
| `validate` | **2.1 (name reserved)** | ad hoc plot loops |
| read-outs | `curve(g, var, par; station, regime, history)`, `transitions(g)` (K×K×T), `correlation(g, var; regime, month)`, `regime_model(g)` (the SPSM HMM), `g[:TX]`, `spec(g)` | `μₜ…`, `θ_ΔT[j][k](t)` |
| stats | `loglikelihood`, `aic`, `bic(g; by = :block)`, `dof`, `nobs`, `likelihood_kind` | — |
| persistence | `save_generator` / `load_generator` | `JLD.save(Dict(...))` |
| data | `read_stations(INRAE() \| ECA() \| MeteoFrance(), …)`, `example_data`, `pretrained` | `collect_data_*` |

**Argument order.** SWG puts the data first, like SciML `solve(prob, alg)`. SPSM puts the model first, like Distributions `fit_mle(d, x)`.

**`curve`** is a single function owned by SPSM. Blocks adds methods for `FittedGenerator`, and the umbrella re-exports it.

---

## 3. Core semantics (resolving the critique)

- **WeatherData input.**
  - Any Tables.jl source is accepted, in wide or long layout, with the date and station columns auto-detected.
  - Aliases (`TMAX`, `tasmax`, …) map to canonical names internally; the original names are restored on output.
  - Class hooks `requested_variables(alg)` and `declared_kinds(alg)` run *before* conversion, so columns that a block names explicitly are never dropped.
  - **Site metadata is kept, not dropped.**
    - `LAT`, `LON` and `ALT` (and a CRS kind, `:lonlat` or `:projected`) are station metadata.
    - Accessors: `coords(data)` and `distances(data; metric = Haversine())`, with one documented metric and no rounding. Legacy `haversine` rounds and uses r = 6372.8 (`src/data/stations.jl:20`); the codepaper uses the Distances.jl default.
    - The distance matrix used at fit time is persisted.
    - A 3-D array constructor `WeatherData(A, dates, sites; vars)` serves gridded data.
  - Unknown columns are dropped with one info line, for example the constant `CO2 = 330.0` that `collect_data_INRAE` emits (`src/data/extracting.jl:139`).
  - `NaN` becomes `missing`. Duplicate (date, station) rows raise an error.
  - `data[a:b]` subsets by date.
- **Calendar.**
  - T = 366 internally for every calendar.
  - `Leap366` equals legacy `dayofyear_Leap` (`src/utilities.jl:15`).
  - `NoLeap365` builds an axis without 29 February and skips index 60. A 360-day calendar maps onto the same index.
  - `simulate` takes consecutive steps.
- **Fit window.** `fit(...; window = a:b)`. Days before `a` serve only as `past`, and the regime chain starts on `a`. This reproduces the legacy alignment for the `==` gates.
- **Initial conditions.** One keyword, `past`, which may be:
  - a table or `WeatherData` ending the day before the start;
  - `past = Date(…)`, which pulls each block's memory and the regime from the training data;
  - omitted, in which case a `burnin` is used.

  Edge cases:
  - Short lags are drawn from the fitted conditional.
  - A censored lag draws w₀ from the truncated normal.
  - With `given`, `past` is required and burn-in is disabled.
- **Conditioning.**
  - `given = (regime = path,)` imposes the regime path.
  - `given = (TX = …)` requires `given.regime` in 2.0. Sampling z from p(z | TX) by forward-filtering backward-sampling is 2.1.
  - Conditioning a parent on its child is rejected using the DAG.
- **RNG.**
  - One stream per (member, block), plus one for the regime path, derived from `(UInt64(seed), member, block id)` with a SplitMix64 step written in Core. `Base.hash` is not used: its value is not stable across processes, and `hash(i, 50000)` with an `Int` seed is a MethodError.
  - The derivation version is stamped into outputs and saved files.
  - Guarantee: identical output given the same Manifest and the same Julia minor version.
  - `members = a:b` reproduces slices of a full run exactly.
- **Scale (37 sites × 50 years × 300 members).**
  - `sink` writes and releases one `Tables.partitions` unit (member, station) at a time.
  - Innovations stay in per-member buffers.
  - `eltype = Float32` is available.
  - The P4 exit criteria include a time and peak-memory budget.
- **Missing data (2.0 assumes complete data).**
  - Fitting requires a gap-free daily series inside the fit window.
  - Imputation stays the user's explicit step, done as today with Impute.jl through `read_stations(...; impute = [...])`.
  - The default `window` is the date range common to all stations.
  - A `missing` value inside the window raises an error that names the station, the variable and the dates, and gives both fixes (`impute = …` or a narrower `window`).
  - **Later (2.x, additive):** multi-sequence fits, where long gaps split the data into sequences and the fit sums their log-likelihoods. SPSM HMMs already take `seq_ends`; AR1, the mixture and `SmoothDistribution` gain it in 5.x.
- **Persistence.**
  - A structured JSON tree: type tag, format version and every field including defaults, built from `to_plain`/`from_plain` hooks and a registry.
  - **Never `eval`.**
  - Frozen regimes are stored as data: coefficients, path, posterior, dates and a source hash.
  - Evaluated curves are stored next to the coefficients, and loading checks that the rebuilt model reproduces them.
  - An unknown type raises "load package X first".
- **Likelihood kind** is a per-block trait. The generator's kind is the weakest of its blocks.
  - `aic`/`bic` refuse composite blocks; CLIC is reserved.
  - `K` is selected only by the Stage-1 BIC. A block's `season` range is selected by block BIC under the shared path, and only for exact or conditional blocks.
  - With frozen regimes, `loglikelihood(g)` sums the blocks given the path.
- **Solver.**
  - Core's OptimizationOptimJL extension defines `default_solver() = LBFGS()`.
  - Without it, the error lists the exact packages to load.
  - Closed-form steps never need a solver: counts, AR(1) sufficient statistics and correlations.

---

## 4. Flagship `RegimeHMM` (Blocks `ProductGenerator` + preset)

**Joint law.** The regime chain uses SPSM's leaving-period convention, A_k(t) = softmax(B(t)β_k). The emission is a product over blocks in DAG order, each block conditioned on z, its own past and its parents' innovations.

**Default block by variable kind:**

| Kind | Default block |
|---|---|
| rain | `Rain()` |
| temperature | `Temperature()` (AR(1) on the variable, as legacy) |
| lower temperature, with TX present | `Difference(:TX, Gamma)` |
| positive | `Positive()` |
| other continuous | `Unbounded(Normal)` |

**An atom at zero is a property of the marginal, not of rain.** Rain is one example of a law that is continuous with an atom at 0; ETPP, wind, snow, sunshine duration or any user variable can have the same law. The generic block is:
```julia
Hurdle(G; at = 0.0, zeros = TwoPart(occurrence = Occurrence(memory = 1)) | Censored(), dynamics, copula, season)
```
- **`TwoPart`** (rain-style): an occurrence process with its own memory and its own multisite dependence (independent given z, a latent-Gaussian copula, or spatial). Values on non-zero days follow G, with a copula restricted to the non-zero set.
- **`Censored`** (censored-latent-Gaussian style): one latent Gaussian field with AR dynamics, censored at Φ⁻¹(p0). Zero/non-zero and the amount come from the same field.
- The other blocks are presets of `Hurdle`:
  - `Rain()` = `Hurdle(Mixture(Exponential, 2); zeros = TwoPart(Occurrence(memory = 1)))`;
  - `Positive(G; zeros = :atom)` = `Hurdle(G; zeros = Censored())`;
  - `CensoredLatentGaussian` uses `Hurdle(G; zeros = Censored())` on RR.

  So every variable can choose either zero mechanism, and every rain model is also available for other variables.
- SPSM supplies the pieces: the seasonal p0 is a seasonal logistic and G is a `SmoothDistribution`, both weighted.

**Blocks:**
- **Rain** (the `TwoPart` preset).
  - Occurrence: logistic in (site, regime, memory category).
  - `Occurrence(from = :regimes)` is the default when frozen regimes cover exactly these stations and the rain variable. It reuses the regime model's own emission, independent or spatial. Otherwise the occurrence is refitted given ẑ, as legacy `fit_mle_RO` does.
  - Amounts: a seasonal mixture on wet days, with a Gaussian copula on the wet set.
  - Sampling draws ε ~ N(0, C_k) over all D sites from one read-only cached Cholesky factor per (k[, month]), then keeps the wet subset. This is exact, thread-safe and needs no per-mask cache.
- **Positive** (QQ, ETPP): a censored latent AR(1), which is the `Censored` preset when zeros are present.
  - Latent recursion: w_n = ρ w_{n−1} + √(1−ρ²) ε_n.
  - Censoring: Y = 0 if w ≤ Φ⁻¹(p0); otherwise Y = G⁻¹((Φ(w) − p0)/(1 − p0)).
  - The fitted marginal is exact and nothing is clipped. ρ is shared across sites by default.
  - **Zero policy** is `zeros = :auto | :atom | :suspect | :none`:
    - `:auto` treats radiation zeros as `:suspect` (they become missing, with a warning; the INRAE QQ zeros coincide with positive ETPP, so they are sensor errors);
    - for ETPP-like kinds, `:auto` treats zeros as an `:atom` when they cluster seasonally.
- **Difference(:TX, G):** Δ = TX − TN follows G. It is coupled to the parent's innovation through r_{s,k}(t), fitted on the innovation scale actually used in simulation (`coupling = :innovation`, the default), with a site copula on the coupling residuals.

### 4.1 Every model is selectable (author decision 8)

| Concern | Legacy model (exact reproduction) | Improved model (default) |
|---|---|---|
| positive variables (QQ, ETPP) | `Unbounded(Normal; dynamics = AR1(), clip = 0.0)` (AR(1) then clipping) | `Positive(Gamma; dynamics = LatentAR1())` (censored latent AR) |
| zeros in a variable | `zeros = :atom` | `zeros = :auto` (radiation zeros `:suspect`) |
| how zeros are generated, for any variable | `Hurdle(G; zeros = TwoPart(Occurrence(memory = m)))` (rain-style, legacy for RR) | `Hurdle(G; zeros = Censored())` (one latent field), selectable per variable |
| non-positive-definite copula | `MultisiteGaussian(by = :regime, independent = (K,))` (legacy: last regime independent) | `MultisiteGaussian(by = :regime)` + `nearest_cor` projection |
| TN given TX | `Difference(:TX, Gamma; coupling = :raw, copula = nothing)` (r from raw cor(TX, TX−TN)) | `Difference(:TX, Gamma)` (innovation-scale r, site copula) |
| copula estimation under latent AR | `MultisiteGaussian(estimator = SameDay())` | `MultisiteGaussian()`, fitted on innovations ε̂ |
| WGEN counting | `WGEN(counting = :legacy)` (month slices, `src/rain/wgen.jl:84-110`) | `WGEN(counting = :calendar)` |
| temperature dynamics | `Temperature(dynamics = AR1())` (default, same as legacy) | `Temperature(dynamics = LatentAR1(), marginal = …)` available |
| rain occurrence across sites | independent given z (default, same as legacy) | `Occurrence(copula = MultisiteGaussian())`, or a spatial regime model through `Regimes(hmm, path)` |

- `RegimeHMM(preset = :legacy)` sets every legacy choice at once. It drives the equivalence tests, which test the legacy **models**, not legacy code paths.
- Every row is documented on the models page with the reason for the improvement.
- Future variants (a t-copula, a space-time Matérn `structure`, an EGPD marginal) are new values of the same keywords and are additive.

**Regimes.**
```julia
Regimes(; K = 4, season = 1, init = Slices(window, n_random, dirichlet) | <SPSM model>,
        em = (maxiter, tol, robust), order_by = :wetness | (:station => id))
Regimes(g::FittedGenerator)      # frozen
Regimes(hmm, path)               # external, e.g. spatial
Regimes(; path = table)          # observed weather types
NoRegimes()
```
- Emission memory and season are read only from the RR `Occurrence`. `resolve` errors on any conflict.
- Frozen or pretrained regimes are **never relabelled**.

**Estimation.**

| Mode | What is maximized | Release |
|---|---|---|
| `TwoStage()` | EM on rain occurrence → Viterbi ẑ → each block by weighted IFM given `OneHot(ẑ)`; a classification likelihood | 2.0 |
| `TwoStage(decode = Posterior())` | the cut-model bound with γ and ξ | 2.1 |
| `JointEM()` | generalized EM, tracking the best iterate | 2.2 |

**Inside a block (IFM):**
1. Fit the margins per site × regime in SPSM.
2. Fit the dynamics.
3. Fit the copula C_k **on innovations ε̂**. Under latent AR with regime switching, same-day levels mix the C of earlier regimes; the paper path has mean regime spells of about 2.5 days. Censored pairs use the conditional pairwise censored likelihood given uncensored lags.
4. Project with `nearest_cor`.

**Sparse cells.** At `resolve`, compute the effective weight per (block, site, regime, season window).
- Below a threshold: `pool = :auto` falls back to the curve pooled over regimes, and `show` prints that choice.
- A regime with zero weight for a block is an error unless `pool = :regimes`.

### 4.2 Planned classes after launch, designed in now

**(a) The HMM_Spatial_codepaper model** (Cognot, Bel, Métivier, Parey): class `SpatialRegimeHMM` in `lib/StochasticWeatherGeneratorsSpatialHMM`. It lowers to `ProductGenerator`, as WGEN does. Its estimation is exactly `TwoStage()`.

The model:
- **Regimes.** K = 4, transitions of trig degree 1. The occurrence emission is a latent Gaussian field thresholded per site, with own-site memory 1 and correlation exp(−h/R_k(t)) whose range R_k(t) is seasonal and per regime. This is SPSM `SmoothARPeriodicHMMSpatial`, fitted by EM with a QMC E-step and a pairwise M-step for R, then Viterbi ẑ.
- **Amount marginals.** Per (k, s, t): uniform on [0.1, 1] mm spliced with a truncated-Beta EGPD tail above 1 mm. Every parameter has trig degree 2, so there are 2 960 coefficients.
- **Amount field.** One non-separable Gneiting–Matérn space-time Gaussian field, shared by all regimes and not seasonal.
  - It is **conditioned on the occurrence mask**: X > Φ⁻¹(p^dry) if and only if the site is wet, with p^dry taken from the occurrence model.
  - Fit: a conditional pairwise censored likelihood (`norm_cdf_2d_vfast`).
  - Simulation: a sequential truncated-MVN sampler with 10 lags.
  - The amount is R = F⁻¹((Φ(X) − p^dry)/(1 − p^dry)).

Sketch (names provisional):
```julia
SpatialRegimeHMM(regimes = Regimes(K = 4, emission = :spatial),   # or Regimes(hmm_spa, ẑ)
    RR = Rain(occurrence = Occurrence(from = :regimes),
              amount = Spliced(Uniform(0.1, 1.0), EGPD; at = 1.0, season = 2),
              copula = ConditionalField(GneitingMatern(); lag = 10, by = :pooled,
                         estimator = PairwiseCensored(given = :occurrence, maxdist = 2/3, maxlag = 10))))
```

**(b) A seasonal, improved MSTWeatherGen** (Obakrim et al. 2025): class `MSTWeatherGen(; preset = :paper | :seasonal)` in `lib/StochasticWeatherGeneratorsMSTWeatherGen`.

The paper's model:
- weather types from PTA compression plus a Gaussian mixture, refitted per user season;
- transitions from sliding-window counts;
- marginals by truncated ordered-quantile normalization per (season, k, variable, site), so rain is a censored latent Gaussian;
- a multivariate Gneiting–Matérn space-time cross-covariance per (season, k), fitted by pairwise likelihood on a pair design;
- simulation by a regime-switching VAR(M) on the latent scale.

The `:seasonal` preset changes four things:
- every parameter is a smooth function of t (SPSM bases), so the season boundaries and their nearest-centroid hack disappear;
- regimes come from the shared `Regimes` (or `Regimes(path = …)` for externally clustered types);
- zeros use `Hurdle(G; zeros = Censored())`;
- the objective is a deterministic, seeded pairwise likelihood with links that keep the covariance valid. The R code instead returns a random penalty for infeasible parameters.

It lowers to a multi-output `LatentField(vars, margins, dependence = MultisiteGaussian(over = (:variable, :site), structure = GneitingMatern(), estimator = PairwiseLikelihood(PairDesign(…))), dynamics = VAR(M))`. The paper-code and printed-equation parametrizations of Gneiting–Matérn differ, so the port must pick one explicitly.

**(c) Shared machinery** in `lib/StochasticWeatherGeneratorsSpaceTime`:
- structures: `Matern`, `ExpExp`, univariate and multivariate `GneitingMatern`;
- `PairwiseLikelihood` / `PairwiseCensored` with `PairDesign`;
- `ConditionalField`, `LatentField`, `VAR(M; from = :covariance)`;
- truncated-MVN sampling.

Heavy dependencies (TruncatedMVN, which pulls NonlinearSolve, and BesselK) stay here and never reach Blocks. The EGPD and `Spliced` families go into SPSM 5.x through `SmoothDistribution`.

**Extension points Core and Blocks 1.0 must reserve now** (signatures and slots only; the implementations ship later):
1. **Site metadata** in `WeatherData` and `FittedGenerator` (§3): coordinates, distances, a persisted distance matrix, and the 3-D array constructor.
2. **Multi-output blocks everywhere.**
   - The engine never assumes one variable per block.
   - `g[:TX]`, `curve` and `correlation` route to the block that owns the variable.
   - A joint block can be a DAG parent.
3. **Blocks expose derived quantities**, not only outputs and innovations. For example, the occurrence block exposes p^dry[s, n] given z and the lag. `inputs` can name a (block, quantity) pair. The occurrence and amount parts of a variable are addressable separately (`RR_occurrence`) in outputs and in `given`.
4. **Dependence protocol.**
   - `AbstractDependence` works over an index set: sites, or (variable × site).
   - A separate `structure::AbstractStructure` slot has the signature `cov!(C, s, θ, coords; lag, vars)` **from day one**, although only lag 0 and one variable are used at launch.
   - Structure parameters can be seasonal or per regime through SPSM `ParamSpec`. R_k(t) = exp(trig) is `TrigBasis => LogLink` with `by = :regime`.
   - Constrained sampling: `draw!(rng, x, dep, θ, ctx; lower, upper)`. The unconstrained case is the launch "draw all sites, keep the wet subset".
   - `memory(dep)` trait. `resolve` rejects a dependence that owns the temporal part combined with per-site `dynamics`.
   - `by` accepts `:pooled`, `:regime` and `(:regime, :season_node)`.
5. **An estimator protocol separate from the structure**: pair design (distance fraction, maximum lag, seeded sampling), soft weights `ctx.W`, and `likelihood_kind = :composite`. Reserve `PairwiseLikelihood`, `PairwiseCensored(given = …)`, and `vcov(g; kind = :godambe)` for CLIC.
6. **Pluggable regime source and chain estimator.**
   - The `Regimes` variants are subtypes of an abstract source. `Regimes(K, emission = :spatial)` and `Regimes(hmm::SmoothARPeriodicHMMSpatial, path)` are the first additions, and a PTA+GMM source comes later.
   - The chain-estimator slot holds `SmoothChain(season)` at launch and `WindowChain(L)` later.
   - `criterion = BIC() | ICL()` on K ranges. ICL is used by the codepaper.
   - `decode(...; rng)`, because QMC Viterbi is stochastic.
7. **Typed `season`:** an Int, `Monthly()`, `Seasons(...)`, or any SPSM basis. Costly objects (factors, VAR matrices) are evaluated at **season nodes**.
8. **Block-owned latent state with L > 1 lags:** `latent_past(b, past)` reconstructs latent lags (PIT on wet days, truncated draws on dry days). Burn-in stays the default.
9. **Factor cache**, keyed by (block, regime, season node): read-only, built lazily, thread-safe and memory-accounted. Nothing assumes one Cholesky factor per (k, month) for every block; MSTWeatherGen needs dimension ≈ 3 000.
10. **RNG streams for fitting too.** `FitContext` carries an rng derived from `(seed, :fit, block id)`, for pair designs and QMC. SPSM 5 passes `rng` to `mvnormcdf`.
11. **Marginal access:** `cdf` and `quantile` at (k, s, t) on fitted marginal blocks, plus `pit(...; ties = :jitter)` for discretized data.
12. **Persistence:** registry tags for structure, dependence and family types; compact typed arrays for quantile tables; factors rebuilt on load and checked against stored hashes.
13. **Output:** `sink` partitions by (member, time chunk) as well as by (member, station). A NetCDF/Zarr sink slot is reserved.
14. **Capabilities:** `sequential_only`, `approximate_conditional`, `handles_missing` (pairwise likelihood tolerates gaps), and a likelihood flag for "exact but Monte-Carlo-noisy", which QMC needs.
15. **Proof of openness.** A CI test defines a toy block and a toy class **outside the repo** against the `public` protocol.

---

## 5. WGEN and CensoredLatentGaussian

**WGEN** is `WGEN(; order = 1, season = Monthly(), amount = nothing, copula = MultisiteGaussian(by = :month, estimator = MomentMatching(nsim = 2000)), counting = :calendar)`.
- It lowers to `ProductGenerator(NoRegimes(), RR = Rain(...))`, and `expand(spec(g))` prints that form.
- Probability tables are closed-form under `Monthly()`.
- `counting = :legacy` reproduces `fit_markov_chain` exactly.
- Ω: a bisection with common random numbers, then `nearest_cor`. Constant months give an independent pair rather than NaN. `estimator = PairwiseLikelihood()` is also available.
- With `amount = nothing`, the output is `RO::Bool` (an occurrence-only capability, so the 2.1 `validate` can report NA for amount statistics).
- `likelihood_kind = :none`.
- Golden test: `WGEN(order = 1, copula = nothing)` must give tables equal to those of `ProductGenerator(NoRegimes(), Rain(Occurrence(season = Monthly())))`.

**CensoredLatentGaussian** (Ailliot 2009; Vaittinada Ayar 2020): `Hurdle(G; zeros = Censored())` applied to RR, so occurrence and amount come from one field.
- `regimes` can be `NoRegimes`, learned, frozen or observed.
- Transform: anamorphosis or `PowerTransform`.
- Fitting is IFM: p0 and G through SPSM `SmoothDistribution`, ρ and C by the pairwise censored likelihood (C on innovations).
- `likelihood_kind = :composite`, and `bic` errors.
- Labelled "two-stage approximation of Ailliot 2009". The joint QMC HMM is experimental.
- It reuses SPSM's bivariate-normal kernel `norm_cdf_2d_vfast` (`src/HHMM_Bernoulli_spatial/fast_bivariate_cdf.jl:19`).

---

## 6. SPSM 5.0: the domain-agnostic engine

### Engine contract
Every item is first added in 4.2 (additive) and becomes the only form in 5.0. Only these names are `public` for Blocks.
```julia
fit_mle(m, y[, w]; n2t, past = nothing, seq_ends = nothing, solver, kw...) -> m′     # MODEL ONLY (Distributions contract)
fit(SeasonalFit, m, y[, w]; kw...) -> SeasonalFit                                   # model + FitHistory + StatsAPI
loglikelihoods(m, y; n2t, past, seq_ends) -> N×K        # every model; the column axis is ALWAYS the state/regime axis
rand([rng,] m, n2t; z = nothing, past = nothing)        # HMMs return (; z, Y)
viterbi(m, Y; n2t, past, seq_ends);  curve(m, name; state);  m(t);  coef(m)   # coef(m) = live ComponentArray
initialize(m0, Y; past, n2t, method = Slices(window, n_random, dirichlet)) -> smooth model
SmoothDistribution(family, T; K = 1, season = 1)        # replaces TimeVaryingFit / PeriodicNLL + fit_loss_optim;
                                                        # any seasonal_family; season per parameter (NamedTuple);
                                                        # fit keywords adtype (NoAD() allowed), penalty
SmoothLatentAR1(T; K = 1, season = 1, x = lag)          # tied variance; x = covariate (TN coupling)
SmoothARPeriodicHMM(K, D, T; memory, season)            # size constructors, same keywords as SWG
fit_mle(hmm, Y, W; fixed = (:A,), past, n2t)            # weighted complete-data M-step (replaces fit_mle_RO)
```

**Contract rules:**
- **Weights are typed:** `AbstractVector{<:AbstractFloat}`, `AbstractMatrix{<:Real}` or `OneHot(z, K)`. A third positional argument of type `AbstractVector{<:Integer}` **throws** "pass OneHot(z, K)", so SPSM 4 calls `fit_mle(ar1, y, z)` fail loudly instead of being fitted with the wrong weights.
- **`past` is uniform:** `past = (; Y = D×m, z = k₀)` in fit, `loglikelihoods`, `viterbi`, `rand` and `SeasonalControls`. A vector of sequences pairs with a vector of pasts.
- **`n2t` is a required keyword.** The ~69 `n_to_t(...)` defaults are removed.
- Every M-step minimizes the **average** weighted NLL.
- `curve(m, :A)` returns K×K×T and `curve(m, :p)` returns K×D×H×T for the Bernoulli HMMs.
- **Solver extension split:**
  - AR1, mixture and `SmoothDistribution` fits trigger on Optimization alone.
  - Only the nonlinear least-squares paths need SimpleNonlinearSolve.
  - `fit_mle` without the extension raises a helpful error, not a bare MethodError (`src/fit_stubs.jl:31-37`).
- **Type names are unchanged** (`Smooth*`, author decision 9). The new types follow the same pattern.
- **`SmoothDistribution` is an open family protocol from 5.0.**
  - It accepts any `seasonal_family`, per-parameter degrees, an `adtype` including `NoAD()` with derivative-free solvers, and an additive `penalty`.
  - This is what lets the codepaper's EGPD arrive later without a breaking change. The EGPD normalization is not differentiable in κ, so it needs `NoAD()`; and its fit uses the penalty `(κ − 1)²/0.1`.
  - `SmoothDistribution` also covers what `david_proj` does today with `PeriodicNLL` + `fit_loss_optim`.
- **The spatial HMM takes an `rng`** in its likelihood, fit and `viterbi`, passed through to `mvnormcdf`, so QMC results become reproducible.

### Exports: 94 → about 43
- **Exported:**
  - bases and links;
  - `ParamSpec`, `curve`, `TimeVaryingFamily`, `seasonal_family`;
  - the `Smooth*` models plus `SmoothDistribution` and `SmoothLatentAR1`;
  - `OneHot`, `fit_mle`, `fit`, `SeasonalFit`, `FitHistory`, `converged`, `loglikelihoods`, `viterbi`, `initialize`;
  - StatsAPI verbs, `components`, `ncomponents`, `mixtures`, `n_to_t`.
- **`public` only:**
  - layout helpers and NLL objects;
  - the EM knobs;
  - solver plumbing (`solve_objective` is the old `fit_loss_optim`);
  - evaluated twins (`AR1`, `ARPeriodicHMM`, …), `SeasonalControls`;
  - shared kernels.
- **Removed:**
  - `TimeVaryingFit`, `mixture_weight_spec`, `mixture_weights`;
  - `Trig2ARPeriodicHMM*`, `randARPeriodicHMM`;
  - `fit_θ*`, `fit_mle_RO`, `fit_mle_all_slices` (replaced by `initialize`);
  - `μₜ ρₜ σₜ αₜ`, `params`, `parameters`, `nparams`;
  - the multisite `rand(::Array{AR1}, …)`, which moves to Blocks.

### Moves
| From → to | Item |
|---|---|
| SWG → SPSM | `_ar1_init`, `mix_ini`, `_fit_seasonal_gamma` → `SmoothDistribution` |
| SPSM → SWG Blocks | multisite AR simulation |
| SWG ext → Blocks | `cor_groupby*` → `SmoothLatentAR1(x = ε)` |
| stays in SPSM | the spatial HMM |

**Deferred and additive:** the `EGPD` and `Spliced` families (5.x, for the codepaper model), `SoftmaxLink` and `Conditional` covariates (5.1), `CorrelationLink` (5.2, needed for seasonal cross-correlations in MSTWeatherGen `:seasonal`), and the `CachedLogDensity` bridge that replaces the override of the undocumented `HMMs.obs_logdensities!` (5.1). TimeVaryingParameters stays vendored.

---

## 7. Release mechanics, migration, equivalence

1. **SPSM 4.2, additive, registered first.**
   - The SWG libraries declare compat `"4.2"`.
   - A per-method table lists what exists in 4.2 and in 5.0.
   - SPSM's `Downstream` CI job runs all four SWG libraries and fails on a ResolutionError instead of skipping, plus a nightly SWG lane against SPSM `main`.
2. **SWG `next` branch.** The libraries and the 2.0 umbrella live there, while `master` keeps the legacy code.
3. **Version labels.**
   - `master` stays on the 1.x line (1.4.0, which can be registered as a last legacy release).
   - On `next`, the root `Project.toml` says `version = "2.0.0"`. 2.0.0 is sequential after 1.x, so AutoMerge accepts it.
   - Remove the remaining 3.0.0 mentions from CHANGELOG, CITATION.cff and the docs.
4. **Cutover branch.** The libraries' compat becomes `"4.2, 5"`, and their CI runs against the SPSM v5 branch through a temporary `[sources]`.
5. **Pre-registration** at least 4 weeks before the release, in the order Core → Blocks → classes (parallel) → umbrella 2.0.0. Fixtures are pinned by git tree.
6. **Release day:**
   - merge and register SPSM 5.0.0;
   - library patches widen their compat;
   - merge `next` into `master`, which deletes `src/rain`, `src/AR1`, `src/temperature` and `ext/SmoothPeriodicStatsModelsExt`;
   - register SWG 2.0.0;
   - one announcement for both packages.
7. **Migration guide**, in both repos:
   - generated mechanically from `names(M)` for **SWG 1.3.1 (registered), SWG 1.4.0 (legacy master) and SPSM 4.1**;
   - a CI test asserts that every old export has a row (keep, rename, move or remove, with its replacement);
   - rows include `fit_mle(m, Ys, Y_pasts)` → `fit_mle(m, Ys; past = Y_pasts)` (used in `examples/tuto_genomics.jl`), `rand(…; seq, z_ini)`, and `viterbi(m, Ys, Y_pasts)`.
8. **Numerical equivalence** against legacy fixtures (`test/reference/make_legacy.jl`, environment pinned by git tree to legacy SWG master + SPSM 4.1.0, synthetic data, no network). The `preset = :legacy` models are compared with the legacy code:
   - **L1, parameters:** rtol 1e-8 (closed form), 1e-6 (regime HMM, occurrence given ẑ), 1e-4 (optimizer fits). WGEN `counting = :legacy` tables use `==`. Ω is compared with a 10⁶-day legacy reference in Monte Carlo SE units. ẑ labels must equal `z_hat_K_4_d_1_m_1.csv` exactly.
   - **L2, pathwise:** an unexported `noise =` hook injects pre-drawn ε/U and must match SPSM 4 `rand(ARs, ϵ, …)`, `rand_cond`, occurrence thresholding and `simulate_markov_gaussian`.
   - **L3, distributional:** about 200 statistics over 300 members, Bonferroni-corrected.
   - **Unavoidable differences** that even `:legacy` cannot remove, each with a property test:
     - AR1 margins drop gap terms;
     - the `cov_ar1` first-residual fix;
     - the TVP average objective;
     - explicit regime index instead of relying on `groupby(:z)` order;
     - `past` and burn-in;
     - common random numbers in WGEN Ω;
     - the new RNG streams.
9. **Crop-tutorial yields.** In P0, ship the legacy weather ensemble that was fed to STICS as a hashed artifact next to the yields. The yield sections load that ensemble and assert the hash. New simulations appear only in the generator sections.

---

## 8. Implementation phases

Path roots: **SPSM** = `…\dev\SmoothPeriodicStatsModels`, **SWG** = `…\dev\StochasticWeatherGenerators`.

**P0. Safety net (no API change)**
- Commit the in-progress 1.4.0 edits that are now uncommitted in the SWG working tree, then branch `next` from `master`.
- Add SWG CI: the §1.1 matrix, Aqua, `@time_imports` budgets, the TagBot `subdir` matrix, and CompatHelper/Dependabot with subdirs.
- Add `SWG\test\reference\make_legacy.jl` plus fixtures.
- Convert `assets\tuto_1\hmm_fit_K_4_d_1_m_1.jld` and `z_hat_K_4_d_1_m_1.csv` into a `pretrained` JSON artifact (10 stations).
- Add the legacy STICS weather artifact.
- Make the Météo-France download lazy (`src\data\extracting.jl:176`).
- Save this plan, the full workflow design and the three research reports (monorepo practice, codepaper model, MSTWeatherGen) to `SWG\docs\design\`, so they are durable. They currently sit in the session scratchpad, which can be cleared.

**P1. SPSM 4.2 (additive)**
- **Files:**
  - `src\fit\ar1.jl` (typed weights, `OneHot`, sequences, `past`);
  - `src\AR\AR1.jl` (keyword `rand`, N×K `loglikelihoods`);
  - `src\fit\mixture.jl`;
  - `src\fit\smooth_hmm.jl` (size constructors, `initialize`, `fixed = (:A,)`, `past`);
  - `src\fit\add_sites.jl`;
  - new `src\fit\distribution.jl`, `src\AR\latent_ar1.jl`, `src\onehot.jl`;
  - the extension split in `ext\SmoothPeriodicStatsModelsOptimizationExt.jl` and `src\fit_stubs.jl`.
- **Reuse:**
  - `update_B!` and the `fit_mle_RO` logic (`src\fit\add_sites.jl:13-36`);
  - `ar1_sufficient_statistics` (`src\fit\ar1.jl:74-76`);
  - `fit_mle_all_slices` (`src\HHMM_Bernoulli\mle_slice.jl:16`);
  - SWG `_fit_seasonal_gamma` (`ext\SmoothPeriodicStatsModelsExt\fit_mle_TN.jl:34-43`).
- **Exit:**
  - SPSM 4 tests unchanged;
  - OneHot ≡ z-vector AR1, bitwise;
  - weighted A-fixed M-step ≡ `fit_mle_RO`;
  - `SmoothDistribution(Gamma)` ≡ SWG `fit_TN` margins;
  - `SmoothLatentAR1` recovers ρ(t);
  - an integer third argument throws.

**P2. Core**
- Implement `WeatherData` (with site metadata and distances), the calendars, the verbs, the ensemble driver, the RNG derivation for fitting and simulation, persistence (type registry, compact arrays), and a toy class in the tests.
- Implement the §4.2 slots that belong to Core: multi-output routing, the regime-source and capability traits, `sink` partitions by time chunk.
- **Reuse:** `dayofyear_Leap` (`src\utilities.jl:15`); `longuest_spell`, `monthly_agg`, `VCX3` (`src\climate_indices.jl`), renaming `longuest_spell` to `longest_spell`.

**P3. Blocks**
- Implement the protocol, `MultisiteGaussian`, the censored latent kernel, `ProductGenerator`, `Regimes`/`NoRegimes`, `TwoStage`, `show`, the `to_plain` registry, and every variant of §4.1.
- Implement `Hurdle` with `TwoPart` and `Censored`; `Rain` and `Positive` are presets of it.
- Implement the §4.2 slots that belong to Blocks:
  - `AbstractDependence` / `AbstractStructure` with the `cov!(…; lag, vars)` signature;
  - constrained `draw!`;
  - the dependence estimator protocol;
  - derived-quantity `inputs`;
  - `latent_past`;
  - the factor cache;
  - `cdf`/`quantile`/`pit` on marginals.
- **Exit test:** a toy block and a toy class defined outside the repo work against the `public` protocol.
- **Reuse:**
  - `Σ_Spearman2Pearson`, `cor_RR` (`src\rain\correlations.jl:34,102`);
  - `cov_ar1` with the gap fix (`src\AR1\fit_mle.jl:6`);
  - the `rand_RR` structure, including the regime-K-independent variant as `independent = (K,)` (`:31`);
  - `rand_cond`;
  - SPSM `norm_cdf_2d_vfast`.

**P4. RegimeHMM and the new tutorials**
- Implement the `RegimeHMM` preset with `preset = :legacy`, plus `read_stations`, `pretrained` and `example_data` in the umbrella.
- Fix the longitude bug in `src\data\stations.jl:53`.
- Write new versions of `tuto_add_station_variable`, `tuto_paper` and `tuto_paper_HMMspatial`; the spatial tutorial goes through `Regimes(hmm, path)`.
- **Exit:** a time and memory budget at 37 sites × 50 years × 300 members.

**P5. WGEN**
- **Reuse:** `fit_markov_chain`, `simulate_markov_gaussian` and the `fit_Ω` logic (`src\rain\wgen.jl:84-212`).

**P6. CensoredGaussian**
- **Exit:** simulate → refit recovery with NoRegimes and with K = 2 under switching regimes.

**P7. Docs and migration guide**
- Rewrite `SWG\docs\make.jl` (per-package API pages, `checkdocs = :public`) and `docs\src\models\*.md` (the §4.1 catalogue; each TODO mapped to a reserved extension name).
- Update the SPSM docs and its 5 tutorials.

**P8. Pre-registration**

**P9. Release day** (§7, step 6)

**P10. Follow-ups**
- SWG 2.1: `validate`, Posterior, FFBS for `given`, `Regimes(emission = :spatial)`, the `stepper` streaming API, multi-sequence fits.
- SPSM 5.1: `SoftmaxLink`, `Conditional`, `CachedLogDensity`, `seq_ends` for AR1, the mixture and `SmoothDistribution`, the `EGPD` and `Spliced` families.

**P11. The codepaper model** (`…SpaceTime` + `…SpatialHMM`, at 0.x)
- **Port** from HMM_Spatial_codepaper:
  - `23precip_intensity/covariance_functions.jl`: the Gneiting–Matérn and ExpExp structures, the conditional pairwise censored likelihood, `simulate_iterative`;
  - `Simulation_study_bootstrap/david_proj/rain_margins.jl`: the EGPD;
  - `gaussian_field.jl`.
- **Fix the convention traps** the research found:
  - the paper code uses B = 1/(1+exp(P)), the opposite sign of SPSM ≥ 4's `logistic(P)`;
  - Y_past layout `order×D` vs SPSM's `D×order`;
  - a day-1 initial condition mismatch;
  - two different Matérn scalings;
  - `GaussianField` reads a global instead of its argument.
- **Converter:** turn the paper artifacts into a `pretrained(:france37_spatial_K4)` artifact, negating θᴮ and the p and ξ of the margins.
- **Exit:**
  - parameters match `02_covariance_estimation.jl:91-96` within optimizer tolerance;
  - the simulation statistics match `03_simulation.jl`;
  - the cost stays within the measured 2 700 s for 500 members × 51 years.

**P12. MSTWeatherGen** (`…MSTWeatherGen`, at 0.x)
- `preset = :paper` first, against the R package on its vignette data.
- Then `:seasonal`, which needs SPSM `CorrelationLink`.

---

## Verification

- **Per phase:** the exit tests above, run with `julia --project=lib/X -e "using Pkg; Pkg.test()"` per library and `julia --project=. -e "using Pkg; Pkg.test()"` for SPSM.
- **Edge-case tests**, as required in addition to round trips:
  - a `missing` value inside the window raises the guided error;
  - stations with different date ranges fall back to the common window, with an info line;
  - 29 February under `Leap366` and `NoLeap365`;
  - one-site wet days and wet sets of varying size;
  - non-PD correlation estimates under both the `nearest_cor` and the `independent` variants;
  - |ρ| → 1;
  - a regime with zero weight for a block;
  - TN ≥ TX or δ ≤ 0 in the input;
  - integer weights passed to SPSM;
  - `past` shorter than the memory, or censored;
  - `given` without `regime`;
  - a frozen regime path that does not cover the data;
  - seeded output independent of the thread count and of `members` slicing;
  - JSON round-trip of frozen regimes, of a user block, of site coordinates and of the distance matrix, plus an unknown type;
  - a toy block and a toy class defined outside the repo;
  - a `Hurdle` on a non-rain variable (ETPP) with both `TwoPart` and `Censored`;
  - every T0/T1 name resolving with `using StochasticWeatherGenerators, Distributions, SmoothPeriodicStatsModels, DataFrames`.
- **End to end:**
  - the T0 crop tutorial runs in ≤ 6 lines;
  - the frozen-overlap report equals the legacy `innerjoin`;
  - `preset = :legacy` passes the L1, L2 and L3 suites, apart from the unavoidable differences;
  - the default (improved) models pass recovery tests on data simulated from themselves;
  - all three classes fit and simulate on the student dataset;
  - the docs build in CI;
  - every tutorial runs.
- **CI lanes:**
  - a "registered siblings" lane strips `[sources]` and resolves the siblings at their lower bounds;
  - Downgrade runs per library;
  - SPSM `Downstream` runs all four libraries.

---

## Open decisions (recommended default in **bold**; can be settled during implementation)

1. **QQ marginal.**
   - **A Gamma hurdle at launch.** A Beta bounded by clear-sky radiation comes in 2.1 and needs a formula from the author.
2. **WGEN Ω default estimator.**
   - **MomentMatching with common random numbers** (faithful to Wilks).
   - `PairwiseLikelihood` is also available.
3. **CLG parametrization.**
   - **Anamorphosis with seasonal ρ.** The author confirms it against Vaittinada Ayar 2020 and Ailliot 2009 before 1.0.
4. **Default K without pretrained regimes.**
   - **K = 4, with a warning below 8 stations.**
