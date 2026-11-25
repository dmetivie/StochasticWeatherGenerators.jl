# SPSM v5 design brief: synthesis of six reader reports

Path roots: **SPSM** = `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels`, **SWG** = `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators`, **TVP** = `SPSM\src\TimeVaryingParameters`. Unless a claim is marked *(inferred)*, the reader read it in the code, and nothing was executed. **[spot-checked]** means I re-read the cited lines myself.

### Spot-check results and corrections

| Claim | Status |
|---|---|
| The AR(1) noise law is fixed because TVP has no covariates (SPSM/src/AR/AR1.jl:53-58) | **[spot-checked] correct** |
| The multisite `rand(ARs, ϵ, n2t, z, Σ; y₁)` ignores Σ (AR1.jl:147-156) | **[spot-checked] correct.** Σ is unused in the body. The default `y₁` reads `AR.σ[1]`, which is σ[1,1] whatever `z[1]` is (AR1.jl:123). |
| `rand(::AR1{<:AbstractMatrix}, N::Integer)` fails | **[spot-checked] confirmed by reading.** AR1.jl:31 sets `T = size(AR.μ,1)`, which is K, then calls `rand(rng, AR, n2t; y₁)`. Only `AR1{<:AbstractVector}` has that method (AR1.jl:59), so this should be a MethodError. Not executed. |
| `fit_mle`/`fit` are excluded from the helpful solver stub | **[spot-checked] correct** (fit_stubs.jl:31-37) |
| The M-step minimises the *average* NLL, and the source explains why | **[spot-checked] correct** (fit/mixture.jl:185, 190-197) |
| TVP `fit` minimises the unnormalised objective | **[spot-checked] correct as far as fit.jl goes:** fit.jl:139-149 does not normalise. I did not re-read PeriodicNLL itself. |
| `rand` argument order is reversed between AR(1) and HMM | **[spot-checked] correct:** `rand(rng, AR, n2t, z)` at AR1.jl:71 vs `rand(rng, hmm::ARPeriodicHMM, z, n2t)` at HHMM_Bernoulli/periodichmm.jl:~69-72 |
| The SWG tutorial `.md` is stale | **[spot-checked] correct:** md:282 has `y_ini=y_ini`, while jl:324 has `Y_past` |
| Version mismatch | **[spot-checked] correct:** Project.toml:3 is `4.1.0`, CHANGELOG.md:7 is `[4.0.1] - Unreleased` |
| Copulas compat | **[spot-checked]** `Copulas = "0.1, 1"`, `HiddenMarkovModels = "0.7"`, `julia = "1.12"` |
| **CORRECTION:** the docs reader says SWG `fit_TN` uses TVP's `fit(fam, …)` | **Wrong.** `_fit_seasonal_gamma` builds a `PeriodicNLL`, divides it by N, calls `fit_loss_optim`, then wraps the result in `TimeVaryingFit(fam, sol.u, sol)` (SWG/ext/SmoothPeriodicStatsModelsExt/fit_mle_TN.jl:34-43). The core reader's version is the right one. |
| **CORRECTION:** "about 99 exports" (docs reader) | **94.** I counted the export lines at SmoothPeriodicStatsModels.jl:51-199; this matches the core reader. |
| **CORRECTION:** the prior-art reader suggests `Compat.@compat public` for Julia 1.10 | **Not needed.** SPSM already requires `julia = "1.12"`, so the `public` keyword is available natively. |
| SWG `fit_TN` groups by `groupby([:z])` and simulation indexes `θ_uni[k]` by the regime label | **[spot-checked] code confirmed** (fit_mle_TN.jl:28, :109). Whether the group order actually differs from 1:K is still *(inferred)*. |

---

## 1. Architecture today

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ L0  PARAMETRIZATION  (vendored TVP, SPSM/src/TimeVaryingParameters)                  │
│     θ_j(t) = link_j⁻¹(B_j(t)·β_j),  t ∈ 1:T integer period index                      │
│     AbstractBasis: Constant, Polynomial, Fourier, TrigBasis(T,d), Indicator,          │
│       BSpline, CyclicBSpline, `+` → CombinedBasis           (bases.jl:7-342)           │
│     Links: scalar→scalar only (IdentityLink, LogLink, LogitLink, IntervalLink)        │
│     ParamSpec{names}: flat coefficient vector + block offsets  (spec.jl:39-59)        │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ L1  FAMILIES / OBJECTIVES                                                             │
│     TimeVaryingFamily(D|closure, spec) → fam(t,θ)::Distribution  (family.jl:31-52)    │
│     PeriodicDesign, PeriodicNLL(fam,design,y,groups[,w]), PointwiseNLL (147-222)      │
│     seasonal_family(D, basis)                         (seasonal_mixture.jl:127-160)   │
│     update_dist  = generic weighted M-step            (fit/mixture.jl:179)            │
│     hand-coded kernels: bernoulli_logit_nll, multinomial_logit_nll (trig_kernels.jl)  │
│     softmax ×4, trig evaluation ×3 (TrigBasis, trig_basis, polynomial_trigo)           │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ L2  MODELS  (Smooth* = holds `ps::ComponentArray` / evaluated twin over 1:T)          │
│     SmoothAR1 / AR1 (K columns = per regime)         ── not an HMM emission           │
│     SmoothSeasonalMixture / Vector{MixtureModel}      ── iid, univariate, via EM.jl    │
│     SmoothARPeriodicHMM / ARPeriodicHMM               ── Bernoulli + memory, hand logit│
│     SmoothsARPeriodicHMM / sARPeriodicHMM             ── per-site order, NamedTuple ps │
│     SmoothARPeriodicHMMSpatial / ARPeriodicHMMSpatial ── latent-Gaussian, QMC E-step   │
│     SmoothPeriodicHMM / PeriodicHMMDists (BlockProduct) ── any family, no memory       │
│     HMM glue: SeasonalControls(t, lag_cat, n, t_prev), CachedEmissions,                │
│       RobustEmissions, _baum_welch!, _update_transitions!, viterbi(a,A,LL)            │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ L3  FITS  (extension SmoothPeriodicStatsModelsOptimizationExt: Optimization+SNLS)      │
│     fit_mle(m, data...; n2t, solver) → (m, FitHistory)       [main convention]        │
│     fit(SeasonalFit, m, data...) → StatsAPI.StatisticalModel (aic/bic/dof)            │
│     exceptions: TVP fit → TimeVaryingFit; fit_loss_optim/OptimMLE → SciML sol;        │
│       fit_mle_RO → (B, θᴮ); fit_θ → (θᴬ, θᴮ); Bernoulli fit_mle_all_slices → evaluated │
│     4 solver plumbings: TVP.solve_objective, solve_unconstrained, OptimMLE, inline    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ L4  DOWNSTREAM SWG GLUE  (SWG/ext/SmoothPeriodicStatsModelsExt + SWG/src)             │
│     regimes: ARPeriodicHMM fit on 37 ECA stations → Viterbi ẑ saved as CSV            │
│     given ẑ:  fit_mle_RO (RO) · fit_mle_RR + cov_RR (RR) · fit_AR1 + cov_ar1          │
│       (TX,QQ,ETPP) · fit_TN + cor_groupbyTXTN (TN via ΔT)                             │
│     simulate: rand(hmm) · rand_RR · rand(ar1s,n2t,z,Σ;y₁,output_ϵ) · rand_cond         │
│     these return 6 different types; the (model, history) pair is thrown away everywhere│
└──────────────────────────────────────────────────────────────────────────────────────┘
Upstream: HiddenMarkovModels 0.7 (E-step, viterbi), ExpectationMaximization 0.3 (mixture
driver), Copulas 1.x (GaussianCopula/SklarDist), Distributions 0.25, PeriodicHiddenMarkovModels 0.4
(n_to_t only), ComponentArrays 0.15, StatsAPI 1.8.
```

---

## 2. Public API inventory (94 exports, SmoothPeriodicStatsModels.jl:51-199)

Legend: **K** = keep public, **I** = make internal (`public`, not exported, for experts), **R** = remove, **N** = rename candidate.

**L0 parametrization (TVP re-exports, lines 51-54)**

| Names | Mark | Reason |
|---|---|---|
| ParamSpec, TrigBasis, ConstantBasis, FourierBasis, PolynomialBasis, IndicatorBasis, BSplineBasis, CyclicBSplineBasis | K | The expert's per-parameter vocabulary. v5 adds vector-valued links next to them. |
| IdentityLink, LogLink, LogitLink, IntervalLink | K | Needed to override links. Defaults should be inferred from the parameter's support. |
| curve | K | The core read-out verb. It must exist for every model (it is missing for the Bernoulli HMMs) and have a single arity. |
| paramnames, design_matrix | I | Useful to experts, noise to novices. |
| ncoef, blockindex, blockrange, nbasis | I | Layout internals. |
| coefs | I/N | Near-collides with `coef`. |

**L1 families / objectives (lines 57-58, 115)**

| Names | Mark | Reason |
|---|---|---|
| TimeVaryingFamily | K | The declarable seasonal law, copula closures included. |
| seasonal_family | K/N | The extension point for new laws. Could become `family(D; μ=…, σ=…)`. |
| PeriodicDesign, PeriodicNLL | I | M-step plumbing. Experts need them to write `update_dist`. |
| PointwiseNLL, group_by_time | I | Plumbing (group_by_time is used in only 2 of ~10 places). |
| distributions | I/N | Too generic a name for an export. |
| TimeVaryingFit | R | Second "fitted" container. Merge into SeasonalFit. |
| mixture_weight_spec | R | Replace with a softmax link inside ParamSpec. |

**L3 params / fit (lines 68, 96, 102-108, 198-199)**

| Names | Mark | Reason |
|---|---|---|
| fit_mle, fit, FitHistory, converged, parameters, SeasonalFit | K | The v4 contract. SeasonalFit is a candidate to become *the* single result type. |
| coef, loglikelihood, dof, nobs, aic, aicc, bic | K | StatsAPI. |
| ClassicEM, StochasticEM | I | Mixture-only knobs. |
| nparams | K/N | `nparams` means the number of coefficients here but the number of named parameters in TVP. |
| params | N | `params(m, t)` sits next to `parameters(m)`, which is confusing. |
| ncolumns, column | I | Layout. |
| default_nlls_solver, default_adtype, default_ensemble, solve_unconstrained, solve_nlls | I | Solver plumbing. |
| fit_loss_optim | I/R | Returns a raw SciML solution. Only SWG uses it. |
| n_to_t | K→N | Keep as the fallback, but v5 should take Dates or a calendar object. |

**L2 models (lines 115-117, 164-199)**

| Names | Mark | Reason |
|---|---|---|
| SmoothAR1 | K/N | Core block. The "Smooth" prefix is redundant once twins are internal. |
| ar1_spec | I | Layout. |
| ar1_sufficient_statistics, ar1_nll | I | Kernels. |
| AR1 | I | Evaluated twin. SWG uses it today. |
| SmoothSeasonalMixture | K/N | Rename to drop the prefix. |
| mixture_weights | N | Fold into `curve(m, :weights)`. |
| mixtures, components, ncomponents | K | Read-outs. |
| component_coefs, component_range | I | Layout. |
| SmoothPeriodicHMM | K | The natural generic seasonal HMM. |
| PeriodicHMMDists, emission_coefs, emission_range | I | Twin and layout. |
| SmoothARPeriodicHMM, SmoothsARPeriodicHMM, SmoothARPeriodicHMMSpatial | N/merge | Candidates to become emission blocks of one generic HMM (§7, D6). |
| degrees | I | |
| ARPeriodicHMM, sARPeriodicHMM, ARPeriodicHMMSpatial | I | Evaluated twins. Note that SWG builds `ARPeriodicHMM(a, A, B)` by hand (tuto .jl:209). |
| Trig2ARPeriodicHMM, Trig2ARPeriodicHMMSpatial | R | Duplicate the `ARPeriodicHMM(a, θᴬ, θᴮ, T)` constructor. |
| randARPeriodicHMM | R | Replace with `(K, D, order, d, T; init)` constructors. |
| SeasonalControls | I | HMMs.jl bridge. |
| loglikelihoods | K | The N×K per-state log-density is the core block protocol. |
| viterbi | K | |
| fit_θ, fit_θ!, fit_θᴬ!, fit_θᴮ! | R/I | Only needed because the Bernoulli slice start returns an evaluated model. |
| fit_mle_RO | R | Replace with `fit_mle(block, Y, z; n2t)` returning a model. SWG pirates it today. |
| fit_mle_all_slices | N | Rename to an `initialize`-type verb that returns a smooth model for every HMM. |
| μₜ, ρₜ, σₜ, αₜ | R | They take t as a fraction of the period, unlike everything else (utilities.jl:34-74). SWG still uses them (fit_mle_TN.jl:103). |

Roughly 40 of the 94 names are I or R.

---

## 3. Pain points, ranked

| # | Issue | Evidence | Who | Sev |
|---|---|---|---|---|
| 1 | TVP has no covariates or conditioning (lagged y, regime z, parent variable). Because of this, the AR(1) NLL is hand-written: it reads blocks by position and takes the third as log σ, and the noise law cannot be changed. | AR1.jl:53-58 [spot-checked], 359-371; tuto_ar1.md:142-144 | both | high |
| 2 | Only SmoothAR1 supports "given z". The mixture, SmoothPeriodicHMM and the spatial model cannot take z. The Bernoulli case goes through `fit_mle_RO`, which returns arrays. SWG splits the data by z by hand. | fit/ar1.jl:66 vs fit/mixture.jl:21, fit/smooth_periodic_hmm.jl:118, fit/add_sites.jl:13-36; SWG fit_mle_rain.jl:41-52 | both | high |
| 3 | The multisite Gaussian-copula AR(1) exists only as a simulator over evaluated `Array{AR1}`, with one constant Σ_k per regime. It has no fit, likelihood or residuals, and no test. The ϵ-method ignores Σ, and the default y₁ ignores z[1]. SWG's `cov_ar1` does the fitting and has gap and n=1 bugs. | AR1.jl:123-156 [spot-checked]; SWG src/AR1/fit_mle.jl:6-30 | both | high |
| 4 | Fit outputs come in at least 5 shapes in SPSM: (m, FitHistory), TimeVaryingFit, SciML solution, (B, θᴮ), evaluated HMM or (hmm, Dict). SWG then produces 6 result types and drops every history. | fit/add_sites.jl:35; mle_slice.jl:54; optim_mle.jl:12-20; TVP fit.jl:150; SWG fit_mle_TN.jl:43, 84-88 | both | high |
| 5 | Only scalar links exist. Softmax rows (transitions, mixture weights) and correlation matrices are not ParamSpec citizens, so the softmax is written 4 times and a seasonal copula would need D(D−1)/2 named scalars. The link declared for the mixture weights is silently ignored. | spec.jl:13-14; family.jl:110-114; hmm_coefficients.jl:20-37; seasonal_mixture.jl:169, 292-309 | expert/maint | high |
| 6 | One call shape means different things. The third positional argument of `fit_mle` is z (AR1), weights (mixture) or Y_past (HMM). `rand` takes (n2t, z) for AR1 but (z, n2t) for HMMs, and both are `Vector{Int}` of the same length. | fit/ar1.jl:66; fit/mixture.jl:21; fit/smooth_hmm.jl:227; AR1.jl:71 vs periodichmm.jl:~69 [spot-checked] | both | high |
| 7 | A fitted seasonal model cannot be nested as an EM component or HMM emission. EM and Distributions expect `fit_mle(d, y, w)` to return a distribution, but SPSM returns (model, history). | EM classic_em.jl:116, 127; Distributions genericfit.jl:8-31; SPSM fit/mixture.jl:21 | expert | high |
| 8 | Novices must build 4-D logit arrays for the Bernoulli HMM (the core of the weather generator), with K as the softmax reference. Starting from data takes three steps (slices → fit_θ → rebuild) and uses the misspelt `Yₜ_extanted`. | HHMM_Bernoulli/smooth.jl:50-53; tuto_hmm.md:131-136, 191-193; mle_slice.jl:18 | novice | high |
| 9 | Simulation state is threaded by hand: z goes to 5 samplers, along with n2t, T, a Σ per variable, the ϵ_TX side channel and per-site loops. The initial-condition conventions conflict: Y_past means the days before the first simulated day, y₁ means the first simulated day itself. | SWG tuto .jl:285-342; periodichmm.jl:93-96 vs AR1.jl:59-63 | both | high |
| 10 | Silent wrong defaults: `n2t = n_to_t(N,T)`, which is wrong on wet-day subsets and the mixture tutorial never passes `n2t`; `y₁ ~ N(0, σ[1])`; transitions of degree 1 in the families form; `dα = 1`; `maxiter = 100`. | fit/mixture.jl:22; tuto_mixture.md:116-132; AR1.jl:30; HMM_Dists/smooth.jl:81; seasonal_mixture.jl:237 | novice | high |
| 11 | The AR(1) takes one contiguous vector, so gaps left by dropmissing are bridged silently. The mixture also takes a single sequence, while the HMMs accept several. | fit/ar1.jl:27, 66; AR1.jl:331-343; SWG fit_mle_AR.jl:1 (TODO) | both | high |
| 12 | Positive variables (QQ, ETPP) have no model. They are clipped with `max(x, 0)` after simulation, so the simulated law is not the fitted one. | AR1.jl:53-58; SWG tuto .jl:296, 338-341 | both | med-high |
| 13 | Containers and read-outs are inconsistent. Coefficient axes come first (AR1, mixture) or last (HMMs). Blocks are named by role (A, B, Y, α) or by parameter (μ, ρ, σ). SmoothsARPeriodicHMM returns a NamedTuple. The evaluated layout moves t (θᴮ[k,j,h,:] vs B[k,t,j,h]). `curve` is missing for the Bernoulli HMMs, and `curve(m, :α)` means Gamma's shape, not the weights. | params.jl:49-53; trig_conversion.jl:7-10; sHHMM_Bernoulli/smooth.jl:49; tuto_hmm.md:85 vs 149 | both | med |
| 14 | Four HMM families with copied code: the state sampler ×4, slice EM ×2, fit contexts ×3, emission M-step ×3, plus a dead ragged `update_B!`. The Bernoulli HMMs bypass TimeVaryingFamily. | HHMM_Bernoulli/periodichmm.jl:98-108 and 3 twins; mle_slice.jl ×2; fit/smooth_hmm.jl:128, 270, 397; hmm_coefficients.jl:41-47 | maint | med |
| 15 | The surface is too large and inconsistent: 94 exports, about 40 of them internal or legacy, Unicode names, two time conventions (integer t vs fraction t/T), and four solver plumbings with two different missing-solver messages. Without the extension, `fit_mle` gives a bare MethodError. | SmoothPeriodicStatsModels.jl:51-199 [counted]; utilities.jl:34-74; fit_stubs.jl:31-37 [spot-checked]; TVP fit.jl:10-35 | novice/maint | med |

Also worth noting:
- SPSM overrides the undocumented `HMMs.obs_logdensities!` (hmm_interface.jl:116 and others). Maintainer risk, high.
- The Copulas compat `"0.1, 1"` is only nominal: the copula test presumes 1.x (runtests.jl:1772-1775).
- SWG's regime-indexed fits rely on `groupby` order (fit_mle_TN.jl:28, :109).
- `rand_RR` hard-codes regime K as independent (SWG generation.jl:31).
- A network download runs at SWG precompile time (extracting.jl:176-178).
- There is no save/load API (2-fitting.md:203-217).
- There are no tied or fixed coefficients (grep found nothing).

---

## 4. The target model, formally

**Indices.** n = day; t_n = n2t[n] ∈ 1:T (T = 366, leap convention); k ∈ 1:K regimes; v ∈ 𝒱 = {RO/RR, TX, TN, QQ, ETPP, …}; s ∈ 1:D_v sites.

**Seasonal parameter (L0).** θ_j(t) = g_j⁻¹(B_j(t)·β_j), where g_j can be a scalar link *or* a vector link (softmax, correlation). The GAP is the vector links.

**(Z) Regime chain.**
- P(Z_1 = k) = a_k.
- P(Z_n = l | Z_{n−1} = k) = A_{kl}(t_{n−1}) = softmax_l(B_A(t_{n−1})·θᴬ_{k,·}), with reference l = K and SPSM's "leaving" convention.
- *Optional:* a dependence on covariates (NHMM). This is a GAP.

**Emission, given the regime path (product form).**

p(y_{1:N} | z_{1:N}) = Π_n Π_{b ∈ blocks} p_b( y_n^b | z_n, t_n, y_{n−1}^b, y_n^{pa(b)} )

The blocks follow a DAG: pa(TN) = {TX}, pa(RR) = {RO}. Given z, separate blocks are independent. SWG relies on this: tuto .jl:20 says the regimes plus seasonality are enough to correlate the variables.

**(RO) Rain occurrence**, per site s:
- h_{s,n} = 1 + Σ_{i=1..m} 2^{m−i} O_{s,n−i} (memory category).
- P(O_{s,n} = 1 | Z_n = k, h) = logistic(B(t_n)·β_{s,k,h}).
- Cross-site dependence: either independent given Z (ARPeriodicHMM), or a latent Gaussian with correlation exp(−d_{ss'}/R_k(t)) (Spatial).

**(RR) Rain amount** on the wet set W_n = {s : O_{s,n} = 1}:
- R_{W_n,n} | Z_n = k ~ SklarDist(GaussCop(C^R_k[W_n, W_n]), (F^R_{s,k,t_n})_{s∈W_n}).
- F^R is a seasonal mixture of exponentials (or any seasonal family).
- Today C^R_k is constant in t and estimated pairwise (Spearman → Pearson).

**(X) Continuous variables with dynamics** (X ∈ {TX, QQ, ETPP}), two candidate forms:
- **(A) AR on the variable** (today): X_{s,n} = μ_{s,k}(t_n) + ρ_{s,k}(t_n) X_{s,n−1} + σ_{s,k}(t_n) ε_{s,n}, with ε_{·,n} | Z_n = k ~ N_D(0, C^X_k). It is Gaussian only and has no positivity.
- **(B) AR on a latent Gaussian** (a Gaussian-copula Markov process): W_{s,n} = Φ⁻¹(F_{s,k,t_n}(X_{s,n})), then W_{s,n} = ρ_{s,k}(t_n) W_{s,n−1} + √(1−ρ²) ε_{s,n}, with ε_{·,n} ~ N(0, C^X_k).
  - Any seasonal marginal F works, positive ones included, and the marginal is invariant.
  - It unifies "copula marginals" with "AR dynamics".
  - Option (B) is my inference from the readers' open questions; no code exists for it.

**(TN) Conditional variable.**
- Δ_{s,n} = TX − TN = F⁻¹_{Γ(α_{s,k}(t), θ_{s,k}(t))}(Φ(ξ_{s,n})), with ξ = r_{s,k}(t_n) ε^{TX}_{s,n} + √(1−r²) η_{s,n}.
- Alternatively, a joint copula over (variable × site) for (TX, TN) (the Kleiber 2013 style).
- Today r is fitted on raw cor(TX, TX−TN), a different quantity from the innovation-scale r used in simulation (fit_mle_TN.jl:75-80 vs :103).

**Open axes for the copula C:**
- per variable vs across variables;
- per regime (today) vs regime × season C_k(t);
- on innovations ε (form A) vs on PIT scores (form B).

**Estimation modes:**
- **(i) Two-stage plug-in** (today): fit Z on RO, decode ẑ with Viterbi, then fit each block given ẑ (one-hot weights), with copulas by IFM.
- **(ii) Soft weights:** γ from forward-backward used as M-step weights.
- **(iii) Joint EM:** every block contributes an N×K log-density.

**Factor → building block**

| Factor | Existing block (file) | Status |
|---|---|---|
| Seasonal transitions A(t), softmax | `_update_transitions!`, `multinomial_logit_nll` (fit/smooth_hmm.jl:175; trig_kernels.jl:65-81); SmoothPeriodicHMM | Exists. Softmax is not a ParamSpec link. |
| E-step / decoding | HMMs.jl `forward_backward!`, `SeasonalControls`, `CachedEmissions`, `viterbi(a,A,LL)` (hmm_interface.jl; viterbi.jl:84) | Exists |
| Product of blocks given z | `BlockProduct` / `PeriodicHMMDists` (HMM_Dists/periodichmm.jl:97-147) | Exists, but no memory, no DAG and no z-given fit |
| Seasonal marginal, any law | `TimeVaryingFamily`, `seasonal_family` | Exists. Per-parameter degree only through ParamSpec. |
| Weighted M-step for a marginal | `update_dist` (fit/mixture.jl:179) | Exists, internal |
| "Given z" for any block | one-hot γ pattern in `fit_mle_RO` (add_sites.jl:29-34) | **GAP**: only AR1 supports it |
| RO with memory, multisite independent | SmoothARPeriodicHMM emissions | Exists, hand-coded, bypasses TVP |
| RO with a latent-Gaussian spatial field | SpatialBernoulli, `pairwise_loglikelihood` | Exists, special EM loop |
| RR marginal (seasonal mixture) | SmoothSeasonalMixture | Exists. Univariate, no z, no multisite. |
| RR wet-set copula: fit | SWG `cov_RR` (correlations.jl:102-155) | **GAP in SPSM** |
| RR wet-set copula: simulate | SWG `rand_RR` (generation.jl:6-59) | **GAP in SPSM** |
| AR(1) per site × regime | SmoothAR1 with K columns, `fit_mle(m, y, z)` | Exists. Gaussian only, one sequence, no N×K log-densities, no weights. |
| Multisite copula on AR innovations: simulate | `rand(ARs, n2t, z, Σ)` (AR1.jl:123) | Exists, evaluated types only, untested |
| Multisite copula on AR innovations: fit/likelihood | SWG `cov_ar1` | **GAP in SPSM** |
| Latent-Gaussian AR with arbitrary marginal (form B) | — | **GAP** |
| Seasonal copula C_k(t) | — (only seasonal r(t), via ρₜ in SWG) | **GAP** (needs a correlation link) |
| Conditional block TN given TX | SWG `fit_TN`, `cor_groupbyTXTN`, `rand_cond` | **GAP in SPSM** (generic parts belong there) |
| Positive variables | — | **GAP** (clipping today) |
| Gaps / multiple sequences for AR | `_sequences` for HMMs (smooth_hmm_sequences.jl:8-39) | **GAP** for AR and mixture |
| Calendar / n2t from Dates | SWG `dayofyear_Leap`, SPSM `n_to_t` | Split across two packages |
| Composite object owning K, T, sites, RNG, initial state | — | **GAP** |
| Model selection | SeasonalFit + bic | Exists per component |

---

## 5. Hard constraints and soft preferences for v5

**Hard constraints (from the evidence):**
1. **HiddenMarkovModels.jl 0.7 interface.** Custom HMMs provide `initialization`, `transition_matrix(hmm, control)`, `obs_distributions(hmm, control)` and the **4-argument** `fit!(hmm, fb, obs, controls; seq_ends)`; the 3-argument fallback drops controls (HMMs baum_welch.jl:101-109).
   - The control at n drives the transition n−1 → n. SPSM stores A at the leaving period and carries `n2t[n−1]` in the control (hmm_interface.jl:70-97). The HMMs docstring describes the timing backwards (abstract_hmm.jl:76-78).
   - `HMMs.rand` fixes controls in advance, so AR or memory models need their own simulators.
   - Either stop relying on the undocumented `obs_logdensities!`, or get it made public upstream.
2. **Distributions.jl contract.** `fit_mle(D or d, x[, w])` returns a *distribution*. A v5 block that should nest inside EM or HMM emissions needs a model-only weighted refit (for example `fit!(block, data, w)` or `fit_mle(block, data, w)`). The user-facing `(model, history)` can stay a separate tier.
3. **Copulas.jl.**
   - `GaussianCopula{d}` has d ≥ 2 and d in the type, so one-site days and varying wet-set sizes need special-casing.
   - `logpdf` refactorises Σ on every call.
   - Use `fit(GaussianCopula, U; weights)` (fast Z W Zᵀ), not the EM extension's generic path.
   - The compat should move to `"1"`: 0.1 collapses diagonal Σ to IndependentCopula and has no weights.
   - Discrete margins in a SklarDist cost 2^k CDF evaluations, so rain must stay factored as occurrence × amount.
4. **StatsAPI.** `fit`, `fit!`, `loglikelihood`, `dof`, `nobs`, `aic`, `bic`, `coef`. SeasonalFit already mirrors Copulas' `CopulaModel`.
5. **Explicit solver.** This is a v4 decision (CHANGELOG 4.0.0). Optimization and SimpleNonlinearSolve stay weak dependencies.
6. **Average, not summed, weighted NLL** in every M-step (fit/mixture.jl:190-197). The TVP `fit` must follow.
7. **ComponentArray coefficients**, one container convention for every model. `parameters(m)` must never be a NamedTuple.
8. **The tested v4 behaviour** listed by the docs reader (runtests.jl): model in/model out, m0 not mutated, monotone EM with `logtots[end]` equal to the loglikelihood, cache invalidation when coefficients are edited, the lossless HMMs interface, the `update_dist` override.
9. **Julia ≥ 1.12** (Project.toml), so `public` can mark the expert surface.

**Soft preferences:**
- One verb per job: `fit_mle` / `fit`, `rand`, `loglikelihood(s)`, `curve`.
- Data in D×N, sites by time, matching HMMs, Copulas and SWG. Time-only data as a vector.
- One initial-condition convention: "the observations before the first step", generalising Y_past.
- Dates in, n2t inferred.
- Per-parameter bases via a NamedTuple keyed by parameter name, with an integer shorthand meaning trig degree.
- Links inferred from the parameter's support.
- Weighted M-steps everywhere, so "given z", "soft γ" and "joint EM" share one code path.
- Missing values handled by sequences or marginalisation; no upstream package does this.

---

## 6. Prior-art lessons (most transferable)

1. **One predictor per distribution parameter** (gamlss `sigma.fo`, momentuHMM `DM`, hmmTMB, brms `lf(sigma ~ …)`). This is already SPSM's core idea (97-related.md:32). Shape: `Marginal(Normal; μ = 2, σ = 1)`, where an integer means a trig degree and a basis object is the expert form. It fits because ParamSpec is exactly this; it only needs name-keyed sugar and per-parameter degree in the convenience constructors (today only ParamSpec allows it).
2. **Hidden-state object separate from observation objects** (hmmTMB `MarkovChain` / `Observation` / `HMM`; HMMs.jl `HMM(init, trans, dists)`). Shape: `Regimes(K; transitions = 1)` plus `(RR = …, TX = …)`. This matches SWG's real workflow: regimes are fitted once and frozen while stations and variables are added.
3. **Product emission, conditionally independent given the state** (depmixS4 `list(formulas)`, Distributions `product_distribution(NamedTuple)`). Shape: a NamedTuple of blocks. The E-step adds the blocks' log-likelihoods and the M-step splits per block. BlockProduct already does this; the step is generalising it to blocks with memory and a z-given fit.
4. **Explicit, declared cross-variable links** (brms `set_rescor`, Rglimclim's sequential conditional GLMs, the RMAWGEN option=2 reparameterisation TX/DTR). Shape: `TN = Temp(given = :TX)` or `Joint(:TX, :TN)`. It replaces the `ϵ_TX` side channel and `rand_cond`. GWEX shows the anti-pattern: the conditioning object is passed again at fit and at simulation time.
5. **Sklar layering fitted by IFM** (Copulas.jl `sklar_method = :ifm`, Wilks 1998, GWEX). Shape: `Multisite(block; copula = GaussianCopula(), by = :regime)`. One concept replaces `cov_RR`, `cov_ar1` and `cor_groupbyTXTN`, and IFM reproduces current SWG practice.
6. **Switching flags per parameter** (statsmodels `switching_variance`, MarSwitching `intercept = "switching"`). Shape: `ρ = 1 => Shared()`. It controls parameter count. Note that shared parameters need a joint weighted M-step across k.
7. **Hurdle / zero-inflation part with its own predictor** (glmmTMB `ziformula`). The rain block is occurrence (with memory) × amount on the wet set, matching SPSM and SWG today. Keep room for a single censored latent-Gaussian alternative (Ailliot 2009).
8. **One `rand` with conditioning** (Turing `model | (x = …)`). Shape: `rand(swg, dates; given = (z = ẑ,), init = (TX = y0,), nsim)`. It replaces `rand_RR`, `rand_cond` and `rand(ar1s, n2t, z, Σ; y₁)`. Conditioning on a child to simulate its parent must be rejected.
9. **Progressive disclosure** (Keras Sequential → Functional → subclassing; MLJ). Tier 0 is a facade, `fit(WeatherGenerator, table; K)`. Tier 1 is explicit composition. Tier 2 is a block interface plus the v4 `fit_mle`. `show` of a Tier 0 result prints its Tier 1 spec.
10. **Bridge release** (ColPrac: "introducing deprecations is not breaking; removing deprecations is breaking"). Since v4.0.0 is two days old, add the new layer in v4.x, deprecate, and remove in v5 once SWG has migrated.

---

## 7. Decisions the author must make

**D1. Where does the composite product model live?**
- (a) In SPSM, generic: Regimes, blocks, copula layer, DAG. SWG keeps only presets (RainBlock, TemperatureBlock), data I/O and plots.
- (b) Entirely in SWG.
- (c) A new package (for example `SeasonalHMMs` or `RegimeSwitchingProducts`).

Evidence:
- SWG's `cov_ar1`, `rand_cond`, `cor_groupby` and `_fit_seasonal_gamma` are generic (swg reader), which points to (a).
- SWG already pirates SPSM's `fit_mle_RO` (SWG ext SmoothPeriodicStatsModelsExt.jl:13).
- hmmTMB (generic) vs momentuHMM (domain-specific) is the precedent for the (a) split.
- (c) costs a third compat chain.

**D2. Estimation mode as the default.**
- (a) Two-stage plug-in ẑ (today).
- (b) Soft γ weights.
- (c) Joint EM over the product.

(a) is cheap and matches the paper. (c) requires every block to provide N×K log-densities with memory plus a weighted M-step, which SmoothAR1 lacks. Designing the block protocol around **weighted M-steps** gives (a) as one-hot weights and makes (b) and (c) cheap later. Recommendation-neutral: the choice is whether (c) is in v5.0 scope.

**D3. One generic seasonal HMM vs specialised ones.**
- (a) Merge all four HMMs into SmoothPeriodicHMM with emission blocks: memory through controls, a batched N×K path, a custom M-step order for the spatial model.
- (b) Keep the specialised types behind a common supertype.

Evidence:
- 4× and 3× duplication (§3 #14).
- The test "SmoothPeriodicHMM with Bernoulli emissions equals order-0 SmoothARPeriodicHMM" (runtests.jl:1485-1516) already ties them.
- Unknown: whether the Bernoulli sufficient-statistics speed can be recovered through an `update_dist` specialisation.

**D4. AR dynamics form for continuous variables.**
- (a) Keep AR on the variable, Gaussian (today).
- (b) Latent-Gaussian AR with arbitrary seasonal marginal (Gaussian-copula Markov).
- (c) Add covariates to TVP so any family can take y_{n−1}.

(b) solves positivity and unifies it with copula marginals. (c) is the general route the AR1.jl:53-58 comment points to, but it costs more engine work.

**D5. Vector-valued parameters in L0.**
- (a) Add SoftmaxLink and a correlation link (Cholesky or angles) to ParamSpec, enabling a seasonal C_k(t).
- (b) Keep correlations constant per regime (replicate axes only).

This decides whether ParamSpec needs matrix-valued parameters (core reader's open question). (a) also removes the 4 softmax copies.

**D6. Keep evaluated twins public?**
- (a) Make them internal; `curve`, `m(t)` and `rand` cover every model.
- (b) Keep them public.

SWG composes `ARPeriodicHMM(a, A, B)` and `AR1` directly today (tuto .jl:209), which is a migration cost.

**D7. Data input.**
- (a) Arrays (D×N) plus Dates or n2t.
- (b) Tables.jl long format (date, station, variable, value) at Tier 0, arrays at Tier 1/2.
- (c) DataFrames-specific.

(b) keeps SPSM free of DataFrames. The adapter could live in SWG.

**D8. Composition syntax.**
- (a) Keyword constructors, `RegimeSwitching(Regimes(4); RR = …, TX = …)`.
- (b) Operator DSL (`*`, `⊗`).
- (c) `@formula` / StatsModels.

HMM, copula and AR structure do not fit a formula well, and Julia operator overloading hurts discoverability. (a) is the safe default, and sugar can be added later without breaking anything.

**D9. Result type and verbs.**
- (a) Keep `fit_mle → (m, history)` as the only verb.
- (b) `fit(spec, data) → Fitted` (model + history + regimes + bic) for novices, with `fit_mle` as the expert path.
- (c) Make SeasonalFit the only result.

Evidence:
- The tuple is thrown away at every SWG call site.
- TimeVaryingFit and SeasonalFit duplicate each other.
- Nesting needs a separate model-only refit (constraint 2).

**D10. Naming conventions.**
- Drop `Smooth*` once twins are internal.
- "Seasonal" vs "Periodic": pick one.
- Use T everywhere, not nT.
- Pick one meaning each for `degree`, `order` and `d`.
- Remove Unicode exports (μₜ, fit_θ…).
- Initial conditions: one keyword, for example `past =` or `init_obs =`, replacing y₁, Y_past, y_past and z_ini.
- Solve the `parameters` / `params` / `coef` / `coefs` collisions.

**D11. Solver default.**
- (a) Keep it mandatory.
- (b) Default to LBFGS when OptimizationOptimJL is loaded, through an extension.

Mandatory hurts Tier 0. A default through an extension adds no hard dependency.

**D12. Depth of the break and migration.**
- (a) v4.x bridge (additive new layer plus `@deprecate`), then v5 removes.
- (b) Clean v5 now with a migration table.
- (c) v5 with temporary shims in SWG.

v4.0.0 shipped on 2026-11-21. Project.toml already says 4.1.0, the CHANGELOG says 4.0.1, and SWG compat requires 4.1, so a 4.x bridge is already partly in motion. Return-shape changes (a tuple becoming a Fitted object) cannot be mapped by `@deprecate` and must be documented hard breaks.

**D13. TVP ownership.** Vendored copy vs depend on the registered TimeVaryingParameters (upstream PR #10). Vector links and covariates have to go into one place. The vendored copy is byte-identical to fork commit 0594e38 except for the solver default (core reader, verified by diff).

**D14. Copula estimator defaults.** Weighted MLE on normal scores (Copulas `fit(GaussianCopula, U; weights)`) vs pairwise rank-based on co-wet days (SWG today, no Copulas equivalent). The choice also covers whether Nataf replaces SWG's `cor2cov` and Spearman→Pearson heuristics (`cor2cov` has no effect anyway: Copulas normalises).

---

## 8. Unknowns (nobody verified)

- Whether DataFrames `groupby(:z)` returns groups in sorted order in practice. This decides whether the current SWG `θ_uni[k]` / `θ_cor[k]` indexing is right (fit_mle_TN.jl:28, :109).
- The actual MethodError for `rand(::AR1{<:AbstractMatrix}, N)`: confirmed by reading, never executed.
- Whether the Bernoulli M-step's sufficient-statistics speed survives a move to `TimeVaryingFamily(Bernoulli)` with an `update_dist` specialisation. No benchmark exists.
- Whether TVP's `fit(fam, y; solver)` works with only OptimizationOptimJL loaded (ForwardDiff availability).
- How `SklarDist` and IFM treat a heterogeneous `MixtureModel([Dirac(0), Exponential()])`: whether its value_support resolves as Discrete or Continuous.
- How the HMMs.jl Distributions extension behaves on `Vector{SklarDist}` emissions. By reading, the weights are taken as `copula_method`, which would give an ArgumentError; not executed.
- Whether the regime-K independence in `rand_RR` is a modelling choice or a workaround for non-PD matrices.
- Whether the TX–ΔT correlation should be re-estimated on the innovation scale (the fitted and simulated quantities differ today).
- The performance of a Gaussian-copula E-step over N×K days with about 37 sites and varying wet subsets (a factorisation per call, and a type per dimension).
- How many downstream users besides SWG depend on v4 names. This decides how hard the break can be.
- Whether Copulas 0.1 is still used anywhere; the copula test would fail on it *(inferred)*.

### Critical Files for Implementation
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\TimeVaryingParameters\spec.jl (and family.jl, fit.jl in the same folder)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\AR\AR1.jl
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\HMM_Dists\periodichmm.jl (with src\HMM_Dists\smooth.jl, src\fit\smooth_hmm.jl, src\hmm_interface.jl)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\SmoothPeriodicStatsModels.jl (export surface)
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\examples\tuto_add_station_variable.jl (target workflow; the `.md` under docs\src\examples is stale)