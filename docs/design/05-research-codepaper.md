# HMM_Spatial_codepaper generator: formal model, pipelines and mapping onto SWG 2.0 / SPSM 5.0

**Path roots used in citations**
- **CP** = `c:\Users\metivier\Dropbox\PC (2)\Documents\GitHub\HMM_Spatial_codepaper`
- **DP** = `CP\Simulation_study_bootstrap\david_proj`. This is a clean rewrite of the pipeline on SPSM 4.1, pinned at `DP\Manifest.toml:1874-1879`.
- **SPSM** = `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels`
- **SWG** = `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators`
- **PLAN** = `C:\Users\metivier\.claude\plans\one-could-imagine-a-toasty-dove.md`

**Sources outside the code.** The HAL preprint page (hal-05523768) blocked automated access, so I could not read it. Apart from the code, the only paper text used is the SWGEN 2025 abstract by Cognot, Bel, Métivier and Parey: https://swgen2025.sciencesconf.org/data/pages/6_cognot_1.pdf. Its key sentences:
- "the spatial rain occurrence is modelled using a Gaussian field with a Matérn covariance, thresholded at each location"
- "Rain amounts are modelled with Extended Generalized Pareto Distributions (EGPD) in each weather states"
- "censored Gaussian process on normalized variables, conditioned on the occurrence process"
- "composite pairwise likelihood and the EM algorithm for the occurrence process, then with marginal and pairwise likelihood for the intensity process"

**Which code is canonical.**
- The final paper model is in `CP\13PeriodicHMMSpatialBernoulli` (`CP\README.md:36`) and `CP\23precip_intensity` (`CP\README.md:69`).
- `CP\complete_estimation_simrealparam.jl` is a work-in-progress script. It uses names it never defines: `covmodel` (:77), `locsdata` (:76), `di` (:98) and `Pdryhay` (:238).

---

## 1. Formal description of the generator

### 1.0 Data and notation
- **Sites.** D = 37 ECA&D stations in France.
- **Days.** n = 1…N over 1973-01-01…2023-12-31 (`CP\23precip_intensity\01_marginal_estimation.jl:26-31`). N = 18 627, the line count of `CP\00data\transformedECAD_zhat.csv`.
- **Season index.** t_n = `dayofyear_Leap(date)` with T = 366 (`01_marginal_estimation.jl:30-31,56`). This equals the plan's `Leap366` (PLAN:182).
- **Rain.** RR in mm is 0.1 × the raw value (`CP\13PeriodicHMMSpatialBernoulli\real_data_getZ.jl:110`). Occurrence is Y = 1{RR ≠ 0} (`onefy`, `real_data_getZ.jl:58,65`).
- **Distances.** h_ij is the haversine distance in km on (LON, LAT), from Distances.jl with its default radius and no rounding (`CP\00data\data_selection.jl:127-129`).

### 1.1 Hidden regime chain
- **States and selection.** Z_n ∈ {1..K} with K = 4. It was chosen by ICL over K ∈ 1:5, degree d ∈ 0:2 and memory m ∈ 0:1 (`CP\13PeriodicHMMSpatialBernoulli\test_real_data.jl:71-75`). The criterion is `icl_val = CLKdegp - log(length(ẑ)) / 2 * n_param_val`, where `CLKdegp` is the complete log-likelihood at the Viterbi path (`test_real_data.jl:385-387`).
- **Transitions are seasonal, with degree 1.** A_kl(t) = exp(P^A_kl(t)) / (1 + Σ_{l'<K} exp(P^A_kl'(t))), with reference state K and trig polynomials (`CP\13PeriodicHMMSpatialBernoulli\PeriodicHMMSpa.jl:48-62`). SPSM uses the same convention (`SPSM\src\hmm_coefficients.jl:20-32`).
- **Leaving-period convention.** `z[n] = rand(Categorical(hmm.A[z[n-1], :, tₙ₋₁]))` (`PeriodicHMMSpa.jl:115-117`), and the same in `SPSM\src\HHMM_Bernoulli_spatial\periodichmmspa.jl:84-86`.
- **Initial law.** a starts uniform (`test_real_data.jl:126`) and is updated by EM (`estimation_functions_BthenR.jl:582`).

### 1.2 Occurrence emission: a spatial latent Gaussian field thresholded per site, with own-site memory
Given Z_n = k:
- **Marginal probability with memory m = 1.** λ_{s,n} = B_{k,t_n,s,1+Y_{s,n−1}}, with B_{k,t,s,h} = 1/(1+exp(P^B_{k,s,h}(t))) and degree 1 (`PeriodicHMMSpa.jl:67`; `estimation_functions_BthenR.jl:304`). The lag selection is `lambdas = (1 .- y_previous) .* hmm.B[z[n], t, :, 1] .+ y_previous .* hmm.B[z[n], t, :, 2]` (`PeriodicHMMSpa.jl:92`).
- **Latent field.** U_n ~ N_D(0, Σ_k(t_n)) with Σ_k(t)_ij = exp(−h_ij / R_k(t)).
- **Range.** R_k(t) = exp(P^R_k(t)) km, seasonal and per regime, degree 1 (`PeriodicHMMSpa.jl:69`; `SPSM\src\HHMM_Bernoulli_spatial\trig_conversion.jl:24`). The fit starts at R0 = 500 km (`test_real_data.jl:123-125`).
- **Thresholding.** Y_{s,n} = 1{U_{s,n} < Φ⁻¹(λ_{s,n})}: `x[:] .= u .< thresholds` (`CP\11SpatialBernoulli\SpatialBernoulli.jl:70-75`; `SPSM\src\HHMM_Bernoulli_spatial\SpatialBernoulli.jl:37-41`).
- **Kernel.**
  - `11SpatialBernoulli` implements a Matérn with sill and order, using the √(2ν)·h/range scaling (`SpatialBernoulli.jl:44-49`).
  - The HMM fixes sill = 1 and order = ½, which gives the exponential kernel: `SpatialBernoulli(hmm.R[z[n], t], 1.0, 1 / 2, lambdas, hmm.h)` (`PeriodicHMMSpa.jl:93`). SPSM has only `expkernel` (`SPSM\...\SpatialBernoulli.jl:31-35`).
  - So "Matérn" in the abstract is Matérn ν = ½ in the final code.
- **Memory.** Only the site's own previous day. *Inference:* U is drawn independently each day given (Z_n, Y_{n−1}) (`PeriodicHMMSpa.jl:89-94`), so the occurrence has no temporal dependence beyond the regime and that lag.
- **Emission likelihood.** The orthant probability, computed by quasi-Monte Carlo with `mvnormcdf(...; m = D * QMC_m)` and QMC_m = 30 (`SPSM\src\HHMM_Bernoulli_spatial\likelihoods.jl:16-43`; `test_real_data.jl:73`).
- **Coefficient count.** (2d+1)(K·2^m·D + K(K−1) + K) (`SPSM\...\likelihoods.jl:70`). For K=4, m=1, d=1, D=37 that is 936 (my arithmetic).

### 1.3 Rain-amount marginal: a uniform small-rain part spliced with an EGPD tail
For a wet day in regime k, at site s and season index t, with a = 0.1 mm and b = 1.0 mm (`01_marginal_estimation.jl:57-58`):
- **Spliced cdf.**
  - For a ≤ r ≤ b: F(r) = p·(r−a)/(b−a).
  - For r > b: F(r) = p + (1−p)·H_κ(G_{σ,ξ}(r−b)).
  - Source: the `MixedUniformTail` pdf, cdf and quantile (`CP\23precip_intensity\EGPD_functions.jl:40-80`).
- **Tail.** G = GPD(0, σ, ξ) and H_κ = `TBeta(κ)`, which is a Beta(κ, κ) truncated to [1/32, 1/2] and rescaled to [0, 1] (`~\.julia\packages\ExtendedExtremes\WyqWu\src\distributions\TruncatedBeta.jl:31-39`). The EGPD cdf is H(G(x)) (`...\ExtendedGeneralizedPareto.jl:35-41`). *Inference:* this is the Gamet–Jalbert truncated-beta EGPD.
- **Links, all trig degree 2, one set per (regime, site).**
  - p = 1/(1+exp(P^p)), σ = exp(P^σ), ξ = 1/(1+exp(P^ξ)) ∈ (0, 1), κ = exp(P^κ) (`CP\23precip_intensity\EGPD_class.jl:67-86`; `01_marginal_estimation.jl:59,78`).
  - So the marginal is **seasonal, per regime and per site**, with 4 parameters × 5 coefficients × K·D = 2 960 coefficients (my arithmetic).
  - ξ < 1 means a finite-mean tail.
- **Threshold mismatch.** b is 1.0 in the final pipeline (`01:58`; `DP\simulation_study.jl:43-44`), but the comment in `EGPD_functions.jl:45` says 0.5.

### 1.4 Amount field: a conditional spatio-temporal Gaussian field
- **Latent field.** X_{s,n} has mean 0, is stationary and is non-separable in space and time. Cov(X_{i,n}, X_{j,n+u}) = C(h_ij, |u|), with:
  - C(h, u) = σ²·ψ(u)^{−(δ+β)}·M_ν(h / (c·ψ(u)^{β/2}));
  - ψ(u) = 1 + (u/a)^{2α};
  - M_ν(x) = 2^{1−ν}/Γ(ν)·x^ν·K_ν(x).
  - Sources: `CP\23precip_intensity\covariance_functions.jl:129-152`; `adbesselkxv(v, x) = x^v * adbesselk(v, x)` at `BesselK\A8QEg\src\besk.jl:42`.
  - This Matérn uses h/c, not √(2ν)·h/c, unlike `11SpatialBernoulli` (`covariance_functions.jl:131`).
  - *Inference:* this is Gneiting's (2002) class in d = 2, where β is the separability parameter.
- **Fixed sill.** σ² = 1: `model_type(1.0, param...)` (`covariance_functions.jl:641`).
- **Fitted values.** `GneitingMatern(1.0, 305.24, 0.6097, 0.99942, 0.75586, 0.59284, 0.44823)`, that is c ≈ 305 km, a ≈ 0.61 day, α ≈ 1, β ≈ 0.76, δ ≈ 0.59, ν ≈ 0.45 (`02_covariance_estimation.jl:91-96`).
- **Separable alternative.** `ExpExp`, C = e^{−h/r_s}·e^{−u/r_t} (`covariance_functions.jl:114-125`), fitted first to (280.24 km, 0.665 d) (`02:77-82`).
- **Not regime-dependent and not seasonal.** One covariance model is fitted (`02:92`), and `GaussianField` holds one model (`covariance_functions.jl:159-163`).
- **What the field is conditioned on: the occurrence mask, the regime and yesterday's occurrence, through the thresholds.**
  - c_{s,n} = Φ⁻¹(p^dry_{s,n}), with p^dry_{s,n} = 1 − B_{Z_n,t_n,s,1+Y_{s,n−1}}, so p^dry comes from the occurrence HMM (`02:60-69`; `03_simulation.jl:243-253`).
  - Constraint box: X_{s,n} > c_{s,n} if and only if Y_{s,n} = 1, written as `lb = ifelse.(Y .== 1, phimPdry, -Inf)` and `ub = ifelse.(Y .== 0, phimPdry, Inf)` (`covariance_functions.jl:699-701`).
- **Simulation: a lag-L Markov sequential truncated-Gaussian sampler with L = 10.**
  - Each day: X_n | X_{n−L:n−1}, Y_n ~ TruncatedMVN(W·x_past, Σ_L; box(Y_n)).
  - W = C_{0,1:L}·C_{1:L}⁻¹ and Σ_L = C_0 − W·C_{0,1:L}ᵀ (`covariance_functions.jl:705-724`).
  - The draw is `TruncatedMVN.sample(TruncatedMVNormal(matm * Zprec, Σ, lb[:, t], ub[:, t]), 1)` (`covariance_functions.jl:766`).
  - Day 1 is drawn from TN(0, C_0) and days 2…L use a growing history (`:738-761`). L = 10 is set at `03_simulation.jl:254`.
- **Amount.** R = 0 if X < c. Otherwise R = F⁻¹_{Z_n,s,t_n}((Φ(X) − p^dry)/(1 − p^dry)), computed as `quantile(di[s][Zs[n,i],t], cdf(Truncated(Normal(), quantile(Normal(), pdry), Inf), XR[s,n,i]))` (`03_simulation.jl:286`; the same map is in `DP\gaussian_field.jl:81-83`).
- **Fitting: a conditional composite (pairwise) censored likelihood, conditional on each pair's occurrence pattern** (`covariance_functions.jl:637-688`).
  - Both sites wet: log φ₂(z₁, z₂; ρ) − log P(X₁ > c₁, X₂ > c₂) (`:657-658`).
  - One site dry: log Φ((c₁ − ρz₂)/√(1−ρ²)) − log P(X₁ ≤ c₁, X₂ > c₂) (`:660-664`). The constant log φ(z₂) is dropped.
  - Both dry: contributes nothing.
  - Pairs: h ≤ ⅔·max h, |t₁ − t₂| ≤ 10, all ordered (i, j, t₁, t₂) except i = j with u = 0 (`:638,652`). ρ is clamped to ±0.9999 and probabilities to [1e-10, 0.9999].
  - Kernel: `norm_cdf_2d_vfast` (the Tsay–Ke approximation) (`CP\utils\fast_bivariate_cdf.jl`; the same function is at `SPSM\src\HHMM_Bernoulli_spatial\fast_bivariate_cdf.jl:19`).
  - Optimizer: Optimization.jl with `AutoForwardDiff` and `Optim.LBFGS` inside box bounds (`:684-687`; bounds at `02:92`).
  - Cost: 3 780 to 6 279 s per 100 iterations (`02:93-95`).
- **Properties (inferences, not in the code).**
  1. The sequential sampler ignores future days' boxes, so it approximates the joint law of X given the whole Y path; it behaves like a filter.
  2. Occurrence (field U, exponential kernel with R_k(t)) and amounts (field X, Gneiting) are **two different fields**, and X is forced to follow U's mask. The wet-day law of simulated R therefore equals F exactly only under single-site conditioning; neighbour masks and lags distort it. The abstract's stated motivation is that this split allows "different spatial and temporal scales", which "is not possible with a single censored Gaussian model".

---

## 2. Estimation and simulation pipelines

### 2.1 Estimation (real data)

| # | Step | Code | Package functions vs custom |
|---|---|---|---|
| E0 | Select stations, compute distances, write the Y, RR and ẑ CSVs | `00data\data_selection.jl:127-135`; `real_data_getZ.jl:110-119` | custom script |
| E1 | Initialize with an independent periodic HMM: random model → slice fit → trig coefficients → Baum–Welch | `test_real_data.jl:109-116` | SPSM **v2.1.2** API (`randARPeriodicHMM`, `fit_mle_all_slices`, `fit_θ!`, `fit_mle`), pinned at `CP\Manifest.toml:3130-3134`. The SPSM 4 equivalent is in `DP\simulation_study.jl:101-105` |
| E2 | Fit the spatial HMM (see below) | `estimation_functions_BthenR.jl:435-665`, called at `test_real_data.jl:133` | **custom**: JuMP/Ipopt for A and B, Optim for R, MvNormalCDF. Now in SPSM 4 as `fit_mle(::SmoothARPeriodicHMMSpatial, Y, Y_past; tdist, QMC_E, maxiters_R)` (`SPSM\src\fit\smooth_hmm.jl:474-485`; `src\fit\spatial.jl:14-86`), which returns the best iterate (`smooth_hmm.jl:464-468`) |
| E3 | Select K, d, m by ICL | `test_real_data.jl:385-387` | custom; uses `complete_loglikelihood` and `nb_param_HMMSpa` |
| E4 | Viterbi ẑ with QMC_m = 30 | `real_data_getZ.jl:94-98` | SPSM `viterbi` (`SPSM\...\viterbi.jl:10-13`) |
| E5 | Fit the margins (see below) | `01:68-78` | custom code plus ExtendedExtremes types |
| E6 | Normalize the observed amounts (see below) | `01:146-192`, written to CSV at `01:236` | custom |
| E7 | Fit the space-time covariance by `fit_mple(V, Mat_h, GneitingMatern, …, Φ⁻¹(Pdry))` | `02:91-97` | custom code plus Optimization.jl |

E2, the hybrid spatial EM:
- The E-step is the full QMC likelihood (`:569,614`).
- The M-step updates, in order:
  - a (`:582`);
  - A, with JuMP/Ipopt (`:585`);
  - B (`:590`), by maximizing **independent marginal** weighted Bernoulli likelihoods (`:111-139`);
  - then R given B, by maximizing a γ-weighted **pairwise** likelihood (`:594-596`, `:718-800`).
- Pairs are those with h < tdist·max h and tdist = 0.3 (`:450`; `test_real_data.jl:59`).
- Settings: 100 iterations, tol = 1e-3, maxiters_R = 100 (`test_real_data.jl:133`). The iterate with the highest log-likelihood is kept (`complete_estimation_simrealparam.jl:51-56`).

E5, the margins:
- First a pooled fit, then a fit per regime given ẑ.
- p is fitted by ordinary least squares on the logit of daily empirical proportions (`EGPD_class.jl:124-153`), not by maximum likelihood.
- The EGPD tail is fitted by `Optim.optimize(fobj, param)`, with the penalty `(κ-1)^2/0.1` weighted by each day's share (`EGPD_functions.jl:275-312`).
- *Inference:* calling `optimize` without a method uses Nelder–Mead. The rewrite states the reason: "the carrier's normalization, `beta_inc` in κ, has no derivative: hence Nelder-Mead" (`DP\rain_margins.jl:111`).

E6, the normalization:
- Small amounts get a jittered PIT, so the discretized 0.1 mm values become continuous (`01:146-176`).
- Then V = Φ⁻¹(p^dry + (1 − p^dry)·U), with p^dry taken from the fitted spatial HMM at ẑ and the observed previous day. Day 1 uses `Yprevious = 1` (`01:178-192`).
- Dry days are NaN, because `cdf` returns NaN below a (`EGPD_functions.jl:60-62`).

**Structure (inference).** This is exactly a two-stage IFM procedure: regimes, then Viterbi, then per-block fits under the hard path ẑ.

**Unused alternatives.**
- `14PeriodicHMMSpatialBernoulli_CLEM` is a pairwise-EM variant with one forward-backward pass per pair of sites (`README.md:60`; `estimation_functions_BandR_v3.jl:408-531`). The final model does not use it.
- `31HMMIndep` is the conditionally independent baseline (`README.md:79`).

### 2.2 Simulation

| # | Step | Code |
|---|---|---|
| S1 | Draw (z, Y) with `my_rand(hmmspa, n2t; seq=true)`, Nb = 500 members | `03_simulation.jl:30,229-237`. SPSM 4 equivalent: `rand(::SmoothARPeriodicHMMSpatial, n2t; Y_past, seq=true)` (`DP\simulation_study.jl:87`; `SPSM\...\periodichmmspa.jl:79-89`) |
| S2 | Compute p^dry[s, n] from (z_n, t_n, Y_{s,n−1}) | `03:243-253` |
| S3 | Draw X with `simulate_iterative(gf, 1:N, 10, Y, Pdry)`, using TruncatedMVN and threads | `03:241-256` |
| S4 | Transform X to amounts R | `03:266-290` |

Timing comment: "2700 s for 500 sims" (`03:259`).

### 2.3 Defects and convention traps found while tracing the code
1. **The sign of θᴮ.**
   - The paper code uses B = 1/(1+exp(P)) (`BthenR.jl:304`).
   - SPSM 4 uses `logistic(P)`, and its changelog says "the emission coefficients `θᴮ` have the opposite sign of those of version 3" (`SPSM\CHANGELOG.md:74-75`).
   - DP runs on SPSM 4.1 and passes the paper's θᴮ unchanged (`DP\simulation_study.jl:58-61`), yet it negates p and ξ for the margins (`:64-65`, "that fit wrote p and ξ as 1 / (1 + exp(η))").
   - *Inference:* the HMM in DP's `load_truth` therefore has every occurrence probability flipped, B ↦ 1 − B. The scripts in CP itself are consistent because they pin SPSM 2.1.2.
2. **Y_past layout.** CP uses `order × D` (`test_real_data.jl:96`); SPSM 4 uses `D × order` (`CHANGELOG.md:78-79`).
3. **Day-1 initial condition.** The first simulated day is forced to `y_ini = 0` (`PeriodicHMMSpa.jl:88,109`), while p^dry for day 1 assumes the previous day was wet, `Yprevious = 1` (`03:243-244`). This is ad hoc; the plan's `past`/`burnin` fixes it.
4. **Margin file name.** `01` saves `full_periodicEGPD_K42.jld2` (`01:79`) but `03` loads `periodicEGPD_K42.jld2` (`03:262`).
5. **`GaussianField(coords, model)`** reads the global `coordll` instead of its `coords` argument (`covariance_functions.jl:166-172`).
6. **Two different Matérn parametrizations** are used (`11\SpatialBernoulli.jl:47` and `covariance_functions.jl:131`).
7. **QMC randomness cannot be seeded through SPSM.** SPSM calls `mvnormcdf` without an `rng` (`SPSM\...\likelihoods.jl:39`), although the function accepts one (`MvNormalCDF\2hxzG\src\functions.jl:160`).

---

## 3. Mapping onto the SWG 2.0 / SPSM 5.0 plan

### 3.1 Piece-by-piece fit

| Paper component | Plan construct | Status | What is needed |
|---|---|---|---|
| Seasonal regime chain, spatial emission, K=4, d=1 | `Regimes(hmm, path)` "external, e.g. spatial" (PLAN:298); "stays in SPSM: the spatial HMM" (PLAN:401) | **Fits as-is**, through SPSM `SmoothARPeriodicHMMSpatial` | A native `Regimes(K, emission = spatial)` is 2.1 ("a spatial regime emission spec", PLAN:511). An ICL criterion is needed (§3.3, L12) |
| Occurrence = the regime emission (B with memory 1, plus R_k(t)) | `Occurrence(from = :regimes)` reuses the regime emission, "independent or spatial" (PLAN:263) | Fits in concept | The block must **expose λ and p^dry** to other blocks (L2) |
| Variant: occurrence refitted given ẑ with a parametric spatial copula | `Occurrence(copula = MultisiteGaussian(structure = ExponentialKernel()))` | **New pieces** | A `structure` keyword; a range that is seasonal and per regime (R_k(t) = exp(trig), `PeriodicHMMSpa.jl:69`); station coordinates; a pairwise estimator with distance cut-off. The paper's M-step at fixed weights is exactly "marginal B, then pairwise R given B" (`BthenR.jl:590-596`) |
| EGPD spliced marginal per (k, s, t) | `amount =` a SPSM `SmoothDistribution(family, T; K, season)` (PLAN:357); "an EGPD marginal" is promised as additive (PLAN:291) | **New family** | An `EGPD` type (DP defines it in about 13 lines, `rain_margins.jl:24-36`, so ExtendedExtremes and its Gadfly/CSV deps can be avoided, `CP\Manifest.toml:943-947`); a spliced family (uniform below b, b + EGPD above); a penalty hook; derivative-free fitting (`NoAD`); per-parameter degree |
| Amount dependence **conditioned on the dry sites** | Rain amounts use a copula on the wet set: "Sampling draws ε ~ N(0, C_k) over all D sites … then keeps the wet subset" (PLAN:264-265) | **Does not fit** | A new dependence value (here called `ConditionalField`) that takes thresholds from the occurrence block, truncates at dry sites, samples truncated multivariate normals, and fits a conditional pairwise censored likelihood |
| Non-separable space-time covariance with 10-day lags | dynamics are per-site AR1/LatentAR1 and the copula is fitted on innovations (PLAN:266-268, 313-317) | **Does not fit**: space and time are coupled | Structure types with a time lag (`GneitingMatern`, `ExpExp`), where the dependence object owns the temporal dependence |
| Censored pairwise kernel | CLG reuses `norm_cdf_2d_vfast` (PLAN:342); "censored pairs use the conditional pairwise censored likelihood" (PLAN:316) | Kernel **reusable** | Add normalization by the pair's occurrence pattern (`covariance_functions.jl:657-664`) |
| Latent lag initialization | `past`; "A censored lag draws w₀ from the truncated normal" (PLAN:193) | Partial | Generalize to L = 10 latent lags |
| Station coordinates and distances | `WeatherData` (PLAN:173-179) has no coordinates, and "Unknown columns are dropped" (PLAN:177) | **Missing** | L1 |
| Likelihood kind | `:composite` blocks make `bic` refuse (PLAN:220-222) | Fits: the amount field is `:composite` | The regime log-likelihood is exact but QMC-noisy (`SPSM\src\fit\smooth_hmm.jl:369-371`). K is chosen by ICL, while the plan says "K is selected only by the Stage-1 BIC" (PLAN:223) |
| RNG | one stream per (member, block) (PLAN:199-203) | Mostly fits | `TruncatedMVN.sample(rng, d, n)` exists (`TruncatedMVN\zpqND\src\TruncatedMVN.jl:100`); the QMC likelihood needs an rng threaded through |
| Dependencies | "`NonlinearSolve` leaves too" (PLAN:86) | **Conflict** | TruncatedMVN does `using NonlinearSolve` (`TruncatedMVN.jl:14`), and the Matérn needs BesselK. Keep both out of Blocks |

### 3.2 Separate model class or a configuration of RegimeHMM?

**Recommendation: a configuration of `ProductGenerator`/`RegimeHMM`, shipped as a sublibrary plus a named preset. Not a new class.**

Reasons (inference, supported by the code):
- The joint law factorizes in DAG order: p(z)·p(Y | z, Y₋)·p(X | Y, z, Y₋, X₋)·δ(R = g(X, z, t)). This is the plan's "product over blocks in DAG order" (PLAN:234).
- Estimation is exactly `TwoStage()`: EM on occurrence, then Viterbi ẑ, then each block by IFM given `OneHot(ẑ)` (PLAN:309), which is §2.1 steps E2 to E7.
- The pieces are useful outside this model:
  - EGPD fits any positive variable;
  - the conditional field fits any `TwoPart` hurdle (the plan's "every rain model is also available for other variables", PLAN:257).

Proposed packaging:
- Put the blocks in `lib/StochasticWeatherGeneratorsSpatial`, depending on Core and Blocks only (the plan's rule, PLAN:77).
- Add a convenience alias that lowers to `ProductGenerator`, as WGEN does (PLAN:328), for example `SpatialRegimeHMM()` or `RegimeHMM(preset = :cognot)`.
- A separate class would only be justified by a joint EM over occurrence and amounts, which the code does not do. The pairwise EM in folder `14` is an estimator option for `Regimes` (for example `em = PairwiseEM()`), not a class.

Illustrative spec (my proposal; none of these names exist yet):
```julia
RegimeHMM(regimes = Regimes(hmm_spa, ẑ),
  RR = Rain(occurrence = Occurrence(from = :regimes),
            amount = Spliced(Uniform(0.1, 1.0), EGPD; at = 1.0, season = 2),
            copula = ConditionalField(GneitingMatern(); lag = 10,
                       estimator = PairwiseCensored(given = :occurrence, maxdist = 2/3, maxlag = 10))))
```

### 3.3 What the launch architecture must do now so this model can be added later without breaking changes
1. **L1. Coordinates in `WeatherData`.**
   - Recognize LAT, LON and ALT as station metadata rather than dropping them; add `coords(data)` and `distances(data; metric)`.
   - The distance is part of the model state: `h::AM` in `SPSM\...\periodichmmspa.jl:24`.
   - Use one documented metric with no rounding. The paper uses Distances.jl's default radius on (LON, LAT); SWG's `haversine` uses r = 6372.8 and `round_Int = true` (`SWG\src\data\stations.jl:20`).
   - Persist the D×D matrix that was used at fit time.
2. **L2. Blocks can read derived quantities of other blocks, not only their outputs or innovations.** For example, the occurrence block (or the regime emission under `from = :regimes`) must provide p^dry[s, n] given z_n and Y_{n−1}, both in `FitContext` (`02:60-69`) and in the simulation context, after today's occurrence has been drawn (`03:243-253`). The `inputs` trait should be able to name a (block, quantity) pair.
3. **L3. The occurrence and amount parts of one variable can be addressed separately**, in outputs and in `given`. The script `03:219-221` simulates amounts given the *observed* occurrence. Reserve names such as `RR_occurrence` now.
4. **L4. Dependence objects sample under constraints.** Define a `draw!(rng, x, dep, θ, ctx; lower, upper)` contract in which the unconstrained case is today's "draw all sites, keep the wet ones" (PLAN:264-265). Do not hard-code subsetting in `Rain`/`Hurdle`.
5. **L5. `AbstractDependence` allows non-separable space-time objects.**
   - Give the structure interface a lag argument from day one, `cov(structure, h, u)` with only u = 0 at launch.
   - Add a `memory(dep)` trait.
   - In `resolve`, treat "dependence that owns the temporal part" and "per-site `dynamics`" as mutually exclusive, instead of as independent keywords.
6. **L6. Latent block state with L > 1 lags.**
   - Block state can hold unobserved vectors.
   - `past = Date(...)` reconstructs them by PIT on wet days and truncated draws on dry days, generalizing PLAN:193. Burn-in remains the default.
7. **L7. A reserved `structure` keyword in `MultisiteGaussian`**, defaulting to `Empirical()`. Parametric structures carry parameters that can be seasonal or per regime through SPSM `ParamSpec` (basis plus link). R_k(t) = exp(trig) is exactly `TrigBasis => LogLink` with `by = :regime`.
8. **L8. An estimator protocol for dependence**, with pair selection (distance fraction, maximum lag, weights `wp`), soft weights (`ctx.W`) and `likelihood_kind = :composite`. Reserve `PairwiseLikelihood` and `PairwiseCensored(given = :occurrence)`.
9. **L9. An open marginal-family protocol in SPSM's `SmoothDistribution`.**
   - Any `seasonal_family` (the mechanism exists: `SPSM\src\seasonal_mixture.jl:110-143`).
   - `adtype = NoAD()` with derivative-free solvers.
   - An additive penalty term.
   - Per-parameter degree.
   - Spliced families.
   - Then EGPD can arrive in SPSM 5.x additively. DP depends on `PeriodicNLL` and `fit_loss_optim` (`rain_margins.jl:107-112`), which the plan demotes to `public` or renames (PLAN:383-390), so either `SmoothDistribution` covers this case or those names stay `public`.
10. **L10. Distribution access on fitted marginal blocks:** `cdf` and `quantile` at (k, s, t), plus a ties policy for discretized data (`pit(...; ties = :jitter)`, following `01:146-154`).
11. **L11. `Regimes(hmm, path)` accepts SPSM `SmoothARPeriodicHMMSpatial`.**
    - Check station order against h.
    - Implement `Occurrence(from = :regimes)` with memory categories `B[k, t, s, h]`.
    - `decode(...; rng)`, because QMC Viterbi is stochastic.
12. **L12. A model-selection criterion keyword** (`criterion = BIC() | ICL()`) on K ranges, and a likelihood flag that separates "composite" from "exact but Monte-Carlo-noisy".
13. **L13. RNG plumbing everywhere.** `FitContext` carries an rng, and SPSM 5 passes `rng` to `mvnormcdf`.
14. **L14. Dependency isolation.** The truncated-MVN sampler (NonlinearSolve) and BesselK live only in the new sublibrary or an extension. The Blocks protocol must be complete enough to support that: add a CI test with a toy block living outside the repo.
15. **L15. Persistence and pretrained models.**
    - Registry tags for structure, dependence and family types.
    - A converter for the paper's artifacts: negate θᴮ (`CHANGELOG.md:74-75`), negate p and ξ of the margins (`DP\simulation_study.jl:64-65`), and transpose `Y_past` (`CHANGELOG.md:78-79`).
    - Store h, L and the source of the thresholds.
    - A candidate pretrained artifact is `:france37_spatial_K4`.
16. **L16. Capabilities and performance.**
    - Blocks declare that they are sequential-only and give approximate conditionals.
    - The ensemble driver must not assume a cached Cholesky factor per (k, month) for every block (PLAN:265).
    - The measured cost, 2 700 s for 500 members × 51 years (`03:259`), should go into the P4 time budget.
17. **L17. Shared (global) parameter scope.** The Gneiting parameters are common to all regimes (`02:92`), so the scope/`by` semantics must express "pooled over regimes" for dependence blocks, not only for margins.

---

### Critical Files for Implementation
- `c:\Users\metivier\Dropbox\PC (2)\Documents\GitHub\HMM_Spatial_codepaper\23precip_intensity\covariance_functions.jl`
- `c:\Users\metivier\Dropbox\PC (2)\Documents\GitHub\HMM_Spatial_codepaper\Simulation_study_bootstrap\david_proj\gaussian_field.jl`
- `c:\Users\metivier\Dropbox\PC (2)\Documents\GitHub\HMM_Spatial_codepaper\Simulation_study_bootstrap\david_proj\rain_margins.jl`
- `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\fit\smooth_hmm.jl`
- `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\HHMM_Bernoulli_spatial\likelihoods.jl`

Sources:
- [SWGEN 2025 abstract: Cognot, Bel, Métivier, Parey, "Spatio-temporal generation of precipitations using a hidden Markov Model with correlated emissions"](https://swgen2025.sciencesconf.org/data/pages/6_cognot_1.pdf)
- [HAL preprint hal-05523768 (blocked automated access, not read)](https://hal.science/hal-05523768v1)