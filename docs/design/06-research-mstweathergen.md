# MSTWeatherGen: formal model, a seasonal and improved version, and what SWG 2.0 needs at launch

## 0. Sources and their status
- **Paper (open access).** Obakrim, Benoit, Allard et al., *SERRA* 2025, doi 10.1007/s00477-024-02897-8. I read the full HTML from https://link.springer.com/article/10.1007/s00477-024-02897-8 ; the page is marked "Open access". Below it is cited as "paper §x".
- **No preprint found.** A web search found no HAL or arXiv version. The SWGEN 2025 abstract (https://swgen2025.sciencesconf.org/data/pages/18_obakrim_1.pdf) repeats the paper's abstract word for word.
- **Code, two repositories:**
  - The paper-era repo is https://github.com/sobakrim/MSTWeatherGen (DESCRIPTION: `Version: 0.1.0.9000`). Line numbers below point to this repo.
  - The maintained repo is https://github.com/Chair-Geolearning/MSTWeatherGen. Its NEWS.md reads "MSTWeatherGen v1.0.0 (Release date: 2026-02-18)" and "v1.0.1 (Release date: 2026-06-15)". The GitHub API reports `pushed_at` 2026-11-17. A recent commit is titled "Fonction de Gneiting Version Denis et avec quelques corrections".
  - Both repos keep the same exported API.
- **Docs.** The reference index is https://sobakrim.github.io/MSTWeatherGen/reference/index.html. The vignette is `vignettes/MSTWeatherGen.Rmd` in both repos.

---

## 1. Formal description

### 1.1 Joint law
- **Model.** Paper §2.1, eq. (1): "Y_i(s,t) = Ψ_{k,i,s}(Z_{k,i}(s,t)), with k = X(t)".
  - X(t) ∈ {1..K} is the weather type.
  - There are K latent p-variate Gaussian random fields Z_k. Each is "zero mean and unit variance" and "second-order stationary in space and time".
  - The cross-covariance is C_{k,ij}(h,u).
- **Daily, discrete time.** "we set 𝒯 = ℕ" (§2.1).

### 1.2 Weather types (regimes)
- **Shared across space and fixed before modelling.** "the temporal domain is discretized into weather types prior to the statistical modeling" and "weather types are common for the whole spatial domain" (§2.2).
- **Method: tensor compression, then a Gaussian mixture.**
  - Paper: "we first reduce the dimensionality of the data using Principal Tensor Analysis (PTA)… we employ a Gaussian mixture model estimated by the expectation maximization (EM) algorithm… number of weather types is determined by Bayesian Information Criterion (BIC)… R package mclust" (§2.2).
  - Code (`R/Weather_types.R`):
    - `data_compression`: `PTAk::PTA3(data, nbPT = 3, nbPT2 = 1, …)`, then `keep = keep[1:4] # Keep the top 4 components`;
    - `weather_types`: `mclust::Mclust(pca, G = n_w, verbose = F)`;
    - `MSTWeatherGen_Estim_season` hard-codes `max_number_wt = 6`.
- **Done separately in each season.** `weather_types` is called on the season-filtered data inside `MSTWeatherGen_Estim_season` (`filter_season_data(...)`, then `weather_types(...)`).
- **Code fact: `names_weather_types` does not affect the clustering.** `weather_types` compresses the full array (`pca <- data_compression(data)`; Chair repo `R/Weather_types.R:77`). The argument is only used for plots and for `find_centroids`.
- **Transitions:**
  - X(t) is a "first-order Markov chain… non-homogeneous" with π_kl(t) (§2.2, eqs. 2–3).
  - π̂_kl(t) is estimated "by their empirical frequencies in a sliding window of length 2L+1" (§3.1, eq. 11); the case study uses "L=30 days" (§6.2).
  - Code: `estimate_transitions(cluster, dates, nb = 30, K)` sets `nb = nb * length(unique(lubridate::year(dates)))` and uses `FNN::get.knn(as.numeric(factor(substr(dates, 6, 11))), nb)`. It returns one K×K matrix per training day.

### 1.3 Seasonality
- **Two mechanisms** (§5): "first, any seasonal cycle in each variable is removed; then, seasons during which the dependencies… are considered constant are defined. Seasons must be provided by the users".
- **Step 1, de-seasonalising.** "Except for precipitation, the central tendency and standard deviation are calculated for each variable and each day of the year. The moving average method is then used to smooth" (§5).
  - Code: `scale_data(data, names, dates, window_size = 10)` is called inside `MSTWeatherGen_Estim_season`.
  - It is only applied `if(length(unique(lubridate::year(dates))) > 5)`.
  - It is off by default (`scale = FALSE` in `MSTWeatherGen_Estim`); the vignette passes `scale = T`.
- **Step 2, refit per season.** "all parameters are estimated separately for each season, including transition probabilities between weather types, transformation functions, and covariance function parameters" (§5). The case study uses DJF/MAM/JJA/SON (§6.2).
- **Season boundaries.**
  - Paper: "The likelihood of Z(t_last) is then computed using the parameters of the new season, and the most likely state… is selected" (§5).
  - **The code does something else:** `most_probable_weather_type` takes `which.min(... mean((c(sim[, names_weather_types]) - centroids[[k]])^2))`, which is the nearest centroid in data space. It then samples from `transitions[[1]][previous_wt, ]`.
- **Simulating dates outside the training period.** `simulate_weathertypes` looks up `which(dates == dates_sim[i-1])` among the training dates. If the date is absent, it falls back to `which.min(abs(dates - dates_sim[i-1]))` with a warning.
  - *Inference:* for future years this freezes the transitions at the last training date.

### 1.4 Marginal transformation: truncated Ordered Quantile Normalization (OQN)
- **Form** (§2.3, eq. 5): Ψ_{k,i,s}(Z) = F⁻¹_{k,i,s}(Φ_T(Z)) if Z > T_{k,i,s}, and 0 otherwise.
  - Φ_T is "the left truncated Gaussian CDF".
  - "if the variable is not zero inflated, we simply set T = −∞".
- **Threshold** (§3.2): T = Φ⁻¹(f₀), "where f₀ is the frequency of zero values".
- **Estimation.** Rank based: g(y_t) = Φ_T⁻¹{(r_t−½)/n + Φ(T)(1−(r_t−½)/n)}, with linear interpolation inside the data range and logistic-GLM tails "logit(Φ_T(Ψ⁻¹(y_i))) = β₀ + β₁ y_i" (§3.2, eqs. 12–15).
- **One transform per (weather type k, variable i, site s).**
  - Code: `estimate_lambda_transformations` loops `lapply(1:K … lapply(names … lapply(1:ns …`.
  - Only the variable named "Precipitation" gets a finite threshold: `q <- qnorm(proportion_zeros)`.
  - `orderNorm_all` pools neighbouring sites when zeros are at least 80% (`while ((length(which(x!=0))<50)) { x = c(x,data[,kn[j]])`).
- **Precipitation is a censored latent Gaussian.** Simulated latent values at or below T become 0 (`apply_inverse_transformations`: `sim[wt==k & sim[,j,v] <= q,j,v] = 0`).

### 1.5 Multivariate space-time covariance
- **Paper version** (§2.4):
  - Eq. (6) splits it into a regional term and a local term: C_{k,ij}(h,u) = C⁽¹⁾_{k,ij}(u) + C⁽²⁾_{k,ij}(h,u).
  - Eq. (7), C⁽¹⁾: multivariate exponential, β⁽¹⁾_{ij} exp(−r⁽¹⁾_{ij}|u|).
  - Eq. (8), C⁽²⁾: Gneiting–Matérn from Allard, Clarotto & Emery 2022 (doi 10.1016/j.spasta.2022.100706, cited in `R/Covariance_and_loglikelihood.R:42`):
    β⁽²⁾_{ij} · exp(−r⁽²⁾_{ij}|u|) / (η_{ij}(u)+1)^{d/2} · C_M(h; √(a²_{ij}/(η_{ij}(u)+1)), ν_{ij}).
  - Eq. (10), the pseudo-variogram: η_{ij}(u) = ((a|u|)^{2b}+1)^c − A_iA_j((d|u|)^{2e}+1)^{−c}.
  - Validity needs 2a²_{ij} = a²_{ii}+a²_{jj} and 2ν_{ij} = ν_{ii}+ν_{jj}. ρ⁽¹⁾ and ρ⁽²⁾ are correlation matrices.
  - Count: "5+5p+p(p−1) parameters per weather state" (§2.4).
  - The structure is isotropic and stationary within (season, k).
- **Code version differs from the printed equations.**
  - Paper-era `Gneiting` returns `A1 * A2 + A3`, with `A3 <- ax * 1 / (((g1 * abs(u))^(2*l1) + 1)^(m) - ci*cj * ((g2 * abs(u))^(2*l2) + 1)^(-m))` (`R/Covariance_and_loglikelihood.R:91-99`).
  - Maintained Chair repo: `SpatioTemp <- Matern(abs(h), r = sqrt(aij^2 / etaij), v = nuij) / (etaij * eta2ij)` and `return(beta2ij * SpatioTemp + beta1ij * Temp)`.
  - So in the code both the regional term and the temporal decay of C⁽²⁾ are generalized-Cauchy "1/pseudo-variogram" factors, with per-variable A_i, B_i, C_i, not exponentials.
  - The code divides by η, while the paper divides by (η+1)^{d/2}. The normalizing constants follow suit: code `sqrt((1-Ai^2)*(1-Aj^2))`, paper √((2−A²_i)(2−A²_j)).
  - *Inference:* each version is internally consistent, so these are two parametrizations. The port must pick one and say so.
  - Code parameter list (`initialize_par_all_if_missing`, `GRF_helper_functions.R:42-47`): 15 shared temporal parameters (`a1,d1,g1,a2,d2,g2,b1,e1,l1,b2,e2,l2,c,f,m`), plus `ai, bi, ci` per variable, plus `rij, vij, ax, dij` per pair.

### 1.6 Estimation (stepwise; pairwise composite likelihood)
- **Per weather type, one after another** (§3.3): "estimated independently for each weather type successively".
- **Pairwise likelihood** (§3.3): "we use the pairwise likelihood (PL), a special case of composite likelihood".
- **Pair design:**
  - n_A centroids from k-means (`selectUniformPointsIndices`, `GRF_helper_functions.R:415-430`);
  - n_B partners per centroid, drawn with probability ∝ 1/(‖s_α−s_β‖²+1) (eq. 17; `selectPoints`, `:446-459`);
  - time lags |t_γ−t_δ| ≤ t_max (eq. 18);
  - case study: n_A, n_B, t_max = 15, 15, 3 (§6.2).
- **Censored bivariate terms** (§3.3, eqs. 20–22): Gaussian density if both values are above threshold; log Φ((T_i − C Z_j)/√(1−C²)) if one is censored; log Φ₂(T_i, T_j; C) if both are. Code: `loglik`, `:510-535`, via `VGAM::pbinorm`.
- **Stepwise schedule in the code** (`estimation_gf`, `:331-401`):
  1. Per-variable spatial Matérn at u = 0: `init_space_par`, which calls `optim(par = c(1, 1), …)` with R's default Nelder–Mead (`:154-163`).
  2. Spatial parameters at u = 0: `optimize_spatial_parameters` (`:383`).
  3. Two alternations of temporal then spatial parameters: `for (v in 1:2) {…}` (`:385-390`).
- **Moment steps, not PL:**
  - The regional term β⁽¹⁾ is initialised from the empirical covariance at the largest distance: `ax = vgm[vgm$lagtime==0&vgm$dist==max(vgm$dist),]` (`:629`).
  - The co-located cross-correlation ρ⁽²⁾ is **moment-matched**: `beta_val <- (cr[v1, v2]) / (cc - ax)` (`Covariance_and_loglikelihood.R:233`), after `Matrix::nearPD(cr-ax)` (`:222`).
  - `cr` is the average over sites of same-day correlations over the whole season, pooled across weather types (`GRF_helper_functions.R:615-619`, computed before `for (k in 1:K)`).
- **Infeasible parameters get a random penalty:** `return(abs(rnorm(1)) * 1e+20)` (`Covariance_and_loglikelihood.R:545`). So the objective is not deterministic.
- **Lagged pairs:** `generate_temporal_index_pairs` builds `cbind(wt_id-i, wt_id, i)` with wt_id ∈ type k (`:504-513`).
  - *Inference:* the earlier day of a lagged pair may belong to another type, so C_k is read as a law conditional on X(t) only.

### 1.7 Simulation
- **Markov approximation of order M** (§4, eq. 21): p(Z(t) | X(t), Z(t−M..t−1)).
- **Regime-switching VAR(M) on the latent scale** (eq. 22): Z(t) = Σ_m B_{k,m} Z(t−m) + L_{k,0} ε(t).
  - B comes from the block-Toeplitz Yule–Walker system built on C_k(0..M).
  - L L′ = C_k(0) − Σ B_m C_k(m).
  - Code: `calculate_Bk_matrices` and `simulate_Z` (`Simulation_helper_functions.R`).
- **Then** apply Ψ⁻¹ per (k, i, s) and add back the seasonal mean and sd (§5; `apply_inverse_transformations`, `rescale_data`).
- **Case-study size:** "M=3 … M×K = 18 matrices B_{k,m} of size n_S p × n_S p = 2988 × 2988" (§6.3), from 498 SAFRAN pixels × 6 variables (§6.1).
  - *Inference:* that is about 71 MB per Float64 matrix, so about 1.3 GB per season. `calculate_AR_coefficients_matrices` builds all seasons up front.
- **Fallbacks and start-up in the code:**
  - If a Cholesky fails, `calculate_AR_coefficients_matrices` copies the factor from a neighbouring type (`while (is.character(bk[[s]][[k]]$bk$Bk_0)) …`).
  - Initial values come from `generate_initial_conditions`, a 20-step burn-in.
- **Inference: switching distorts the latent variance.** After a switch l→k, the lags were generated under C_l, so Var Z(t) ≠ C_k(0) and Ψ_k⁻¹ is applied to a latent value that does not have unit variance. This is the same issue plan line 316 notes for the latent AR.
- **Authors' own stated limits:**
  - "uniform weather types across the entire domain… may not be realistic for very large domains";
  - "new non-stationary… covariance models need to be designed";
  - Cholesky "may become computationally intensive… spectral methods… could alleviate" (§7.1).

### 1.8 R API
```r
swg <- MSTWeatherGen_Estim(data, dates, by_season = TRUE, seasons, scale = FALSE, precipitation = T,
                           names = NULL, names_weather_types = NULL, coordinates, max_it, tmax, n1, n2)
# -> list(swg = <per season: list(dates, wt, scale_parm, transitions, lmbd, gf_par)>, by_season, names, names_weather_types)
bk  <- calculate_AR_coefficients_matrices(parm = swg, coordinates, AR_lag)
sim <- MSTWeatherGen_Sim(dates_sim, dates_original, data, seasons = NULL, parm, AR_lag = 1, bk)   # 3D array
```
- Sources: `R/MSTWeatherGen.R` for the signatures, and the `return(list(dates = dates, wt = wt, scale_parm = …))` of `MSTWeatherGen_Estim_season`.
- **Input layout:**
  - data is "a 3D array (time×location×variable)", precipitation "has to be the first variable";
  - coordinates is a "two column matrix";
  - seasons is `list(s1 = list(min_day = 1, max_day = 29, min_month = 12, max_month = 2), …)` (vignette).
- **Exports** (NAMESPACE): the three functions above and 4 `plot_*` functions. The Chair repo adds `S3method(summary,swg)`, `setCores` and `getCores`.

---

## 2. What a "seasonal and improved" MSTWeatherGen would be in this ecosystem

The proposal is one class with two presets: `preset = :paper` for faithful reproduction and an improved default. This follows the selectability principle of plan §4.1 (plan lines 275–291).

| Concern | `:paper` (faithful) | Improved default (SWG/SPSM-native) |
|---|---|---|
| Seasons | User seasons; every parameter refitted per season | One model. Every parameter is a smooth function of t through SPSM bases (`season = d` is a trig degree). SPSM's `IndicatorBasis` (`src/TimeVaryingParameters/bases.jl:169`) expresses the paper's DJF… seasons as one basis, so `:paper` is the same model with a piecewise basis |
| Regimes | PTA+GMM per season, K ≤ 6 by BIC, used as an observed path | The shared `Regimes` (plan line 293ff): a periodic HMM with A_k(t) = softmax(B(t)β_k), frozen, or `Regimes(; path = table)` for externally clustered types. Emission choices: the RegimeHMM rain-occurrence stage, or a diagonal-Gaussian periodic HMM on PTA/PCA scores (`PeriodicHMMDists` takes any law per block of rows, `src/HMM_Dists/periodichmm.jl:16-21`). *Inference:* SmoothPeriodicHMM fit support for these emissions should be checked |
| Transitions | Sliding-window counts, L = 30 | SPSM smooth softmax. Window counts stay available as a chain estimator for `:paper` |
| Season boundary | Nearest-centroid hack | Not needed: one chain over the whole year |
| Dates outside training | Transition matrix of the closest training date | A(t) is defined for every day-of-year |
| Marginals | Daily standardization, then truncated OQN per (season, k, i, s) | Per (site, var, k): `SmoothDistribution(family, T; K, season)` (plan line 357), or a seasonal-window `Anamorphosis` (empirical CDF in a ±w-day window, OQN logistic tails). Separate de-seasonalization becomes unnecessary |
| Zeros | Threshold T = Φ⁻¹(f₀) per (season, k, s), precipitation only | `Hurdle(G; zeros = Censored())` (plan lines 246–255) with seasonal p₀_{s,k}(t), so T(t) = Φ⁻¹(p₀(t)). This is the same zero mechanism as CensoredLatentGaussian and works for any variable (wind, ETP) |
| Covariance | Gneiting–Matérn per (season, k) | Same structure with seasonal parameters θ_k(t): log-ranges a_{ii}(t), r_{ii}(t) as exp(X(t)β) (precedent: SPSM spatial range `exp(X_R[t, :] ⋅ θᴿ[k, :])`, `src/HHMM_Bernoulli_spatial/trig_conversion.jl:6,24`); A_i, b, c, e via `IntervalLink` (`links.jl:19`); ν fixed per (k, i); ρ⁽¹⁾_k(t), ρ⁽²⁾_k(t) through SPSM `CorrelationLink` (deferred to 5.2, plan line 403). Validity holds pointwise in t because the sufficient conditions are imposed through the links |
| ρ⁽²⁾ | Moment-matched from season-pooled correlations | Estimated by PL per regime, weighted by ctx.W. No `nearPD` needed because the correlation link is valid by construction |
| PL | Random penalty, Nelder–Mead, unseeded pair sampling | Deterministic objective with links; LBFGS (Core `default_solver()`); pair design drawn from a seeded fit-time RNG stream. Each PL term is weighted by W[t,k] of the later day (StateScope, plan line 147). The censored Φ₂ term reuses SPSM `norm_cdf_2d_vfast` (`fast_bivariate_cdf.jl:19`); SPSM already has a weighted pairwise likelihood (`likelihoods.jl:118-169`). *Inference:* `besselk` is probably not ForwardDiff-differentiable in ν, so fix ν or use half-integer closed forms |
| Dynamics | VAR(M) from C_k at the current regime | Same "covariance-first" VAR(M) as the default for faithfulness. A `switching` option could fit PL on same-regime spells only, or share B across regimes and switch only the innovation covariance (*inference*: this removes the variance distortion of §1.7) |
| Time-varying factors | K×M matrices per season | B_{k,m}(t) and L_{k,0}(t) evaluated at season nodes (for example 12 or 24 per year) and cached; θ is smooth but its factors are piecewise. Later: a Vecchia or nearest-neighbour conditional, or spectral simulation (paper §7.1) |
| Initial conditions | 20-step burn-in | The plan's `past` / `burnin`. A censored past draws from a truncated multivariate normal |
| Likelihood kind | — | `:composite`; `bic` refuses, CLIC reserved (plan lines 221–222) |

---

## 3. Mapping onto the plan's architecture

### 3.1 Verdict
Build `StochasticWeatherGeneratorsMSTWeatherGen` as a class library that depends on **Core and Blocks**, not on Core alone.
- **Why not Core only.** Plan line 150 allows a Core-only class, but that class would have to re-implement:
  - `Regimes` (learned, frozen, external, observed);
  - SPSM marginals given OneHot(ẑ);
  - the Hurdle/Censored zero mechanism;
  - the censored pairwise kernel;
  - the per-block RNG streams;
  - persistence hooks.
- **What it becomes.** `MSTWeatherGen(; preset, order = 3, …)` lowers to:
  ```
  ProductGenerator(Regimes(…), Joint = LatentField(vars = (:RR, :TX, :TN, :QQ, :HU, :FF),
      margins = (RR = Hurdle(Anamorphosis(); zeros = Censored()), …),
      dependence = MultisiteGaussian(over = (:variable, :site), structure = GneitingMatern(), by = (:regime,),
                                     estimator = PairwiseLikelihood(PairDesign(nA = 15, nB = 15, tmax = 3))),
      dynamics = VAR(3; from = :covariance)))
  ```
  This is the same pattern as WGEN lowering to `ProductGenerator(NoRegimes(), …)` (plan line 328).
- **Where the new types live.** The MST library provides `LatentField`, `GneitingMatern`, `PairwiseLikelihood`, `VAR` and the PTA+GMM regime source. Blocks must already expose the **slots and protocols** listed below.

### 3.2 What the launch architecture must provide so MSTWeatherGen stays additive

**Must have now** (without these, the MST library would force changes to Core or Blocks types, the protocol, or the saved-file format):

1. **Site metadata in `WeatherData` and `FittedGenerator`.**
   - Store `sites` (id, lon, lat, optional alt, CRS kind `:lonlat | :projected`, optional grid index and mask), with accessors `coords(data)` and `distances(data; metric = Haversine())`.
   - Legacy already has `haversine` and `LLA(LAT, LON, ALT)` (`SWG/src/data/stations.jl:19,35-36`).
   - Add a constructor matching the R input: `WeatherData(A::Array{T,3}, dates, sites; vars)`.
   - Reserve the JSON field now, plus a `sites` keyword in `simulate` for ungauged sites (2.x).
   - The plan's WeatherData (lines 173–179) has no coordinates today.
2. **Multi-output blocks.**
   - `outputs(b)` returns a tuple of variables (the trait exists, plan line 147).
   - `simulate_block!` writes a (variables × sites) slab; `innovations` returns (variable × site).
   - `g[:TX]`, `curve(g, :TX, …)` and `correlation(g, :TX)` route to the block that owns `:TX`.
   - Other blocks (for example `Difference(:TX)`) can use a joint block as their DAG parent.
   - The engine must not assume one variable per block anywhere.
3. **Dependence protocol that goes beyond `MultisiteGaussian`.**
   - Keep `AbstractDependence` (plan lines 83–85), with an index set: sites, or (variable × site).
   - Add a separate `structure` slot, defaulting to `Unstructured()`, behind an abstract `AbstractStructure`.
   - Structure interface: `cov!(C, s, θ, coords; lag::Int, vars)`, `nparams`, `links`, `isvalid`. The *lag and variable arguments must be in the launch signature* even though launch only implements lag 0 for one variable.
   - Keep the estimator separate from the structure (`PairwiseRank`, `MomentMatching`, later `PairwiseLikelihood`).
   - `by` must accept `(:regime, :season_node)`.
   - Plan line 291 already promises "a space-time Matérn `structure`" as a keyword value; this gives it a type hierarchy.
4. **Pluggable regime source and chain estimator.**
   - The `Regimes` variants (plan lines 295–300) become subtypes of an abstract source, so the MST library can add `Regimes(source = PTAGMM(K = 1:6))`.
   - The chain estimator is a slot: `SmoothChain(season = d)`, with `WindowChain(L)` added for `:paper`.
   - Observed paths may carry season-specific label sets. *Inference:* for `:paper`, the K_total = Σ K_s states can use a block-masked A(t).
5. **Season keyword type.**
   - `season =` accepts an Int (trig degree), `Monthly()`, `Seasons(DJF, …)`, or any SPSM `AbstractBasis` (`IndicatorBasis`, `CyclicBSplineBasis`, `bases.jl:169,258`).
   - Add a notion of **evaluation nodes** for objects too costly to rebuild every day of the year (factors, VAR matrices).
6. **Initial conditions delegated to the block.**
   - Each block provides `memory(b) ≥ M` and a hook `latent_past(b, past)`, so latent lags (with censored components drawn from a truncated multivariate normal) come from the block.
   - The plan only covers the univariate censored lag (plan lines 191–193).
7. **Factor cache.**
   - Keyed by (block, regime, season node), read-only and shared across threads, built lazily, with memory accounting and Float32 support.
   - This generalizes "one read-only cached Cholesky factor per (k[, month])" (plan line 265) to dimension about 3000.
8. **RNG streams for fitting.**
   - Core's SplitMix derivation (plan line 200) also issues streams keyed by `(seed, :fit, block id)`, used for pair designs and initialization, so that fits are reproducible.
9. **Composite-likelihood plumbing.**
   - `likelihood_kind = :composite` already exists (plan lines 147, 221–222).
   - Make the estimator types `public`, and reserve `vcov(g; kind = :godambe)` so CLIC can come later.

**Should have now** (cheap at launch):

10. **Gridded output.**
    - Allow `sink` partitions by (member, time chunk) as well as (member, station); 498 pixels × 6 variables fits poorly in a per-station layout (plan lines 204–208).
    - Reserve a NetCDF/Zarr sink extension slot.
11. **Large numeric arrays in persistence.**
    - Store quantile tables (sites × variables × regimes × nodes) as compact typed arrays in the JSON format version.
    - Do not store factors; rebuild them on load and check them against stored hashes (plan lines 215–220).
12. **Shared nonparametric `Anamorphosis` marginal in Blocks.**
    - It has a tail-extrapolation hook (OQN logistic tails) and is shared with CensoredLatentGaussian ("Transform: anamorphosis or PowerTransform", plan line 338).
    - `Hurdle(…; zeros = Censored())` must work as a *component* inside a joint block, with threshold indexed by (site, regime, t).
13. **`handles_missing` capability per block.** Pairwise likelihood copes with missing pairs, so the MST library could accept gaps even though 2.0 requires gap-free data (plan lines 209–214).
14. **Multivariable statistics in `validate` (2.1).** `stats` should take functions of several variables, such as FWI or heatwaves, which is how the paper validates (§6.3.4–6.3.5).

**Left to the MST library later** (additive once items 1–9 exist):
- The `GneitingMatern` structure, in the paper or code parametrization (§1.5), chosen explicitly;
- `PairwiseLikelihood` with `PairDesign`;
- `VAR(M; from = :covariance)`;
- the PTA+GMM regime source and `WindowChain`;
- truncated OQN;
- Vecchia or spectral simulation;
- simulation at new sites.
- Extra dependencies: SpecialFunctions (besselk) and a clustering package. *Inference:* PTA (`PTAk::PTA3`) has no Julia equivalent, so `:paper` would document PCA on the unfolded tensor as a deviation, or vendor PTA.

### Critical Files for Implementation
- `C:\Users\metivier\.claude\plans\one-could-imagine-a-toasty-dove.md` (§1 lines 83–85, §2 T2 lines 142–151, §3 lines 173–228, §4 lines 246–321)
- `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\HHMM_Bernoulli_spatial\likelihoods.jl` (weighted pairwise likelihood, lines 118–169)
- `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\HHMM_Bernoulli_spatial\trig_conversion.jl` (seasonal range per regime, lines 6 and 24)
- `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\SmoothPeriodicStatsModels\src\TimeVaryingParameters\bases.jl` (`IndicatorBasis` and `CyclicBSplineBasis`, lines 169 and 258)
- `c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\src\data\stations.jl` (`haversine` and `LLA`, lines 19 and 35–36)