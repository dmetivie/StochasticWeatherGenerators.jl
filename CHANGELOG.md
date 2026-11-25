# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow
[Semantic Versioning](https://semver.org/).

## [1.4.0] - Unreleased

This version follows [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl)
version 4: the seasonal fits of this package are now fits of its models, and they take the same
conventions. The API of the previous versions is not kept.

### Changed

- Requires SmoothPeriodicStatsModels 4.1 and Julia 1.12.
- Observations are `D × N` matrices, sites in rows and days in columns, in every function:
  `wgen`, `fit_wgen`, `fit_markov_chain`, `fit_Ω`, `simulate_markov_gaussian` and `cov_ar1`.
  The initial condition of `rand(::wgen, ...)` and `simulate_markov_gaussian` is the keyword
  `Y_past` (it was `Y_ini`, as `Y_past` is called in SmoothPeriodicStatsModels), a `D × p`
  matrix, and the simulation it returns is `D × N`. Matrices in the former layout must be
  transposed.
- The fits through SmoothPeriodicStatsModels (`fit_AR1`, `fit_TN`, `fit_mle_RO`, `fit_mle_RR`,
  `cor_groupby`, `cor_groupbyTXTN`) take the optimization algorithm as a `solver` keyword, and
  forward their other keywords to the fit. `cor_groupby` and `cor_groupbyTXTN` no longer
  hard-code Ipopt. The solver stack is a weak dependency of SmoothPeriodicStatsModels: load it
  and the package of your solver, for example
  `using Optimization, SimpleNonlinearSolve, OptimizationOptimJL` and `solver = LBFGS()`.
- `fit_AR1` fits a `SmoothAR1` and returns the evaluated `AR1`. The JuMP-based fit is gone.
- `fit_TN` returns one `TimeVaryingFit` per hidden state instead of a raw solution. `fit(t)` is
  the `Gamma` of day `t`, and `coef` and `curve` read its coefficients.
- `rand_cond` reads the marginal of each hidden state from `θ_uni[k](t)`, which is what `fit_TN`
  returns, instead of rebuilding a `Gamma` from a coefficient vector.
- `mix_ini` returns a `SmoothSeasonalMixture` instead of the tuple `(mixtures, θ_al0, θ_Y0)`,
  and `fit_mle_RR` fits it with `SmoothPeriodicStatsModels.fit_mle`. Its `silence` and
  `warm_start` keywords are gone. The degree is read off the length of `θσ10`.
- `joint_rain`, `Σ_Spearman2Pearson`, `Σ_Kendall2Pearson` and `corTail` take `dims`, as
  `Statistics.cor` does. The default `dims = 1` is the layout they had. A `cor_method` passed to
  the correlation functions of the rain amounts is called as `cor_method(M; dims = 2)`, so it
  must accept `dims`.
- `monthly_agg(df, col)` returns one row per calendar year from the first to the last year of
  `df`, with `missing` for the months it does not contain. It used to stack the years it found,
  which shifted the rows or threw a `DimensionMismatch` when the first or last year was
  incomplete. The keyword is `matrixform`; the misspelled `maxtrixform` still works.
- `VCX3(y, idxs; nb = 3)` takes `nb` as a keyword, as `VCX3(df)` already did. `VCX3(y, idxs, nb)`
  still works.
- `simulate_markov_gaussian` and `rand(::wgen, ...)` do not modify `Y_past` anymore, and accept
  years that are not consecutive.

### Added

- A random number generator can be given first to `rand(::wgen, ...)`, `fit_wgen`, `fit_Ω`,
  `simulate_markov_gaussian`, `rand_RR` and `rand_cond`. Without one they use the global
  generator, as before.
- Plot functions, defined when Makie is loaded (`MakieExt`): `errorline`, `errorscatter`,
  `errorlinehist`, `errorscatterhist` and their `!` forms. `map_with_stations` is defined when
  GeoMakie and NaturalEarth are loaded as well (`GeoMakieExt`). Called without a backend, they
  point to the package to load.
- `init` in `fit_AR1`, the starting point of the fit as in `SmoothAR1`. By default it is read
  off the data: from the zero start, a series far from zero such as a temperature ended at
  `ρ = 1`, where the fit cannot move.
- Tests (`test/runtests.jl`).
- Documentation: a Get started page, the tutorial of multisite rainfall with a hidden Markov
  model and a Gaussian latent variable, and `seasonal_family` mixtures in the rainfall page.

### Removed

- `collect_data_MeteoFrance` and `download_data_MeteoFrance` are commented out: the data.gouv.fr
  API changed in 2024 and they no longer worked. The package also downloaded a file from
  data.gouv.fr when it was loaded, which is gone.

### Fixed

- `cor_groupby` and `cor_groupbyTXTN` could return a correlation of ±1 for a state whatever the
  data, with a successful solver status: the fit minimized the sum of the squared errors, and its
  first step landed where `ρ(t)` is saturated. It minimizes the mean now. In the multivariate
  tutorial, 10 of the 16 (station, state) correlations between `TX` and `TX - TN` were affected.
- `fit_AR1` ended at a poor optimum, with a `ρ(t)` carrying the seasonal cycle, for a series with
  a strong seasonal mean. Its default start now follows the seasonal mean of the series.
- `cov_ar1`: the residuals of every station were computed from the series of the first station
  (linear indexing into the matrix), so the covariances did not describe the stations.
- `rand_cond` drew the other normal variable with `1 - ρ²` as its standard deviation instead of
  its variance, which gave the wrong conditional spread.
- `rand_cond` read the coefficients of `fit_TN` as degree 1, whatever degree was fitted.
- `fit_mle_RO` and `fit_mle_RR` did not forward their extra keywords: `kwargs...` stood among
  the positional arguments of the `DataFrame` methods.
- `wgen(p, D)` and `wgen(p, D, idxs)` built a model for one site instead of `D`.
- `distance_x_to_y` and `distance_x_to_all_stations` threw a `MethodError` (`LLA` with a
  `Float64` and an `Int` field), and `distance_x_to_y` used the longitude of the second station
  for the first.
- `cor_RR(dfs)` and `cov_RR(dfs)` threw an error when two stations had one rain day in common,
  and `cov_RR` threw one whenever a covariance was `missing`; they return `missing` now, or
  `impute_missing` when it is given.
- `Σ_Spearman2Pearson` and `Σ_Kendall2Pearson` returned `[1 missing; 1 missing]` for fewer than
  two observations: the diagonal is 1 now.
- `fit_TN`, `cor_groupby` and `cor_groupbyTXTN` return their fits in the order of the hidden
  states, and warn when a state of `1:K` is missing from the data.
