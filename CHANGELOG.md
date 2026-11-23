# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions follow
[Semantic Versioning](https://semver.org/).

## [3.0.0] - 2026-11-23

Version 3 follows [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl)
version 4: the seasonal fits of this package are now fits of its models, and they take the same
conventions. The API of version 2 is not kept.

### Changed

- Requires SmoothPeriodicStatsModels 4.1 (was 2.0 and 2.1) and Julia 1.12.
- Observations are `D × N` matrices, sites in rows and days in columns, in every function:
  `wgen`, `fit_wgen`, `fit_markov_chain`, `fit_Ω`, `simulate_markov_gaussian` and `cov_ar1`.
  The initial condition `Y_ini` of `rand(::wgen, ...)` is `D × p`, and so is the simulation it
  returns (`D × N`). Matrices of version 2 must be transposed.
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
- Documentation: the tutorial of multisite rainfall with a hidden Markov model and a Gaussian
  latent variable, and `seasonal_family` mixtures in the rainfall page.

### Fixed

- `cov_ar1`: the residuals of every station were computed from the series of the first station
  (linear indexing into the matrix), so the covariances did not describe the stations.
- `rand_cond` drew the other normal variable with `1 - ρ²` as its standard deviation instead of
  its variance, which gave the wrong conditional spread.
- `rand_cond` read the coefficients of `fit_TN` as degree 1, whatever degree was fitted.
- `fit_mle_RO` and `fit_mle_RR` did not forward their extra keywords: `kwargs...` stood among
  the positional arguments of the `DataFrame` methods.
