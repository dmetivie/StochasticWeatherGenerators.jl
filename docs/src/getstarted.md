# 🚀 Get started

`StochasticWeatherGenerators.jl` defines, fits and simulates Stochastic Weather Generators.
What a model needs to be fitted depends on what you load: the package is split in a light core and in several [package extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)) that Julia activates when their trigger packages are loaded.
This page explains what to load for what, and the conventions shared by all the functions.

## Install

```julia
import Pkg
Pkg.add("StochasticWeatherGenerators")
```

## What to load

| You want | Load |
|---|---|
| The WGEN model, rain correlations, Markov chains, readers of weather data, climate indices | `using StochasticWeatherGenerators` |
| The seasonal fits: `fit_mle_RO`, `fit_mle_RR`, `mix_ini`, `fit_AR1`, `fit_TN`, `rand_cond`, `cor_groupby`, `cor_groupbyTXTN` | `using SmoothPeriodicStatsModels`, `using Optimization, SimpleNonlinearSolve` and a solver package such as `using OptimizationOptimJL` |
| The plots `errorline`, `errorscatter`, `errorlinehist`, `errorscatterhist` | `using Makie`, or a backend such as `using CairoMakie` |
| The map `map_with_stations` | the plots above and `using GeoMakie, NaturalEarth` |

The seasonal models (periodic hidden Markov models, AR(1), mixtures) live in [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl).
The seasonal fits of this package are fits of these models, and they go through two layers of extensions:

1. Loading `SmoothPeriodicStatsModels` activates the extension of this package that defines `fit_AR1`, `fit_TN`, `fit_mle_RO`, `fit_mle_RR`, `mix_ini`, `rand_cond`, `cor_groupby` and `cor_groupbyTXTN`.
   Before, these functions exist but have no method.
2. Loading `Optimization` and `SimpleNonlinearSolve` activates the fitting code of `SmoothPeriodicStatsModels` itself, which these functions call.
   Without them, `SmoothPeriodicStatsModels` can simulate and evaluate likelihoods, but a fit fails with an error.

!!! tip "The usual preamble for the seasonal fits"
    ```julia
    using StochasticWeatherGenerators
    using SmoothPeriodicStatsModels
    using Optimization, SimpleNonlinearSolve # activate the fits
    using OptimizationOptimJL # an optimization algorithm, e.g. LBFGS
    ```

### The `solver` keyword

The fitting functions do not choose an optimization algorithm for you: pass any algorithm of [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) as the keyword `solver`, for example `LBFGS()` from `OptimizationOptimJL`.
Without it, the fit throws an `ArgumentError`. The other keywords are passed on to the fit of `SmoothPeriodicStatsModels`.

```julia
ar1 = fit_AR1(df, :TX, degree; solver = LBFGS())
```

## Conventions

- **Observations are `D × N` matrices**: `D` sites in rows and `N` days in columns. This holds for the input and the output of `fit_wgen`, `rand`, `fit_markov_chain` and `simulate_markov_gaussian`.
  The functions that mirror `Statistics.cor` (`Σ_Spearman2Pearson`, `Σ_Kendall2Pearson`, `corTail`, `joint_rain`) take a `dims` keyword with the same meaning, `dims = 2` for this layout.
- **Time is the day of the year in `1:366`.** `dayofyear_Leap(date)` gives it, with index 60 for the 29th of February (unused in non-leap years), so that day `t` is the same calendar date in every year. The period of the seasonal models is `T = 366`, and `n2t` is the vector of the day index of each observation.
- **Hidden states are `z ∈ 1:K`.** They are given in a column `z` of the station `DataFrame`s. Functions that fit one model per state return a vector whose index is the state, so every state of `1:K` must appear in the data.
- **Station data are `DataFrame`s** with a `DATE` column and one column per variable (`RR` rain amount, `RO` rain occurrence, `TX` and `TN` temperature, ...). The readers of the [Weather Data](data.md) page return this format.
- **Random numbers**: the functions that draw random values (`rand`, `fit_wgen`, `rand_RR`, `rand_cond`, ...) accept a random number generator as an optional first argument, and use the global one otherwise.

## A first model: WGEN

The WGEN model needs only the core of the package. It is a Markov chain of order `p` at each of the `D` sites, with the dependence between sites given by a Gaussian copula, one set of parameters per month.

Fit it to a `D × N` matrix of rain occurrences and simulate years from it. Here the data are random, for illustration.

```@example getstarted
using StochasticWeatherGenerators, Dates, Random

rng = Xoshiro(1)
D, p = 3, 1
dates = Date(2000):Day(1):Date(2009, 12, 31)
Y = rand(rng, D, length(dates)) .< 0.3 # D × N, rain occurrence
idx_months = [findall(d -> month(d) == m, dates) for m in 1:12] # the days of each month

model = fit_wgen(rng, Y, idx_months, p)
Y_past = rand(rng, Bool, D, p) # the p days before the first simulated day, D × p
Y_sim = rand(rng, model, 2020:2022; Y_past)
size(Y_sim)
```

## A first seasonal fit: AR(1)

A seasonal AR(1) has periodic coefficients, trigonometric polynomials of the day of the year. Its fit needs the preamble above.

```@example getstarted
using SmoothPeriodicStatsModels
using Optimization, SimpleNonlinearSolve, OptimizationOptimJL

T = 366
dates = Date(2000):Day(1):Date(2009, 12, 31)
n2t = dayofyear_Leap.(dates) # day of the year of each observation
y = zeros(length(dates)) # a temperature-like series
for n in 2:length(y)
    y[n] = 0.6 * y[n-1] + randn(rng)
end
y .+= 15 .+ 8 .* sin.(2π .* n2t ./ T)

degree = 1 # degree of the trigonometric polynomials
ar1 = fit_AR1(y, n2t, degree, T; solver = LBFGS())
round.(ar1.ρ[[1, 100, 200, 300]], digits = 2) # the fitted autocorrelation on four days of the year
```

With a hidden state sequence `z`, `fit_AR1(y, z, n2t, degree, K, T; solver)` fits one AR(1) per state, and with a station `DataFrame`, `fit_AR1(df, :TX, degree; solver)` reads the series, the day of the year and the states from its columns.

## Where to go next

- The [Rainfall](models/rain.md) and [Temperature & Others](models/others.md) pages describe the models and list their functions.
- The tutorial [Multisite rainfall (HMM)](@ref TutoSHHMM) reproduces the paper: seasonal hidden Markov model, WGEN benchmark, and diagnostics.
- The tutorial [Multivariate SWG](@ref TutoCrop) adds stations and variables (temperature, solar irradiance, ...) conditionally to the hidden states.
- [Weather Data](data.md) reads station files.
