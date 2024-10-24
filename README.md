# StochasticWeatherGenerators

[<img src="https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/docs/src/assets/logo.svg" width="25%" alt = "StochasticWeatherGenerators.jl"/>
](https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/docs/src/assets/logo.svg)

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/)

A Julia package, to define, fit and use Stochastic Weather Generators (SWG).

Currently, the package offers:

- The daily multisite rain SWG proposed in the [*Interpretable Seasonal Hidden Markov Model for spatio-temporal stochastic rain generation in France*](https://hal.inrae.fr/hal-04621349) paper. This SWG relies on a "Seasonal Autoregressive Hidden Markov Models" (SHMM) with interpretable hidden states. Note that the seasonal models HMM, AR etc are currently implemented in a separate package [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl).

- The possibility to add stations to the SHMM and variables w.r.t. the hidden states. In particular, a multisite SWG with 5 weather variables (Rain, Temperature Max, Temperature Min, Evapotranspiration, Solar Irradiance) was tested. The structure of the added variables is very simplistic but do the job for the proof of concept. See the associated tutorial in the documentation.

> [!NOTE] Stochastics Weather Generators
> Stochastics Weather Generators are probabilistic weather models. Like random number generators, they can quickly generate multiple random sequences, except that the produced sequences correctly reproduce some statistics of interest, e.g. spatial-temporal correlations, extremes, etc.
> They can be used to study climate variability in particular extremes and "stress" models that takes weather inputs e.g. crop models. See the [IPCC note](https://www.ipcc-data.org/guidelines/pages/weather_generators.html) on SWG for more context.

> [!IMPORTANT]
> The objective of this package is not only to show *my* model, but also to propose several classic (and newer) SWG model. Hence, **feel free to open an issue or open PR** with ideas and models.
> This would allow easy model comparison and, in some cases, combination.
> I'll try to implement the simple (and historic) model, i.e. the [Richardson - Water resources research, 1981](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/wr017i001p00182).

Go check the [documentation](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/) and the fully [reproducible tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/) associated with the paper.
