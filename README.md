# StochasticWeatherGenerators

[<img src="https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/docs/src/assets/logo.svg" width="25%" alt = "StochasticWeatherGenerators.jl"/>
](https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/docs/src/assets/logo.svg)

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/)

A Julia package, to define, fit and use a Stochastic Weather Generator (SWG) as proposed in the [*Interpretable Seasonal Hidden Markov Model for spatio-temporal stochastic rain generation in France*](https://hal.inrae.fr/hal-04621349) paper. 
This SWG relies on some "Seasonal Hidden Markov Models" currently implemented in the package [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl).

> [!NOTE]
> The objective of this package is not only to show *my* model, but also to propose several classic (and newer) SWG model. Hence, **feel free to open an issue or open PR** with ideas and models.
> This would allow easy model comparison and, in some cases, combination.
> I'll try to implement the simple (and historic) model, i.e. the [Richardson - Water resources research, 1981](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/wr017i001p00182).

Go check the [documentation](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/) and the fully [reproducible tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/) associated with the paper.