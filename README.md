# StochasticWeatherGenerators

[<img src="https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/docs/src/assets/logo.svg" width="25%" alt = "StochasticWeatherGenerators.jl"/>
](https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/docs/src/assets/logo.svg)

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/)

A Julia package to define, fit and generate from Stochastic Weather Generators (SWG).
It provides the interface to the models and data.

To install it, just `add` it as any other Julia package, i.e. in the Julia REPL:

```julia
import Pkg
Pkg.add("StochasticWeatherGenerators")
# or
pkg> add StochasticWeatherGenerators
```

[<img src="https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/docs/src/assets/fig_swg_dice.svg" width="100%" alt = "Illustration of SWG"/>
](https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/docs/src/assets/fig_swg_dice.svg)

> [!NOTE]
> Stochastic Weather Generators (SWGs) are statistical tools that create realistic sequences of weather data by mimicking patterns found in observations. They are used to study climate variability and provide synthetic data for impact models in fields like hydrology and agriculture.
> For more details, see the [documentation](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/#Stochastic-Weather-Generators) or the [IPCC note on weather generators](https://www.ipcc-data.org/guidelines/pages/weather_generators.html).

## Models

Currently, the package offers:

- The daily multisite rainfall SWG **WGEN model** proposed in [*Multisite Generalization of a Daily Stochastic Precipitation Generation Model*](https://www.sciencedirect.com/science/article/pii/S0022169498001863) by Wilks, D. S.

- The daily multisite rainfall SWG **SHHMM model** proposed in the [*Interpretable Seasonal Hidden Markov Model for spatio-temporal stochastic rain generation in France*](https://hal.inrae.fr/hal-04621349) paper. This SWG relies on a "Seasonal Autoregressive Hidden Markov Models" (SHMM) with interpretable hidden states. Note that the seasonal models HMM, AR etc are currently implemented in a separate package [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl).

- The possibility to add stations to the SHMM and variables with respect to the hidden states. In particular, a multisite SWG with 5 weather variables (Rain, Temperature Max, Temperature Min, Evapotranspiration, Solar Irradiance) was tested. The structure of the added variables is very simplistic but does the job for the proof of concept. See the associated tutorial in the documentation. This was used to generate a [Hackathon dataset](https://doi.org/10.57745/C3FNBY).

> [!IMPORTANT]
> The objective of this package is not only to show *my* model, but also to propose several classic (and newer) SWG models. Hence, **feel free to open an issue or open a PR** with ideas and models.
> This would allow easy model comparison and, in some cases, combination.
> I'll try to implement the simple (and historic) model, i.e. the [Richardson - Water resources research, 1981](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/wr017i001p00182).

Go check the [documentation](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/) and the fully [reproducible tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/) associated with the paper.
