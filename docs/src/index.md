# StochasticWeatherGenerators.jl

A Julia package, to define, fit and use Stochastic Weather Generators (SWG).

Currently, the package offers:

- The daily multisite rainfall SWG **WGEN model** proposed in [1998_wilks_MultisiteGeneralizationDaily](@cite) and [2009_srisrikanthan_NestedMultisiteDaily](@cite)

- The daily multisite rainfall SWG **SHHMM model** proposed in the [2024_gobet_InterpretableSeasonalHiddena](@cite) paper. This SWG relies on a "Seasonal Autoregressive Hidden Markov Models" (SHMM) with interpretable hidden states. Note that the seasonal models HMM, AR etc are currently implemented in a separate package [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl).

- The possibility to add stations to the SHMM and variables with respect to the hidden states. In particular, a multisite SWG with 5 weather variables (Rain, Temperature Max, Temperature Min, Evapotranspiration, Solar Irradiance) was tested. The structure of the added variables is very simplistic but do the job for the proof of concept. See the associated tutorial in the documentation.

!!! tip "Stochastics Weather Generators"
    Stochastics Weather Generators are probabilistic weather models. Like random number generators, they can quickly generate multiple random sequences, except that the produced sequences correctly reproduce some statistics of interest, e.g. spatial-temporal correlations, extremes, etc.
    They can be used to study climate variability in particular extremes and "stress" models that takes weather inputs e.g. crop models. See the [IPCC note](https://www.ipcc-data.org/guidelines/pages/weather_generators.html) on SWG for more context.

!!! note "More models ? => Contribute"
    The objective of this package is not only to show *one* model, but also to propose several classic (and newer) SWG model. Hence, **feel free to open an issue or open PR** with ideas and models.
    This would allow easy model comparison and, in some cases, combination.

!!! warning
    Package in construction! 
