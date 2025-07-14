# StochasticWeatherGenerators.jl

A Julia package to define, fit and generate from Stochastic Weather Generators (SWGs).

## Stochastic Weather Generators

Stochastic Weather Generators are probabilistic weather models. Like random number generators, they can quickly generate multiple random sequences, except that the produced sequences correctly reproduce some statistics of interest, e.g. spatial-temporal correlations, extremes, etc.
They can be used to study climate variability, in particular extremes, and "stress" models that take weather inputs to quantify climate risk (present and future) typically in hydrology, agriculture, ecology, energy, insurance.
See the [IPCC note](https://www.ipcc-data.org/guidelines/pages/weather_generators.html) on SWG for more context.

![Illustration SWG](https://github.com/user-attachments/assets/7d4a66ab-ab28-4373-85fc-c9a176e26c82)

## Models

Currently, the package offers a few models:

- The daily multisite rainfall SWG **WGEN model** proposed in [1998_wilks_MultisiteGeneralizationDaily](@cite) and [2009_srisrikanthan_NestedMultisiteDaily](@cite)

- The daily multisite rainfall SWG **SHHMM model** proposed in the [2024_gobet_InterpretableSeasonalHiddena](@cite) paper. This SWG relies on a "Seasonal Autoregressive Hidden Markov Models" (SHMM) with interpretable hidden states.

- Some seasonal models HMM, AR(1), rain amounts, ..., are available as a [package extension](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)) with the package [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl). This means that to use these models, this package must also be installed and loaded. Seasonal models are models with yearly periodic parameters. For now, the parameters are transformations of trigonometric functions with arbitrary degree. This allows fitting all the data at once instead of fitting each month/season independently. This is useful for small datasets with high variability, e.g., rainfall data.

- The possibility to add stations to the SHMM and variables with respect to the hidden states. In particular, a multisite SWG with 5 weather variables (Rain, Temperature Max, Temperature Min, Evapotranspiration, Solar Irradiance) was tested. The structure of the added variables is very simplistic but does the job for the proof of concept. See the associated tutorial in the documentation.

!!! note "More models ? => Contribute"
    The objective of this package is not only to show *one* model, but also to propose several classic (and newer) SWG models.
    Hence, **feel free to open an issue or open a PR** with ideas and models.
    This would allow easy model comparison and, in some cases, combination.
