# StochasticWeatherGenerators.jl

A Julia package to define, fit and generate from Stochastic Weather Generators (SWGs).

## Stochastic Weather Generators

Stochastic Weather Generators (SWGs) are statistical models designed to simulate realistic sequences of weather variables, such as rainfall or temperature, by capturing key statistical properties observed in training data. SWGs are calibrated to reproduce important features like spatial and temporal correlations, frequency and intensity of extremes, and seasonal patterns.
Simply put, they are like random number generators but for weather data.

```@raw html
<img src="https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/docs/src/assets/fig_swg_dice.svg" width="95%" alt = "Illustration of a SWG"/>
```

SWGs are used to assess climate variability and change, especially for analyzing rare or extreme events. They provide synthetic weather data for driving impact models in fields such as hydrology, agriculture, ecology, energy, and insurance, enabling robust risk assessments under current and future climate scenarios.

For more background, see the [IPCC note on weather generators](https://www.ipcc-data.org/guidelines/pages/weather_generators.html).

### Difference with climate models

Climate models are based on physical principles, relying on partial differential equations (PDEs) to simulate the dynamics of the atmosphere, oceans, and land surfaces. These models discretize the entire Earth into a grid and solve for many variables (e.g., temperature, pressure, humidity) at each grid point.

- This approach is computationally intensive and typically very slow, especially for generating large ensembles or long time series.
- Climate models often struggle to accurately reproduce rare or extreme events due to limitations in spatial resolution and the complexity of underlying processes.

These models are essential for understanding long-term climate trends, feedback mechanisms, and interactions between different components of the Earth system. They are used for climate projections, scenario analysis, and understanding the impacts of climate change. In particular, simulations from these models are used in the IPCC reports.
In Julia there is the [CliMA organization](https://github.com/CliMA) that is building a new Earth system model.

In contrast, **Stochastic Weather Generators** provide a **fast** and **flexible** way to generate synthetic weather sequences only for **a few variables of interest** and for the **desired spatio-temporal resolution** making them suitable for impact studies.

### Where is AI?

SWGs are spatiotemporal generative models. 
Currently, they are mostly based on classic statistical models, such as Markov chains, Hidden Markov Models, autoregressive models and geostatistical models. Generative Deep Learning models are not as widely used yet in this field as in other domains like image or text generation. A few reasons for this are:
- The relatively small size of weather datasets compared to image or text datasets, e.g., 60 years of daily data is only 21,900 data points for one station.
- Moreover, this data is very non i.i.d. (independent and identically distributed), i.e., there are trend effects, seasonal effects.
- 


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
