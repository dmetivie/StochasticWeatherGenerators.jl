# StochasticWeatherGenerators.jl

A Julia package to define, fit and generate from Stochastic Weather Generators (SWGs).

## Stochastic Weather Generators

Stochastic Weather Generators (SWGs) are statistical models designed to simulate realistic sequences of weather variables, such as rainfall or temperature, by capturing key statistical properties observed in training data. SWGs are calibrated to reproduce important features like spatial and temporal correlations, frequency and intensity of extremes, and seasonal patterns.
Simply put, they are like random number generators but for weather data.

```@raw html
<img src="https://github-production-user-asset-6210df.s3.amazonaws.com/46794064/466196788-7d4a66ab-ab28-4373-85fc-c9a176e26c82.svg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250715%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250715T082127Z&X-Amz-Expires=300&X-Amz-Signature=1740099d90d8bcf2f812091d38fadeba9e332b370e86ad6693ae0729f953be1f&X-Amz-SignedHeaders=host" width="95%" alt = "Illustration of a SWG"/>
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
