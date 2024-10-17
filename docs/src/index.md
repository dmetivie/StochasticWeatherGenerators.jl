# StochasticWeatherGenerators.jl

Documentation for StochasticWeatherGenerators.jl

!!! warning
    In construction! Note that the main functions to fit HMM, AR etc are currently in [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl). This will change when these packages are rebased.

A Julia package, to define, fit and use a Stochastic Weather Generator (SWG) as proposed in the [*Interpretable Seasonal Hidden Markov Model for spatio-temporal stochastic rain generation in France*](https://hal.inrae.fr/hal-04621349) paper. 
This SWG relies on some "Seasonal Hidden Markov Models" currently implemented in the package [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl).

!!! note
    The objective of this package is not only to show *my* model, but also to propose several classic (and newer) SWG model. Hence, **feel free to open an issue or open PR** with ideas and models.
    This would allow easy model comparison and, in some cases, combination.
    I'll try to implement the simple (and historic) model, i.e. the [Richardson - Water resources research, 1981](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/wr017i001p00182).

Go check the [documentation](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/) and the fully [reproducible tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/) associated with the paper.

!!! note "Stochastics Weather Generators"
    Stochastics Weather Generators are probabilistic weather models. Like random number generators, they can quickly generate multiple random sequences, except that the produced sequences correctly reproduce some statistics of interest, e.g. spatial-temporal correlations, extremes, etc.
    They can be used to study climate variability.


## API

### Fit function

```@docs
fit_mle_RO
fit_mle_RR
fit_TN
fit_AR1
```

### Climate indexes

```@docs
VCX3
monthly_agg
corTail
longuest_spell
pmf_spell

```

### Simulations

```@docs
rand_RR
rand_cond
```

### Correlation utilities

#### For temperature

```@docs
cor_groupby
cor_groupbyTXTN
cov_ar1
```

#### For rain

```@docs
cor_RR
cov_RR
Σ_Spearman2Pearson
Σ_Kendall2Pearson
joint_rain
```

### Map utilities

```@docs
distance_x_to_y
dms_to_dd
```

### Data manipulation

```@docs
collect_data_ECA
select_in_range_df
shortname
```

### Generic utilities

```@docs
my_color
```
