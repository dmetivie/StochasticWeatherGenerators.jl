# StochasticWeatherGenerators.jl

Documentation for StochasticWeatherGenerators.jl

!!! warning
    In construction! Note that the main functions to fit HMM, AR etc are currently in [SmoothPeriodicStatsModels.jl](https://github.com/dmetivie/SmoothPeriodicStatsModels.jl). This will change when these packages are rebased.

## API

### Fit function

```@docs
fit_mle_stations
fit_mle_RR
```

### Climate indexes

```@docs
VCX3
cum_monthly
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
