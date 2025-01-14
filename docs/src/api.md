# API

## Fit function

```@docs
fit_mle_RO
fit_mle_RR
fit_TN
fit_AR1
```

## Climate indexes

```@docs
VCX3
monthly_agg
corTail
longuest_spell
pmf_spell

```

## Simulations

```@docs
rand_RR
rand_cond
```

## Correlation utilities

### For temperature

```@docs
cor_groupby
cor_groupbyTXTN
cov_ar1
```

### For rain

```@docs
cor_RR
cov_RR
Σ_Spearman2Pearson
Σ_Kendall2Pearson
joint_rain
```

## Map utilities

```@docs
distance_x_to_y
dms_to_dd
```

## Data manipulation

```@docs
collect_data_ECA
collect_data_INRAE
collect_data_MeteoFrance 
download_data_MeteoFrance 
clean_data 
select_in_range_df
shortname
```

## Generic utilities

```@docs
my_color
```
