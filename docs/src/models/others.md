# Temperature & Other variables

So far only a seasonal `AR(1)` model is coded, i.e., AR(1) with periodic coefficients.
It can be used for Temperature but also for some other "simple" variables.

For a showcase on how to use these on a multivariate example, see the [AppliMultivariate SWG: Application to crop model](@ref TutoCrop) tutorial.

```@docs
fit_AR1
cov_ar1
```

## Temperature

There is a simple model to generate correlated $T_{\mathrm{max}}$ and $T_{\mathrm{max}}$.

```@docs
fit_TN
rand_cond
cor_groupby
cor_groupbyTXTN
```

!!! todo
    - Add `AR(p>1)` models.
    - Add state-of-the-art temperature models, e.g. with good extremes.

## Others

For positive variables, one can truncate when generating the `AR(1)` model, but it is not a convincing modeling approach (but it kind of works when in a hurry).

!!! todo
    - Add a true truncated AR model.
    - Add more appropriate and domain specific models
    - Other multivariate models: [2018_sparks_IMAGEMultivariateMultisite, 2010_flecher_StochasticDailyWeather](@cite)


## References

```@bibliography
Pages = ["others.md"]
```
