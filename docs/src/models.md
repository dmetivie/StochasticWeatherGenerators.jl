# 📘 Stochastic Weather Generator Models

```@contents
Pages = ["models.md"]
Depth = 5
```

## Rain

TODO:

- Censored models as in [2009_ailliot_SpaceTimeModelling](@cite) , [2020_vaittinadaayar_SpacetimeSimulationPrecipitation](@cite), etc.
- Something like [2021_najibi_IdentifyingWeatherRegimes](@cite). The authors even created a [Git](https://github.com/cadillac-desert/wgen) of their model.

### WGEN — Multisite latent Gaussian Rain Occurrence Model

#### Overview

The WGEN model is a weather generator approach using **latent Gaussian variables** to generate multisite rain occurrences with order $p$ Markov chain dependence at each site.
The model was first proposed in [1998_wilks_MultisiteGeneralizationDaily](@cite) and extended to order 4 Markov chains in [2009_srisrikanthan_NestedMultisiteDaily](@cite).

##### Model Components

- **Single site Temporal Occurrence**: Binary rain/no-rain determined by Markov chain of order $p$ at each station.
- **Spatial correlation**: Gaussian latent variable with spatial covariance i.e. Gaussian copula.
- **Amount Layer**: Can be added trained on top of the occurrence layer using parametric distributions (e.g., Gamma, mixture of exponential) fitted to positive rain days. See [2018_evin_StochasticGenerationMultisite](@cite) for a more complex rainfall amount model and associated R package [GWEX](https://github.com/guillaumeevin/GWEX) (TODO implementation).
- **Seasonality**: All parameters are fitted monthly (current implementation concatenates each month into the likelihood

TODO: sum the likelihood of each month, or take the mean/median of each month)

##### Assumptions & Limitations

- Computational complexity scales as $\mathcal{O}(S^2)$ where $S$ is the station number, making it less suitable for high-dimensional station networks.
- Lacks explicit hidden states or large-scale regime representation.
- Time and space are fitted separately
- Ideal for small regions.

TODO: model selection

##### Mathematical description

TODO

#### Usage

TODO (see tutorial)

```@docs
wgen
fit_wgen
fit_Ω
rand(model::wgen, years::AbstractArray{<:Integer}; Y_ini)
```

For simple Markov chain of order $p$

```@docs
fit_markov_chain
simulate_markov_gaussian
```

### SHHMM — Seasonal Hierarchical Hidden Markov Model

#### Overview

`SHHMM` is a **multisite stochastic weather generator** (SWG) based on a **seasonal Hidden Markov Model (HMM)**.
The model was proposed in [2024_gobet_InterpretableSeasonalHiddena](@cite) by one author of this package.
It is designed to generate **temporally and spatially correlated precipitation sequences** over large regions (e.g., France), while maintaining full interpretability.
It has a **Hierarchical Structure** as it combines a discrete HMM for occurrences with a **seasonal rainfall amount model** conditional on the hidden states.

##### Model Components

- **Hidden Weather Regimes**: The model infers interpretable hidden states (weather regimes) from daily **rain occurrence** data, without relying on exogenous variables.
- **Rain Occurrence Layer**: Discrete Bernoulli HMM modeling rain/no-rain across stations.
- **Local Memory**: Incorporates autoregressive memory for dry/wet persistence (Auto-Regressive HMM). The local memory can be per site. TODO: doc
- **Rain Amount Layer**: Gaussian copula-based model to assign amounts conditioned on the occurrence and hidden states. The parametric form can be an arbitrary continuous distribution from `Distributions.jl`.
- **Seasonality**: All model parameters vary continuously over the year.

##### Assumptions & Limitations

- Relies on the **conditional independence** assumption — dense station networks may violate this.
- Computational complexity is $\mathcal{O}(S)$ and $\mathcal{O}(K^2)$ (number of hidden states).
- Tail behavior/extreme events can challenge the stability or identifiability of the hidden states.
- Trade-off between **model interpretability** and spatial resolution.
- **Ideal** to studying climate variability over large regions.

TODO: model selection

##### Mathematical description

TODO

```@raw html
<img src="https://github.com/dmetivie/StochasticWeatherGenerators.jl/assets/46794064/5fe1d677-877d-4fd5-83ac-29d30f728ca5" width="95%" alt = "Schematic of the Autoregressive Seasonal Hidden Markov Model"/>
```

#### Usage

TODO (see tutorial)

##### Fit

Fit functions condition on **given** hidden states. Fix `K=1` or `z=ones(size(Y,A))` if you do not have hidden states.

```@docs
fit_mle_RO
fit_mle_RR
```

TODO add `SmoothPeriodicStatsModels.fit_mle`

##### Spatial correlations

```@docs
cor_RR
cov_RR
Σ_Spearman2Pearson
Σ_Kendall2Pearson
joint_rain
```

##### Generation

Function to generate rain amounts

```@docs
SmoothPeriodicStatsModels.rand
rand_RR
```

## Temperature & Other variables

So far `AR(1)` models with truncated values for positive variables.
See

```@docs
fit_AR1
cov_ar1
```

There is a simple model to generate correlated $T_{\mathrm{max}}$ and $T_{\mathrm{max}}$.

```@docs
fit_TN
rand_cond
cor_groupby
cor_groupbyTXTN
```

## References

```@bibliography
1998_wilks_MultisiteGeneralizationDaily
2009_srisrikanthan_NestedMultisiteDaily
2018_evin_StochasticGenerationMultisite
2024_gobet_InterpretableSeasonalHiddena
```
