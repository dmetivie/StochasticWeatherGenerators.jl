
# Rainfall

```@contents
Pages = ["rain.md"]
Depth = 5
```

!!! todo
    Amongst the rich literature of SWG there are many interesting models. For example
    - Censored Latent Gaussian models as in [2020_vaittinadaayar_SpacetimeSimulationPrecipitation, 2009_ailliot_SpaceTimeModelling](@cite), etc. In these models, the rainfall occurrence and amounts are generated using the same underlying Gaussian latent variable. It makes possible having very complex spatiotemporal covariance structures.
    - The author of [2021_najibi_IdentifyingWeatherRegimes](@cite) created a [Git repo of their model](https://github.com/cadillac-desert/wgen), which is always cool and interesting. -> investigate

## WGEN — Multisite latent Gaussian Rain Occurrence Model

### Overview

The WGEN model is a weather generator approach using **latent Gaussian variables** to generate multisite rain occurrences with order $p$ Markov chain dependence at each site.
The model was first proposed in [1998_wilks_MultisiteGeneralizationDaily](@cite) and extended to order 4 Markov chains in [2009_srisrikanthan_NestedMultisiteDaily](@cite).

!!! warning "WGEN model"
    In the literature, the name WGEN refers to different models e.g. to [1981_richardson_StochasticSimulationDaily](@cite) !

#### Model Components

- **Single site Temporal Occurrence**: Binary rain/no-rain determined by Markov chain of order $p$ at each station.
- **Spatial correlation**: Gaussian latent variable with spatial covariance, i.e. Gaussian copula.
- **Amount Layer**: Can be added trained on top of the occurrence layer using parametric distributions (e.g., Gamma, mixture of exponential) fitted to positive rain days. See [2018_evin_StochasticGenerationMultisite](@cite) for a more complex rainfall amount model and associated R package [GWEX](https://github.com/guillaumeevin/GWEX) (TODO implementation).
- **Seasonality**: All parameters are fitted monthly.

!!! todo
    - Rewrite the Markov chain efficiently (the generation is slow).
    - Current implementation concatenates each month into the likelihood to fit -> add different fit options:
        - Sum the likelihood of each month
        - Estimate at each month of each year and take the mean/median of each month

#### Assumptions & Limitations

- Computational complexity scales as $\mathcal{O}(S^2)$ where $S$ is the station number, making it less suitable for high-dimensional station networks.
- Lacks explicit hidden states or large-scale regime representation.
- Time and space are fitted separately
- Ideal for small regions.

!!! todo
    Model selection

#### Mathematical description

TODO

### Usage

See the [WGEN model section](@ref TutoWGEN) of the [tutorial](@ref TutoSHHMM) for an example of fit and simulation.

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

## SHHMM — Seasonal Hierarchical Hidden Markov Model

### Overview

`SHHMM` is a **multisite stochastic weather generator** (SWG) based on a **seasonal Hidden Markov Model (HMM)**.
The model was proposed in [2024_gobet_InterpretableSeasonalHiddena](@cite) by one author of this package.
It is designed to generate **temporally and spatially correlated precipitation sequences** over large regions (e.g., France), while maintaining full interpretability.
It has a **Hierarchical Structure** as it combines a discrete HMM for occurrences with a **seasonal rainfall amount model** conditional on the hidden states.

#### Model Components

- **Hidden Weather Regimes**: The model infers interpretable hidden states (weather regimes) from daily **rain occurrence** data, without relying on exogenous variables.
- **Rain Occurrence Layer**: Discrete Bernoulli HMM modeling rain/no-rain across stations.
- **Local Memory**: Incorporates autoregressive memory for dry/wet persistence (Auto-Regressive HMM). The local memory can be per site. TODO: doc
- **Rainfall amount Layer**: Gaussian copula-based model to assign amounts conditioned on the occurrence and hidden states. The parametric form can be an arbitrary continuous distribution from `Distributions.jl`.
- **Seasonality**: All model parameters vary continuously over the year.

#### Assumptions & Limitations

- Relies on the **conditional independence** assumption — dense station networks may violate this.
- Computational complexity is $\mathcal{O}(S)$ and $\mathcal{O}(K^2)$ (number of hidden states).
- Tail behavior/extreme events can challenge the stability or identifiability of the hidden states.
- Trade-off between **model interpretability** and spatial resolution.
- **Ideal** to studying climate variability over large regions.

!!! todo
    Add model selection details

#### Mathematical description

TODO

```@raw html
<img src="https://github.com/dmetivie/StochasticWeatherGenerators.jl/assets/46794064/5fe1d677-877d-4fd5-83ac-29d30f728ca5" width="95%" alt = "Schematic of the Autoregressive Seasonal Hidden Markov Model"/>
```

### Usage

For a complete example see the [this section](@ref TutoSHHMM) that both serve as a tutorial to fit, simulate and visualize the SHHMM model **and** the reproducible supplementary material of the paper [2024_gobet_InterpretableSeasonalHiddena](@cite).
Some element of the paper are not included in the tutorial to limit runtime of CI (model selection, Weather Regimes illustration, comparison with climate models).

#### Fit

Fit functions condition on **given** hidden states. Fix `K=1` or `z=ones(size(Y,A))` if you do not have hidden states.

```@docs
fit_mle_RO
fit_mle_RR
```

TO get an initial condition for `fit_mle_RR`, one can use (for a given hidden state `k`):

```@docs
mix_ini
```

TODO add `SmoothPeriodicStatsModels.fit_mle`

#### Spatial correlations

```@docs
cor_RR
cov_RR
Σ_Spearman2Pearson
Σ_Kendall2Pearson
joint_rain
```

#### Generation

Function to generate rainfall amounts

```@docs
rand_RR
```

## References

```@bibliography
Pages = ["rain.md"]
```
