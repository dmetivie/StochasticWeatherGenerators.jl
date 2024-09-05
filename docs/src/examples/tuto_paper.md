```@meta
EditURL = "../../../examples/tuto_paper.jl"
```

````@example tuto_paper
using Markdown#hide
cd(@__DIR__)#hide
````

# Multisite daily Stochastic Weather Generator

This tutorial describes the numerical applications described in the paper [*Interpretable Seasonal Hidden Markov Model for spatio-temporal stochastic rain generation in France*](https://hal.inrae.fr/hal-04621349) by [Emmanuel Gobet](http://www.cmap.polytechnique.fr/~gobet/) (CMAP - École Polytechnique), [David Métivier](https://davidmetivier.mistea.inrae.fr/) (MISTEA -- INRAE) and [Sylvie Parey](https://fr.linkedin.com/in/sylvie-parey-60285194) (R&D -- EDF).
It shows a fully reproducible example on how to use the package `StochasticWeatherGenerators.jl` to reproduce, step-by-step, exactly (almost) all the figures of the paper.

The paper describes the construction of a Stochastic Weather Generator with an Autoregressive Seasonal Hidden Markov Model (SHMM). The SHMM is trained with French weather stations, and the hidden states are interpreted as weather regimes. The model is validated with simulations, especially for its ability to reproduce extreme weather, e.g. droughts.
In the paper, the model is also used with Climate Change RCP scenarios (not shown here).

```@raw html
<img src="https://github.com/dmetivie/StochasticWeatherGenerators.jl/assets/46794064/5fe1d677-877d-4fd5-83ac-29d30f728ca5" width="90%" alt = "Schematic of the Autoregressive Seasonal Hidden Markov Model"/>
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

