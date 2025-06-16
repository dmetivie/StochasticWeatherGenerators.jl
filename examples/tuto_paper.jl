using Markdown#hide
import Pkg;
cd(@__DIR__)#hide
Pkg.activate("../docs/");#src

md"""
# [Multisite rainfall HMM based SWG (paper)](@id TutoSHHMM)
"""

md"""
This tutorial describes the numerical applications described in the paper [*Interpretable Seasonal Hidden Markov Model for spatio-temporal stochastic rain generation in France*](https://hal.inrae.fr/hal-04621349) by [Emmanuel Gobet](http://www.cmap.polytechnique.fr/~gobet/) (CMAP - École Polytechnique), [David Métivier](https://davidmetivier.mistea.inrae.fr/) (MISTEA -- INRAE) and [Sylvie Parey](https://fr.linkedin.com/in/sylvie-parey-60285194) (R&D -- EDF).
It shows a fully reproducible example on how to use the package `StochasticWeatherGenerators.jl` to reproduce, step-by-step, exactly (almost) all the figures of the paper.

The paper describes the construction of a Stochastic Weather Generator with an Autoregressive Seasonal Hidden Markov Model (SHMM). The SHMM is trained with French weather stations, and the hidden states are interpreted as weather regimes. The model is validated with simulations, especially for its ability to reproduce extreme weather, e.g. droughts. 
In the paper, the model is also used with Climate Change RCP scenarios (not shown here).
"""

md"""
```@raw html
<img src="https://github.com/dmetivie/StochasticWeatherGenerators.jl/assets/46794064/5fe1d677-877d-4fd5-83ac-29d30f728ca5" width="95%" alt = "Schematic of the Autoregressive Seasonal Hidden Markov Model"/>
```
"""

md"""
## Set up
"""

md"""
### Package and functions
"""

#!nb # !!! note "For Julia new user"
#!nb #     There are several ways to `add` a package before `using`, one way is for this tutorial to copy-paste (it might take a while):
#!nb #     ```julia
#!nb #     import Pkg
#!nb #     Pkg.add(["CSV", "JLD", "DelimitedFiles", "DataFrames", "DataFramesMeta", "StatsBase", "Random", "Distributions", "StatsPlots", "LaTeXStrings"])
#!nb #     ```

md"""
## [WGEN model](@id TutoWGEN)

We will compare to the WGEN model that propose Markov chain of order 4 for rain occurences (fitted monthly) and laten gaussian model for multisite occurences (fitted monthly).
- Wilks, D. S. "Multisite generalization of a daily stochastic precipitation generation model". Journal of Hydrology, (1998). https://doi.org/10.1016/S0022-1694(98)00186-3.
- Srikanthan, Ratnasingham, et Geoffrey G. S. Pegram. "A nested multisite daily rainfall stochastic generation model". Journal of Hydrology 2009. https://doi.org/10.1016/j.jhydrol.2009.03.025.
"""

wgen_order = 4
md"""
Package list and version
"""

import Pkg; Pkg.status()