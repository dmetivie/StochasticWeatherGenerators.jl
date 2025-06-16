using Markdown#hide

md"""
# [Multivariate SWG: Application to crop model](@id TutoCrop)
"""

import Pkg;
cd(@__DIR__)#hide
Pkg.activate("../docs/");#src

md"""
This tutorial has two objectives
1. Train a multivariate multisite (simplistic) model, reusing the hidden states trained in the other tutorial.
2. How to use this SWG with a crop model to generate annual crop yield for maize.

In the first part, the tutorial shows how to easily train weather stations given the hidden states sequence `z` obtained in the [previous tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/).
We will show how to make a (simplistic) multisite SWG with multiple correlated weather variables such as daily Rain `RR` ($\mathrm{m}\mathrm{m}$), daily Temperature minimum `TN` (°C), maximum `TX` (°C), total daily solar irradiance `QQ` (MJ/$\mathrm{m}^2$) and daily evapotranspiration Penman `ETPP` ($\mathrm{m}\mathrm{m}/\mathrm{m}^2$). 
This model will be trained with respect to the given hidden states, and the parameters will be periodic and vary smoothly during a calendar year.

The hidden states and the seasonality are enough to correlate well the weather variables without extra codependency between simulated variables.

This multisite, multivariable model has been used as input of the [STIC crop model](https://www.sciencedirect.com/science/article/pii/S1161030102001107) to generate data of annual crop yield for maize in the [GenHack 3 Hackathon](https://www.polytechnique.edu/en/genhack-3-hackathon-generative-modelling). See the associated published dataset [https://doi.org/10.57745/C3FNBY](https://doi.org/10.57745/C3FNBY). In the second part, we show what steps to follow to generate a similar dataset.
"""

md"""
The GIF below illustrates a 50 years generated multivariate weather times series at one site with the associated yearly maize[^Error]. The code to produce this GIF is [at the end of this tutorial](@ref YieldvsWeather). 
The vertical dashed lines show a key period of the year for maize growth with respect to precipitation. This key period is identified in the [Sensitivity Analysis](@ref YieldSensitivity) section of this tutorial.

[^Error]: Note that, for some simulations with STICS an error was thrown (which I cannot explain at the moment and **might** be due to a STICS issue).
"""
[Illustration of Yield vs Generated Weather series]("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/refs/heads/master/assets/tuto_2/yeild_vs_TS_station_1_simu_153.gif")

md"""
## Set up
"""

md"""
### Package and functions
"""

using CSV, DelimitedFiles# File Read/Load/Save/dwl
import JLD# File Read/Load/Save/dwl
import Downloads
using DataFrames, DataFramesMeta # DataFrames
using Dates
using Random, Distributions
using LaTeXStrings
using StatsBase

md"""
The two main packages for this tutorial are not yet registered in the official Julia registry, since they are not quite fully ready. 
They can be either `add`ed through [my local Julia registry](https://github.com/dmetivie/LocalRegistry) with the [LocalRegistry.jl](https://github.com/GunnarFarneback/LocalRegistry.jl) package i.e. 
```julia
using Pkg
pkg"registry add https://github.com/dmetivie/LocalRegistry"
Pkg.add("SmoothPeriodicStatsModels")
Pkg.add("StochasticWeatherGenerators")
```
"""

using SmoothPeriodicStatsModels
using StochasticWeatherGenerators

md"""
## Data extraction and settings

To get the interesting weather variables, we use weather station provided by a the French research institute for agronomy and environment (INRAE).
This data is available through the INRAE CLIMATIK platform[^climatik] ([https://agroclim.inrae.fr/climatik/](https://agroclim.inrae.fr/climatik/), in French) managed by the AgroClim laboratory of Avignon, France.
Unfortunately, these data are not yet open access (they should be soon).
Météo France do have a version of this data and it is accessible through an API on the website [Data.Gouv.fr](https://www.data.gouv.fr/en/datasets/).
This package provide a simple command to extract the data of one station (given its STAtionID) from the API.
```julia
# Download the four stations used in this tutorial from MeteoFrance collection
dfs = collect_data_MeteoFrance.([49215002, 80557001, 40272002, 63345002])
```
See the [Data section of the documentation](@ref DataMeteofrance) section.
"""


md"""
However, the data there does not exactly match the available on CLIMATIK, (less data, different values ...).
For now I stored the CLIMATIK data on a private repo until the Météo France data is fixed.

!!! warning
    While testing this function, it appears that MeteoFrance API might bug sometimes returning an error for some stations (and working for others).
    In that case, you can check the API directly [here](https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-quotidiennes/#).

[^climatik]: Delannoy, David; Maury, Olivier; Décome, Jérémie, 2022, “CLIMATIK : système d’information pour les données du réseau agroclimatique INRAE”, [https://doi.org/10.57745/AJNXEN](https://doi.org/10.57745/AJNXEN), Recherche Data Gouv, V1
"""

md"""
## Stochastic Weather Generator + Crop model STICS

In this section, we demonstrate how stochastic weather simulations can serve as inputs for a crop model to study the climate sensitivity of a crop model.
This is a proof of concept showing how SWG can be useful when combined with other models.

Specifically, we use the [STICS crop model](https://stics.inrae.fr/eng)[^STICS]. To download STICS, go to this [page](https://stics.inrae.fr/telechargement) (in French...) and follow these instructions:

1. Register on the dedicated [website](https://w3.avignon.inrae.fr/forge/account/register), or [log in](https://w3.avignon.inrae.fr/forge/login) if you already have an account.

2. You will receive an e-mail confirming the creation of your account.

3. Download files [here](https://w3.avignon.inrae.fr/forge/projects/stics_main_projecv/files) after logging in

The download files will have 
- STICS executable, here we will use the lower level one `stics_modulo` and not the Java version which is slower. In this tutorial, we used the version `STICS-10.0.0`.
- Parameter files of different crop.

[^STICS]: Brisson et al. (2003). An overview of the crop model STICS. European Journal of agronomy, 18(3-4), 309-332.
"""

md"""
### Description

For this tutorial, we use the default STICS parameters for maize with the following modifications: no irrigation (to highlight the effect of hydric stress), `pgrainmaxi = 0.35`, and `nbgrmin = 0`, `nbgrmax = 4500` (minimum and maximum number of fruits per m$^2$).

Typically, the final yield ranges between 0 and 15 t/ha and is highly dependent on rainfall and temperature.
"""

md"""
### Running STICS

In the file `file_stics`, we implemented functions to streamline calls to the STICS executable (either `.exe` or `.sh`), which can be obtained from the STICS downloads. For each simulation, the script updates the STICS weather files based on the input data frames (`dfs_simus`). It extracts the final yield (along with some other quantities not used here). If STICS encounters an error, the YIELD value is set to `missing`.

Since repeated calls to STICS are time-intensive, results are saved for reuse. Below, we show the executed code:
"""
md"""
### [Sensitivity Analysis of Maize to Rainfall](@id YieldSensitivity)

To determine which rainfall period between April and October has the most significant influence on maize yield, the growing season is segmented into nine distinct periods.
"""


md"""
## [Yield vs the Weather Time series](@id YieldvsWeather)

This section shows how the yield is related to the weather time series at a given station.
"""


md"""
Package list and version
"""

import Pkg;
Pkg.status();