using Markdown#hide
import Pkg;
cd(@__DIR__)#hide
Pkg.activate("../docs/");#src

md"""
# [Multisite rainfall HMM based SWG (paper)](@id TutoSHHMM)
"""

md"""
This tutorial describes the numerical applications described in the paper [*Interpretable Seasonal Multisite Hidden Markov Model for stochastic rain generation in France*](https://hal.inrae.fr/hal-04621349) by [Emmanuel Gobet](http://www.cmap.polytechnique.fr/~gobet/) (CMAP - École Polytechnique), [David Métivier](https://davidmetivier.mistea.inrae.fr/) (MISTEA -- INRAE) and [Sylvie Parey](https://fr.linkedin.com/in/sylvie-parey-60285194) (R&D -- EDF).
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

using CSV, JLD, DelimitedFiles # File Read/Load/Save

using DataFrames, DataFramesMeta # DataFrames

using Dates

using StatsBase, Random

using Distributions

md"""
The main package is `StochasticWeatherGenerators.jl`, which provides the interface to the models and data.
`SmoothPeriodicStatsModels.jl` contains the smooth periodic models (fit and sampling methods) used in this tutorial, namely the Seasonal Hidden Markov Model (SHMM), and the seasonal mixture models for rainfall amounts.
"""

using StochasticWeatherGenerators # interface to use with SmoothPeriodicStatsModels.jl

using SmoothPeriodicStatsModels # Name might change. Small collection of smooth periodic models e.g. AR, HMM

#-
Random.seed!(1234)

md"""
### Settings for plotting

Some settings and packages to have nice plots.
"""

using StatsPlots, LaTeXStrings
using StatsPlots.PlotMeasures # To play with margin in Plots

gr() # plotly() # for interactive plots
default(fontfamily="Computer Modern")
cur_colors = get_color_palette(:auto, 100);
my_palette(K) = palette(vcat(cur_colors[1], [cur_colors[c] for c in 3:4], cur_colors[2]), K)

file_for_plot_utilities = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/utilities_plot.jl")
include(file_for_plot_utilities)

md"""
To plot maps, we use `GeoMakie.jl` + `NaturalEarth.jl`. 
Note that using `cartopy` with `PyCall.jl` also works very well.

For the following code to work you will need to add the following packages
```julia
import Pkg
Pkg.add("NaturalEarth", "GeoMakie", "CairoMakie")
```
"""

file_for_maps_with_geomakie = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/utilities_geo_makie_features.jl") # download file from a GitHub repo
include(file_for_maps_with_geomakie)

md"""
### Global Parameters
"""

md"""
The period of the Seasonal HMM is chosen as the number of days in a year. Our choice is 366 however we carefully skip February 29 when needed.
"""

T = 366

md"""
Define the French area for map (Longitude and latitude) plot and the precision of the map `precision_scale`
"""

precision_scale = 50 # meter

LON_min = -5 # West

LON_max = 10 # East

LAT_min = 41 # South

LAT_max = 52 # North

md"""
`conversion_factor` for rainfall amounts `RR` in 0.1 mm to mm 
"""

conversion_factor = 0.1 # 0.1 mm -> mm 

md"""
## HMM Hyperparameters
"""

md"""
Number of hidden states
"""

K = 4

my_pal = my_palette(K); # just colors I like for plotting weather regime!

md"""
Degree `𝐃𝐞𝐠` of the trigonometric expansion 
It could be an array different for each station and variable. Not implemented yet though.
"""

𝐃𝐞𝐠 = 1

md"""
Local memory order i.e. at station $j$, $\mathbb{P}(Y_n^{(j)} = y_n^{(j)} \mid Z = k, Y_{n-1:n-\texttt{local memory}}^{(j)} = y_{n-1:n-\texttt{local memory}}^{(j)})$
"""

local_order = 1

#!nb # !!! note
#!nb #     The `local_order` and/or `𝐃𝐞𝐠` could be a vector/matrix of size `D` and different for each station, and also different depending on wet or dry past. Not yet implemented.

size_order = 2^local_order

println("K = $K, ", "degree = $𝐃𝐞𝐠, ", "local_order = $local_order")

md"""
## Data
"""
save_tuto_path = "../../assets/tuto_1/tuto_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)" #src
isdir(save_tuto_path) ? nothing : mkdir(save_tuto_path) #src

md"""
### Select relevant stations from the `station.txt` file
"""

md"""
Here we
- Remove white space at the right of the `CN`, STANAME which is caused by imperfect CVS importation
- Select only the stations with 100% valid data for the period `Date(1955,12,31) .≤ :DATE .≤ Date(2019,12,31)`
- Shorten station names 	
"""
begin
    station_file = Base.download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/weather_files/stations.txt")
    station_all = CSV.read(station_file, DataFrame, header=18, normalizenames=true, ignoreemptyrows=true)
    station_all = @chain station_all begin
        @transform(:CN = rstrip.(:CN), :STANAME = rstrip.(:STANAME))
        ## @subset(:CN .∈ tuple(["FR", "BE", "LU", "CH"])) # Choose that if you want to look at all stations in France, Belgium, Luxembourg and Switzerland.
        @subset(:STAID .∈ tuple([32, 33, 34, 36, 39, 203, 322, 323, 434, 736, 737, 738, 740, 742, 745, 749, 750, 755, 756, 757, 758, 786, 793, 2192, 2203, 2205, 2207, 2209, 11244, 11245, 11247, 11249]))
        @transform(:STANAME = shortname.(:STANAME))
    end
end

selected_station_name = ["BOURGES", "TOULOUSE", "MARIGNANE", "LUXEMBOURG", "LILLE", "EMBRUN", "BASTIA", "LA HAGUE", "CHASSIRON", "ORLY"]

#!nb # !!! note "Hypothesis: Conditional Independence of Rain Occurrences"
#!nb #     You can change the selected stations. However, keep in mind that for the model to work, the **conditional independence hypothesis** must hold between stations i.e. $\mathbb{P}(Y_1 = y_1, \cdots, Y_S = y_s\mid Z = k) = \prod_{s=1}^S \mathbb{P}(Y_s = y_s)$.
#!nb #     Hence stations must be sufficiently far apart. Check out this [MNIST example](https://dmetivie.github.io/ExpectationMaximization.jl/dev/examples/#MNIST-dataset:-Bernoulli-Mixture) to see Bernoulli mixtures in action!

station = @subset(station_all, :STANAME .∈ tuple(selected_station_name))

STAID = station.STAID #[32, 33, 39, 203, 737, 755, 758, 793, 11244, 11249]; 

station_name = station.STANAME

md"""
Sort stations (index) by latitude. It is useful for plotting from North to South.
"""

staid_lat = sortperm(station.LAT, rev=true);

md"""
Station number
"""

D = length(STAID)

md"""
### Date range
"""

date_start = Date(1956)

md"""
Date including the previous days used in the initial condition (in case `local_memory > 0`)
"""

date_start_w_memory = date_start - Day(local_order)

date_end = Date(2020) - Day(1)

every_year = date_start:Day(1):date_end

every_year_w_memory = date_start_w_memory:Day(1):date_end

n2t = dayofyear_Leap.(every_year)

N = length(n2t)

md"""
### Treat data
"""

md"""
Load into a `DataFrame` the (ECA) RR files (rain). It filters by date and valid data.
It also adds a column `:RO` for rain occurrences (0: dry, 1: wet).
"""

begin
    data_stations = collect_data_ECA.(STAID, date_start_w_memory, date_end, "https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/weather_files/ECA_blend_rr/RR_", portion_valid_data=1, skipto=22, header=21, url=true)
    for i = eachindex(data_stations)
        @transform!(data_stations[i], :RO = onefy.(:RR))
    end
end


md"""
Binary matrix version of the rain event at the `D` stations.
"""

RR = reduce(hcat, [data_stations[j].RR[1+local_order:end] for j = 1:D]) * 0.1

Yall = BitMatrix(reduce(hcat, [data_stations[j].RO for j = 1:D]))

Y_past = BitMatrix(Yall[1:local_order, :]) # rand(Bool, local_order, D)

ξ = [1; zeros(K - 1)];  # 1 jan 1956 was most likely a type Z = 1 wet day all over France

Y = Yall[1+local_order:end, :]

md"""
### Map of stations
"""

md"""
Convert LAT/LON coordinates from DMS to DD which seems most widely accepted format.
"""

LAT_idx = dms_to_dd.(station.LAT)

LON_idx = dms_to_dd.(station.LON)

long_spell = [longuest_spell(y) for y in eachcol(Y)]

FR_map_spell = map_with_stations(LON_idx, LAT_idx, long_spell; station_name=station_name, show_value=true, colorbar_show=true, precision_scale = precision_scale, colorbar_label = "Days")

#-
savefigcrop(FR_map_spell, "FR_longest_dry_spell_$(D)_station_histo", save_tuto_path); #src

md"""
## Fit the seasonal HMM
"""

#!nb # !!! note "Hypothesis: Smooth parameter evolution"
#!nb #     We assume all models e.g. HMM, rain mixture to have parameters evolving smoothly with periodicity $T$ for $t \in [1, T]$.
#!nb #     For example a Bernoulli parameter will write
#!nb #     ```math 
#!nb #     p(t) = \dfrac{1}{1 + e^{P(t)}} \in [0, 1],
#!nb #     ``` 
#!nb #     with 
#!nb #     ```math
#!nb #         P_c(t) = c_0 + \sum_{j=1}^{\texttt{Deg}} \left(c_{2j-1}\cos\left(\dfrac{2\pi}{T}j t\right) + c_{2j}\sin\left(\dfrac{2\pi}{T}j t\right)\right).
#!nb #     ```

md"""
### Fit slice: naive estimation
"""

#!nb # !!! note
#!nb #     Before inferring the HMM parameters with the EM (Baum-Welch) algorithm, we do a first naive inference that will be used as initial condition for the EM.

md"""
The reference station `ref_station` is used to sort the hidden states obtained via the slide initialization
Here we choose `j=1` $\to$ `STAID=32` $\to$ `BOURGES` because it is a central station for France
"""

ref_station = 1

md"""
This generates a random Periodic HMM that we then fit slice by slice (day by day). See paper.
"""
hmm_random = randARPeriodicHMM(K, T, D, local_order; ξ=ξ, ref_station=ref_station);

@time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13]);

θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, 𝐃𝐞𝐠);

md"""
### Fit with Baum Welch using the slice estimate as a starting point

With the Slice estimate as a good starting point for the full (seasonal) Baum Welch EM algorithm we fit the model!
"""

#!nb # !!! tip
#!nb #     To accelerate the fitting procedure (especially for larger models or when testing various model hyperparameters), one can do
#!nb #     ```julia
#!nb #     using Distributed
#!nb #     addprocs(10) # number of worker to add
#!nb #     @everywhere using SmoothPeriodicStatsModels # load the pkg on each worker
#!nb #     ```
#!nb #     Then the fitting loop inside `fit_mle` will be distributed. See the [official Julia doc](https://docs.julialang.org/en/v1/stdlib/Distributed/#man-distributed) for more info.
#!nb #     On smaller models it does not worth it since adding workers add some compilation and communication time.
@time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past, maxiter=10000, robust=true; display=:iter, silence=true, tol=1e-3, θ_iters=true, n2t=n2t);

#-

save(joinpath(save_tuto_path, "hmm_fit_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).jld"), "hmm", hmm_fit, "hist", hist, "Q_param", θq_fit, "Y_param", θy_fit); #src

md"""
Run the following code to load a saved hmm
```julia
hmm_infos = load(joinpath(save_tuto_path,"hmm_fit_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).jld"))
hmm_fit = hmm_infos["hmm"]
hist = hmm_infos["hist"]
θq_fit = hmm_infos["Q_param"]
θy_fit = hmm_infos["Y_param"]
```
"""

md"""
### Visualization of the HMM parameters
"""

md"""
#### Transition matrix
"""


begin
    pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing, legend_columns=4, tickfont=12, legendfontsize=16) for k in 1:K]
    for k in 1:K
        [plot!(pA[k], hmm_fit.A[k, l, :], c=my_color(l, K), label=L"Q_{%$(k)\to %$(l)}", legend=:top, lw=1.75) for l in 1:K]
        hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
        xticks!(pA[k], vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(string.(monthabbr.(1:12)), ""), xlims=(0, 367), ylims=(0, 1))
    end
    pallA = plot(pA..., size=(1000, 500))
end

#-
savefig(pallA, joinpath(save_tuto_path, "Q_transition_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf")); #src

md"""
#### Rain probabilities
"""

begin
    mm = 1
    jt = D
    pB = [plot(legendfont=14, title="$(station_name[j])", titlefontsize=17, tickfont=14, legendfontsize = 16) for j in 1:jt]
    for j in 1:jt
        [plot!(pB[j], succprob.(hmm_fit.B[k, :, j, mm]), c=my_color(k, K), label=islabel(j, 3, L"\mathbb{P}(Y = \textrm{wet}\mid Z = %$k, H = \textrm{dry})"), lw=2) for k in 1:K]
        hline!(pB[j], [0.5], c=:black, label=:none, s=:dot)
        xticks!(
            pB[j],
            vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366),
            vcat(string.(first.(monthabbr.(1:12))))
        )
        xlims!(pB[j], (0, 367))
        ylims!(pB[j], (0, 1))
    end
    pallB = plot(pB[staid_lat]..., size=(3000 / 2, 1000 / 1), layout=(2, 5))
end

#-
savefig(pallB, joinpath(save_tuto_path, "proba_rain_all_station_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf")); #src

md"""
#### Spatial Rain probability 
"""

memory_past_cat = 1

md"""
h = 1 (day before dry) or 2 (day before wet)
$\mathbb{P}(Y = \text{Rain}\mid Z = k, H = h)$ with `h = memory_past_cat`

For now there are some scale rendering issues due to an [GeoMakie.jl issue](https://github.com/MakieOrg/GeoMakie.jl/issues/268) so it might be tiny.
"""

p_FR_map_mean_prob = map_with_stations(LON_idx, LAT_idx, [[mean(succprob.(hmm_fit.B[k, :, j, memory_past_cat])) for j in 1:length(STAID)] for k in 1:K], colorbar_show=true, colorbar_label = L"\mathbb{P}(Y = \text{Rain}\mid Z = k, H = 1)", precision_scale = precision_scale)

#-
savefigcrop(p_FR_map_mean_prob, "FR_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)_mean_proba_cat_1", save_tuto_path); #src

md"""
### Inference of the historical hidden states
"""

md"""
####  Viterbi algorithm
"""

ẑ = viterbi(hmm_fit, Y, Y_past; n2t=n2t)

data_stations_z = map(data_stations) do df
    @transform(df, :z = [fill(missing, local_order); ẑ])
end

ẑ_per_cat = [findall(ẑ .== k) for k in 1:K]

CSV.write(joinpath(save_tuto_path, "z_hat_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).csv"), DataFrame([:DATE, :z] .=> [data_stations[1].DATE[1+local_order:end], ẑ])); #src

md"""
#### Visualization of the historical sequences of hidden states
"""

year_range = unique(year.(data_stations[1][1+local_order:end, :DATE]));

idx_year = [findall(x -> year.(x) == m, data_stations[1][1+local_order:end, :DATE]) for m in year_range];

select_year = unique(sort([4:10:length(year_range); 21; 48; 64]))

begin
    year_nb = length(select_year)
    z_hat_mat = zeros(year_nb, 366)

    for (i, y) in enumerate(select_year)
        if isleapyear(year_range[y])
            z_hat_mat[i, :] = ẑ[idx_year[y]]
        else
            z_hat_mat[i, :] = [ẑ[idx_year[y]]; 0]
        end
    end
    thick = 1
    heatmap(z_hat_mat, colorbar=:none, c=my_palette(K), minorticks=:false, framestyle=:xbox, grid=:none, thickness_scaling=thick)
    xticks!(vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(string.(monthabbr.(1:12)), ""), xlims=(0, 367), xtickfontsize=14 / thick, ytickfontsize=14 / thick)
    hline!((1:year_nb) .+ 0.5, c=:black, legend=:none, lw=4)
    ylims!(0.5, year_nb + 0.5)
    pviterbi = yticks!(1:year_nb, string.(year_range[select_year]), size=(1000, 600))
end

#-
savefig(pviterbi, joinpath(save_tuto_path, "temporal_1959_2009_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf")); #src

md"""
## Adding Rainfall amounts to the model
"""

md"""
### Marginal distribution

We fit the marginals of the rainfall amount $R>0$ at each station $s$ and for each hidden state $Z$ independently. 
We use a mixture of exponential functions 
```math
g(r) = w \dfrac{e^{-{\frac {r}{\vartheta_1}}}}{\vartheta_1} + (1-w)  \dfrac{e^{-{\frac {r}{\vartheta_2}}}}{\vartheta_2}.
```
whose parameters $w(t)$, $\vartheta_1(t)$ and $\vartheta_2(t)$ are smooth periodic functions of the day of the year.
"""

@time "FitMLE RR" mix_allE = fit_mle_RR.(data_stations_z, local_order, mix₀=mix_ini(T));

save(joinpath(save_tuto_path, "rain_mix.jld"), "mix2Exp", mix_allE); #src

md"""
Thanks to [Distributions.jl PR #1389 (September 2nd, 2021)](https://github.com/JuliaStats/Distributions.jl/pull/1389) and Julia multiple dispatch, the quantile function of Mixtures can be very efficiently computed.
"""

md"""
### Rain correlations

We fit a Gaussian copula to each pair of stations for joint rainy days only.
"""

#!nb # !!! warning
#!nb #     For some hidden states corresponding to dry weather, it might happen that for some pair of stations, there are not enough simultaneous rain occurrences in a rain category $Z = k$ to estimate a correlation coefficient.
#!nb #     In that case a `missing` coefficient is returned by `cov_RR` or it returns the value `impute_missing` if specified (to avoid missing).
#!nb #     The option `force_PosDef` (enabled by default) ensures having positive definite matrix. This is necessary to use gaussian copula.

Σ²RR = cov_RR(data_stations_z, K)

md"""
## Simulation

Now we are ready to generate samples from the SWG model.
"""

md"""
`Nb` is the number of realization. In [2024_gobet_InterpretableSeasonalHiddena](@cite) `Nb = 5_000` was used.
"""
Nb = 1_000

md"""
Sample the (seasonal) HMM model and output the sequence of hidden states and multi-site dry/wet.
"""
begin
    zs = zeros(Int, N, Nb)
    ys = zeros(Bool, N, D, Nb)
    @time "Simulations Z, Y" for i in 1:Nb
        zs[:, i], ys[:, :, i] = rand(hmm_fit, n2t; y_ini=Yall[1:local_order, :], z_ini=1, seq=true)
    end
end

md"""
Given the hidden states and dry/wet, it generates the rainfall amounts at each station (correlated with a Gaussian Copula).
"""

begin
    rs = zeros(D, N, Nb)
    @time "Simulations RR>0" for i in 1:Nb
        rs[:, :, i] = rand_RR(mix_allE, n2t, zs[:, i], ys[:, :, i]', Σ²RR)
    end
end

md"""
## [WGEN model](@id TutoWGEN)

We will compare to the WGEN model that propose Markov chain of order 4 for rain occurrences (fitted monthly) and laten gaussian model for multisite occurrences (fitted monthly).
- Wilks, D. S. "Multisite generalization of a daily stochastic precipitation generation model". Journal of Hydrology, (1998). https://doi.org/10.1016/S0022-1694(98)00186-3.
- Srikanthan, Ratnasingham, et Geoffrey G. S. Pegram. "A nested multisite daily rainfall stochastic generation model". Journal of Hydrology 2009. https://doi.org/10.1016/j.jhydrol.2009.03.025.
"""

wgen_order = 4
idx_months = [findall(x -> month.(x) == m, data_stations[1][1+local_order:end, :DATE]) for m in 1:12]
wgen4_model = fit_wgen(Y, idx_months, wgen_order)

ys_wgen = similar(ys)
@time "Simulation Y wgen 4" for i in 1:Nb
    ys_wgen[:, :, i] = rand(wgen4_model, 1956:2019; Y_ini=vcat(rand(Bool, wgen_order - 1, D), Y_past))
end

md"""
## Results
"""

md"""
### Spell distribution

`select_month` to choose the month where to compute the spell distributions (summer month, winter, etc.)
`select_month = 1:12` corresponds to all months.
"""
select_month = 1:12

idx_months = [findall(x -> month.(x) == m, data_stations[1][1+local_order:end, :DATE]) for m in 1:12]

idx_month_vcat = vcat(idx_months[select_month]...)

idx_all = [intersect(yea, mon) for yea in idx_year, mon in idx_months];

md"""
##### Historic spells
"""

len_spell_hist = [pmf_spell(Y[idx_month_vcat, j], dw) for j in 1:D, dw in 0:1];

md"""
##### Simulation spells
"""

len_spell_simu = [pmf_spell(ys[idx_month_vcat, j, i], dw) for i in 1:Nb, j in 1:D, dw in 0:1];

md"""
#### Dry spell
"""

make_range(y, step=1) = range(extrema(y)..., step=step)

begin
    dry_or_wet = 1 # dry
    p_spell_dry = [plot(ylims=(1e-4, 1e-0), tickfont=11, legendfontsize=13) for j = 1:D]
    for j = 1:D
        all_spells = len_spell_simu[:, j, dry_or_wet]
        spell_range = 1:1:(1+maximum(vcat(reduce(vcat, all_spells), len_spell_hist[j, dry_or_wet])))

        errorlinehist!(p_spell_dry[j], all_spells, groupcolor=:grey, legend=:topright, label=islabel(j, staid_lat[[1]], L"Simu $q_{0,100}$"), norm=:probability, bins=spell_range, errortype=:percentile, percentiles=[0, 100], fillalpha=0.4, centertype=:median)

        errorlinehist!(p_spell_dry[j], all_spells, groupcolor=:red, label=islabel(j, staid_lat[[1]], L"Simu $q_{25,75}$"), norm=:probability, bins=make_range(reduce(vcat, all_spells)), errortype=:percentile, percentiles=[25, 75], fillalpha=0.5, centertype=:median)

        histo_spell = len_spell_hist[j, dry_or_wet]
        errorlinehist!(p_spell_dry[j], [histo_spell], label=islabel(j, staid_lat[[1]], "Obs"), groupcolor=:blue, lw=1.5, norm=:probability, bins=spell_range, errortype=:percentile, alpha = 0.8)
        xlims!(p_spell_dry[j], 0, 2 + maximum(1.5maximum.(histo_spell)))
        yaxis!(:log10)
    end

    [xlabel!(p_spell_dry[j], "Nb of days", xlabelfontsize=12) for j in staid_lat[6:10]]
    [ylabel!(p_spell_dry[j], "PMF", ylabelfontsize=12) for j in staid_lat[[1, 6]]]
    [title!(p_spell_dry[j], station_name[j], titlefontsize=13) for j = 1:D]
    pall_spell_dry = plot(p_spell_dry[staid_lat]..., size=(3000 / 2.5, 1000 / 1.5), layout=(2, 5), left_margin=0.5cm, bottom_margin=0.275cm)
end

#-
savefig(pall_spell_dry, joinpath(save_tuto_path, "spell_steppost_dry_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf")); #src

md"""
#### Wet spell
"""

begin
    dry_or_wet = 2 # wet
    p_spell_wet = [plot(ylims=(1e-4, 1e-0), tickfont=11, legendfontsize=13) for j = 1:D]
    for j = 1:D
        all_spells = len_spell_simu[:, j, dry_or_wet]
        spell_range = 1:1:(1+maximum(vcat(reduce(vcat, all_spells), len_spell_hist[j, dry_or_wet])))

        errorlinehist!(p_spell_wet[j], all_spells, groupcolor=:grey, legend=:topright, label=islabel(j, staid_lat[[1]], L"Simu $q_{0,100}$"), norm=:probability, bins=spell_range, errortype=:percentile, percentiles=[0, 100], fillalpha=0.4, centertype=:median)

        errorlinehist!(p_spell_wet[j], all_spells, groupcolor=:red, label=islabel(j, staid_lat[[1]], L"Simu $q_{25,75}$"), norm=:probability, bins=spell_range, errortype=:percentile, percentiles=[25, 75], fillalpha=0.5, centertype=:median)

        histo_spell = len_spell_hist[j, dry_or_wet]
        errorlinehist!(p_spell_wet[j], [histo_spell], label=islabel(j, staid_lat[[1]], "Obs"), groupcolor=:blue, lw=1.5, norm=:probability, bins=spell_range, errortype=:percentile, alpha = 0.8)
        xlims!(p_spell_wet[j], 0, 2 + maximum(1.5maximum.(histo_spell)))
        yaxis!(:log10)
    end

    [xlabel!(p_spell_wet[j], "Nb of days", xlabelfontsize=12) for j in staid_lat[6:10]]
    [ylabel!(p_spell_wet[j], "PMF", ylabelfontsize=12) for j in staid_lat[[1, 6]]]
    [title!(p_spell_wet[j], station_name[j], titlefontsize=13) for j = 1:D]
    pall_spell_wet = plot(p_spell_wet[staid_lat]..., size=(3000 / 2.5, 1000 / 1.5), layout=(2, 5), left_margin=0.5cm, bottom_margin=0.275cm)
end

#-
savefig(pall_spell_wet, joinpath(save_tuto_path, "spell_steppost_wet_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf")); #src

md"""
### Seasonal areal dry spells
"""

RRmax = 0
ROR = [mean(r .> RRmax) for r in eachrow(RR)]
RORs = [[mean(r .> RRmax) for r in eachrow(rr)] for rr in eachslice(ys, dims=3)]
RORswgen = [[mean(r .> RRmax) for r in eachrow(rr)] for rr in eachslice(ys_wgen, dims=3)]

JJA = [6, 7, 8]
MAM = [3, 4, 5]
SON = [9, 10, 11]
DJF = [12, 1, 2]
SEASONS = [DJF, MAM, JJA, SON]
seasonname = ["DJF", "MAM", "JJA", "SON"]

idx_seasons = [findall(month.(data_stations[1][1+local_order:end, :DATE]) .∈ tuple(season)) for season in SEASONS]

let
    perc = 0.2
    QQ = [5, 95]

    p_spell_rors = [plot(ylims=(5e-4, 1e-0), xlims=(-0.01,25), tickfont=11, legendfontsize=13, legend=:left) for i in eachindex(idx_seasons)]
    xlabel!.(p_spell_rors[3:end], "Nb of days", xlabelfontsize=12)
    ylabel!.(p_spell_rors[[1, 3]], "PMF", ylabelfontsize=12)
    for m in eachindex(idx_seasons)
        len_ror_hist = pmf_spell(ROR[idx_seasons[m]] .≤ perc, 1)
        len_ror_simu = [pmf_spell(RORs[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]
        len_ror_simuwgen = [pmf_spell(RORswgen[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]

        errorlinehist!(p_spell_rors[m], [len_ror_hist], groupcolor=:blue, lw=2, norm=:probability, bins=make_range(len_ror_hist), errortype=:percentile,
        label=label = islabel(m, 1, "Obs"),
        legend=:bottom)
        yaxis!(:log10)

        sim_range = make_range(reduce(vcat, len_ror_simuwgen))
        errorlinehist!(p_spell_rors[m], len_ror_simuwgen, groupcolor=:green, legend=:topright,
            label=islabel(m, 1, "WGEN 4"),
            norm=:probability, bins=sim_range, errortype=:percentile, percentiles=QQ, fillalpha=0.25, centertype=:median, linewidth=2)

        sim_range = make_range(reduce(vcat, len_ror_simu))
        errorlinehist!(p_spell_rors[m], len_ror_simu, groupcolor=:grey, legend=:topright,
            label=islabel(m, 1, "SHHMM"),
            norm=:probability, bins=sim_range, errortype=:percentile, percentiles=QQ, fillalpha=0.3, centertype=:median, alpha=1, linewidth=2)
        annotate!(p_spell_rors[m], median(sim_range), 1.5, seasonname[m])
        yticks!(10.0 .^ (-4:-0))
    end

    pall = plot(p_spell_rors..., layout=(2, 2), size=(1000, 600), top_margin=0.34cm, left_margin=0.3cm, bottom_margin=0.22cm)
    file_name = "ROR_spell_season_perc_$(perc)_Q_$(QQ[1])_$(QQ[2])_no_inset"
    file_name = replace(file_name, "." => "p")
    savefig(pall, joinpath(save_tuto_path, file_name * ".pdf")) #src
    pall
end

md"""
### Rain
"""

md"""
#### Interpretation: Mean Rain per weather regime $R > 0 \mid Z = k$.

We plot the empirical (strictly) positive **mean** rainfall amounts per weather regime. The results are smoothed using a `cyclic_moving_average` with a time window of $\pm 15$ days and the Epanechnikov kernel.
"""

begin
    p_rainpercat = [plot(tickfont=12, ylabelfontsize=14, titlefontsize=14, legendfontsize=13) for j = 1:D]
    for j = 1:D
        df_j = @chain data_stations_z[j] begin
            dropmissing
            @transform(:day = dayofyear_Leap.(:DATE))
            @subset(:RR .> 0)
            @by([:day, :z], :MEAN_RR = mean(:RR))
            groupby(:z)
        end
        ## Uncomment to see how the double exponential mixtures compare to the empirical data.
        ## [plot!(p_rainpercat[j], 1:T, t -> conversion_factor * mean(mix_allE[j][k, t]), label=:none, c=my_color(k, K), lw=1.5, legend = :topleft) for k in 1:K]
        for k in 1:K
            cycle_avg = replace(cyclic_moving_average(df_j[k].MEAN_RR, df_j[k].day, T, 15), 0 => missing)
            @df df_j[k] plot!(p_rainpercat[j], 1:T, conversion_factor * cycle_avg, c=my_color(k, K), alpha=1, label=islabel(j, staid_lat[[4]], L"Z = %$k"), lw=1.5)
        end
        ylims!(p_rainpercat[j], 0, Inf)
    end
    [ylabel!(p_rainpercat[j], "Rain (mm)") for j in staid_lat[[1, 6]]]
    [xticks!(
        p_rainpercat[j],
        vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366),
        vcat(string.(first.(string.(monthabbr.(1:12)))))
    ) for j in 1:D]
    [title!(p_rainpercat[j], station_name[j]) for j = 1:D]
    plt_rain_cat_mix = plot(p_rainpercat[staid_lat]..., size=(3000 / 2.2, 1000 / 1.5), layout=(2, 5), left_margin=25px)
end

#-
savefig(plt_rain_cat_mix, joinpath(save_tuto_path, "mean_positive_rain_per_cat_from_mixture_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf")); #src

md"""
#### Univariate Rain distributions
"""

md"""
Historical vs `Nb` simulations distribution
"""

begin
    p_uniR = [plot(yaxis=:log10, ylims=(0.2e-4, 1e-0), tickfont=11, legendfontsize=13, titlefontsize=13) for j = 1:D]
    for j = 1:D
        dists_RR_positive_j = conversion_factor * [filter(!iszero, rs[j, :, i]) for i in 1:Nb]
        Rmax = ceil(max(dists_RR_positive_j .|> maximum |> maximum, conversion_factor * filter(!iszero, data_stations[j].RR) |> maximum))
        BINS = 0:2:Rmax # fixing the bins is very important to ensure fair comparison. Note that changing the bin step changes the aspect of the distributions.
        errorlinehist!(p_uniR[j], dists_RR_positive_j, groupcolor=:grey, legend=:topright, label=islabel(j, staid_lat[[1]], L"Simu $q_{0,100}$"), norm=:pdf, errortype=:percentile, percentiles=[0, 100], fillalpha=0.4, centertype=:median, bins = BINS)

        errorlinehist!(p_uniR[j], dists_RR_positive_j, groupcolor=:red, label=islabel(j, staid_lat[[1]], L"Simu $q_{25,75}$"), norm=:pdf, errortype=:percentile, percentiles=[25, 75], fillalpha=0.5, centertype=:median, bins = BINS)

        errorlinehist!(p_uniR[j], [conversion_factor * filter(!iszero, data_stations[j].RR)], label=islabel(j, staid_lat[[1]], "Obs"), groupcolor=:blue, lw=1.5, norm=:pdf, errortype=:percentile, bins = BINS, alpha = 0.7)

        xlims!(p_uniR[j], 0.0, Inf)
    end
    [plot!(p_uniR[j], xlabel="Rain (mm)") for j in staid_lat[6:10]]
    [plot!(p_uniR[j], ylabel="PDF") for j in staid_lat[[1, 6]]]

    [title!(p_uniR[j], station_name[j]) for j = 1:D]

    pall_R = plot(p_uniR[staid_lat]..., size=(3000 / 2.5, 1000 / 1.5), layout=(2, 5), bottom_margin=11px, left_margin=15px)
end

#-
savefig(pall_R, joinpath(save_tuto_path, "dist_R_positive_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf")); #src

md"""
Aggregated 5 days `RR` distribution
"""
agg_window = 5
df_res = [
    @chain df[1+local_order:end, :] begin
    @transform(:agg = first.(divrem.(1:N, agg_window)))
    @by(:agg, :AGG = sum(:RR) * 0.1)
    end
        for df in data_stations_z
            ]


agg_i_full = first.(divrem.(1:N, agg_window))
idx_agg = [findall(agg_i_full .== val) for val in unique((agg_i_full))]
agg_rs = [[sum(rs[j, ii, i]) for ii in idx_agg] for j in 1:D, i in 1:Nb] * conversion_factor
agg_RR = [[sum(RR[ii, j]) for ii in idx_agg] for j in 1:D]
begin
    p_uniR = [plot(yaxis=:log10, ylims=(3e-5, 2e-1), tickfont=11, legendfontsize=13, titlefontsize=13, legend=:bottom) for j = 1:D]
    [plot!(p_uniR[j], xlabel="$(agg_window) days rain (mm)") for j in staid_lat[6:10]]
    [plot!(p_uniR[j], ylabel="PDF") for j in staid_lat[[1, 6]]]
    [title!(p_uniR[j], station_name[j]) for j = 1:D]
    for j = 1:D
        dists_RR_positive_j = agg_rs[j, :]
        Rmax = ceil(max(agg_rs[j, :] .|> maximum |> maximum, agg_RR[j] |> maximum))
        errorlinehist!(p_uniR[j], dists_RR_positive_j, groupcolor=:grey, legend=:topright, label=islabel(j, 20, L"Simu $q_{0,100}$"), norm=:pdf, errortype=:percentile, percentiles=[0, 100], fillalpha=0.4, centertype=:median, bins=0:3:Rmax)

        errorlinehist!(p_uniR[j], dists_RR_positive_j, groupcolor=:red, label=islabel(j, 20, L"Simu $q_{25,75}$"), norm=:pdf, errortype=:percentile, percentiles=[25, 75], fillalpha=0.5, centertype=:median, bins=0:3:Rmax)

        errorlinehist!(p_uniR[j], [agg_RR[j]], label=islabel(j, 20, "Obs"), groupcolor=:blue, lw=1.5, norm=:pdf, errortype=:percentile, bins=0:3:Rmax, fillalpha=0.8)

        xlims!(p_uniR[j], 0.0, Inf)
        yticks!(10.0 .^ (-5:-1))
        lens!(p_uniR[j], [0, 25], [0.0, 0.2], inset=(1, bbox(0.48, -0, 0.475, 0.25)), linewidth=0)
    end

    pall_aggR = plot(p_uniR[staid_lat]..., size=(3000 / 2.5, 1000 / 1.5), layout=(2, 5), bottom_margin=14px, left_margin=15px)
    plot!(pall_aggR, (1:3)', inset=(1, bbox(2.4, 0.42, 0.6, 0.3)), subplot=2D + 1, legendfontsize=14, framestyle=:none, label=[L"Simu $q_{0,100}$" L"Simu $q_{25,75}$" "Obs"], c=[:gray :red :blue], foreground_color_legend=nothing, lw=2)
end
savefig(pall_aggR, joinpath(save_tuto_path, "dist_aggR_window_$(agg_window)_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf")); #src

md"""
#### Autocorrelation
"""

acfrange = 0:15
@views aa = [autocor(rs[j, :, i], acfrange) for j in 1:D, i in 1:Nb]

begin
    p_spell_wet = [plot(xlabelfontsize=16, ylabelfontsize=16, tickfont=11, legendfontsize=16) for j = 1:D]
    for j = 1:D
        errorline!(p_spell_wet[j], acfrange, stack(aa[:, j], dims=1)', groupcolor=:gray, label=islabel(j, staid_lat[[1]], L"Simu $q_{0,100}$"), errortype=:percentile, percentiles=[0, 100], fillalpha=0.8, lw=2, centertype=:median)
        plot!(p_spell_wet[j], acfrange, autocor(RR[:, j], acfrange), label=islabel(j, staid_lat[[1]], "Obs"), lw=2.0, c=1, markers=:circle, alpha=0.8)
    end

    [xlabel!(p_spell_wet[j], "Lag", xlabelfontsize=12) for j in staid_lat[6:10]]
    [ylabel!(p_spell_wet[j], "ACF", ylabelfontsize=12) for j in staid_lat[[1, 6]]]
    [title!(p_spell_wet[j], station_name[j], titlefontsize=13) for j = 1:D]
    pall_ACF = plot(p_spell_wet[staid_lat]..., size=(3000 / 2.5, 1000 / 1.5), layout=(2, 5), left_margin=0.42cm, bottom_margin=0.32cm)
end
savefig(pall_ACF, joinpath(save_tuto_path, "ACF_RR.pdf")) #src

md"""
#### Monthly quantile amount
"""

rh = reduce(hcat, [df[1+local_order:end, :RR] for df in data_stations])

month_rain_simu = [monthly_agg(rs[j, :, i], idx_all) for j in 1:D, i in 1:Nb];

month_rain_histo = [monthly_agg(rh[:, j], idx_all) for j in 1:D]

qs = [0.9, 0.5, 0.1]

@time "Plot monthly quantile" begin
    p_month_RR = [scatter(xtickfontsize=10, ytickfontsize=11, ylabelfontsize=12, legendfontsize = 12, foreground_color_legend=nothing) for j = 1:D]
    for j = 1:D
        for (α, per) in enumerate([[0, 100], [25, 75]])
            for (cc, q) in enumerate(qs)
                errorline!(p_month_RR[j], [quantile(month_rain_simu[j, i][:, m], q) * conversion_factor for m in 1:12, i in 1:Nb], label=(α == 1 ? islabel(j, 9,L"Simu  $q_{%$(Int(q*100))}$") : :none), fillalpha=0.18 * α^2, centertype=:median, errortype=:percentile, percentiles=per, groupcolor=my_palette(length(qs))[cc])
            end
        end
        for q in qs
            scatter!(p_month_RR[j], m -> quantile(month_rain_histo[j][:, m], q) * conversion_factor, 1:12, label=(q == qs[1] ? islabel(j, 3,"Obs") : :none), legend = :topleft, ms=2.5, c=:blue)
            plot!(p_month_RR[j], m -> quantile(month_rain_histo[j][:, m], q) * conversion_factor, 1:12, label=:none, c=:blue, lw=1.75)
        end
        xticks!(p_month_RR[j], 1:12, string.(first.(monthabbr.(1:12))))
        ylims!(p_month_RR[j], 0, Inf)
    end
    [ylabel!(p_month_RR[j], "Rain (mm)") for j in staid_lat[[1, 6]]]

    [title!(p_month_RR[j], station_name[j], titlefontsize=12) for j = 1:D]
    pall_month_RR = plot(p_month_RR[staid_lat]..., size=(1190, 500), layout=(2, 5), left_margin=19px)
end

#-
savefigcrop(pall_month_RR, "EDF_like_$(Nb)_simu_monthly_quantile_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)", save_tuto_path); #src

md"""
### Correlations
"""

md"""
##### Rain event dry/wet
"""

cor_bin_hist = cor(reduce(hcat, [df.RO for df in data_stations]));

cor_bin_mean_simu = mean(cor(ys[:, :, i]) for i in 1:Nb);


begin
    plots_cor_bin = [plot(-0.1:0.1:0.8, -0.1:0.1:0.8, aspect_ratio=true, label=:none, xlabelfontsize=16, ylabelfontsize=16, tickfont=11, legendfontsize=13) for _ in 1:1]
    scatter!(plots_cor_bin[1], vec_triu(cor_bin_hist), vec_triu(cor_bin_mean_simu), label="Correlations", xlabel="Observations", ylabel="Simulations", c=2)
    [xlims!(plots_cor_bin[i], -0.1, 1) for i in 1:1]
    [ylims!(plots_cor_bin[i], -0.1, 1) for i in 1:1]
    annotate!(0.2, 0.7, "MSE ≃ $(round(mean(abs2, vec_triu(cor_bin_hist) - vec_triu(cor_bin_mean_simu)), digits = 4))")
    plot_cor_bin = plot(plots_cor_bin...)
end

#-
savefigcrop(plot_cor_bin, "full_cor_binary_hist_vs_$(Nb)_mean_simu_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)", save_tuto_path); #src

md"""
The largest pair correlation error for rain occurrence comes from the pair 
"""

println("$(station_name[findmax(cor_bin_hist - cor_bin_mean_simu)[2][1]]) and $(station_name[findmax(cor_bin_hist - cor_bin_mean_simu)[2][2]])")

md"""
##### Rainfall amount
"""

cor_hist = cor(reduce(hcat, [df.RR for df in data_stations]));

corT_hist = corTail(reduce(hcat, [df.RR for df in data_stations]));

cor_mean_simu = mean(cor(rs[:, :, i]') for i in 1:Nb);

corT_mean_simu = mean(corTail(rs[:, :, i]') for i in 1:Nb);

begin
    plots_cor = [plot(-0.1:0.1:0.8, -0.1:0.1:0.8, aspect_ratio=true, label=:none, xlabelfontsize=16, ylabelfontsize=16, tickfont=11, legendfontsize=13) for _ in 1:2]
    scatter!(plots_cor[1], vec_triu(cor_hist), vec_triu(cor_mean_simu), label="Correlations", xlabel="Observations", ylabel="Simulations", c=2)
    annotate!(plots_cor[1], 0.3, 0.7, "MSE ≃ $(round(mean(abs2, vec_triu(cor_hist) - vec_triu(cor_mean_simu)), digits = 4))")

    scatter!(plots_cor[2], vec_triu(corT_hist), vec_triu(corT_mean_simu), label="Tail index", xlabel="Observations", ylabel="Simulations", c=3)
    annotate!(plots_cor[2], 0.3, 0.7, "MSE ≃ $(round(mean(abs2, vec_triu(corT_hist) - vec_triu(corT_mean_simu)), digits = 4))")

    [xlims!(plots_cor[i], -0.1, 1) for i in 1:2]
    [ylims!(plots_cor[i], -0.1, 1) for i in 1:2]
    plot_cor_all = plot(plots_cor..., size=(800, 400), left_margin=15px, right_margin = 8px)
end

#-
savefigcrop(plots_cor[1], "full_cor_hist_vs_$(Nb)_mean_simu_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)", save_tuto_path); #src
savefigcrop(plots_cor[2], "full_corT_hist_vs_$(Nb)_mean_simu_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)", save_tuto_path); #src
savefigcrop(plot_cor_all, "full_cor_both_hist_vs_$(Nb)_mean_simu_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)", save_tuto_path); #src

md"""
The largest pair correlation error for rain (zero and non zero amounts) comes from the pair 
"""

println("$(station_name[findmax(cor_hist - cor_mean_simu)[2][1]]) and $(station_name[findmax(cor_hist - cor_mean_simu)[2][2]])")

md"""
##### Gaussian copula hypothesis

For a pair of stations, we transform the marginal $R_s>0$ to $\mathcal{N}(0,1)$. We compare the obtained bi-variate Normal distribution with the Mahalanobis distance to the theoretical $\chi^2(2)$-distriubtion.
"""
corΣ = cov2cor.(Σ²RR)
begin
    j1 = 10
    j2 = 8
    plt_qqp_copula = plot(0:25, 0:25, aspect_ratio=:equal, legendfontsize=14, c=:black, label=:none, tickfont=12, ylabelfontsize=13, xlabelfontsize=13)
    df_12 = leftjoin(data_stations_z[j1], data_stations_z[j2], on=:DATE, makeunique=true)
    @subset!(df_12, :RR .> 0, :RR_1 .> 0)
    for k in 1:K
        df_X = @chain df_12 begin
            @subset(:z .== k)
            dropmissing
            @aside u = StochasticWeatherGenerators.Copulas.pseudos(permutedims(hcat(_.RR, _.RR_1)))
            @transform(:X = quantile(Normal(), u[1,:]), :X_1 = quantile(Normal(), u[2,:]))
        end
        X = hcat(df_X.X, df_X.X_1)
        cor_sigma = [1 corΣ[k][j1,j2]; corΣ[k][j1,j2] 1]
        Σ⁻¹ = inv(cor_sigma)

        X2 = [(x' * Σ⁻¹ * x) for x in eachrow(X)] |> sort
        ecdfX2 = ecdf(X2)(X2) * length(X2) / (length(X2) + 1)

        plot!(quantile(Chisq(2), ecdfX2), X2, xlabel=L"$\chi^2(2)$-quantile", c=my_color(k, K), ylabel="Observed squared Mahalanobis distance", label=L"Z = %$k ", legend=:topleft, lw=2)
    end
    title!("$(station_name[j1])-$(station_name[j2]) $(ifelse(j1 == 10 && j2 == 8, "(334 km)", ""))")
    xlims!(0, 22)
    ylims!(0, 22)
end

#-
savefigcrop(plt_qqp_copula, "qq_copula_$(station_name[j1])_$(ifelse(j2 == 8, "LA_HAGUE", station_name[j2]))_Z_full_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)", save_tuto_path); #src

md"""
## Reproducibility
"""
using InteractiveUtils
InteractiveUtils.versioninfo()

md"""
Package list and version
"""

import Pkg; Pkg.status()