using Markdown#hide
import Pkg;
cd(@__DIR__)#hide
Pkg.activate("../docs/");#src

md"""
# [Multisite spatial rainfall HMM with Gaussian conditional dependence](@id TutoSHMMSpa)
"""

md"""
This tutorial describes the spatial HMM model for multisite rainfall occurrences.
It extends [this tutorial](@ref TutoSHHMM) by replacing conditionally independent Bernoulli emissions at each station with a `SpatialBernoulli` copula model that captures spatial dependence between stations via an exponential covariance kernel.

It is based on the paper [**Spatio-temporal generation of precipitation using a Hidden Markov Model, extended extreme distributions and conditional Gaussian fields**](https://hal.science/hal-05523768/).

Here, the model is fixed to `K = 4` hidden states and `local_order = 1` (one day of local memory). 
It is applied to the same set of 10 stations in France as in the previous tutorial. However, contrary to the previous model where conditional independence was assumed, implying that stations would be far enough apart, here it is possible to use stations that are closer together.
"""

md"""
## Set up
"""

md"""
### Package and functions
"""

using CSV, DelimitedFiles # File Read/Load/Save

using JLD

using DataFrames, DataFramesMeta # DataFrames

using Dates

using StatsBase, Random

using Distributions

md"""
The main package is `StochasticWeatherGenerators.jl`, which provides the interface to the models and data.
`SmoothPeriodicStatsModels.jl` contains the smooth periodic models used in this tutorial, in particular the
`ARPeriodicHMMSpatial` model and its fitting routines.
"""

using StochasticWeatherGenerators # interface to use with SmoothPeriodicStatsModels.jl

using SmoothPeriodicStatsModels # Small collection of smooth periodic models e.g. AR, HMM, HMMSpatial

md"""
Since `SmoothPeriodicStatsModels.jl` v3 all the fitting code lives in a package extension: the models and
`rand` are always available, while `fit_mle` and friends are only defined once the SciML solver stack below
is loaded: Optimization and SimpleNonlinearSolve are the extension trigger. Every fit then takes its
optimization algorithm explicitly as `solver`.
`LineSearches.jl` is only needed for the solver we configure by hand further down.
"""

using Optimization, SimpleNonlinearSolve # load the fitting routines
using OptimizationOptimJL # the optimization algorithm, LBFGS
using LineSearches
#-
Random.seed!(123)

md"""
### Settings for plotting

Plots are made with [Makie.jl](https://docs.makie.org/) and its `CairoMakie` backend.
Loading Makie defines the plot functions of `StochasticWeatherGenerators.jl` (`errorline`, `errorlinehist`, ...).
Loading `GeoMakie.jl` and `NaturalEarth.jl` as well defines `map_with_stations` to plot maps.
"""

using CairoMakie, LaTeXStrings
using GeoMakie, NaturalEarth # maps

md"""
The theme sets the LaTeX fonts and the font size of every figure. Each figure is then sized by the `width` and `height` of its axes.
"""

plots_colors = [RGBf(0.0, 0.606, 0.979), RGBf(0.889, 0.436, 0.278), RGBf(0.242, 0.643, 0.304), RGBf(0.764, 0.444, 0.824),
    RGBf(0.676, 0.556, 0.094), RGBf(0.0, 0.666, 0.681), RGBf(0.931, 0.367, 0.576), RGBf(0.777, 0.51, 0.146)] # Plots.jl default colors

set_theme!(theme_latexfonts(); fontsize=16, palette=(color=plots_colors,))

my_palette(K) = plots_colors[my_color.(1:K, K)] # blue for k = 1 (∼wetter) and orange for k = K (∼driest)

month_days = vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366) # x ticks for day of the year plots

month_letters = vcat(string.(first.(monthabbr.(1:12))), "");

include("utilities_plot.jl") #src

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
## HMM Hyperparameters
"""

md"""
Number of hidden states
"""

K = 4

my_pal = my_palette(K); # just colors I like for plotting weather regime!

md"""
Degree `𝐃𝐞𝐠` of the trigonometric expansion 
"""

𝐃𝐞𝐠 = 1

md"""
Local memory order i.e. at station $j$, $\mathbb{P}(Y_n^{(j)} = y_n^{(j)} \mid Z = k, Y_{n-1:n-\texttt{local memory}}^{(j)} = y_{n-1:n-\texttt{local memory}}^{(j)})$
"""

local_order = 1

size_order = 2^local_order

md"""
Initial range parameter $R_0$ for the exponential covariance kernel of the `SpatialBernoulli` distribution (in km).
"""

R0 = [400, 400, 300, 200]

md"""
Distance threshold fraction used to build the neighborhood weight matrix `wp` (only pairs of stations closer than `tdist * max_distance` are considered).
"""

tdist = 0.3

md"""
Number of quasi-Monte Carlo samples for the multivariate normal CDF estimation in the spatial likelihood.
"""

QMC_m = 300

println("K = $K, ", "degree = $𝐃𝐞𝐠, ", "local_order = $local_order")

md"""
## Data
"""

save_tuto_path = "../../assets/tuto_HMMspa/tuto_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)" #src
save_tuto_path = joinpath(pkgdir(StochasticWeatherGenerators), "assets", "tuto_3", "tuto_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)") #src
isdir(save_tuto_path) ? nothing : mkpath(save_tuto_path) #src

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
    station_file = joinpath(pkgdir(StochasticWeatherGenerators), "weather_files/stations.txt")
    station_all = CSV.read(station_file, DataFrame, header=18, normalizenames=true, ignoreemptyrows=true)
    station_all = @chain station_all begin
        @transform(:CN = rstrip.(:CN), :STANAME = rstrip.(:STANAME))
        @subset(:STAID .∈ tuple([32, 33, 34, 36, 39, 203, 322, 323, 434, 736, 737, 738, 740, 742, 745, 749, 750, 755, 756, 757, 758, 786, 793, 2192, 2203, 2205, 2207, 2209, 11244, 11245, 11247, 11249]))
        @transform(:STANAME = shortname.(:STANAME))
    end
end

selected_station_name = ["BOURGES", "TOULOUSE", "MARIGNANE", "LUXEMBOURG", "LILLE", "EMBRUN", "BASTIA", "LA HAGUE", "CHASSIRON", "ORLY"]

station = @subset(station_all, :STANAME .∈ tuple(selected_station_name))

STAID = station.STAID

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
### Pairwise distance matrix
"""

md"""
Convert LAT/LON coordinates from DMS to DD (decimal degrees).
"""

LAT_idx = dms_to_dd.(station.LAT)

LON_idx = dms_to_dd.(station.LON)

md"""
Build the pairwise distance matrix `my_distance` (in km) between stations using the haversine formula.
"""

my_locations = hcat(LON_idx, LAT_idx) # (D × 2) matrix: [longitude, latitude]

md"""
We define the haversine great-circle distance (in km) and build the $D \times D$ distance matrix.
"""

function haversine_km(coords1, coords2; R=6371.0)
    lon1, lat1 = deg2rad(coords1[1]), deg2rad(coords1[2])
    lon2, lat2 = deg2rad(coords2[1]), deg2rad(coords2[2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)^2 + cos(lat1) * cos(lat2) * sin(dlon / 2)^2
    return 2R * asin(sqrt(a))
end

my_distance = [haversine_km(my_locations[i, :], my_locations[j, :]) for i in 1:D, j in 1:D]

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

Yall = BitMatrix(stack([data_stations[j].RO for j = 1:D]; dims=1)) # D × N, the emission convention

Y_past = BitMatrix(Yall[:, 1:local_order]) # rand(Bool, D, local_order)

ξ = [1; zeros(K - 1)];  # 1 jan 1956 was most likely a type Z = 1 wet day all over France

Y = Yall[:, 1+local_order:end]

md"""
### Map of stations
"""

long_spell = [longuest_spell(y) for y in eachrow(Y)] # one row per station

FR_map_spell = map_with_stations(LON_idx, LAT_idx, long_spell; station_name=station_name, show_value=true, colorbar_show=true, precision_scale=precision_scale, colorbar_label="Days")

#-
savefigcrop(FR_map_spell, "FR_longest_dry_spell_$(D)_station_histo", save_tuto_path); #src

md"""
## Step 1: Fit the non-spatial seasonal HMM

We first fit the standard (non-spatial) autoregressive periodic HMM using the Baum-Welch EM algorithm.
The result provides the transition parameters `θᴬ` and the Bernoulli emission parameters `θᴮ`, which
serve as the starting point for the spatial extension.
"""

md"""
### Fit slice: naive estimation
"""

#!nb # !!! note
#!nb #     Before inferring the HMM parameters with the EM (Baum-Welch) algorithm, we do a first naive inference that will be used as initial condition for the EM.

md"""
The reference station `ref_station` is used to sort the hidden states obtained via the slice initialisation.
Here we choose `j=1` $\to$ `STAID=32` $\to$ `BOURGES` because it is a central station for France.
"""

ref_station = 1

md"""
The following code loads the HMM fitted in the previous tutorial [Multisite HMM with independent Bernoulli emissions](@ref TutoSHHMM).
It will be used as the initial condition for the spatial HMM fitting.
"""

hmm_ind_infos = JLD.load(joinpath(pkgdir(StochasticWeatherGenerators), "assets", "tuto_1", "hmm_fit_K_4_d_1_m_1.jld"))

hmm_ind = hmm_ind_infos["hmm"]
θq_fit_ind = hmm_ind_infos["Q_param"]
θy_fit_ind = hmm_ind_infos["Y_param"];

md"""
The Spatial model need also the range parameter `θᴿ` for the exponential covariance kernel of the `SpatialBernoulli` distribution.
The range parameter trigonometric coefficients `θᴿ` are initialised with a constant $\log(R_0)$,
which corresponds to a time-invariant spatial range of $R_0$ km.
"""

size_degree_of_P = 2 * 𝐃𝐞𝐠 + 1

θᴿ = zeros(K, size_degree_of_P)
θᴿ[:, 1] .= log.(R0)

md"""
### Fit the spatial HMM with EM

`fit_mle` runs the EM algorithm for `SmoothARPeriodicHMMSpatial`. It updates the transition parameters
`θᴬ`, the Bernoulli emission parameters `θᴮ`, and the spatial range parameters `θᴿ` jointly.

Every M-step goes through the same four keywords: `solver` (the algorithm handed to `solve`,
required), `adtype` (the AD backend, `AutoForwardDiff()` by default), `ensemble` (`EnsembleSerial()`,
`EnsembleThreads()` or `EnsembleDistributed()` for the per-state/per-site problems) and `solve_kwargs`
(a `NamedTuple` forwarded verbatim to `solve`). Any `Optimization.jl`-compatible solver works, so the
choice below is just the one that behaves best on the range update.
`maxiters_R` stays a separate keyword because it caps the inner iterations of the `R` M-step only.

For time sake in this tutorial, we load the fitted model but show bellow the code to run the EM algorithm (that can take an hour or so).
"""
md"""
```julia
solver = OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking())

hmm_spa_guess = SmoothARPeriodicHMMSpatial(fill(1 / K, K), θq_fit_ind, θy_fit_ind, θᴿ, T, my_distance; QMC_m=QMC_m)

@time "FitMLE HMMSpa (EM)" hmm_spa_fit, history = fit_mle(hmm_spa_guess, Y, Y_past; solver=solver, ensemble=EnsembleThreads(), n2t=n2t, maxiter=60, tol=1e-3, maxiters_R=100, display=:iter, tdist=tdist, ps_iters=true);
# on my machine
# EM has not converged after 100 iterations, logtot = -115271.5855184366
# FitMLE HMMSpa (EM): 440.382967 seconds (2.44 G allocations: 357.809 GiB, 13.96% gc time, 13.02% compilation time: <1% of which was recompilation)
JLD.save(joinpath(pkgdir(StochasticWeatherGenerators), "assets", "tuto_3", "hmm_fit_K_4_d_1_m_1.jld"), "logtots", history.logtots, "ps_iters", history.ps_iters, "best_iteration", history.best_iteration);
```
"""

hmm_infos = load(joinpath(pkgdir(StochasticWeatherGenerators), "assets", "tuto_3","hmm_fit_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).jld"))

logtots = hmm_infos["logtots"]
ps_iters = hmm_infos["ps_iters"];

md"""
In classic EM algorithm, the log-likelihood is guaranteed to increase at each iteration. However, here a modified EM (see paper) and 1) increase was not demonstrated theoretically (though it might be true) 2) the spatial likelihood is estimated with a quasi-Monte Carlo method, which adds some noise to the likelihood estimation. 
Therefore, it is possible to observe some non-monotonicity in the log-likelihood convergence curve.

Hence, `fit_mle` returns the iteration with the highest log-likelihood, `history.best_iteration`
(`0` would be the starting point):
"""

iter_best = hmm_infos["best_iteration"]
#-
logtots[iter_best]

md"""
`hmm_spa_fit` above is that model. Since the fit is not run here, we rebuild it from the saved
coefficient iterates: `ps_iters[1]` is the starting point, so iteration `i` is `ps_iters[i + 1]`.
"""

ps_best = ps_iters[iter_best+1]
hmm_spa = ARPeriodicHMMSpatial(SmoothARPeriodicHMMSpatial(fill(1 / K, K), ps_best.A, ps_best.B, ps_best.R, T, my_distance; QMC_m=QMC_m))

#-

md"""
### Log-likelihood convergence
"""

begin
    p_logtots = Figure()
    ax = Axis(p_logtots[1, 1], xlabel="EM iteration", ylabel="Log-likelihood", title="Convergence of the spatial HMM EM", width=500, height=300)
    lines!(ax, logtots, linewidth=2)
    vlines!(ax, iter_best, color=:red, linestyle=:dash, label="best iter")
    axislegend(ax; position=:rc, framevisible=false)
    resize_to_layout!(p_logtots)
    p_logtots
end

#-
save(joinpath(save_tuto_path, "logtots_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf"), p_logtots); #src

md"""
## Visualisation of the fitted model parameters
"""

md"""
### Transition matrix
"""

begin
    pallA = Figure()
    for k in 1:K
        row, col = fldmod1(k, 2)
        ax = Axis(pallA[row, col], limits=(0, 367, 0, 1), xticks=(month_days, month_letters), width=400, height=200)
        col > 1 && hideydecorations!(ax, grid=false, ticks=false)
        for l in 1:K
            lines!(ax, 1:T, hmm_spa.A[k, l, :], color=my_palette(K)[l], linewidth=1.75, label=L"Q_{%$(k)\to %$(l)}")
        end
        hlines!(ax, 0.5, color=:black, linestyle=:dot)
        axislegend(ax, position=:ct, nbanks=K, framevisible=false, colgap=6, patchlabelgap=2)
    end
    resize_to_layout!(pallA)
    pallA
end

#-
save(joinpath(save_tuto_path, "Q_transition_spa_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf"), pallA); #src

md"""
### Spatial range $R(t)$

The seasonal range $R_k(t)$ controls the spatial correlation length for each hidden state $k$.
A larger range means stronger spatial dependence between stations.
"""

begin
    pr = Figure()
    ax = Axis(pr[1, 1], title="Spatial range per hidden state", ylabel="Range (km)", xticks=(month_days, month_letters), limits=(0, 367, 100, 350), width=500, height=300)
    for k in 1:K
        lines!(ax, 1:T, hmm_spa.R[k, :], color=my_palette(K)[k], label=L"k = %$k", linewidth=2)
    end
    Legend(pr[1, 2], ax, framevisible=false)
    resize_to_layout!(pr)
    pr
end

#-
save(joinpath(save_tuto_path, "range_spa_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf"), pr); #src

md"""
### Rain occurrence probabilities $\mathbb{P}(Y = \text{wet} \mid Z = k, H = h)$
"""

begin
    mm = 1 # h = 1: day before was dry
    pallB = Figure()
    for (idx, j) in enumerate(staid_lat)
        row, col = fldmod1(idx, 5)
        ax = Axis(pallB[row, col], title=station_name[j], limits=(0, 367, 0, 1), xticks=(month_days, month_letters), width=230, height=170)
        col > 1 && hideydecorations!(ax, grid=false, ticks=false)
        for k in 1:K
            lines!(ax, 1:T, hmm_spa.B[k, :, j, mm], color=my_palette(K)[k], linewidth=2)
        end
        hlines!(ax, 0.5, color=:black, linestyle=:dot)
    end
    Legend(pallB[:, 6], [LineElement(color=my_palette(K)[k], linewidth=2) for k in 1:K],
        [L"\mathbb{P}(Y = \textrm{wet}\mid Z = %$k, H = \textrm{dry})" for k in 1:K], framevisible=false)
    resize_to_layout!(pallB)
    pallB
end

#-
save(joinpath(save_tuto_path, "proba_rain_all_station_spa_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf"), pallB); #src

md"""
### Spatial map of mean rain probability
"""

memory_past_cat = 1

md"""
Mean rain probability per hidden state $k$: $\overline{\mathbb{P}}(Y = \text{Rain}\mid Z = k, H = 1)$
"""

p_FR_map_mean_prob = map_with_stations(LON_idx, LAT_idx, [[mean(hmm_spa.B[k, :, j, memory_past_cat]) for j in 1:D] for k in 1:K], colorbar_show=true, colorbar_label=L"\mathbb{P}(Y = \text{Rain}\mid Z = k, H = 1)", precision_scale=precision_scale)

#-
savefigcrop(p_FR_map_mean_prob, "FR_spa_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)_mean_proba_cat_1", save_tuto_path); #src

md"""
## Inference of the historical hidden states
"""

md"""
### Viterbi algorithm
"""

ẑ = viterbi(hmm_spa, Y, Y_past; n2t=n2t)

ẑ_per_cat = [findall(ẑ .== k) for k in 1:K]

CSV.write(joinpath(save_tuto_path, "z_hat_spa_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).csv"), DataFrame([:DATE, :z] .=> [data_stations[1].DATE[1+local_order:end], ẑ])); #src

md"""
### Visualisation of the historical sequence of hidden states
"""

year_range = unique(year.(data_stations[1][1+local_order:end, :DATE]));

idx_year = [findall(x -> year.(x) == m, data_stations[1][1+local_order:end, :DATE]) for m in year_range];

select_year = unique(sort([4:10:length(year_range); 21; 48; 64]))

begin
    year_nb = length(select_year)
    z_hat_mat = fill(NaN, 366, year_nb) # day × year, NaN on day 366 of non leap years

    for (i, y) in enumerate(select_year)
        z_hat_mat[eachindex(idx_year[y]), i] = ẑ[idx_year[y]]
    end
    pviterbi = Figure()
    ax = Axis(pviterbi[1, 1], xticks=(month_days, vcat(string.(monthabbr.(1:12)), "")), yticks=(1:year_nb, string.(year_range[select_year])),
        yreversed=true, xgridvisible=false, ygridvisible=false, width=850, height=450)
    heatmap!(ax, 1:366, 1:year_nb, z_hat_mat, colormap=my_palette(K), colorrange=(1, K), nan_color=:white)
    hlines!(ax, (1:year_nb-1) .+ 0.5, color=:black, linewidth=4)
    xlims!(ax, 0.5, 366.5)
    Legend(pviterbi[1, 2], [PolyElement(color=my_palette(K)[k]) for k in 1:K], [L"Z = %$k" for k in 1:K], framevisible=false)
    resize_to_layout!(pviterbi)
    pviterbi
end

#-
save(joinpath(save_tuto_path, "temporal_viterbi_spa_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf"), pviterbi); #src

md"""
Sample the spatial HMM model and output the sequence of hidden states and multi-site dry/wet.
"""

Nb = 1000 

begin
    zs = zeros(Int, N, Nb)
    ys = zeros(Bool, D, N, Nb)
    @time "Simulations Z, Y (spatial HMM)" for i in 1:Nb
        zs[:, i], ys[:, :, i] = rand(hmm_spa, n2t; Y_past, z_ini=1, seq=true)
    end
end

md"""
## The WGEN model

We will compare to the WGEN model that propose Markov chain of order 4 for rain occurrences (fitted monthly) and laten gaussian model for multisite occurrences (fitted monthly).
- Wilks, D. S. "Multisite generalization of a daily stochastic precipitation generation model". Journal of Hydrology, (1998). https://doi.org/10.1016/S0022-1694(98)00186-3.
- Srikanthan, Ratnasingham, et Geoffrey G. S. Pegram. "A nested multisite daily rainfall stochastic generation model". Journal of Hydrology 2009. https://doi.org/10.1016/j.jhydrol.2009.03.025.
"""

wgen_order = 4
idx_months = [findall(x -> month.(x) == m, data_stations[1][1+local_order:end, :DATE]) for m in 1:12]
wgen4_model = fit_wgen(Y, idx_months, wgen_order)

ys_wgen = similar(ys)
@time "Simulation Y wgen 4" for i in 1:Nb
    ys_wgen[:, :, i] = rand(wgen4_model, 1956:2019; Y_past=hcat(rand(Bool, D, wgen_order - local_order), Y_past))
end

md"""
### Non-spatial HMM simulations

Simulate from the independent (non-spatial) `ARPeriodicHMM` fitted in Step 1, for comparison.
"""

ys_ind = similar(ys)
zs_ind = zeros(Int, N, Nb)
@time "Simulations Z, Y (HMM_ind)" for i in 1:Nb
    zs_ind[:, i], ys_ind[:, :, i] = rand(hmm_ind, n2t; Y_past, z_ini=1, seq=true)
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

len_spell_hist = [pmf_spell(Y[j, idx_month_vcat], dw) for j in 1:D, dw in 0:1];

md"""
##### Simulation spells
"""

len_spell_simu = [pmf_spell(ys[j, idx_month_vcat, i], dw) for i in 1:Nb, j in 1:D, dw in 0:1];

md"""
#### Dry spell
"""

make_range(y, step=1) = range(extrema(y)..., step=step)

legend_simu = ([PolyElement(color=:grey, alpha=0.4), PolyElement(color=:red, alpha=0.5), LineElement(color=:red), LineElement(color=:blue)],
    [L"Simu $q_{0,100}$", L"Simu $q_{25,75}$", L"Simu $q_{50}$", "Obs"]) # legend of the simulation envelopes vs observations

begin
    dry_or_wet = 1 # dry
    pall_spell_dry = Figure()
    for (idx, j) in enumerate(staid_lat)
        row, col = fldmod1(idx, 5)
        ax = Axis(pall_spell_dry[row, col], title=station_name[j], xlabel=(row == 2 ? "Nb of days" : ""), ylabel=(col == 1 ? "PMF" : ""),
            yscale=log10, width=200, height=170)
        all_spells = len_spell_simu[:, j, dry_or_wet]
        histo_spell = len_spell_hist[j, dry_or_wet]
        spell_range = 1:1:(1+maximum(vcat(reduce(vcat, all_spells), histo_spell)))

        errorlinehist!(ax, all_spells, bins=spell_range, normalization=:probability, errortype=:percentile, percentiles=[0, 100], centertype=:median, color=:red, secondarycolor=:grey, secondaryalpha=0.4)

        errorlinehist!(ax, all_spells, bins=spell_range, normalization=:probability, errortype=:percentile, percentiles=[25, 75], centertype=:median, color=:red, secondaryalpha=0.5)

        errorlinehist!(ax, [histo_spell], bins=spell_range, normalization=:probability, errortype=:percentile, color=:blue, linewidth=1.5, alpha=0.8)
        xlims!(ax, 0, 2 + 1.5maximum(histo_spell))
        ylims!(ax, 1e-4, 1)
        col > 1 && hideydecorations!(ax, grid=false, ticks=false, minorgrid=false, minorticks=false)
    end
    Legend(pall_spell_dry[:, 6], legend_simu..., framevisible=false)
    resize_to_layout!(pall_spell_dry)
    pall_spell_dry
end

#-
save(joinpath(save_tuto_path, "spell_steppost_dry_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf"), pall_spell_dry); #src

md"""
#### Wet spell
"""

begin
    dry_or_wet = 2 # wet
    pall_spell_wet = Figure()
    for (idx, j) in enumerate(staid_lat)
        row, col = fldmod1(idx, 5)
        ax = Axis(pall_spell_wet[row, col], title=station_name[j], xlabel=(row == 2 ? "Nb of days" : ""), ylabel=(col == 1 ? "PMF" : ""),
            yscale=log10, width=200, height=170)
        all_spells = len_spell_simu[:, j, dry_or_wet]
        histo_spell = len_spell_hist[j, dry_or_wet]
        spell_range = 1:1:(1+maximum(vcat(reduce(vcat, all_spells), histo_spell)))

        errorlinehist!(ax, all_spells, bins=spell_range, normalization=:probability, errortype=:percentile, percentiles=[0, 100], centertype=:median, color=:red, secondarycolor=:grey, secondaryalpha=0.4)

        errorlinehist!(ax, all_spells, bins=spell_range, normalization=:probability, errortype=:percentile, percentiles=[25, 75], centertype=:median, color=:red, secondaryalpha=0.5)

        errorlinehist!(ax, [histo_spell], bins=spell_range, normalization=:probability, errortype=:percentile, color=:blue, linewidth=1.5, alpha=0.8)
        xlims!(ax, 0, 2 + 1.5maximum(histo_spell))
        ylims!(ax, 1e-4, 1)
        col > 1 && hideydecorations!(ax, grid=false, ticks=false, minorgrid=false, minorticks=false)
    end
    Legend(pall_spell_wet[:, 6], legend_simu..., framevisible=false)
    resize_to_layout!(pall_spell_wet)
    pall_spell_wet
end

#-
save(joinpath(save_tuto_path, "spell_steppost_wet_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).pdf"), pall_spell_wet); #src

md"""
### Seasonal areal dry spells
"""

RRmax = 0
ROR = [mean(r .> RRmax) for r in eachcol(Y)]
RORs = [[mean(r .> RRmax) for r in eachcol(rr)] for rr in eachslice(ys, dims=3)]
RORs_ind = [[mean(r .> RRmax) for r in eachcol(rr)] for rr in eachslice(ys_ind, dims=3)]
RORswgen = [[mean(r .> RRmax) for r in eachcol(rr)] for rr in eachslice(ys_wgen, dims=3)]

JJA = [6, 7, 8]
MAM = [3, 4, 5]
SON = [9, 10, 11]
DJF = [12, 1, 2]
SEASONS = [DJF, MAM, JJA, SON]
seasonname = ["DJF", "MAM", "JJA", "SON"]

idx_seasons = [findall(month.(data_stations[1][1+local_order:end, :DATE]) .∈ tuple(season)) for season in SEASONS]

pall_ROR = let
    perc = 0.1
    QQ = [5, 95]

    fig = Figure()
    for m in eachindex(idx_seasons)
        row, col = fldmod1(m, 2)
        ax = Axis(fig[row, col], title=seasonname[m], xlabel=(row == 2 ? "Nb of days" : ""), ylabel=(col == 1 ? "PMF" : ""),
            yscale=log10, limits=(-0.01, 25, 5e-4, 1), width=350, height=200)
        len_ror_hist = pmf_spell(ROR[idx_seasons[m]] .≤ perc, 1)
        len_ror_simu = [pmf_spell(RORs[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]
        len_ror_simu_ind = [pmf_spell(RORs_ind[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]
        len_ror_simuwgen = [pmf_spell(RORswgen[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]

        errorlinehist!(ax, len_ror_simuwgen, bins=make_range(reduce(vcat, len_ror_simuwgen)), normalization=:probability, errortype=:percentile, percentiles=QQ,
            centertype=:median, color=:green, secondaryalpha=0.25, linewidth=2, label="WGEN 4")

        errorlinehist!(ax, len_ror_simu_ind, bins=make_range(reduce(vcat, len_ror_simu_ind)), normalization=:probability, errortype=:percentile, percentiles=QQ,
            centertype=:median, color=:grey, secondaryalpha=0.25, linewidth=2, label="HMM_ind")

        errorlinehist!(ax, len_ror_simu, bins=make_range(reduce(vcat, len_ror_simu)), normalization=:probability, errortype=:percentile, percentiles=QQ,
            centertype=:median, color=:orange, secondaryalpha=0.3, linewidth=2, label="HMM_spa")

        errorlinehist!(ax, [len_ror_hist], bins=make_range(len_ror_hist), normalization=:probability, errortype=:percentile, color=:blue, linewidth=2, label="Obs")
        col > 1 && hideydecorations!(ax, grid=false, ticks=false, minorgrid=false, minorticks=false)
    end
    Legend(fig[:, 3], content(fig[1, 1]), framevisible=false)
    resize_to_layout!(fig)
    file_name = "ROR_spell_season_perc_$(perc)_Q_$(QQ[1])_$(QQ[2])_no_inset" #src
    file_name = replace(file_name, "." => "p") #src
    save(joinpath(save_tuto_path, file_name * ".pdf"), fig) #src
    fig
end

md"""
### Correlations
"""

md"""
##### Rain event dry/wet
"""

cor_bin_hist = cor(stack([df.RO for df in data_stations]; dims=1), dims=2);

cor_bin_mean_simu = mean(cor(ys[:, :, i], dims=2) for i in 1:Nb);

cor_bin_mean_simu_ind = mean(cor(ys_ind[:, :, i], dims=2) for i in 1:Nb);

cor_bin_mean_simu_wgen = mean(cor(ys_wgen[:, :, i], dims=2) for i in 1:Nb);

begin
    plot_cor_bin = Figure()
    ax = Axis(plot_cor_bin[1, 1], xlabel="Observations", ylabel="Simulations", limits=(-0.1, 1, -0.1, 1), aspect=DataAspect(), width=350, height=350)
    ablines!(ax, 0, 1, color=:black)
    scatter!(ax, vec_triu(cor_bin_hist), vec_triu(cor_bin_mean_simu), color=:orange, label="Correlations vs SHHMM-Spa")
    ## scatter!(ax, vec_triu(cor_bin_hist), vec_triu(cor_bin_mean_simu_ind), color=:gray, label="Correlations vs HMM")
    ## scatter!(ax, vec_triu(cor_bin_hist), vec_triu(cor_bin_mean_simu_wgen), color=:green, label="Correlations vs WGEN")
    text!(ax, 0.2, 0.7, text=L"\mathrm{MSE} \simeq %$(round(mean(abs2, vec_triu(cor_bin_hist) - vec_triu(cor_bin_mean_simu)), digits = 4))")
    axislegend(ax, position=:rb)
    resize_to_layout!(plot_cor_bin)
    plot_cor_bin
end

#-
savefigcrop(plot_cor_bin, "full_cor_binary_hist_vs_$(Nb)_mean_simu_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)", save_tuto_path); #src
