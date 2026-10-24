```@meta
EditURL = "../../../examples/tuto_paper_HMMspatial.jl"
```

````@example tuto_paper_HMMspatial
using Markdown#hide
import Pkg;
cd(@__DIR__)#hide
````

# [Multisite spatial rainfall HMM with conditional dependence](@id TutoSHMMSpa)

This tutorial describes the spatial HMM model for multisite rainfall occurrences.
It extends [this tutorial](@ref TutoSHHMM) by replacing conditionally independent Bernoulli emissions at each station with a `SpatialBernoulli` copula model that captures
spatial dependence between stations via an exponential covariance kernel.

Here, the model is fixed to `K = 4` hidden states and `local_order = 1` (one day of local memory).

## Set up

### Package and functions

````@example tuto_paper_HMMspatial
using CSV, JLD2, DelimitedFiles # File Read/Load/Save

using DataFrames, DataFramesMeta # DataFrames

using Dates

using StatsBase, Random

using Distributions
````

The main package is `StochasticWeatherGenerators.jl`, which provides the interface to the models and data.
`SmoothPeriodicStatsModels.jl` contains the smooth periodic models used in this tutorial, in particular the
`ARPeriodicHMMSpatial` model and its fitting routines.

````@example tuto_paper_HMMspatial
using StochasticWeatherGenerators # interface to use with SmoothPeriodicStatsModels.jl

using SmoothPeriodicStatsModels # Small collection of smooth periodic models e.g. AR, HMM, HMMSpatial
using OptimizationOptimJL, LineSearches
````

````@example tuto_paper_HMMspatial
Random.seed!(1234)
````

### Settings for plotting

Some settings and packages to have nice plots.

````@example tuto_paper_HMMspatial
using StatsPlots, LaTeXStrings
using StatsPlots.PlotMeasures # To play with margin in Plots

gr() # plotly() # for interactive plots
default(fontfamily="Computer Modern")
cur_colors = get_color_palette(:auto, 100);
my_palette(K) = palette(vcat(cur_colors[1], [cur_colors[c] for c in 3:4], cur_colors[2]), K)

file_for_plot_utilities = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/utilities_plot.jl")
include(file_for_plot_utilities)
````

To plot maps, we use `GeoMakie.jl` + `NaturalEarth.jl`.

````@example tuto_paper_HMMspatial
file_for_maps_with_geomakie = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/utilities_geo_makie_features.jl") # download file from a GitHub repo
include(file_for_maps_with_geomakie)
````

### Global Parameters

The period of the Seasonal HMM is chosen as the number of days in a year. Our choice is 366 however we carefully skip February 29 when needed.

````@example tuto_paper_HMMspatial
T = 366
````

Define the French area for map (Longitude and latitude) plot and the precision of the map `precision_scale`

````@example tuto_paper_HMMspatial
precision_scale = 50 # meter

LON_min = -5 # West

LON_max = 10 # East

LAT_min = 41 # South

LAT_max = 52 # North
````

## HMM Hyperparameters

Number of hidden states

````@example tuto_paper_HMMspatial
K = 4

my_pal = my_palette(K); # just colors I like for plotting weather regime!
nothing #hide
````

Degree `𝐃𝐞𝐠` of the trigonometric expansion

````@example tuto_paper_HMMspatial
𝐃𝐞𝐠 = 1
````

Local memory order i.e. at station $j$, $\mathbb{P}(Y_n^{(j)} = y_n^{(j)} \mid Z = k, Y_{n-1:n-\texttt{local memory}}^{(j)} = y_{n-1:n-\texttt{local memory}}^{(j)})$

````@example tuto_paper_HMMspatial
local_order = 0

size_order = 2^local_order
````

Initial range parameter $R_0$ for the exponential covariance kernel of the `SpatialBernoulli` distribution (in km).

````@example tuto_paper_HMMspatial
R0 = [400, 400, 300, 200]
````

Distance threshold fraction used to build the neighborhood weight matrix `wp` (only pairs of stations closer than `tdist * max_distance` are considered).

````@example tuto_paper_HMMspatial
tdist = 0.3
````

Number of quasi-Monte Carlo samples for the multivariate normal CDF estimation in the spatial likelihood.

````@example tuto_paper_HMMspatial
QMC_m = 30

println("K = $K, ", "degree = $𝐃𝐞𝐠, ", "local_order = $local_order")
````

## Data

### Select relevant stations from the `station.txt` file

Here we
- Remove white space at the right of the `CN`, STANAME which is caused by imperfect CVS importation
- Select only the stations with 100% valid data for the period `Date(1955,12,31) .≤ :DATE .≤ Date(2019,12,31)`
- Shorten station names

````@example tuto_paper_HMMspatial
begin
    station_file = Base.download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/weather_files/stations.txt")
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
````

Sort stations (index) by latitude. It is useful for plotting from North to South.

````@example tuto_paper_HMMspatial
staid_lat = sortperm(station.LAT, rev=true);
nothing #hide
````

Station number

````@example tuto_paper_HMMspatial
D = length(STAID)
````

### Pairwise distance matrix

Convert LAT/LON coordinates from DMS to DD (decimal degrees).

````@example tuto_paper_HMMspatial
LAT_idx = dms_to_dd.(station.LAT)

LON_idx = dms_to_dd.(station.LON)
````

Build the pairwise distance matrix `my_distance` (in km) between stations using the haversine formula.

````@example tuto_paper_HMMspatial
my_locations = hcat(LON_idx, LAT_idx) # (D × 2) matrix: [longitude, latitude]
````

We define the haversine great-circle distance (in km) and build the $D \times D$ distance matrix.

````@example tuto_paper_HMMspatial
function haversine_km(coords1, coords2; R=6371.0)
    lon1, lat1 = deg2rad(coords1[1]), deg2rad(coords1[2])
    lon2, lat2 = deg2rad(coords2[1]), deg2rad(coords2[2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)^2 + cos(lat1) * cos(lat2) * sin(dlon / 2)^2
    return 2R * asin(sqrt(a))
end

my_distance = [haversine_km(my_locations[i, :], my_locations[j, :]) for i in 1:D, j in 1:D]
````

### Date range

````@example tuto_paper_HMMspatial
date_start = Date(1956)
````

Date including the previous days used in the initial condition (in case `local_memory > 0`)

````@example tuto_paper_HMMspatial
date_start_w_memory = date_start - Day(local_order)

date_end = Date(2020) - Day(1)

every_year = date_start:Day(1):date_end

every_year_w_memory = date_start_w_memory:Day(1):date_end

n2t = dayofyear_Leap.(every_year)

N = length(n2t)
````

### Treat data

Load into a `DataFrame` the (ECA) RR files (rain). It filters by date and valid data.
It also adds a column `:RO` for rain occurrences (0: dry, 1: wet).

````@example tuto_paper_HMMspatial
begin
    data_stations = collect_data_ECA.(STAID, date_start_w_memory, date_end, "https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/weather_files/ECA_blend_rr/RR_", portion_valid_data=1, skipto=22, header=21, url=true)
    for i = eachindex(data_stations)
        @transform!(data_stations[i], :RO = onefy.(:RR))
    end
end
````

Binary matrix version of the rain event at the `D` stations.

````@example tuto_paper_HMMspatial
Yall = BitMatrix(reduce(hcat, [data_stations[j].RO for j = 1:D]))

Y_past = BitMatrix(Yall[1:local_order, :]) # rand(Bool, local_order, D)

ξ = [1; zeros(K - 1)];  # 1 jan 1956 was most likely a type Z = 1 wet day all over France

Y = Yall[1+local_order:end, :]
````

### Map of stations

````@example tuto_paper_HMMspatial
long_spell = [longuest_spell(y) for y in eachcol(Y)]

FR_map_spell = map_with_stations(LON_idx, LAT_idx, long_spell; station_name=station_name, show_value=true, colorbar_show=true, precision_scale=precision_scale, colorbar_label="Days")
````

## Step 1: Fit the non-spatial seasonal HMM

We first fit the standard (non-spatial) autoregressive periodic HMM using the Baum-Welch EM algorithm.
The result provides the transition parameters `θᴬ` and the Bernoulli emission parameters `θᴮ`, which
serve as the starting point for the spatial extension.

### Fit slice: naive estimation

!!! note
    Before inferring the HMM parameters with the EM (Baum-Welch) algorithm, we do a first naive inference that will be used as initial condition for the EM.

The reference station `ref_station` is used to sort the hidden states obtained via the slice initialisation.
Here we choose `j=1` $\to$ `STAID=32` $\to$ `BOURGES` because it is a central station for France.

````@example tuto_paper_HMMspatial
ref_station = 1
````

This generates a random Periodic HMM that we then fit slice by slice (day by day). See paper.

````@example tuto_paper_HMMspatial
hmm_random = randARPeriodicHMM(K, T, D, local_order; ξ=ξ, ref_station=ref_station);

@time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13]);

θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, 𝐃𝐞𝐠);
nothing #hide
````

### Fit with Baum-Welch using the slice estimate as a starting point

````@example tuto_paper_HMMspatial
@time "FitMLE SHMM (Baum Welch)" hmm_ind, θᴬ_fit, θᴮ_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past, maxiter=10000, robust=true; display=:iter, silence=true, tol=1e-3, θ_iters=true, n2t=n2t);
nothing #hide
````

## Step 2: Initialise and fit the spatial HMM

### Initialise the range parameters

The range parameter trigonometric coefficients `θᴿ` are initialised with a constant $\log(R_0)$,
which corresponds to a time-invariant spatial range of $R_0$ km.

````@example tuto_paper_HMMspatial
size_degree_of_P = 2 * 𝐃𝐞𝐠 + 1

θᴿ = zeros(K, size_degree_of_P)
θᴿ[:, 1] .= log.(R0)
````

### Build the initial `ARPeriodicHMMSpatial`

The function `Trig2ARPeriodicHMMSpatial` converts trigonometric coefficient arrays into the model
matrices `A` (transition), `B` (Bernoulli probabilities) and `R` (spatial range), and returns an
`ARPeriodicHMMSpatial` object.

````@example tuto_paper_HMMspatial
hmm_spa_init = Trig2ARPeriodicHMMSpatial(fill(1 / K, K), θᴬ_fit, θᴮ_fit, θᴿ, T, my_distance)
````

### Fit the spatial HMM with EM

`fit_mle!` runs the EM algorithm for `ARPeriodicHMMSpatial`.  It updates the transition parameters
`θᴬ`, the Bernoulli emission parameters `θᴮ`, and the spatial range parameters `θᴿ` jointly.

The `solver` keyword specifies the inner optimiser used for the range update step (M-step for `R`).

````@example tuto_paper_HMMspatial
solver = OptimizationOptimJL.LBFGS(linesearch=LineSearches.BackTracking())

@time "FitMLE HMMSpa (EM)" history, all_θᴬ_iter, all_θᴮ_iter, all_θᴿ_iter = fit_mle!(hmm_spa_init, θᴬ_fit, θᴮ_fit, θᴿ, Y, Y_past; solver=solver, n2t=n2t, maxiter=100, tol=1e-3, maxiters_R=100, display=:iter, tdist=tdist, QMC_m=100);
nothing #hide
````

In classic EM algorithm, the log-likelihood is guaranteed to increase at each iteration. However, here a modified EM (see paper) and 1) increase was not demonstrated theoretically (though it might be true) 2) the spatial likelihood is estimated with a quasi-Monte Carlo method, which adds some noise to the likelihood estimation.
Therefore, it is possible to observe some non-monotonicity in the log-likelihood convergence curve.

Hence, we recover the iteration with the highest log-likelihood:

````@example tuto_paper_HMMspatial
iter_best = argmax(history["logtots"])
````

Reconstruct the best model from the stored parameter iterates:

````@example tuto_paper_HMMspatial
hmm_spa = Trig2ARPeriodicHMMSpatial(fill(1 / K, K), all_θᴬ_iter[iter_best+1], all_θᴮ_iter[iter_best+1], all_θᴿ_iter[iter_best+1], T, my_distance)
````

Run the following code to load a saved spatial HMM:
```julia
hmmspa_infos    = load(joinpath(save_tuto_path, "hmm_spa_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).jld2"))
hmm_spa         = hmmspa_infos["hmm"]
history_logtots = hmmspa_infos["logtots"]
```

### Log-likelihood convergence

````@example tuto_paper_HMMspatial
begin
    p_logtots = plot(history["logtots"], xlabel="EM iteration", ylabel="Log-likelihood", title="Convergence of the spatial HMM EM", legend=:none, lw=2)
    vline!(p_logtots, [iter_best], c=:red, s=:dash, label="best iter")
end
````

## Visualisation of the fitted model parameters

### Transition matrix

````@example tuto_paper_HMMspatial
begin
    pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing, legend_columns=4, tickfont=12, legendfontsize=16) for k in 1:K]
    for k in 1:K
        [plot!(pA[k], hmm_spa.A[k, l, :], c=my_color(l, K), label=L"Q_{%$(k)\to %$(l)}", legend=:top, lw=1.75) for l in 1:K]
        hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
        xticks!(pA[k], vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(string.(monthabbr.(1:12)), ""), xlims=(0, 367), ylims=(0, 1))
    end
    pallA = plot(pA..., size=(1000, 500))
end
````

### Spatial range $R(t)$

The seasonal range $R_k(t)$ controls the spatial correlation length for each hidden state $k$.
A larger range means stronger spatial dependence between stations.

````@example tuto_paper_HMMspatial
begin
    pr = plot(title="Spatial range per hidden state", ylabel="Range (km)", tickfont=12, legendfontsize=14)
    for k in 1:K
        plot!(pr, 1:T, hmm_spa.R[k, :], c=my_color(k, K), label=L"k = %$k", lw=2)
    end
    xticks!(pr, vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(string.(monthabbr.(1:12)), ""), xlims=(0, 367))
end
````

### Rain occurrence probabilities $\mathbb{P}(Y = \text{wet} \mid Z = k, H = h)$

````@example tuto_paper_HMMspatial
begin
    mm = 1 # h = 1: day before was dry
    jt = D
    pB = [plot(legendfont=14, title="$(station_name[j])", titlefontsize=17, tickfont=14, legendfontsize=16) for j in 1:jt]
    for j in 1:jt
        [plot!(pB[j], hmm_spa.B[k, :, j, mm], c=my_color(k, K), label=islabel(j, 3, L"\mathbb{P}(Y = \textrm{wet}\mid Z = %$k, H = \textrm{dry})"), lw=2) for k in 1:K]
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
````

### Spatial map of mean rain probability

````@example tuto_paper_HMMspatial
memory_past_cat = 1
````

Mean rain probability per hidden state $k$: $\overline{\mathbb{P}}(Y = \text{Rain}\mid Z = k, H = 1)$

````@example tuto_paper_HMMspatial
p_FR_map_mean_prob = map_with_stations(LON_idx, LAT_idx, [[mean(hmm_spa.B[k, :, j, memory_past_cat]) for j in 1:D] for k in 1:K], colorbar_show=true, colorbar_label=L"\mathbb{P}(Y = \text{Rain}\mid Z = k, H = 1)", precision_scale=precision_scale)
````

## Inference of the historical hidden states

### Viterbi algorithm

````@example tuto_paper_HMMspatial
ẑ = viterbi(hmm_spa, Y, Y_past; n2t=n2t, QMC_m=QMC_m)

ẑ_per_cat = [findall(ẑ .== k) for k in 1:K]
````

### Visualisation of the historical sequence of hidden states

````@example tuto_paper_HMMspatial
year_range = unique(year.(data_stations[1][1+local_order:end, :DATE]));

idx_year = [findall(x -> year.(x) == m, data_stations[1][1+local_order:end, :DATE]) for m in year_range];

select_year = unique(sort([4:10:length(year_range); 21; 48; 64]))

begin
    year_nb = length(select_year)
    z_hat_mat = zeros(year_nb, 366)

    for (i, y) in enumerate(select_year)
        if isleapyear(year_range[y])
            z_hat_mat[i, :] = ẑ[idx_year[y]]
        else
            z_hat_mat[i, :] = [ẑ[idx_year[y]]; 0]
        end
    end
    thick = 1
    heatmap(z_hat_mat, colorbar=:none, c=my_palette(K), minorticks=:false, framestyle=:xbox, grid=:none, thickness_scaling=thick)
    xticks!(vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(string.(monthabbr.(1:12)), ""), xlims=(0, 367), xtickfontsize=14 / thick, ytickfontsize=14 / thick)
    hline!((1:year_nb) .+ 0.5, c=:black, legend=:none, lw=4)
    ylims!(0.5, year_nb + 0.5)
    pviterbi = yticks!(1:year_nb, string.(year_range[select_year]), size=(1000, 600))
end
````

Sample the spatial HMM model and output the sequence of hidden states and multi-site dry/wet.

````@example tuto_paper_HMMspatial
Nb = 1000

begin
    zs = zeros(Int, N, Nb)
    ys = zeros(Bool, N, D, Nb)
    @time "Simulations Z, Y (spatial HMM)" for i in 1:Nb
        zs[:, i], ys[:, :, i] = rand(hmm_spa, n2t; y_ini=Yall[1:local_order, :], z_ini=1, seq=true)
    end
end
````

## The WGEN model

We will compare to the WGEN model that propose Markov chain of order 4 for rain occurrences (fitted monthly) and laten gaussian model for multisite occurrences (fitted monthly).
- Wilks, D. S. "Multisite generalization of a daily stochastic precipitation generation model". Journal of Hydrology, (1998). https://doi.org/10.1016/S0022-1694(98)00186-3.
- Srikanthan, Ratnasingham, et Geoffrey G. S. Pegram. "A nested multisite daily rainfall stochastic generation model". Journal of Hydrology 2009. https://doi.org/10.1016/j.jhydrol.2009.03.025.

````@example tuto_paper_HMMspatial
wgen_order = 4
idx_months = [findall(x -> month.(x) == m, data_stations[1][1+local_order:end, :DATE]) for m in 1:12]
wgen4_model = fit_wgen(Y, idx_months, wgen_order)

ys_wgen = similar(ys)
@time "Simulation Y wgen 4" for i in 1:Nb
    ys_wgen[:, :, i] = rand(wgen4_model, 1956:2019; Y_ini=vcat(rand(Bool, wgen_order - local_order, D), Y_past))
end
````

### Non-spatial HMM simulations

Simulate from the independent (non-spatial) `ARPeriodicHMM` fitted in Step 1, for comparison.

````@example tuto_paper_HMMspatial
ys_ind = similar(ys)
zs_ind = zeros(Int, N, Nb)
@time "Simulations Z, Y (HMM_ind)" for i in 1:Nb
    zs_ind[:, i], ys_ind[:, :, i] = rand(hmm_ind, n2t; y_ini=Yall[1:local_order, :], z_ini=1, seq=true)
end
````

## Results

### Spell distribution

`select_month` to choose the month where to compute the spell distributions (summer month, winter, etc.)
`select_month = 1:12` corresponds to all months.

````@example tuto_paper_HMMspatial
select_month = 1:12

idx_months = [findall(x -> month.(x) == m, data_stations[1][1+local_order:end, :DATE]) for m in 1:12]

idx_month_vcat = vcat(idx_months[select_month]...)

idx_all = [intersect(yea, mon) for yea in idx_year, mon in idx_months];
nothing #hide
````

##### Historic spells

````@example tuto_paper_HMMspatial
len_spell_hist = [pmf_spell(Y[idx_month_vcat, j], dw) for j in 1:D, dw in 0:1];
nothing #hide
````

##### Simulation spells

````@example tuto_paper_HMMspatial
len_spell_simu = [pmf_spell(ys[idx_month_vcat, j, i], dw) for i in 1:Nb, j in 1:D, dw in 0:1];
nothing #hide
````

#### Dry spell

````@example tuto_paper_HMMspatial
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
````

#### Wet spell

````@example tuto_paper_HMMspatial
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
````

### Seasonal areal dry spells

````@example tuto_paper_HMMspatial
RRmax = 0
ROR = [mean(r .> RRmax) for r in eachrow(Y)]
RORs = [[mean(r .> RRmax) for r in eachrow(rr)] for rr in eachslice(ys, dims=3)]
RORs_ind = [[mean(r .> RRmax) for r in eachrow(rr)] for rr in eachslice(ys_ind, dims=3)]
RORswgen = [[mean(r .> RRmax) for r in eachrow(rr)] for rr in eachslice(ys_wgen, dims=3)]

JJA = [6, 7, 8]
MAM = [3, 4, 5]
SON = [9, 10, 11]
DJF = [12, 1, 2]
SEASONS = [DJF, MAM, JJA, SON]
seasonname = ["DJF", "MAM", "JJA", "SON"]

idx_seasons = [findall(month.(data_stations[1][1+local_order:end, :DATE]) .∈ tuple(season)) for season in SEASONS]

let
    perc = 0.1
    QQ = [5, 95]

    p_spell_rors = [plot(ylims=(5e-4, 1e-0), xlims=(-0.01,25), tickfont=11, legendfontsize=13, legend=:left) for i in eachindex(idx_seasons)]
    xlabel!.(p_spell_rors[3:end], "Nb of days", xlabelfontsize=12)
    ylabel!.(p_spell_rors[[1, 3]], "PMF", ylabelfontsize=12)
    for m in eachindex(idx_seasons)
        len_ror_hist = pmf_spell(ROR[idx_seasons[m]] .≤ perc, 1)
        len_ror_simu = [pmf_spell(RORs[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]
        len_ror_simu_ind = [pmf_spell(RORs_ind[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]
        len_ror_simuwgen = [pmf_spell(RORswgen[i][idx_seasons[m]] .≤ perc, 1) for i in 1:Nb]

        errorlinehist!(p_spell_rors[m], [len_ror_hist], groupcolor=:blue, lw=2, norm=:probability, bins=make_range(len_ror_hist), errortype=:percentile,
        label=label = islabel(m, 1, "Obs"),
        legend=:bottom)
        yaxis!(:log10)

        sim_range = make_range(reduce(vcat, len_ror_simuwgen))
        errorlinehist!(p_spell_rors[m], len_ror_simuwgen, groupcolor=:green, legend=:topright,
            label=islabel(m, 1, "WGEN 4"),
            norm=:probability, bins=sim_range, errortype=:percentile, percentiles=QQ, fillalpha=0.25, centertype=:median, linewidth=2)

        sim_range = make_range(reduce(vcat, len_ror_simu_ind))
        errorlinehist!(p_spell_rors[m], len_ror_simu_ind, groupcolor=:grey, legend=:topright,
            label=islabel(m, 1, "HMM_ind"),
            norm=:probability, bins=sim_range, errortype=:percentile, percentiles=QQ, fillalpha=0.25, centertype=:median, linewidth=2)

        sim_range = make_range(reduce(vcat, len_ror_simu))
        errorlinehist!(p_spell_rors[m], len_ror_simu, groupcolor=:orange, legend=:topright,
            label=islabel(m, 1, "HMM_spa"),
            norm=:probability, bins=sim_range, errortype=:percentile, percentiles=QQ, fillalpha=0.3, centertype=:median, alpha=1, linewidth=2)
        annotate!(p_spell_rors[m], median(sim_range), 1.5, seasonname[m])
        yticks!(10.0 .^ (-4:-0))
    end

    pall = plot(p_spell_rors..., layout=(2, 2), size=(1000, 600), top_margin=0.34cm, left_margin=0.3cm, bottom_margin=0.22cm)
    file_name = "ROR_spell_season_perc_$(perc)_Q_$(QQ[1])_$(QQ[2])_no_inset"
    file_name = replace(file_name, "." => "p")
    pall
end
````

### Correlations

##### Rain event dry/wet

````@example tuto_paper_HMMspatial
cor_bin_hist = cor(reduce(hcat, [df.RO for df in data_stations]));

cor_bin_mean_simu = mean(cor(ys[:, :, i]) for i in 1:Nb);

cor_bin_mean_simu_ind = mean(cor(ys_ind[:, :, i]) for i in 1:Nb);

cor_bin_mean_simu_wgen = mean(cor(ys_wgen[:, :, i]) for i in 1:Nb);

begin
    plots_cor_bin = [plot(-0.1:0.1:0.8, -0.1:0.1:0.8, aspect_ratio=true, label=:none, xlabelfontsize=16, ylabelfontsize=16, tickfont=11, legendfontsize=13) for _ in 1:1]
    scatter!(plots_cor_bin[1], vec_triu(cor_bin_hist), vec_triu(cor_bin_mean_simu), label="Correlations vs SHHMM-Spa", xlabel="Observations", ylabel="Simulations", c=:orange)
    # scatter!(plots_cor_bin[1], vec_triu(cor_bin_hist), vec_triu(cor_bin_mean_simu_ind), label="Correlations vs HMM", c=:gray)
    # scatter!(plots_cor_bin[1], vec_triu(cor_bin_hist), vec_triu(cor_bin_mean_simu_wgen), label="Correlations vs WGEN", c=:green)
    [xlims!(plots_cor_bin[i], -0.1, 1) for i in 1:1]
    [ylims!(plots_cor_bin[i], -0.1, 1) for i in 1:1]
    annotate!(0.2, 0.7, "MSE ≃ $(round(mean(abs2, vec_triu(cor_bin_hist) - vec_triu(cor_bin_mean_simu)), digits = 4))")
    plot_cor_bin = plot(plots_cor_bin...)
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

