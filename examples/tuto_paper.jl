using Markdown#hide
cd(@__DIR__)#hide

md"""
# Multisite daily Stochastic Weather Generator
"""

md"""
SWG as described in Métivier, Gobet and Parey
"""
md"""
## Set up
"""

md"""
### Package and functions
"""

using CSV, JLD, DelimitedFiles # File Read/Load/Save

using DataFrames, DataFramesMeta # DataFrames

using Dates

using StatsBase, Random

using Distributions

using StatsPlots, LaTeXStrings
using Plots.PlotMeasures # To play with margin in Plots

md"""
The two main package are not yet registered to the official Julia registry. 
They can be either `add`ed through [my local Julia registry](https://github.com/dmetivie/MyJuliaRegistry) with the [LocalRegistry.jl](https://github.com/GunnarFarneback/LocalRegistry.jl).
Or simply `add`ed via url i.e.
```julia
import Pkg
Pkg.add("https://github.com/dmetivie/SmoothPeriodicStatsModels.jl")
Pkg.add("https://github.com/dmetivie/StochasticWeatherGenerators.jl")
```
"""

using SmoothPeriodicStatsModels # Name might change. Small collection of smooth periodic models e.g. AR, HMM

using StochasticWeatherGenerators # interface to use with SmoothPeriodicStatsModels.jl

#-

Random.seed!(1234)

md"""
### Settings for plotting

Some settings to have nice plots and save cropped version of plots (necessary in some cases).
"""

gr() # plotly() # for interactive plots
default(thickness_scaling=1.2, fontfamily="Computer Modern", linewidth=2, label=nothing, size=(1000, 600))
scalefontsizes(1.5)
cur_colors = get_color_palette(:auto, 100);
my_palette(K) = palette(vcat(cur_colors[1], [cur_colors[c] for c in 3:4], cur_colors[2]), K)

file_for_plot_utilities = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/plot_utilities.jl")
include(file_for_plot_utilities)

md"""
For map plot, we use `GeoMakie.jl` + a hack with `NaturalEarth.jl`. This is quite experimental.
I used `cartopy` before with `PyCall.jl` which works very well.
"""

file_for_maps_with_geomakie = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/geo_makie_features.jl") # download file from a GitHub repo
include(file_for_maps_with_geomakie)

md"""
### Data files
"""

md"""
### Global Parameters
"""

md"""
Number of day in a year (choice here is 366)
"""

T = 366

md"""
Define the French area for map (Longitude and latitudes) plot and the precision of the map `precision_scale`
"""

precision_scale = "50m"

LON_min = -5 # West

LON_max = 10 # East

LAT_min = 41 # South

LAT_max = 52 # North

md"""
`conversion_factor` for rain amounts `RR` in 0.1 mm to mm 
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
It could be an array different for each station and variables. Not implemented yet though.
"""

𝐃𝐞𝐠 = 2

md"""
Local memory order i.e. at station $j$, $\mathbb{P}(Y_n^{(j)} = y_n^{(j)} \mid Z = k, Y_{n-1}^{(j)} = y_{n-1}^{(j)}, \cdots, Y_{n-\texttt{local memory}}^{(j)} = y_{n-\texttt{local memory}}^{(j)})$
"""

local_order = 1

md"""
!!! note  
	The `local_order` could be a vector/matrix of size `D` and different for each station, and also different depending on wet or dry past. Indeed it has been noted, TODO add ref, that dry spells have longer memory.
"""

size_order = 2^local_order

println("K = $K, ", "local_order = $local_order, ", "degree = $𝐃𝐞𝐠")

md"""
## Data
"""

md"""
### Select relevant stations from the `station.txt` file
"""

md"""
Here we
- Remove white space at the right of the CN, STANAME which is caused by imperfect CVS importation
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
md"""
!!! note
    You can change the selected stations. However, keep in mind that for the model to work, the conditional independence hypothesis must holds between stations i.e. $\mathbb{P}(Y_1 = y_1, \cdots, Y_S = y_s\mid Z = k) = \prod_{s=1}^S \mathbb{P}(Y_s = y_s)$.
    Hence stations must be sufficiently far apart.
"""

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
Load into DataFrames the (ECA) RR files (rain). It filters by date and valid data.
It also add a column of rain event (0: dry, 1: wet).
"""

begin
    data_stations = collect_data_ECA.(STAID, date_start_w_memory, date_end, "https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/weather_files/ECA_blend_rr/RR_", portion_valid_data=1, skipto=22, header=21, url=true)
    for i = eachindex(data_stations)
        @transform!(data_stations[i], :bin = onefy.(:RR))
    end
end


md"""
Binary matrix version of rain event at the `D` stations.
"""

Yall = BitMatrix(reduce(hcat, [data_stations[j].bin for j = 1:D]))

Y_past = BitMatrix(Yall[1:local_order, :]) # rand(Bool, local_order, D)

ξ = [1; zeros(K - 1)];  # 1 jan 1956 was most likely a type Z = 1 wet day all over France

Y = Yall[1+local_order:end, :]

md"""
### Map of stations
"""

md"""
Convert LAT DMS into DD which seems most widly accepted format.
"""

LAT_idx = dms_to_dd.(station.LAT)

LON_idx = dms_to_dd.(station.LON)

long_spell = [longuest_spell(y) for y in eachcol(Y)]


map_with_stations(LON_idx, LAT_idx, long_spell; station_name=station_name, show_value=true, colorbar_show=true)

md"""
## Fit the seasonal HMM
"""

md"""
### Fit slice: naive estimation
"""

md"""
!!! note
    Before inferring the HMM parameters with EM (Baum-Welch) algorithm, we do a first naive inference that will be used as initial condition for the EM.
"""

md"""
Reference station `ref_station` used to sort hidden state for the slide initialization
Here we choose `j=1` $\to$ `STAID=32` $\to$ `BOURGES` because it is a central station for France
"""

ref_station = 1

md"""
This generate a random Periodic HMM that we then fit slice by slice (day by day). See paper.
"""
hmm_random = randhierarchicalPeriodicHMM(K, T, D, local_order; ξ=ξ, ref_station=ref_station);

@time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13]);

θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, 𝐃𝐞𝐠);

md"""
### Fit with Baum Welch using slice estimate as starting point

With the Slice estimate as a good starting point for the full (seasonal) Baum Welch EM algorithm we fit the model!
"""

@time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past,
    maxiter=10000, robust=true; display=:iter, silence=true, tol=1e-3, θ_iters=true, n2t=n2t);
## On my computer
## Iteration 73: logtot = -116791.100655, max(|θᴬᵢ-θᴬᵢ₋₁|) = 0.0002 & max(|θᴮᵢ-θᴮᵢ₋₁|) = 0.00103
## EM converged in 73 iterations, logtot = -116791.10065504618
## FitMLE SHMM (Baum Welch): 36.161685 seconds (185.76 M allocations: 32.581 GiB, 6.77% gc time, 10.09% compilation time)

save_tuto_path = "../../assets/tuto_1"

# save(joinpath(save_tuto_path,"hmm_fit_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).jld"), "hmm", hmm_fit, "hist", hist, "Q_param", θq_fit, "Y_param", θy_fit)

md"""
Uncomment to load previously computed hmm
```julia
# hmm_infos = load("save_tuto_path/hmm_fit.jld")
# hmm_fit = hmm_infos["hmm"]
# hist = hmm_infos["hist"]
# θq_fit = hmm_infos["Q_param"]
# θy_fit = hmm_infos["Y_param"]
```
"""

md"""
### Visualisation of the HMM parameters
"""

md"""
#### Transition matrix
"""


begin
    pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing) for k in 1:K]
    for k in 1:K
        [plot!(pA[k], hmm_fit.A[k, l, :], c=my_color(l, K), label=L"Q_{%$(k)\to %$(l)}", legend=:topleft) for l in 1:K]
        hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
        xticks!(pA[k], vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(string.(monthabbr.(1:12)), ""), xlims=(0, 367), xtickfontsize=10, ylims=(0, 1))
    end
    pallA = plot(pA..., size=(1000, 500))
end
savefig(pallA, joinpath(save_tuto_path,"Q_transition_memo_1_K_4_d_2.pdf"))

md"""
#### Rain probabilities
"""

begin
    mm = 1
    jt = D
    pB = [plot(legendfont=14, title="$(station_name[j])", titlefontsize=16) for j in 1:jt]
    for j in 1:jt
        [plot!(pB[j], succprob.(hmm_fit.B[k, :, j, mm]), c=my_color(k, K), label=:none) for k in 1:K]
        hline!(pB[j], [0.5], c=:black, label=:none, s=:dot)
        xticks!(
            pB[j],
            vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366),
            vcat(string.(monthabbr.(1:12)), ""), xtickfontsize=10
        )
        xlims!(pB[j], (0, 367))
        ylims!(pB[j], (0, 1))
    end
    pallB = plot(pB[staid_lat]..., size=(3000 / 1.25, 1000 / 1.25), layout=(2, 5))
end
savefig(pallB, joinpath(save_tuto_path,"proba_rain_all_station.pdf"))

md"""
#### Spatial Rain probability 
"""

memory_past_cat = 1

md"""
h = 1 (day before dry) or 2 (day before wet)
$\mathbb{P}(Y = \text{Rain}\mid Z = k, H = h)$ with h = $(memory_past_cat)
"""

p_FR_map_mean_prob = map_with_stations(LON_idx, LAT_idx, [[mean(succprob.(hmm_fit.B[k, :, j, memory_past_cat])) for j in 1:length(STAID)] for k in 1:K], colorbar_show=true)

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

## CSV.write(joinpath(save_tuto_path,"z_hat_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order).csv"), DataFrame([:DATE, :z] .=> [data_stations[1].DATE[1+local_order:end], ẑ]))

md"""
#### Visualisation of the Historical sequences of hidden states
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
    pviterbi = yticks!(1:year_nb, string.(year_range[select_year]))
end
savefig(pviterbi, joinpath(save_tuto_path,"temporal_1959_2009.pdf"))

md"""
## Adding Rain amounts to the model
"""

md"""
### Marginal fit

We fit the marginals at each station independently. 
We use a mixture of exponential functions whose parameters evolve smoothly and periodically
TODO: put equation
"""
@time "FitMLE RR" mix_allE = fit_mle_RR.(data_stations_z, K, local_order, mix₀=StochasticWeatherGenerators.mix_ini(T))
## FitMLE RR: 66.104980 seconds (339.13 M allocations: 47.931 GiB, 5.53% gc time, 4.18% compilation time)

save(joinpath(save_tuto_path, "rain_mix.jld"), "mix2Exp", mix_allE)

md"""
Note that we don't need anymore to fit quantile functions, as [Distributions.jl PR #1389 (September 2nd, 2021)](https://github.com/JuliaStats/Distributions.jl/pull/1389) handles that.
I did my approach (to save interpolate quantile) few months prior to this PR. It would have saved me some times!
"""

md"""
### Rain correlations

We fit a Gaussian copula to each pair of stations for joint rainy days only.
"""

md"""
!!! caution
    When the number of hidden states is getting larger, it migth happen that for some pair of stations there are no simulteneous rain occurence in a rain category $Z = k$.
    In that case a `missing` coefficient is returned.
"""

begin
    Σ²RR = cov_RR(data_stations_z, K)
    if K == 4
        Σ²RR[2][6, 3] = Σ²RR[4][6, 3]
        Σ²RR[2][3, 6] = Σ²RR[4][6, 3]
    end
    Σ²RR = convert.(Matrix{Float64}, Σ²RR)
end

if K == 4
    @warn "For Embrun j=6 and Marignane j=3 the hidden state Z=2 and Z=4 are pretty similar (dry), so we replace the `missing` coefficient of Z=2 with the one of Z = 4"
end

md"""
## Simulation

Now we are ready to generate samples from the SWG model.
"""

md"""
`Nb` is the number of realization
"""
Nb = 1000

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
## Simulations Z, Y: 34.998679 seconds (328.41 M allocations: 32.166 GiB, 8.24% gc time, 1.16% compilation time)

md"""
Given the hidden states and dry/wet, it generates the rain amounts at each stations (correlated with a Gaussian Copula).
"""

begin
    rs = zeros(D, N, Nb)
    @time "Simulations RR" for i in 1:Nb
        rs[:, :, i] = rand_RR(mix_allE, n2t, zs[:, i], ys[:, :, i]', Σ²RR)
    end
end
## Simulations RR: 164.912113 seconds (299.73 M allocations: 43.020 GiB, 2.67% gc time, 0.54% compilation time)

md"""
## Results
"""

md"""
### Spell distribution

`select_month` to chose the month where to compute the spells distributions (summer month, winter, etc.)
`select_month = 1:12` corresponds to all month.
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
    dw_dry = 1 # dry
    p_spell_dry = [plot(ylims=(1e-4, 1e-0), ytickfontsize=9, xtickfontsize=10) for j = 1:D]
    for j = 1:D
        all_spells = len_spell_simu[:, j, dw_dry]
        errorlinehist!(p_spell_dry[j], all_spells, groupcolor=:grey, legend = :topright, label=islabel(j, staid_lat[[1]], L"Simu $q_{0,100}$"), norm=:probability, bins=make_range(reduce(vcat,all_spells)), errortype = :percentile, percentiles = [0,100], fillalpha = 0.4, centertype = :median)

        errorlinehist!(p_spell_dry[j], all_spells, groupcolor=:red, label=islabel(j, staid_lat[[1]], L"Simu $q_{25,75}$"), norm=:probability, bins=make_range(reduce(vcat,all_spells)), errortype = :percentile, percentiles = [25,75], fillalpha = 0.5, centertype = :median)

        histo_spell = len_spell_hist[j, dw_dry]
        errorlinehist!(p_spell_dry[j], [histo_spell], label=islabel(j, staid_lat[[1]], "Obs"), groupcolor=:blue, lw=1.5, norm=:probability, bins=make_range(histo_spell), errortype = :percentile)
        xlims!(p_spell_dry[j], 0, 2 + maximum(1.5maximum.(histo_spell)))
        yaxis!(:log10)
    end

    [xlabel!(p_spell_dry[j], "Nb of days", xlabelfontsize=10) for j in staid_lat[6:10]]
    [ylabel!(p_spell_dry[j], "PMF", ylabelfontsize=10) for j in staid_lat[[1, 6]]]
    [title!(p_spell_dry[j], station_name[j], titlefontsize=13) for j = 1:D]
    pall_spell_dry = plot(p_spell_dry[staid_lat]..., size=(3000 / 2.5, 1000 / 1.5), layout=(2, 5))
end
savefig(joinpath(save_tuto_path,"spell_steppost_dry_c1.pdf"))

md"""
#### Wet spell
"""

begin
    dw_dry = 2 # wet
    p_spell_dry = [plot(ylims=(1e-4, 1e-0), ytickfontsize=9, xtickfontsize=10, legendfontsize = 11) for j = 1:D]
    for j = 1:D
        all_spells = len_spell_simu[:, j, dw_dry]
        errorlinehist!(p_spell_dry[j], all_spells, groupcolor=:grey, legend = :topright, label=islabel(j, staid_lat[[1]], L"Simu $q_{0,100}$"), norm=:probability, bins=make_range(reduce(vcat,all_spells)), errortype = :percentile, percentiles = [0,100], fillalpha = 0.4, centertype = :median)

        errorlinehist!(p_spell_dry[j], all_spells, groupcolor=:red, label=islabel(j, staid_lat[[1]], L"Simu $q_{25,75}$"), norm=:probability, bins=make_range(reduce(vcat,all_spells)), errortype = :percentile, percentiles = [25,75], fillalpha = 0.5, centertype = :median)

        histo_spell = len_spell_hist[j, dw_dry]
        errorlinehist!(p_spell_dry[j], [histo_spell], label=islabel(j, staid_lat[[1]], "Obs"), groupcolor=:blue, lw=1.5, norm=:probability, bins=make_range(histo_spell), errortype = :percentile)
        xlims!(p_spell_dry[j], 0, 2 + maximum(1.5maximum.(histo_spell)))
        yaxis!(:log10)
    end

    [xlabel!(p_spell_dry[j], "Nb of days", xlabelfontsize=10) for j in staid_lat[6:10]]
    [ylabel!(p_spell_dry[j], "PMF", ylabelfontsize=10) for j in staid_lat[[1, 6]]]
    [title!(p_spell_dry[j], station_name[j], titlefontsize=13) for j = 1:D]
    pall_spell_dry = plot(p_spell_dry[staid_lat]..., size=(3000 / 2.5, 1000 / 1.5), layout=(2, 5))
end
savefig(joinpath(save_tuto_path,"spell_steppost_wet_c1.pdf"))

md"""
### Rain
"""

md"""
#### Interpretation: Mean Rain per weather regime $R > 0 \mid Z = k$.

We plot the empirical (strictly) positive **mean** rain amounts per weather regime. The results are smoothed using a `cyclic_moving_average` with a time window of $\pm 15$ days and the Epanechnikov kernel.
"""

begin
    p_rainpercat = [plot(ytickfontsize=12, xtickfontsize=12) for j = 1:D]
    for j = 1:D
        df_j = @chain data_stations_z[j] begin
            dropmissing
            @transform(:day = dayofyear_Leap.(:DATE))
            @subset(:RR .> 0)
            @by([:day, :z], :MEAN_RR = mean(:RR))
            groupby(:z)
        end
        ## Uncomment to see how the double exponential mixtures compares to the empirical data.
        ## [plot!(p_rainpercat[j], 1:T, t -> conversion_factor * mean(mix_allE[j][k, t]), label=:none, c=my_color(k, K), lw=1.5, legend = :topleft) for k in 1:K]
        for k in 1:K
            cycle_avg = replace(cyclic_moving_average(df_j[k].MEAN_RR, df_j[k].day, T,  15), 0=>missing)
            @df df_j[k] plot!(p_rainpercat[j], 1:T, conversion_factor * cycle_avg, c=my_color(k, K), alpha = 1, label=islabel(j, staid_lat[[4]], L"Z = %$k"), lw=1.5)
        end
        ylims!(p_rainpercat[j], 0, Inf)
    end
    [ylabel!(p_rainpercat[j], L"Rain ($\mathrm{mm/m}^2$)", ylabelfontsize=14) for j in staid_lat[[1, 6]]]
    [xticks!(
        p_rainpercat[j],
        vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366),
        vcat(string.(first.(string.(monthabbr.(1:12))))), xtickfontsize=10
    ) for j in 1:D]
    [title!(p_rainpercat[j], station_name[j], titlefontsize=14) for j = 1:D]
    plt_rain_cat_mix = plot(p_rainpercat[staid_lat]..., size=(3000 / 2.05, 1000 / 1.85), layout=(2, 5), left_margin = 20px)
end

savefig(plt_rain_cat_mix, joinpath(save_tuto_path,"mean_positive_rain_per_cat_from_mixture.pdf"))

md"""
#### Univariate Rain distributions
"""

md"""
Historical vs $(Nb) simulations distribution
"""

begin
    p_uniR = [plot(yaxis=:log10, ylims=(1e-4, 1e-0), ytickfontsize=9, xtickfontsize=10, legendfontsize = 11) for j = 1:D]
    for j = 1:D
        dists_RR_positive_j = conversion_factor * [filter(!iszero, rs[j, :, i]) for i in 1:Nb]
        errorlinehist!(p_uniR[j], dists_RR_positive_j, groupcolor=:grey, legend = :topright, label=islabel(j, staid_lat[[1]], L"Simu $q_{0,100}$"), norm=:pdf, errortype = :percentile, percentiles = [0,100], fillalpha = 0.4, centertype = :median)

        errorlinehist!(p_uniR[j], dists_RR_positive_j, groupcolor=:red, label=islabel(j, staid_lat[[1]], L"Simu $q_{25,75}$"), norm=:pdf, errortype = :percentile, percentiles = [25,75], fillalpha = 0.5, centertype = :median)

        errorlinehist!(p_uniR[j], [conversion_factor * filter(!iszero, data_stations[j].RR)], label=islabel(j, staid_lat[[1]], "Obs"), groupcolor=:blue, lw=1.5, norm=:pdf, errortype = :percentile)

        xlims!(p_uniR[j], 0.0, Inf)
    end
    [plot!(p_uniR[j], xlabel=L"Rain (mm/m$^2$)", xlabelfontsize=10) for j in staid_lat[6:10]]
    [plot!(p_uniR[j], ylabel="PDF", ylabelfontsize=10) for j in staid_lat[[1, 6]]]

    [title!(p_uniR[j], station_name[j], titlefontsize=13) for j = 1:D]

    pall_R = plot(p_uniR[staid_lat]..., size=(3000 / 2.5, 1000 / 1.5), layout=(2, 5), bottom_margin = 3px)
end
savefig(pall_R, joinpath(save_tuto_path, "dist_R_positive.pdf"))

md"""
#### Monthly quantile amount
"""

rh = reduce(hcat, [df[1+local_order:end, :RR] for df in data_stations])

month_rain_simu = [cum_monthly(rs[j, :, i], idx_all) for j in 1:D, i in 1:Nb];

month_rain_histo = [cum_monthly(rh[:, j], idx_all) for j in 1:D]

qs = [0.9, 0.5, 0.1]

@time "Plot monthly quantile" begin
    p_month_RR = [scatter(ytickfontsize=11) for j = 1:D]
    for j = 1:D
        for (α, per) in enumerate([[0, 100], [25, 75]])
            for (cc, q) in enumerate(qs)
                errorline!(p_month_RR[j], [quantile(month_rain_simu[j, i][:, m], q) * conversion_factor for m in 1:12, i in 1:Nb], label=:none, fillalpha=0.18 * α^2, centertype=:median, errortype=:percentile, percentiles=per, groupcolor=my_palette(length(qs))[cc])
            end
        end
        for q in qs
            scatter!(p_month_RR[j], m -> quantile(month_rain_histo[j][:, m], q) * conversion_factor, 1:12, label=:none, ms=2, c=:blue)
            plot!(p_month_RR[j], m -> quantile(month_rain_histo[j][:, m], q) * conversion_factor, 1:12, label=:none, c=:blue)
        end
        xticks!(p_month_RR[j], 1:12, string.(first.(monthabbr.(1:12))), xtickfontsize=8)
    end
    [ylabel!(p_month_RR[j], "Rain (mm/month)", ylabelfontsize=12) for j in staid_lat[[1, 6]]]

    [title!(p_month_RR[j], station_name[j], titlefontsize=12) for j = 1:D]
    pall_month_RR = plot(p_month_RR[staid_lat]..., size=(1190, 500), layout=(2, 5))
end
savefigcrop(pall_month_RR, joinpath(save_tuto_path, "EDF_like_$(Nb)_simu_monthly_quantile_K_$(K)_d_$(𝐃𝐞𝐠)_m_$(local_order)"))

md"""
### Correlations
"""

md"""
##### Rain event dry/wet
"""

cor_bin_hist = cor(reduce(hcat, [df.bin for df in data_stations]));

cor_bin_mean_simu = mean(cor(ys[:, :, i]) for i in 1:Nb);


begin
    plots_cor_bin = [plot(-0.1:0.1:0.8, -0.1:0.1:0.8, aspect_ratio=true, label=:none, xlabelfontsize=16, ylabelfontsize=16, xtickfontsize=10, ytickfontsize=10) for _ in 1:1]
    scatter!(plots_cor_bin[1], vec_triu(cor_bin_hist), vec_triu(cor_bin_mean_simu), label="Correlations", xlabel="Observations", ylabel="Simulations", c=2)
    [xlims!(plots_cor_bin[i], -0.1, 1) for i in 1:1]
    [ylims!(plots_cor_bin[i], -0.1, 1) for i in 1:1]
    annotate!(0.2, 0.7, "MSE ≃ $(round(mean(abs2, vec_triu(cor_bin_hist) - vec_triu(cor_bin_mean_simu)), digits = 4))")
    plot_cor_bin = plot(plots_cor_bin...)
end
savefigcrop(plot_cor_bin, joinpath(save_tuto_path, "full_cor_binary_hist_vs_1000_mean_simu"))

md"""
The largest pair correlation error for rain occurence comes from the pair $(station_name[findmax(cor_bin_hist - cor_bin_mean_simu)[2][1]]) and $(station_name[findmax(cor_bin_hist - cor_bin_mean_simu)[2][2]])
"""

md"""
##### Rain amount
"""

cor_hist = cor(reduce(hcat, [df.RR for df in data_stations]));

corT_hist = corTail(reduce(hcat, [df.RR for df in data_stations]));

cor_mean_simu = mean(cor(rs[:, :, i]') for i in 1:Nb);

corT_mean_simu = mean(corTail(rs[:, :, i]') for i in 1:Nb);

begin
    plots_cor = [plot(-0.1:0.1:0.8, -0.1:0.1:0.8, aspect_ratio=true, label=:none, xlabelfontsize=16, ylabelfontsize=16, xtickfontsize=10, ytickfontsize=10) for _ in 1:2]
    scatter!(plots_cor[1], vec_triu(cor_hist), vec_triu(cor_mean_simu), label="Correlations", xlabel="Observations", ylabel="Simulations", c=2)
    annotate!(plots_cor[1], 0.3, 0.7, "MSE ≃ $(round(mean(abs2, vec_triu(cor_hist) - vec_triu(cor_mean_simu)), digits = 4))")

    scatter!(plots_cor[2], vec_triu(corT_hist), vec_triu(corT_mean_simu), label="Tail index", xlabel="Observations", ylabel="Simulations", c=3)
    annotate!(plots_cor[2], 0.3, 0.7, "MSE ≃ $(round(mean(abs2, vec_triu(corT_hist) - vec_triu(corT_mean_simu)), digits = 4))")

    [xlims!(plots_cor[i], -0.1, 1) for i in 1:2]
    [ylims!(plots_cor[i], -0.1, 1) for i in 1:2]
    plot(plots_cor...)
end
savefigcrop(plots_cor[1], joinpath(save_tuto_path, "full_cor_hist_vs_1000_mean_simu"))
savefigcrop(plots_cor[2], joinpath(save_tuto_path, "full_corT_hist_vs_1000_mean_simu"))

md"""
The largest pair correlation error for rain (zero and non zero amounts) comes from the pair $(station_name[findmax(cor_hist - cor_mean_simu)[2][1]]) and $(station_name[findmax(cor_hist - cor_mean_simu)[2][2]])
"""

md"""
##### Gaussian copula hypothesis

For a pair of stations, we transform the marginal $R_s>0$ to $\mathcal{N}(0,1)$. We compare the obtained bi-variate Normal distribution to the theoritical $\chi(2)$-distriubtion.
"""

begin
    j1 = 1
    j2 = 5
    plt_qqp_copula = plot(0:5, 0:5, aspect_ratio = :equal, legendfontsize = 14, c=:black)
    for k in 1:K
    df_12 = leftjoin(data_stations_z[j1], data_stations_z[j2], on = :DATE, makeunique = true)
    df_X = @chain df_12 begin
        @subset(:RR .> 0, :RR_1 .> 0, :z .== k)
        dropmissing
        @aside cdf_ = ecdf(_.RR)
        @aside cdf_1 = ecdf(_.RR_1)
        @transform(:X = quantile(Normal(), cdf_(:RR)*length(:RR)/(1+length(:RR))), 
                    :X_1 = quantile(Normal(), cdf_1(:RR_1)*length(:RR_1)/(1+length(:RR_1))))
    end
    X = hcat(df_X.X, df_X.X_1)
    Σ⁻¹ = inv(cov(X))    
    X2 = [sqrt(x' * Σ⁻¹ * x) for x in eachrow(X)] |> sort
    ecdfX2 = ecdf(X2)(X2)*length(X2)/(length(X2)+1)

    plot!(quantile(Chi(2),ecdfX2), X2, xlabel = L"$\chi (2)$-quantile",c = my_color(k, K), ylabel = "Observed Mahalanobis distance", label = L"Z = %$k ", legend = :topleft, lw = 2)
end
    title!("$(station_name[j1]) vs $(station_name[j2])")
    xlims!(0,4.5)
    ylims!(0,4.5)
end

savefigcrop(plt_qqp_copula, joinpath(save_tuto_path, "qq_copula_$(station_name[j1])_$(station_name[j2])_Z_full"))
