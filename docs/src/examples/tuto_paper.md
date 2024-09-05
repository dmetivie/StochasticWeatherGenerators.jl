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
[<img src="https://github.com/dmetivie/StochasticWeatherGenerators.jl/assets/46794064/5fe1d677-877d-4fd5-83ac-29d30f728ca5" width="75%" alt = "Schematic of the Autoregressive Seasonal Hidden Markov Model"/>](https://github.com/dmetivie/StochasticWeatherGenerators.jl/assets/46794064/5fe1d677-877d-4fd5-83ac-29d30f728ca5)
In the paper, the model is also used with Climate Change RCP scenarios (not shown here).

## Set up

### Package and functions

!!! note "For Julia new user"
    There are several ways to `add` a package before `using`, one way is for this tutorial to copy-paste (it might take a while):
    ```julia
    import Pkg
    Pkg.add(["CSV", "JLD", "DelimitedFiles", "DataFrames", "DataFramesMeta", "StatsBase", "Random", "Distributions", "StatsPlots", "LaTeXStrings"])
    ```

````@example tuto_paper
using CSV, JLD, DelimitedFiles # File Read/Load/Save

using DataFrames, DataFramesMeta # DataFrames

using Dates

using StatsBase, Random

using Distributions
````

The two main packages for this tutorial are not yet registered in the official Julia registry, since they are not quite fully ready.
They can be either `add`ed through [my local Julia registry](https://github.com/dmetivie/LocalRegistry) with the [LocalRegistry.jl](https://github.com/GunnarFarneback/LocalRegistry.jl) package i.e.
```julia
using LocalRegistry
using Pkg
pkg"registry add https://github.com/dmetivie/LocalRegistry"
Pkg.add("SmoothPeriodicStatsModels")
Pkg.add("StochasticWeatherGenerators")
```

Or directly on the master branch with `add`ed via url i.e.

```julia
import Pkg
Pkg.add(url = "https://github.com/dmetivie/SmoothPeriodicStatsModels.jl")
Pkg.add(url = "https://github.com/dmetivie/StochasticWeatherGenerators.jl")
```

````@example tuto_paper
using SmoothPeriodicStatsModels # Name might change. Small collection of smooth periodic models e.g. AR, HMM

using StochasticWeatherGenerators # interface to use with SmoothPeriodicStatsModels.jl
````

````@example tuto_paper
Random.seed!(1234)
````

### Settings for plotting

Some settings and packages to have nice plots.

````@example tuto_paper
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
Note that using `cartopy` with `PyCall.jl` also works very well.

For the following code to work you will need to add the following packages
```julia
import Pkg
Pkg.add("NaturalEarth", "GeoMakie", "CairoMakie")
```

````@example tuto_paper
file_for_maps_with_geomakie = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/utilities_geo_makie_features.jl") # download file from a GitHub repo
include(file_for_maps_with_geomakie)
````

### Global Parameters

Number of days in a year (choice here is 366)

````@example tuto_paper
T = 366
````

Define the French area for map (Longitude and latitude) plot and the precision of the map `precision_scale`

````@example tuto_paper
precision_scale = 50 # meter

LON_min = -5 # West

LON_max = 10 # East

LAT_min = 41 # South

LAT_max = 52 # North
````

`conversion_factor` for rain amounts `RR` in 0.1 mm to mm

````@example tuto_paper
conversion_factor = 0.1 # 0.1 mm -> mm
````

## HMM Hyperparameters

Number of hidden states

````@example tuto_paper
K = 4

my_pal = my_palette(K); # just colors I like for plotting weather regime!
nothing #hide
````

Degree `𝐃𝐞𝐠` of the trigonometric expansion
It could be an array different for each station and variable. Not implemented yet though.

````@example tuto_paper
𝐃𝐞𝐠 = 2
````

Local memory order i.e. at station $j$, $\mathbb{P}(Y_n^{(j)} = y_n^{(j)} \mid Z = k, Y_{n-1:n-\texttt{local memory}}^{(j)} = y_{n-1:n-\texttt{local memory}}^{(j)})$

````@example tuto_paper
local_order = 1
````

!!! note
    The `local_order` could be a vector/matrix of size `D` and different for each station, and also different depending on wet or dry past. Not yet implemented.

````@example tuto_paper
size_order = 2^local_order

println("K = $K, ", "local_order = $local_order, ", "degree = $𝐃𝐞𝐠")
````

## Data

### Select relevant stations from the `station.txt` file

Here we
- Remove white space at the right of the `CN`, STANAME which is caused by imperfect CVS importation
- Select only the stations with 100% valid data for the period `Date(1955,12,31) .≤ :DATE .≤ Date(2019,12,31)`
- Shorten station names

````@example tuto_paper
begin
    station_file = Base.download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/weather_files/stations.txt")
    station_all = CSV.read(station_file, DataFrame, header=18, normalizenames=true, ignoreemptyrows=true)
    station_all = @chain station_all begin
        @transform(:CN = rstrip.(:CN), :STANAME = rstrip.(:STANAME))
        # @subset(:CN .∈ tuple(["FR", "BE", "LU", "CH"])) # Choose that if you want to look at all stations in France, Belgium, Luxembourg and Switzerland.
        @subset(:STAID .∈ tuple([32, 33, 34, 36, 39, 203, 322, 323, 434, 736, 737, 738, 740, 742, 745, 749, 750, 755, 756, 757, 758, 786, 793, 2192, 2203, 2205, 2207, 2209, 11244, 11245, 11247, 11249]))
        @transform(:STANAME = shortname.(:STANAME))
    end
end

selected_station_name = ["BOURGES", "TOULOUSE", "MARIGNANE", "LUXEMBOURG", "LILLE", "EMBRUN", "BASTIA", "LA HAGUE", "CHASSIRON", "ORLY"]
````

!!! note "Hypothesis: Conditional Independence of Rain Occurrences"
    You can change the selected stations. However, keep in mind that for the model to work, the **conditional independence hypothesis** must hold between stations i.e. $\mathbb{P}(Y_1 = y_1, \cdots, Y_S = y_s\mid Z = k) = \prod_{s=1}^S \mathbb{P}(Y_s = y_s)$.
    Hence stations must be sufficiently far apart. Check out this [MNIST example](https://dmetivie.github.io/ExpectationMaximization.jl/dev/examples/#MNIST-dataset:-Bernoulli-Mixture) to see Bernoulli mixtures in action!

````@example tuto_paper
station = @subset(station_all, :STANAME .∈ tuple(selected_station_name))

STAID = station.STAID #[32, 33, 39, 203, 737, 755, 758, 793, 11244, 11249];

station_name = station.STANAME
````

Sort stations (index) by latitude. It is useful for plotting from North to South.

````@example tuto_paper
staid_lat = sortperm(station.LAT, rev=true);
nothing #hide
````

Station number

````@example tuto_paper
D = length(STAID)
````

### Date range

````@example tuto_paper
date_start = Date(1956)
````

Date including the previous days used in the initial condition (in case `local_memory > 0`)

````@example tuto_paper
date_start_w_memory = date_start - Day(local_order)

date_end = Date(2020) - Day(1)

every_year = date_start:Day(1):date_end

every_year_w_memory = date_start_w_memory:Day(1):date_end

n2t = dayofyear_Leap.(every_year)

N = length(n2t)
````

### Treat data

Load into a `DataFrame` the (ECA) RR files (rain). It filters by date and valid data.
It also adds a column `:bin` for rain events (0: dry, 1: wet).

````@example tuto_paper
begin
    data_stations = collect_data_ECA.(STAID, date_start_w_memory, date_end, "https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/weather_files/ECA_blend_rr/RR_", portion_valid_data=1, skipto=22, header=21, url=true)
    for i = eachindex(data_stations)
        @transform!(data_stations[i], :bin = onefy.(:RR))
    end
end
````

Binary matrix version of the rain event at the `D` stations.

````@example tuto_paper
Yall = BitMatrix(reduce(hcat, [data_stations[j].bin for j = 1:D]))

Y_past = BitMatrix(Yall[1:local_order, :]) # rand(Bool, local_order, D)

ξ = [1; zeros(K - 1)];  # 1 jan 1956 was most likely a type Z = 1 wet day all over France

Y = Yall[1+local_order:end, :]
````

### Map of stations

Convert LAT/LON coordinates from DMS to DD which seems most widely accepted format.

````@example tuto_paper
LAT_idx = dms_to_dd.(station.LAT)

LON_idx = dms_to_dd.(station.LON)

long_spell = [longuest_spell(y) for y in eachcol(Y)]

FR_map_spell = map_with_stations(LON_idx, LAT_idx, long_spell; station_name=station_name, show_value=true, colorbar_show=true, precision_scale = precision_scale)
````

## Fit the seasonal HMM

!!! note "Hypothesis: Smooth parameter evolution"
    We assume all models e.g. HMM, rain mixture to have parameters evolving smoothly with periodicity $T$ for $t \in [1, T]$.
    For example a Bernoulli parameter will write
    ```math
    p(t) = \dfrac{1}{1 + e^{P(t)}} \in [0, 1],
    ```
    with
    ```math
        P_c(t) = c_0 + \sum_{j=1}^{\texttt{Deg}} \left(c_{2j-1}\cos\left(\dfrac{2\pi}{T}j t\right) + c_{2j}\sin\left(\dfrac{2\pi}{T}j t\right)\right).
    ```

### Fit slice: naive estimation

!!! note
    Before inferring the HMM parameters with the EM (Baum-Welch) algorithm, we do a first naive inference that will be used as initial condition for the EM.

The reference station `ref_station` is used to sort the hidden states obtained via the slide initialization
Here we choose `j=1` $\to$ `STAID=32` $\to$ `BOURGES` because it is a central station for France

````@example tuto_paper
ref_station = 1
````

This generates a random Periodic HMM that we then fit slice by slice (day by day). See paper.

````@example tuto_paper
hmm_random = randhierarchicalPeriodicHMM(K, T, D, local_order; ξ=ξ, ref_station=ref_station);

@time "FitMLE SHMM (Slice)" hmm_slice = fit_mle_all_slices(hmm_random, Y, Y_past; n2t=n2t, robust=true, rand_ini=true, Dirichlet_α=0.8, history=false, n_random_ini=1, Yₜ_extanted=[-12, -7, 0, 6, 13]);

θᴬ_slice, θᴮ_slice = fit_θ!(hmm_slice, 𝐃𝐞𝐠);
nothing #hide
````

### Fit with Baum Welch using the slice estimate as a starting point

With the Slice estimate as a good starting point for the full (seasonal) Baum Welch EM algorithm we fit the model!

````@example tuto_paper
@time "FitMLE SHMM (Baum Welch)" hmm_fit, θq_fit, θy_fit, hist, histo_A, histo_B = fit_mle(hmm_slice, θᴬ_slice, θᴮ_slice, Y, Y_past,
    maxiter=10000, robust=true; display=:final, silence=true, tol=1e-3, θ_iters=true, n2t=n2t);
nothing #hide
````

Uncomment to load previously computed hmm
```julia
# hmm_infos = load("save_tuto_path/hmm_fit.jld")
# hmm_fit = hmm_infos["hmm"]
# hist = hmm_infos["hist"]
# θq_fit = hmm_infos["Q_param"]
# θy_fit = hmm_infos["Y_param"]
```

### Visualization of the HMM parameters

#### Transition matrix

````@example tuto_paper
begin
    pA = [plot(legendfont=14, foreground_color_legend=nothing, background_color_legend=nothing, legend_columns=4, tickfont=12, legendfontsize=16) for k in 1:K]
    for k in 1:K
        [plot!(pA[k], hmm_fit.A[k, l, :], c=my_color(l, K), label=L"Q_{%$(k)\to %$(l)}", legend=:top, lw=1.75) for l in 1:K]
        hline!(pA[k], [0.5], c=:black, label=:none, s=:dot)
        xticks!(pA[k], vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366), vcat(string.(monthabbr.(1:12)), ""), xlims=(0, 367), ylims=(0, 1))
    end
    pallA = plot(pA..., size=(1000, 500))
end
````

#### Rain probabilities

````@example tuto_paper
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
````

#### Spatial Rain probability

````@example tuto_paper
memory_past_cat = 1
````

h = 1 (day before dry) or 2 (day before wet)
$\mathbb{P}(Y = \text{Rain}\mid Z = k, H = h)$ with `h = memory_past_cat`

For now there are some scale rendering issues due to an [GeoMakie.jl issue](https://github.com/MakieOrg/GeoMakie.jl/issues/268) so it might be tiny.

````@example tuto_paper
p_FR_map_mean_prob = map_with_stations(LON_idx, LAT_idx, [[mean(succprob.(hmm_fit.B[k, :, j, memory_past_cat])) for j in 1:length(STAID)] for k in 1:K], colorbar_show=true, colorbar_title = L"\mathbb{P}(Y = \text{Rain}\mid Z = k, H = 1)", precision_scale = precision_scale)
````

### Inference of the historical hidden states

####  Viterbi algorithm

````@example tuto_paper
ẑ = viterbi(hmm_fit, Y, Y_past; n2t=n2t)

data_stations_z = map(data_stations) do df
    @transform(df, :z = [fill(missing, local_order); ẑ])
end

ẑ_per_cat = [findall(ẑ .== k) for k in 1:K]
````

#### Visualization of the historical sequences of hidden states

````@example tuto_paper
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
````

## Adding Rain amounts to the model

### Marginal distribution

We fit the marginals of the rain amount $R>0$ at each station $s$ and for each hidden state $Z$ independently.
We use a mixture of exponential functions
```math
g(r) = w \dfrac{e^{-{\frac {r}{\vartheta_1}}}}{\vartheta_1} + (1-w)  \dfrac{e^{-{\frac {r}{\vartheta_2}}}}{\vartheta_2}.
```
whose parameters $w(t)$, $\vartheta_1(t)$ and $\vartheta_2(t)$ are smooth periodic functions of the day of the year.

````@example tuto_paper
@time "FitMLE RR" mix_allE = fit_mle_RR.(data_stations_z, K, local_order, mix₀=StochasticWeatherGenerators.mix_ini(T));
nothing #hide
````

Thanks to [Distributions.jl PR #1389 (September 2nd, 2021)](https://github.com/JuliaStats/Distributions.jl/pull/1389) and Julia multiple dispatch, the quantile function of Mixtures can be very efficiently computed.

### Rain correlations

We fit a Gaussian copula to each pair of stations for joint rainy days only.

  !!! warning
  For some hidden states corresponding to dry weather, it might happen that for some pair of stations, there are no simultaneous rain occurrences in a rain category $Z = k$.
  In that case a `missing` coefficient is returned.

````@example tuto_paper
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
````

## Simulation

Now we are ready to generate samples from the SWG model.

`Nb` is the number of realization

````@example tuto_paper
Nb = 1000
````

Sample the (seasonal) HMM model and output the sequence of hidden states and multi-site dry/wet.

````@example tuto_paper
begin
    zs = zeros(Int, N, Nb)
    ys = zeros(Bool, N, D, Nb)
    @time "Simulations Z, Y" for i in 1:Nb
        zs[:, i], ys[:, :, i] = rand(hmm_fit, n2t; y_ini=Yall[1:local_order, :], z_ini=1, seq=true)
    end
end
````

Given the hidden states and dry/wet, it generates the rain amounts at each station (correlated with a Gaussian Copula).

````@example tuto_paper
begin
    rs = zeros(D, N, Nb)
    @time "Simulations RR>0" for i in 1:Nb
        rs[:, :, i] = rand_RR(mix_allE, n2t, zs[:, i], ys[:, :, i]', Σ²RR)
    end
end
````

## Results

### Spell distribution

`select_month` to choose the month where to compute the spell distributions (summer month, winter, etc.)
`select_month = 1:12` corresponds to all months.

````@example tuto_paper
select_month = 1:12

idx_months = [findall(x -> month.(x) == m, data_stations[1][1+local_order:end, :DATE]) for m in 1:12]

idx_month_vcat = vcat(idx_months[select_month]...)

idx_all = [intersect(yea, mon) for yea in idx_year, mon in idx_months];
nothing #hide
````

##### Historic spells

````@example tuto_paper
len_spell_hist = [pmf_spell(Y[idx_month_vcat, j], dw) for j in 1:D, dw in 0:1];
nothing #hide
````

##### Simulation spells

````@example tuto_paper
len_spell_simu = [pmf_spell(ys[idx_month_vcat, j, i], dw) for i in 1:Nb, j in 1:D, dw in 0:1];
nothing #hide
````

#### Dry spell

````@example tuto_paper
make_range(y, step=1) = range(extrema(y)..., step=step)

begin
    dry_or_wet = 1 # dry
    p_spell_dry = [plot(ylims=(1e-4, 1e-0), tickfont=11, legendfontsize=13) for j = 1:D]
    for j = 1:D
        all_spells = len_spell_simu[:, j, dry_or_wet]
        errorlinehist!(p_spell_dry[j], all_spells, groupcolor=:grey, legend=:topright, label=islabel(j, staid_lat[[1]], L"Simu $q_{0,100}$"), norm=:probability, bins=make_range(reduce(vcat, all_spells)), errortype=:percentile, percentiles=[0, 100], fillalpha=0.4, centertype=:median)

        errorlinehist!(p_spell_dry[j], all_spells, groupcolor=:red, label=islabel(j, staid_lat[[1]], L"Simu $q_{25,75}$"), norm=:probability, bins=make_range(reduce(vcat, all_spells)), errortype=:percentile, percentiles=[25, 75], fillalpha=0.5, centertype=:median)

        histo_spell = len_spell_hist[j, dry_or_wet]
        errorlinehist!(p_spell_dry[j], [histo_spell], label=islabel(j, staid_lat[[1]], "Obs"), groupcolor=:blue, lw=1.5, norm=:probability, bins=make_range(histo_spell), errortype=:percentile)
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

````@example tuto_paper
begin
    dry_or_wet = 2 # wet
    p_spell_wet = [plot(ylims=(1e-4, 1e-0), tickfont=11, legendfontsize=13) for j = 1:D]
    for j = 1:D
        all_spells = len_spell_simu[:, j, dry_or_wet]
        errorlinehist!(p_spell_wet[j], all_spells, groupcolor=:grey, legend=:topright, label=islabel(j, staid_lat[[1]], L"Simu $q_{0,100}$"), norm=:probability, bins=make_range(reduce(vcat, all_spells)), errortype=:percentile, percentiles=[0, 100], fillalpha=0.4, centertype=:median)

        errorlinehist!(p_spell_wet[j], all_spells, groupcolor=:red, label=islabel(j, staid_lat[[1]], L"Simu $q_{25,75}$"), norm=:probability, bins=make_range(reduce(vcat, all_spells)), errortype=:percentile, percentiles=[25, 75], fillalpha=0.5, centertype=:median)

        histo_spell = len_spell_hist[j, dry_or_wet]
        errorlinehist!(p_spell_wet[j], [histo_spell], label=islabel(j, staid_lat[[1]], "Obs"), groupcolor=:blue, lw=1.5, norm=:probability, bins=make_range(histo_spell), errortype=:percentile)
        xlims!(p_spell_wet[j], 0, 2 + maximum(1.5maximum.(histo_spell)))
        yaxis!(:log10)
    end

    [xlabel!(p_spell_wet[j], "Nb of days", xlabelfontsize=12) for j in staid_lat[6:10]]
    [ylabel!(p_spell_wet[j], "PMF", ylabelfontsize=12) for j in staid_lat[[1, 6]]]
    [title!(p_spell_wet[j], station_name[j], titlefontsize=13) for j = 1:D]
    pall_spell_wet = plot(p_spell_wet[staid_lat]..., size=(3000 / 2.5, 1000 / 1.5), layout=(2, 5), left_margin=0.5cm, bottom_margin=0.275cm)
end
````

### Rain

#### Interpretation: Mean Rain per weather regime $R > 0 \mid Z = k$.

We plot the empirical (strictly) positive **mean** rain amounts per weather regime. The results are smoothed using a `cyclic_moving_average` with a time window of $\pm 15$ days and the Epanechnikov kernel.

````@example tuto_paper
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
        # Uncomment to see how the double exponential mixtures compare to the empirical data.
        # [plot!(p_rainpercat[j], 1:T, t -> conversion_factor * mean(mix_allE[j][k, t]), label=:none, c=my_color(k, K), lw=1.5, legend = :topleft) for k in 1:K]
        for k in 1:K
            cycle_avg = replace(cyclic_moving_average(df_j[k].MEAN_RR, df_j[k].day, T, 15), 0 => missing)
            @df df_j[k] plot!(p_rainpercat[j], 1:T, conversion_factor * cycle_avg, c=my_color(k, K), alpha=1, label=islabel(j, staid_lat[[4]], L"Z = %$k"), lw=1.5)
        end
        ylims!(p_rainpercat[j], 0, Inf)
    end
    [ylabel!(p_rainpercat[j], L"Rain (mm/m$^2$)") for j in staid_lat[[1, 6]]]
    [xticks!(
        p_rainpercat[j],
        vcat(dayofyear_Leap.(Date.(2000, 1:12)), 366),
        vcat(string.(first.(string.(monthabbr.(1:12)))))
    ) for j in 1:D]
    [title!(p_rainpercat[j], station_name[j]) for j = 1:D]
    plt_rain_cat_mix = plot(p_rainpercat[staid_lat]..., size=(3000 / 2.2, 1000 / 1.5), layout=(2, 5), left_margin=25px)
end
````

#### Univariate Rain distributions

Historical vs Nb simulations distribution

````@example tuto_paper
begin
    p_uniR = [plot(yaxis=:log10, ylims=(1e-4, 1e-0), tickfont=11, legendfontsize=13, titlefontsize=13) for j = 1:D]
    for j = 1:D
        dists_RR_positive_j = conversion_factor * [filter(!iszero, rs[j, :, i]) for i in 1:Nb]
        errorlinehist!(p_uniR[j], dists_RR_positive_j, groupcolor=:grey, legend=:topright, label=islabel(j, staid_lat[[1]], L"Simu $q_{0,100}$"), norm=:pdf, errortype=:percentile, percentiles=[0, 100], fillalpha=0.4, centertype=:median)

        errorlinehist!(p_uniR[j], dists_RR_positive_j, groupcolor=:red, label=islabel(j, staid_lat[[1]], L"Simu $q_{25,75}$"), norm=:pdf, errortype=:percentile, percentiles=[25, 75], fillalpha=0.5, centertype=:median)

        errorlinehist!(p_uniR[j], [conversion_factor * filter(!iszero, data_stations[j].RR)], label=islabel(j, staid_lat[[1]], "Obs"), groupcolor=:blue, lw=1.5, norm=:pdf, errortype=:percentile)

        xlims!(p_uniR[j], 0.0, Inf)
    end
    [plot!(p_uniR[j], xlabel=L"Rain (mm/m$^2$)") for j in staid_lat[6:10]]
    [plot!(p_uniR[j], ylabel="PDF") for j in staid_lat[[1, 6]]]

    [title!(p_uniR[j], station_name[j]) for j = 1:D]

    pall_R = plot(p_uniR[staid_lat]..., size=(3000 / 2.5, 1000 / 1.5), layout=(2, 5), bottom_margin=11px, left_margin=15px)
end
````

#### Monthly quantile amount

````@example tuto_paper
rh = reduce(hcat, [df[1+local_order:end, :RR] for df in data_stations])

month_rain_simu = [cum_monthly(rs[j, :, i], idx_all) for j in 1:D, i in 1:Nb];

month_rain_histo = [cum_monthly(rh[:, j], idx_all) for j in 1:D]

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
    [ylabel!(p_month_RR[j], L"Rain (mm/m$^2$)") for j in staid_lat[[1, 6]]]

    [title!(p_month_RR[j], station_name[j], titlefontsize=12) for j = 1:D]
    pall_month_RR = plot(p_month_RR[staid_lat]..., size=(1190, 500), layout=(2, 5), left_margin=19px)
end
````

### Correlations

##### Rain event dry/wet

````@example tuto_paper
cor_bin_hist = cor(reduce(hcat, [df.bin for df in data_stations]));

cor_bin_mean_simu = mean(cor(ys[:, :, i]) for i in 1:Nb);


begin
    plots_cor_bin = [plot(-0.1:0.1:0.8, -0.1:0.1:0.8, aspect_ratio=true, label=:none, xlabelfontsize=16, ylabelfontsize=16, tickfont=11, legendfontsize=13) for _ in 1:1]
    scatter!(plots_cor_bin[1], vec_triu(cor_bin_hist), vec_triu(cor_bin_mean_simu), label="Correlations", xlabel="Observations", ylabel="Simulations", c=2)
    [xlims!(plots_cor_bin[i], -0.1, 1) for i in 1:1]
    [ylims!(plots_cor_bin[i], -0.1, 1) for i in 1:1]
    annotate!(0.2, 0.7, "MSE ≃ $(round(mean(abs2, vec_triu(cor_bin_hist) - vec_triu(cor_bin_mean_simu)), digits = 4))")
    plot_cor_bin = plot(plots_cor_bin...)
end
````

The largest pair correlation error for rain occurence comes from the pair

````@example tuto_paper
println("$(station_name[findmax(cor_bin_hist - cor_bin_mean_simu)[2][1]]) and $(station_name[findmax(cor_bin_hist - cor_bin_mean_simu)[2][2]])")
````

##### Rain amount

````@example tuto_paper
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
    plot(plots_cor..., size=(800, 400), left_margin=15px)
end
````

The largest pair correlation error for rain (zero and non zero amounts) comes from the pair

````@example tuto_paper
println("$(station_name[findmax(cor_hist - cor_mean_simu)[2][1]]) and $(station_name[findmax(cor_hist - cor_mean_simu)[2][2]])")
````

##### Gaussian copula hypothesis

For a pair of stations, we transform the marginal $R_s>0$ to $\mathcal{N}(0,1)$. We compare the obtained bi-variate Normal distribution with the Mahalanobis distance to the theoretical $\chi^2(2)$-distriubtion.

````@example tuto_paper
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
    title!("$(station_name[j1]) vs $(station_name[j2])")
    xlims!(0, 22)
    ylims!(0, 22)
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

