using Markdown#hide
import Pkg;
Pkg.activate("docs/");#src
cd(@__DIR__)#hide

md"""
# Adding stations and weather variables
"""

md"""
This tutorial shows how to easily train weather stations given the hidden states sequence `z` obtained in the [previous tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/).
We will show how to make a (simplistic) mutlisite SWG with multiple correlated weather variables such as daily Rain `RR` ($\mathrm{m}\mathrm{m}/\mathrm{m}^2$), daily Temperature minimum `TN` (°C), maximum `TX` (°C), total daily solar irradiance `QQ` (MJ/$\mathrm{m}^2$) and daily evapotranspiration Penman `ETPP` ($\mathrm{m}\mathrm{m}/\mathrm{m}^2$). 
This model will be trained with respect to the given hidden states and the parameters will be periodic and vary smoothly during a calendar year.

The hidden states and the seasonality are enough to correlate well the weather variables without extra codependency between simulated variables.

This multisite, multivariable model has been used as input of the [STIC crop model](https://www.sciencedirect.com/science/article/pii/S1161030102001107) to generate data of annual crop yield for maize in the [GenHack 3 hackaton](https://www.polytechnique.edu/en/genhack-3-hackathon-generative-modelling).
"""

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
For example
"""
collect_data_MeteoFrance(49215002)[1:10,:]

md"""
However, the data there does not exactly match the available on CLIMATIK, (less data, different values ...).
For now I stored the CLIMATIK data on a private repo until the Météo France data is fixed.

!!! warning
    While testing this function, it appears that MeteoFrance API might bug sometimes returning an error for some stations (and working for others).
    In that case, you can check the API directly [here](https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-quotidiennes/#).

[^climatik]: Delannoy, David; Maury, Olivier; Décome, Jérémie, 2022, “CLIMATIK : système d’information pour les données du réseau agroclimatique INRAE”, [https://doi.org/10.57745/AJNXEN](https://doi.org/10.57745/AJNXEN), Recherche Data Gouv, V1
"""

local_order = 1
memory_order = 2^local_order
K = 4
degree = 1
T = 366
save_tuto_path = "../../assets/tuto_2/tuto_K_$(K)_d_$(degree)_m_$(local_order)" #src

md"""
### Weather stations

We select four INRAE weather stations Montreuil-Bellay, Mons-en-Chaussée, Saint-Martin-de-Hinx and Saint-Gènes-Champanelle.
Note that these are not the training stations used in the [paper tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/), however we will still use the weather regime trained there. The rain occurrence generation will be conditionally independent to this weather regime across all these four stations.
"""

md"""
Station French department number.
"""

station_dep = [49, 80, 40, 63]
station_name = ["Montreuil-Bellay", "Mons-en-Chaussée", "Saint-Martin-de-Hinx", "Saint-Gènes-Champanelle"]
station_gps = [(LAT = 47.13, LON = -0.1415), (LAT = 49.875, LON = 3.031), (LAT = 43.5708, LON = -1.29933), (LAT = 45.723, LON = 3.019) ]
D = length(station_name)

path_INRAE_stations = "C:/Users/metivier/Dropbox/PC (2)/Documents/GitLab/weather_data_mistea/INRAE_stations" #src
station_path = joinpath.(path_INRAE_stations, string.("INRAE_STATION_", [49215002, 80557001, 40272002, 63345002], ".csv")) #src
station_path = string.("https://forgemia.inra.fr/david.metivier/weather_data_mistea/-/raw/main/INRAE_stations/INRAE_STATION_", [49215002, 80557001, 40272002, 63345002], ".csv") .|> download

station_ndep = string.(station_name, " (", station_dep, ")")


md"""
### Load
"""

md"""
Load the AutoRegressive Seasonal HMM computed in this [tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/). 
"""

file_hmm = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/refs/heads/master/assets/tuto_1/hmm_fit_K_4_d_1_m_1.jld")
begin
    hmm_infos = JLD.load(file_hmm)
    hmm_fit_full = hmm_infos["hmm"]
    hist = hmm_infos["hist"]
    θq_fit = hmm_infos["Q_param"]
end

md"""
Load the sequence of estimated hidden states for the historical sequence.
"""
file_df_z = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/refs/heads/master/assets/tuto_1/z_hat_K_4_d_1_m_1.csv")
df_z = CSV.File(file_df_z) |> DataFrame
df_z[1:10, :] # Show the first lines of the dataframes

md"""
Load and filter the data by date. There are a few missing values that we impute using `Impute.Interpolate`.
"""

begin
    data_stations_full = collect_data_INRAE.(station_path; show_warning=[:RR, :TX], impute_missing=[:RR, :TX])
    for df in data_stations_full
        @transform!(df, :RO = onefy.(:RR))
    end
end

data_stations = [innerjoin(df, df_z; on=:DATE) for df in data_stations_full]

md"""
We compute the mean temperature during the growth month of maize (May to September) at each location.
"""

mean_summer_TX = round.([@combine(@subset(df, month.(:DATE) .∈ tuple([5,6,7,8,9])), :meanTX=mean(:TX))[1,1] for df in data_stations_full], digits = 1)
file_for_maps_with_geomakie = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/utilities_geo_makie_features.jl") 
include(file_for_maps_with_geomakie)
FR_map_spell = map_with_stations(last.(station_gps), first.(station_gps), mean_summer_TX; station_name=string.("     ",station_ndep), show_value=true, colorbar_show=true, precision_scale = 50, colorbar_label = "°C")

md"""
## Fitting 
We now fit the observed weather variables to seasonal models w.r.t. the hidden variables i.e. the models we fit depend continuously on the day of the year $t\in [1,366]$ and on the provided hidden state $Z \in [1,K]$.
"""

md"""
### Rain Occurrences
Fit the Rain Occurrences of INRAE stations with respect to the given hidden states sequence.
These distributions are then cast into the HMM emission distribution. The pre-trained transitons matrices `Qₜ` are kept the same.
We could have added these new emission distributions to the existing one, however here we focus only on these four stations.
"""

θ = zeros(K, D, memory_order, 2degree + 1)
B = Array{Bernoulli}(undef, K, T, D, memory_order)
@time "Fit RO" for (j, df) in enumerate(data_stations)
    B[:, :, j, :], θ[:, j, :, :] = fit_mle_RO(df, local_order, degree)
end
θy_fit = θ # hmm_infos["Y_param"] for old stations
a = zeros(K)
a[data_stations[1].z[1]] = 1.0 # Initial state of the HMM
hmm_fit = ARPeriodicHMM(a, hmm_fit_full.A, B)

md"""
### Rain Amounts

Here for simplicity we select the double exponential model for the rain amount. 
We aim for a generic interface where one could easily change this choice to other univariate distribution e.g. `mix₀ = MixtureModel([Exponential(1), Gamma(1,2)], [1/2,1/2])` or `mix₀ = Pareto(1)`. 
For now this is not supported (because of the seasonal fit).
"""

@time "Fit Rain amounts" mix_allE = fit_mle_RR.(data_stations, local_order; mix₀=StochasticWeatherGenerators.mix_ini(T));

md"""
The Gaussian copula covariance matrices are then estimated.
"""
Σ²RR = cov_RR(data_stations, K)

md"""
### Temperature

We first fit the daily maximal temperature (we could have started with minimal temperatures).
We use the same principle as for `RR` i.e. first fit each univariate distribution and then the copula.
"""

md"""
#### Temperature Max
"""

@time "Fit TX" ar1sTX = fit_AR1.(data_stations, :TX, degree)
Σ²TX = cov_ar1(data_stations, ar1sTX, :TX)


md"""
#### Minimal Temperature
To directly fit the `TN` one could do the following
```julia
ar1sTN = fit_AR1.(data_stations, :TN, degree)
Σ²TN = cov_ar1(dropmissing.(data_stations), ar1sTN, :TN)
```
However this produce `TN` independantly of `TX` (and we can have `TN>TX`). 
To prevent that we fit the positive difference `ΔT = TX-TN` with a Gamma distribution.
We will then simulate the `TN` conditionally to the `TX`
"""

@time "Fit TN residuals" θ_ΔT = fit_TN.(data_stations, 1, T; print_level=0) # 1 is the degree
θ_cor = cor_groupbyTXTN.(data_stations, T)
f(θ) = Gamma(θ[1], θ[2]) # other options e.g. MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
f(t, θ) = f([σₜ(t, θ[1:(2+1)]), σₜ(t, θ[(2+2):end])])


md"""
### Solar Irradiance (QQ)

`QQ` must be positive, we will truncate at simulation time negative instances.
"""
@time "Fit QQ" ar1sQQ = fit_AR1.(data_stations, :QQ, degree)
Σ²QQ = cov_ar1(dropmissing.(data_stations), ar1sQQ, :QQ)

md"""
###  Evapotranspiration Penman (ETPP)

`ETPP` must be positive, we will truncate at simulation time negative instances.
"""
@time "Fit ETPP" ar1sETPP = fit_AR1.(data_stations, :ETPP, degree)
Σ²ETPP = cov_ar1(dropmissing.(data_stations), ar1sETPP, :ETPP)

md"""
## Simulation
"""

md"""
### Initial conditions
"""

y_ini = [@subset(df, :DATE .== Date(2000) - Day(1)).RO[1] for df in data_stations]'
tx_ini = [@subset(df, :DATE .== Date(2000)).TX[1] for df in data_stations]
z_ini = @subset(data_stations[1], :DATE .== Date(2000)).z[1]

tn_ini = [@subset(df, :DATE .== Date(2000)).TN[1] for df in data_stations]
qq_ini = [@subset(df, :DATE .== Date(2000)).QQ[1] for df in data_stations]
et_ini = [@subset(df, :DATE .== Date(2000)).ETPP[1] for df in data_stations]

md"""
### Generation
"""
indicatrix(x) = x > zero(x) ? x : zero(x) # used to manually suppress negative `QQ` and `ETPP`

Random.seed!(50000)

#-

NSIMU = 300
NYEAR = 50

year_start = 1986
date_range = Date(year_start):Day(1):Date(year_start + NYEAR - 1, 12, 31)
n2t = dayofyear_Leap.(date_range)
years = unique(year.(date_range))
N = length(n2t)

#-

@time "Total simulation $NSIMU of $NYEAR years" begin
    zs = zeros(Int, N, NSIMU)
    ys = zeros(Bool, N, D, NSIMU)
    rs = zeros(D, N, NSIMU)
    txs = zeros(D, N, NSIMU)
    ϵ_TX = zeros(D, N, NSIMU)
    ΔTs = zeros(D, N, NSIMU)
    qqs = zeros(D, N, NSIMU)
    ets = zeros(D, N, NSIMU)

    @time "HMM" for i in 1:NSIMU
        zs[:, i], ys[:, :, i] = rand(hmm_fit, n2t; y_ini=y_ini, z_ini=z_ini, seq=true)
    end
    @time "Rain" for i in 1:NSIMU
        rs[:, :, i] = rand_RR(mix_allE, n2t, zs[:, i], ys[:, :, i]', Σ²RR)
    end
    @time "T_max" for i in 1:NSIMU
        txs[:, :, i], ϵ_TX[:, :, i] = rand(ar1sTX, n2t, zs[:, i], Σ²TX; y₁=tx_ini, output_ϵ=true)
    end

    @time "T_min" for i in 1:NSIMU
        ΔTs[:, :, i] = reduce(hcat, [rand_cond(ϵ_TX[j, :, i], zs[:, i], θ_ΔT[j], θ_cor[j], n2t, T) for j in 1:D]) |> permutedims
    end
    tns = txs - ΔTs #rand(ar1sTN, n2t, zs, Σ²TN; y₁=tn_ini)
    @time "Solar Radiation" for i in 1:NSIMU
        qqs[:, :, i] = indicatrix.(rand(ar1sQQ, n2t, zs[:, i], Σ²QQ; y₁=qq_ini))
    end
    @time "Evapotranspiration" for i in 1:NSIMU
        ets[:, :, i] = indicatrix.(rand(ar1sETPP, n2t, zs[:, i], Σ²ETPP, y₁=et_ini))
    end
end

md"""
### Results

Cast the results into `NSIMU = 300` $\times$ `D=4` `DataFrame`s of `NYEAR = 50` years each, with columns `[:DATE, :RR, :RN, :TX, :QQ, :ETPP, :STAID]` where `STAID` is the unique station identifier. Also cast everything into a `Dict` for convenience.
"""
dicts_simu = Dict(:RR => rs, :TN => tns, :TX => txs, :QQ => qqs, :ETPP => ets)
dfs_simus = [[DataFrame(:DATE => date_range, :RR => rs[j, :, i], :TN => tns[j, :, i], :TX => txs[j, :, i], :QQ => qqs[j, :, i], :ETPP => ets[j, :, i], :STAID => fill(data_stations[j].STAID[1], length(n2t))) for j in 1:D] for i in 1:NSIMU];

md"""
## Plots
"""

md"""
### Settings for plotting

Some settings and packages to have nice plots.
"""

using StatsPlots
using StatsPlots.PlotMeasures # To play with margin in Plots

gr() # plotly() # for interactive plots
default(fontfamily="Computer Modern")
cur_colors = get_color_palette(:auto, 100);
my_palette(K) = palette(vcat(cur_colors[1], [cur_colors[c] for c in 3:4], cur_colors[2]), K)

file_for_plot_utilities = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/utilities_plot.jl")
include(file_for_plot_utilities)
vars = [:RR, :TN, :TX, :QQ, :ETPP]

md"""
### Plot correlations

In this section, we show different correlations between variables and stations. 
"""

md"""
#### Multisite Correlation for each variables
"""

@time begin
    plt_cor_mutlisite = [plot(-1:0.1:1, -1:0.1:1, lw=2, label=:none, aspect_ratio=:equal, foreground_color_legend=nothing, background_color_legend=nothing, legend_columns=2, tickfont=12, legendfontsize=12, xlabelfontsize=16, ylabelfontsize=16) for i in 1:length(vars)]
    for (i, vari) in enumerate(vars)
        annotate!(plt_cor_mutlisite[i], 0.5, 1.15, ("$(string(vari))", 16))
        X = dicts_simu[vari]
        for station_1 in 1:D-1
            @views X₁ = X[station_1, :, :]
            for station_2 in station_1+1:D
                df1 = dropmissing(innerjoin(data_stations[station_1], data_stations[station_2], on=:DATE, makeunique=true))
                @views X₂ = X[station_2, :, :]
                @views mean_cor = mean(cor(X₁[:, r], X₂[:, r]) for r in 1:NSIMU)
                arr = ([cor(df1[:, vari], df1[:, Symbol(string(vari, "_1"))])], [mean_cor])
                scatter!(plt_cor_mutlisite[i], arr, label=ifelse(i == 2, "$(station_dep[station_1]) vs $(station_dep[station_2])", :none), markersize=6)
            end
        end
        xlabel!("Observation")
        i ∈ [1, 4] ? ylabel!("Simulation") : nothing
    end
    plt_cor_mutlisites = plot(plt_cor_mutlisite..., size=(1000, 1000), layout=(3, 3), top_margin=7px)
end
#-
savefigcrop(plt_cor_mutlisites, "cor_space_K_$(K)_d_$(degree)_m_$(local_order)", save_tuto_path); #src

md"""
#### Correlations between weather variables at each sites.
"""

markers = filter((m -> begin
        m in Plots.supported_markers()
    end), Plots._shape_keys)

begin
    plt_var = [plot(-1:0.1:1, -1:0.1:1, lw=2, label=:none, aspect_ratio=:equal, foreground_color_legend=nothing, background_color_legend=nothing, legend_columns=1, tickfont=13, legendfontsize=14, xlabelfontsize=16, ylabelfontsize=16, titlefontsize=16) for j in 1:D]
    for station_1 in 1:D
        title!(plt_var[station_1], "$(station_ndep[station_1])")
        for (i, var_1) in enumerate(vars)
            for j in (i+1):length(vars)
                var_2 = vars[j]
                df1 = dropmissing(data_stations[station_1])
                @views mean_cor = mean(cor(dicts_simu[var_1][station_1, :, r], dicts_simu[var_2][station_1, :, r]) for r in 1:NSIMU)
                scatter!(plt_var[station_1], [cor(df1[:, var_1], df1[:, var_2])], [mean_cor], label=ifelse(station_1 == 1, "$var_1 vs $var_2", :none), m=markers[i], ms=8)
            end
        end
    end
    plt_vars = plot(plt_var..., size=(1000, 1000))
end

#-
savefigcrop(plt_vars, "cor_var_K_$(K)_d_$(degree)_m_$(local_order)", save_tuto_path) #src

md"""
### Univariate distributions

In this section we plot the marginal distribution of each variables at each sites.
"""

md"""
#### Marginal Distributions
"""

begin
    plt_dist_univ = Dict()
    plts_dist_univ = Dict()
    xlabel_string = Dict(:RR => L"Rain ($\mathrm{m}\mathrm{m}/\mathrm{m}^2$)", :TX => L"$T_{\mathrm{max}}$ (°C)", :TN => L"$T_{\mathrm{min}}$ (°C)", :QQ => L"Solar irradiance (MJ/$\mathrm{m}^2$)", :ETPP => L"Evapotranspiration ($\mathrm{m}\mathrm{m}/\mathrm{m}^2$)")
    for XX in vars
        plt_dist_univ[XX] = [plot(tickfont=13, legendfontsize=14, xlabelfontsize=16, ylabelfontsize=16, titlefontsize=16) for j = 1:D]
        for (j, df) in enumerate(data_stations)
            dist_j = XX == :RR ? [filter(!iszero, x) for x in eachcol(dicts_simu[XX][j, :, :])] : [x for x in eachcol(dicts_simu[XX][j, :, :])] # RR>0

            errorlinehist!(plt_dist_univ[XX][j], dist_j, groupcolor=:grey, legend=:topright, label=islabel(j, [1], L"Simu $q_{0,100}$"), norm=:pdf, errortype=:percentile, percentiles=[0, 100], fillalpha=0.4, centertype=:median)

            errorlinehist!(plt_dist_univ[XX][j], dist_j, groupcolor=:red, label=islabel(j, [1], L"Simu $q_{25,75}$"), norm=:pdf, errortype=:percentile, percentiles=[25, 75], fillalpha=0.5, centertype=:median)

            dist_j_histo = XX == :RR ? filter(!iszero, dropmissing(df)[!, XX]) : dropmissing(df)[!, XX] # RR>0

            errorlinehist!(plt_dist_univ[XX][j], [dist_j_histo], label=islabel(j, [1], "Obs"), groupcolor=:blue, lw=1.5, norm=:pdf, errortype=:percentile)
            XX == :RR ? ylims!(plt_dist_univ[XX][j], 1e-4, 0, yaxis=:log10) : nothing
            XX == :RR && j == 2 ? xlims!(plt_dist_univ[XX][j], -1, 100, yaxis=:log10) : nothing # some simulated RR are super extreme and messing with the xaxis
        end
        [xlabel!(plt_dist_univ[XX][j], xlabel_string[XX]) for j in [3, 4]]
        [ylabel!(plt_dist_univ[XX][j], "PDF") for j in [1, 3]]
        station_ndep
        [title!(plt_dist_univ[XX][j], station_ndep[j]) for j = 1:D]

        plts_dist_univ[XX] = plot(plt_dist_univ[XX]..., size=(1000, 700), bottom_margin=11px, left_margin=15px)
        savefigcrop(plts_dist_univ[XX], "pdf_$(XX)_K_$(K)_d_$(degree)_m_$(local_order)", save_tuto_path) #src
    end
end

md"""
##### Rainfall `RR`
"""
plts_dist_univ[:RR]

md"""
##### Temperature max `TX`
"""
plts_dist_univ[:TX]

md"""
##### Temperature min `TN`
"""
plts_dist_univ[:TN]

md"""
##### Solar irradiance `QQ` 
"""
plts_dist_univ[:QQ]

md"""
##### Evapotranspiration `ETPP`
"""
plts_dist_univ[:ETPP]

md"""
#### Monthly statistics
"""
year_range = unique(year.(date_range))
idx_year = [findall(x -> year.(x) == m, date_range) for m in year_range]
idx_month = [findall(x -> month.(x) == m, date_range) for m in 1:12]
idx_all = [intersect(yea, mon) for yea in idx_year, mon in idx_month]
agg_fun(XX) = ifelse(XX == :RR, sum, mean)
month_rain_simu = Dict(key => [monthly_agg(xx[j, :, i], idx_all, agg_fun(key)) for j in 1:D, i in 1:NSIMU] for (key, xx) in dicts_simu)
month_rain_histo = Dict(key => [monthly_agg(@subset(df, :DATE .≥ Date(1986)), key, agg_fun(key)) for df in data_stations] for key in vars)

qs = [0.9, 0.5, 0.1]

@time "Plot monthly quantile" begin
    plt_month = Dict()
    plts_month = Dict()
    ylabel_string = Dict(:RR => L"Cumulated Rain ($\mathrm{m}\mathrm{m}/\mathrm{m}^2$)", :TX => L"$T_{\mathrm{max}}$ (°C)", :TN => L"$T_{\mathrm{min}}$ (°C)", :QQ => L"Solar irradiance (MJ/$\mathrm{m}^2$)", :ETPP => L"Evapotranspiration ($\mathrm{m}\mathrm{m}/\mathrm{m}^2$)")
    for XX in vars
        plt_month[XX] = [plot(xtickfontsize=10, ytickfontsize=11, ylabelfontsize=12, legendfontsize=12, titlefontsize=12, foreground_color_legend=nothing) for j = 1:D]
        for j = 1:D
            for (α, per) in enumerate([[0, 100], [25, 75]])
                for (cc, q) in enumerate(qs)
                    errorline!(plt_month[XX][j], [quantile(month_rain_simu[XX][j, i][:, m], q) for m in 1:12, i in 1:NSIMU], label=(α == 1 ? islabel(j, 4, L"Simu  $q_{%$(Int(q*100))}$") : :none), fillalpha=0.18 * α^2, centertype=:median, errortype=:percentile, percentiles=per, groupcolor=my_palette(length(qs))[cc])
                end
            end
            for q in qs
                scatter!(plt_month[XX][j], m -> quantile(skipmissing(month_rain_histo[XX][j][:, m]), q), 1:12, label=(q == qs[1] ? islabel(j, 3, "Obs") : :none), legend=:topleft, ms=2.5, c=:blue)
                plot!(plt_month[XX][j], m -> quantile(skipmissing(month_rain_histo[XX][j][:, m]), q), 1:12, label=:none, c=:blue, lw=1.75)
            end
            xticks!(plt_month[XX][j], 1:12, string.(first.(monthabbr.(1:12))))
            XX == :RR ? ylims!(plt_month[XX][j], 0, Inf) : nothing
        end
        [ylabel!(plt_month[XX][j], ylabel_string[XX]) for j in [1, 3]]

        [title!(plt_month[XX][j], station_ndep[j]) for j = 1:D]
        plts_month[XX] = plot(plt_month[XX]..., size=(1000, 700), left_margin=19px)
        savefigcrop(plts_month[XX], "EDF_like_$(XX)_$(NSIMU)_simu_monthly_quantile_K_$(K)_d_$(degree)_m_$(local_order)", save_tuto_path) #src
    end
end

md"""
##### Rainfall `RR`
"""
plot(plts_month[:RR])

md"""
##### Temperature max `TX`
"""
plts_month[:TX]

md"""
##### Temperature min `TN`
"""
plts_month[:TN]

md"""
##### Solar irradiance `QQ` 
"""
plts_month[:QQ]

md"""
##### Evapotranspiration `ETPP`
"""
plts_month[:ETPP]

md"""
## Stochastic Weather Generator + Crop model STICS

In this section, we demonstrate how stochastic weather simulations can serve as inputs for a crop model to study the climate sensitivity of a crop model.
This is a proof of concept showing how SWG can be useful when combine with other models.

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
using Suppressor # to handle the STICS hard-printed output
stics_path = joinpath("C:/Users/metivier/Dropbox/PC (2)/Documents/GitLab/weather_data_mistea/stics_files/JavaSTICS-1.5.1-STICS-10.0.0", "bin", "stics_modulo.exe") #src
work_path = joinpath("C:/Users/metivier/Dropbox/PC (2)/Documents/GitLab/weather_data_mistea/stics_files", "maize") #src
include("utilities_stics.jl") #src

file_for_stics_utilities = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/utilities_stics.jl")
include(file_for_stics_utilities)

md"""
```julia
stics_path = joinpath(STICS_PATH, "JavaSTICS-1.5.1-STICS-10.0.0", "bin", "stics_modulo.exe") # or stics_modulo for Linux
work_path = joinpath("stics_files", "maize") # folder where all maize files are

@time res_YIELDs = map(Iterators.product(1:300, 1:D)) do (i, j)
    GC.gc() # empirically needed
    @show i,j
    run_stics_mod_yield(dfs_simus[i][j], infos=100, stics_path=stics_path, work_path=work_path).YIELD
end
# 15650.246490 seconds (1.03 G allocations: 108.568 GiB, 1.13% gc time, 0.01% compilation time: <1% of which was recompilation)
cd(@__DIR__)
JLD.save("results_yield.jld", Dict("res_YIELDs" => res_YIELDs))
```
"""

file_yield = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/refs/heads/master/assets/tuto_2/results_yield.jld")
res_YIELDs = JLD.load(file_yield)["res_YIELDs"]

md"""
### Yield distributions with uncertainty

The following plot shows the typical distribution of maize yield at four location over a 50-year span. We added the envelope of these distributions to highlight uncertainty. Each location has a very different weather (see previous sections).
"""
begin
    plt_dist_yield = [plot(tickfont=13, legendfontsize=14, xlabelfontsize=16, ylabelfontsize=16, titlefontsize=16, legend=:topright) for j = 1:D]
    for j in 1:D
        dist_j = Vector{Vector{Float64}}(filter.(!ismissing, res_YIELDs[:, j]))

        errorlinehist!(plt_dist_yield[j], dist_j, groupcolor=:gray, label=islabel(j, [1], L"Simu $q_{0,100}$"), norm=:pdf, errortype=:percentile, percentiles=[0, 100], fillalpha=0.5, centertype=:median)

        errorlinehist!(plt_dist_yield[j], dist_j, groupcolor=:red, label=islabel(j, [1], L"Simu $q_{25,75}$"), norm=:pdf, errortype=:percentile, percentiles=[25, 75], fillalpha=0.5, centertype=:median)
        xlims!(plt_dist_yield[j], 0, 16)
    end
    [xlabel!(plt_dist_yield[j], "YIELD (t/ha)") for j in [3, 4]]
    [ylabel!(plt_dist_yield[j], "PDF") for j in [1, 3]]
    station_ndep
    [title!(plt_dist_yield[j], station_ndep[j]) for j = 1:D]

    plts_dist_yield = plot(plt_dist_yield..., size=(1000, 700), bottom_margin=11px, left_margin=15px)
end
savefigcrop(plts_dist_yield, "pdf_YIELD_K_$(K)_d_$(degree)_m_$(local_order)", save_tuto_path) #src

md"""
### Sensitivity of maize on rainfall during key growth periods

To identify which rainfall period between April and October has the greatest impact on maize yield, we divide the growing season into four periods. For each period, we calculate the mean rainfall, both conditionally (blue) and unconditionally (orange), on final yields exceeding the median. Each distribution is displayed along with its interquartile range.

For the selected station, the most critical period is from June 12 to July 27, as evidenced by the significant difference between the two distributions. Specifically, the mean rainfall, when conditioned on a high yield, is approximately 20% higher during this period. In contrast, for the other periods, the mean rainfall remains nearly identical, indicating a much lower sensitivity.
"""
week_date = (date_begin=Date(1996, 4, 27), date_end=Date(1996, 10, 27), N_period=4)
groups = chunky(Dates.value(week_date.date_end - week_date.date_begin) + 1, week_date.N_period)

dfs_stat = [[stats_fortnightly(dfsim, week_date, years) for dfsim in dfs_simus[i]] for i in axes(res_YIELDs, 1)]
for i in eachindex(dfs_stat)
    for (j, df) in enumerate(dfs_stat[i])
        @transform!(df, :YIELD = res_YIELDs[i, j])
    end
end

begin
    j = 1 # station number
    dfi = [dropmissing(dfs_stat[i][j]) for i in eachindex(dfs_stat)]
    df_sub = [@subset(df, :YIELD .> quantile(:YIELD, 0.5)) for df in dfi]
    plt_scat = [plot(legend=:topright, tickfont=14, legendfontsize=14, xlabelfontsize=14, ylabelfontsize=14, titlefontsize=14) for i in 1:week_date.N_period]
    for i in 1:week_date.N_period
        errorlinehist!(plt_scat[i], [dfi[ii][:, Symbol("MEAN_RR_$i")] for ii in eachindex(dfi)], groupcolor=2, label=islabel(i, 1, L"R"), norm=:pdf, errortype=:percentile, percentiles=[25, 75], fillalpha=0.1, lw = 2, centertype=:median)

        errorlinehist!(plt_scat[i], [df_sub[ii][:, Symbol("MEAN_RR_$i")] for ii in eachindex(dfi)], groupcolor=1, label=islabel(i, 1, L"R \mid (\mathrm{Yield} > \mathrm{median})"), norm=:pdf, errortype=:percentile, percentiles=[25, 75], fillalpha=0.1, lw = 2, centertype=:median)

        ## stephist!(plt_scat[i], dfi[:, Symbol("MEAN_RR_$i")], label=islabel(i, 2, L"R"), norm=:pdf, lw=1.75)
        ## stephist!(plt_scat[i], df_sub[:, Symbol("MEAN_RR_$i")], label=islabel(i, 2, L"R \mid (\mathrm{Yield} > \mathrm{median})"), norm=:pdf, lw=1.75)
        vline!([mean.([dfi[ii][:, Symbol("MEAN_RR_$i")] for ii in eachindex(dfi)]) |> mean], label=:none, s=:dot, c=2, lw=2)
        vline!([mean.([df_sub[ii][:, Symbol("MEAN_RR_$i")] for ii in eachindex(dfi)]) |> mean], label=:none, s=:dot, c=1, lw=2)
        period_range = period_range_func(groups[i])
        title!("$(day(period_range[1])) $(monthabbr(period_range[1])) - $(day(period_range[2])) $(monthabbr(period_range[2]))")
        xlabel!("Rain (mm/period)")
        ylabel!("PDF")

    end
    annotate!(plt_scat[3], 5.5, 1.1, station_ndep[j])
    plt_sensitivity = plot(plt_scat..., size=(1000, 700), left_margin=3Plots.PlotMeasures.mm)
end
savefigcrop(plt_sensitivity, "RR_per_period_cond_yield_dep_49.pdf", save_tuto_path) #src

md"""
## Reproducibility
"""
import InteractiveUtils
InteractiveUtils.versioninfo()

md"""
Package list and version
"""

import Pkg;
Pkg.status();