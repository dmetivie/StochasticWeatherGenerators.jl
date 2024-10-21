using Markdown#hide
cd(@__DIR__)#hide

md"""
# Adding stations and weather variables
"""

md"""
This short tutorial shows how to easily add weather stations given the hidden states sequence `z` obtained in the [previous tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/).
We will also explore how to add (very simplistic) mutlisite models for other weather variables such as daily Temperature minimum `TN`, maximum `TX`, solar irradiance `QQ` and evapotranspiration `ETPP`. 
These new models will be trained with respect to the given hidden states and their parameters will be periodic and vary smoothly during a calendar year.
For now models will be trained at each site and correlated spatially with Gaussian Copulas.

The hope is that the hidden states and the seasonality will correlate enough the variables for the model to be realistic.
"""

md"""
## Set up
"""

md"""
### Package and functions
"""

using CSV, DelimitedFiles, JLD# File Read/Load/Save/dwl
import Downloads
using DataFrames, DataFramesMeta # DataFrames
using Dates
using Random, Distributions

using SmoothPeriodicStatsModels
using StochasticWeatherGenerators


md"""
# Data extraction and settings
"""

md"""
## Path
"""

# path_save = "/home/metivierd/scratch/David_Scratch/stics/simu/swg"
path_INRAE_stations = "C:/Users/metivier/Dropbox/PC (2)/Documents/X_related/GenHack_2023/test_data_SAA"

local_order = 1
memory_order = 2^local_order
K = 4
degree = 2
T = 366
save_tuto_path = "../../../assets/tuto_2/tuto_K_$(K)_d_$(degree)_m_$(local_order)" #src

md"""
## Select stations
"""

md"""
Station French department number.
"""
station_dep = [49, 80, 40, 63]
station_name = ["Montreuil-Bellay", "Mons-en-Chaussée", "Saint-Martin-de-Hinx", "Saint-Gènes-Champanelle"]

station_path = joinpath.(path_INRAE_stations, string.("Demandes_station_pour_GENHack_INRAE_STATION_", [49215002, 80557001, 40272002, 63345002], ".csv"))

station_ndep = string.(station_name, " (", station_dep, ")")

D = length(station_name)


md"""
## Load
"""

md"""
Load the AutoRegressive Seasonal HMM computed in this [tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/). 
"""

file_hmm = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/refs/heads/master/assets/tuto_1/hmm_fit_K_4_d_1_m_1.jld")
begin
    hmm_infos = load(file_hmm)
    hmm_fit_full = hmm_infos["hmm"]
    hist = hmm_infos["hist"]
    θq_fit = hmm_infos["Q_param"]
end

md"""
Load the sequence of estimated hidden states for the historical sequence.
"""
file_df_z = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/refs/heads/master/assets/tuto_1/z_hat_K_4_d_1_m_1.csv")
df_z = CSV.File(file_df_z) |> DataFrame

md"""
Filter by date and valid data. There are a few missing values that we impute using `Impute.Interpolate`.
"""

begin
    data_stations_full = collect_data_INRAE.(station_path; show_warning=[:RR, :TX], impute_missing=[:RR, :TX])
    for df in data_stations_full
        @transform!(df, :RO = onefy.(:RR))
    end
end

data_stations = [innerjoin(df, df_z; on=:DATE) for df in data_stations_full]

md"""
# Fitting the weather variables
"""

md"""
## Rain Occurrences
Fit the Rain Occurrences of INRAE stations with respect to the given hidden states sequence.
These distributions are then cast into the HMM emission distribution. The pre-trained transitons matrices Qₜ are kept the same.
We could have added these new emission distributions to the existing one, however here we focus only on these four stations.
"""

θ = zeros(K, D, memory_order, 2degree + 1)
B = Array{Bernoulli}(undef, K, T, D, memory_order)
@time "Fit RO" for (j, df) in enumerate(data_stations)
    B[:, :, j, :], θ[:, j, :, :] = fit_mle_RO(df, local_order, degree)
end
θy_fit = θ # hmm_infos["Y_param"]
a = zeros(K)
a[data_stations[1].z[1]] = 1.0 # Initial state of the HMM
hmm_fit = ARPeriodicHMM(a, hmm_fit_full.A, B)

md"""
## Rain 

To change univariate fitted model change the initial distribution `mix₀` e.g. `mix₀ = MixtureModel([Exponential(1), Gamma(1,2)], [1/2,1/2])`. 
Here for simplicity we select the double exponential.
"""

@time "Fit Rain amounts" mix_allE = fit_mle_RR.(data_stations, K, local_order, mix₀=StochasticWeatherGenerators.mix_ini(T))

md"""
The Gaussian copula covariance function is then estimated.
"""
Σ²RR = cov_RR(data_stations, K)

md"""
## Temperature

We first fit the daily maximal temperature (we could have started with minimal temperatures).
We use the same principle as for `RR` i.e. first fit each univariate distribution and then the copula.
"""

md"""
### Temperature Max
"""

@time "Fit TX" ar1sTX = fit_AR1.(data_stations, :TX, degree, T, K)
Σ²TX = cov_ar1(data_stations, ar1sTX, :TX, K)


md"""
### Minimal Temperature
To directly fit the `TN` one could do the following
```julia
ar1sTN = fit_AR1.(data_stations, :TN, degree, T, K)
Σ²TN = cov_ar1(dropmissing.(data_stations), ar1sTN, :TN, K)
```
However this produce `TN` independantly of `TX` (and we can have `TN>TX`). 
To prevent that we fit the positive difference `ΔT = TX-TN` with a Gamma distribution.
We will then simulate the `TN` conditionally to the `TX`
"""

@time "Fit TN residuals" θ_ΔT = fit_TN.(data_stations, 1, T; print_level=0) # 1 is the degree
θ_cor = cor_groupbyTXTN.(data_stations, T)
f(θ) = Gamma(θ[1], θ[2])# MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
f(t, θ) = f([σₜ(t, θ[1:(2+1)]), σₜ(t, θ[(2+2):end])])


md"""
## Solar Irradiance (QQ)

`QQ` must be positive, we will truncate at simulation time negative instances.
"""
@time "Fit QQ" ar1sQQ = fit_AR1.(data_stations, :QQ, degree, T, K)
Σ²QQ = cov_ar1(dropmissing.(data_stations), ar1sQQ, :QQ, K)

md"""
##  Evapotranspiration Penman (ETPP)

`ETPP` must be positive, we will truncate at simulation time negative instances.
"""
@time "Fit ETPP" ar1sETPP = fit_AR1.(data_stations, :ETPP, degree, T, K)
Σ²ETPP = cov_ar1(dropmissing.(data_stations), ar1sETPP, :ETPP, K)

md"""
# Simulation
"""

md"""
## Initial conditions
"""

y_ini = [@subset(df, :DATE .== Date(2000) - Day(1)).RO[1] for df in data_stations]'
tx_ini = [@subset(df, :DATE .== Date(2000)).TX[1] for df in data_stations]
z_ini = @subset(data_stations[1], :DATE .== Date(2000)).z[1]

tn_ini = [@subset(df, :DATE .== Date(2000)).TN[1] for df in data_stations]
qq_ini = [@subset(df, :DATE .== Date(2000)).QQ[1] for df in data_stations]
et_ini = [@subset(df, :DATE .== Date(2000)).ETPP[1] for df in data_stations]

md"""
## Generation
"""
indicatrix(x) = x > zero(x) ? x : zero(x)
Random.seed!(50000)

md"""
### RANGE settings
"""

NYEAR = 1_000

year_start = 1950
date_range = Date(year_start):Day(1):Date(year_start + NYEAR - 1, 12, 31)
n2t = dayofyear_Leap.(date_range)
years = unique(year.(date_range))

md"""
### Rand
"""
@time "Total simulation $NYEAR years" begin
    @time "HMM" zs, ys = rand(hmm_fit, n2t; y_ini, z_ini=z_ini, seq=true)
    @time "Rain" rs = rand_RR(mix_allE, n2t, zs, ys', Σ²RR)
    @time "T_max" txs, ϵ_TX = rand(ar1sTX, n2t, zs, Σ²TX; y₁=tx_ini, output_ϵ=true)

    @time "T_min" ΔTs = reduce(hcat, [rand_cond(ϵ_TX[j, :], zs, θ_ΔT[j], θ_cor[j], n2t, T) for j in 1:D]) |> permutedims
    tns = txs - ΔTs #rand(ar1sTN, n2t, zs, Σ²TN; y₁=tn_ini)

    @time "Solar Radiation" qqs = indicatrix.(rand(ar1sQQ, n2t, zs, Σ²QQ; y₁=qq_ini))
    @time "Evapotranspiration" ets = indicatrix.(rand(ar1sETPP, n2t, zs, Σ²ETPP, y₁=et_ini))
end

md"""
## DataFrames results

Cast the results into `D=4` `DataFrame`s of `NYEAR = 1000` years each, with columns `[:DATE, :RR, :RN, :TX, :QQ, :ETPP, :STAID]` where `STAID` is the unique station identifier.
"""

dfs_simu = [DataFrame(:DATE => date_range, :RR => rs[j, :], :TN => tns[j, :], :TX => txs[j, :], :QQ => qqs[j, :], :ETPP => ets[j, :], :STAID => fill(data_stations[j].STAID[1], length(n2t))) for j in 1:D]

md"""
To save the data frames you can do something like
```julia
map(enumerate(station_dep)) do (i,dep)
    CSV.write(joinpath(save_tuto_path,"swg_simu_station_dep_$(dep)_years_$(length(years)).csv"), dfs_simu[i])
end
```
"""

md"""
# Plots
"""

md"""
## Settings for plotting

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
## Plot correlations

Note that we should/could consider all estimations e.g. correlations, distribution, on the same time spans than the observations here 34 years to get estimate with same error/variance (and repeat this `NYEARS` times). This is what is done in the other tutorial.
"""

vars = [:RR, :TN, :TX, :QQ, :ETPP]

md"""
### Multisite Correlation for each variables
"""

begin
    plt_cor_mutlisite = [plot(-0.1:0.1:1, -0.1:0.1:1, lw=2, label=:none, aspect_ratio=:equal, foreground_color_legend=nothing, background_color_legend=nothing, legend_columns=2, tickfont=12, legendfontsize=12, xlabelfontsize=16, ylabelfontsize=16) for i in 1:length(vars)]
    for (i, vari) in enumerate(vars)
        annotate!(plt_cor_mutlisite[i], 0.5, 1.15, ("$(string(vari))", 16))
        for station_1 in 1:D-1
            for station_2 in station_1+1:D
                df1 = dropmissing(innerjoin(data_stations[station_1], data_stations[station_2], on=:DATE, makeunique=true))
                df2 = dropmissing(innerjoin(dfs_simu[station_1], dfs_simu[station_2], on=:DATE, makeunique=true))
                arr = ([cor(df1[:, vari], df1[:, Symbol(string(vari, "_1"))])], [cor(df2[:, vari], df2[:, Symbol(string(vari, "_1"))])])
                scatter!(plt_cor_mutlisite[i], arr, label=ifelse(i == 2, "$(station_dep[station_1]) vs $(station_dep[station_2])", :none), markersize=6)
            end
        end
        xlabel!("Observation")
        i ∈ [1, 4] ? ylabel!("Simulation") : nothing
        ylims!(-0.1, 1.1)
    end
    plt_cor_mutlisites = plot(plt_cor_mutlisite..., size=(1000, 1000), layout=(3, 3), top_margin=7px)
end
#-
savefigcrop(plt_cor_mutlisites, "cor_space_K_$(K)_d_$(degree)_m_$(local_order)", save_tuto_path); #src

md"""
### Correlations between weather variables at each sites.
"""

markers = filter((m -> begin
        m in Plots.supported_markers()
    end), Plots._shape_keys)
begin
    station_1 = 1
    plt_cor_var = plot(-0.5:0.1:1, -0.5:0.1:1, lw=2, label=:none, aspect_ratio=:equal, foreground_color_legend=nothing, background_color_legend=nothing, legend_columns=2, tickfont=12, legendfontsize=12, xlabelfontsize=16, ylabelfontsize=16)
    xlims!(-0.2, 1)
    ylims!(-0.2, 1)
    for (i, var_1) in enumerate(vars)
        title!(plt[station_1], "$(station_ndep[station_1])")
        for j in (i+1):length(vars)
            var_2 = vars[j]
            df1 = dropmissing(data_stations[station_1])
            df2 = dropmissing(dfs_simu[station_1])
            scatter!(plt[station_1], [cor(df1[:, var_1], df1[:, var_2])], [cor(df2[:, var_1], df2[:, var_2])], label=ifelse(station_1 == 0, "$var_1 vs $var_2", :none), m=markers[i], ms=5)
        end
        xlabel!("Observation")
        ylabel!("Simulation")
    end
end

begin
    plt_var = [plot(-0.2:0.1:1, -0.2:0.1:1, lw=2, label=:none, aspect_ratio=:equal, foreground_color_legend=nothing, background_color_legend=nothing, legend_columns=1, tickfont=13, legendfontsize=14, xlabelfontsize=16, ylabelfontsize=16, titlefontsize=16) for j in 1:D]
    for station_1 in 1:D
        title!(plt_var[station_1], "$(station_ndep[station_1])")
        for (i, var_1) in enumerate(vars)
            for j in (i+1):length(vars)
                var_2 = vars[j]
                df1 = dropmissing(data_stations[station_1])
                df2 = dropmissing(dfs_simu[station_1])
                scatter!(plt_var[station_1], [cor(df1[:, var_1], df1[:, var_2])], [cor(df2[:, var_1], df2[:, var_2])], label=ifelse(station_1 == 1, "$var_1 vs $var_2", :none), m=markers[i], ms=8)
            end
        end
    end
    plt_vars = plot(plt_var..., size=(1000, 1000))
end
#-
savefigcrop(plt_vars, "cor_var_K_$(K)_d_$(degree)_m_$(local_order)", save_tuto_path)

md"""
## Univariate distributions
"""


md"""
## TEst univariate stuff
"""
md"""
### Temperature max `TX`
"""
begin
    plt_TX = [plot(title="$(station_ndep[i])", tickfont=13, legendfontsize=14, xlabelfontsize=16, ylabelfontsize=16, titlefontsize=16) for i in 1:D]
    for (i, df) in enumerate(data_stations)
        @df df stephist!(plt_TX[i], :TX, norm=:pdf, label="Obs", lw=1.5)
        @df dfs_simu[i] stephist!(plt_TX[i], :TX, norm=:pdf, label="Sim", lw=1.5)
        i ∈ [3, 4] ? xlabel!(L"$T_{\mathrm{max}}$ (°C)") : nothing
        i ∈ [1, 3] ? ylabel!("PDF") : nothing
    end
    plt_TXs = plot(plt_TX..., size=(800, 600))
end
#-
savefigcrop(plt_TXs, "pdf_TX_K_$(K)_d_$(degree)_m_$(local_order)", save_tuto_path) #sc

md"""
### Temperature max `TX`
"""
begin
    plt_TN = [plot(title="$(station_ndep[i])", tickfont=13, legendfontsize=14, xlabelfontsize=16, ylabelfontsize=16, titlefontsize=16) for i in 1:D]
    for (i, df) in enumerate(data_stations)
        @df df stephist!(plt_TN[i], :TN, norm=:pdf, label="Obs", lw=1.5)
        @df dfs_simu[i] stephist!(plt_TN[i], :TN, norm=:pdf, label="Sim", lw=1.5)
        i ∈ [3, 4] ? xlabel!(L"$T_{\mathrm{min}}$ (°C)") : nothing
        i ∈ [1, 3] ? ylabel!("PDF") : nothing
    end
    plt_TNs = plot(plt_TN..., size=(800, 600))
end
savefigcrop(plt_TNs, "pdf_TN_K_$(K)_d_$(degree)_m_$(local_order)", save_tuto_path)
md"""
#### Evapotranspiration
"""
begin
    plt_QQ = [plot(title="$(station_ndep[i])", tickfont=13, legendfontsize=14, xlabelfontsize=16, ylabelfontsize=16, titlefontsize=16) for i in 1:D]
    for (i, df) in enumerate(data_stations)
        @df df stephist!(plt_QQ[i], :QQ, norm=:pdf, label="Obs", lw=1.5)
        @df dfs_simu[i] stephist!(plt_QQ[i], :QQ, norm=:pdf, label="Sim", lw=1.5)
        i ∈ [3, 4] ? xlabel!(L"Solar irradiance (MJ/$m^2$)") : nothing
        i ∈ [1, 3] ? ylabel!("PDF") : nothing
    end
    plt_QQs = plot(plt_QQ..., size=(800, 600))
end
savefigcrop(plt_QQs, "pdf_QQ_K_$(K)_d_$(degree)_m_$(local_order)", save_tuto_path)

md"""
#### Evapotranspiration
"""
begin
    plt_ETPP = [plot(title="$(station_ndep[i])", tickfont=13, legendfontsize=14, xlabelfontsize=16, ylabelfontsize=16, titlefontsize=16) for i in 1:D]
    for (i, df) in enumerate(data_stations)
        @df df stephist!(plt_ETPP[i], :ETPP, norm=:pdf, label="Obs", lw=1.5)
        @df dfs_simu[i] stephist!(plt_ETPP[i], :ETPP, norm=:pdf, label="Sim", lw=1.5)
        i ∈ [3, 4] ? xlabel!(L"Evapotranspiration (mm/$m^2$)") : nothing
        i ∈ [1, 3] ? ylabel!("PDF") : nothing
    end
    plt_ETPPs = plot(plt_ETPP..., size=(800, 600))
end
savefigcrop(plt_ETPPs, "pdf_ETPP_K_$(K)_d_$(degree)_m_$(local_order)", save_tuto_path)

md"""
!!! warning
    TODO: Do simulations on the same 34 years to estimate properly the variability
"""
quantileagg(q) = x -> quantile(skipmissing(x), q) # std, mean
begin
    colors = [1, 3, 2]
    v = :TX
    plt_stats = [plot(title="$(station_ndep[j])", tickfont=13, legendfontsize=14, xlabelfontsize=16, ylabelfontsize=16, titlefontsize=16) for j in 1:D]
    for (j, df) in enumerate(data_stations)
        if j ∈ [4]
            df = @subset(df, :DATE .≥ Date(1986))
        elseif j == 2
            df = @subset(df, :DATE .≥ Date(1986))
        end
        for (cc, q) in enumerate([0.1, 0.5, 0.9])
            dfchain = summonthly(df)
            dfc = @combine(dfchain, :OUT = quantileagg(q)($v))
            dfchain = monthly_agg(dfs_simu[j], v, quantileagg(q))
            errorline!(plt_stats[j], 1:12, dfchain, label=ifelse(j == 1, L"Simu  $q_{%$(Int(q*100))}$", :none), centertype=:median, errortype=:percentile, percentiles=[0, 100], groupcolor=colors[cc], fillalpha=0.3)
            errorline!(plt_stats[j], 1:12, dfchain, label=:none, centertype=:median, errortype=:percentile, percentiles=[25, 75], groupcolor=colors[cc], fillalpha=0.5)
            errorline!(plt_stats[j], 1:12, dfc.OUT, label=ifelse(cc == 3 && j == 1, "Obs", :none), c=:blue, groupcolor=:blue, lw=2, legend=:bottom)
        end
        xticks!(plt_stats[j], 1:12, string.(monthabbr.(1:12)))
        j ∈ [1, 3] ? ylabel!(L"$T_{\mathrm{max}}$ (°C)") : nothing
    end
    plot(plt_stats..., size=(1000, 800), left_margin=10px)
end
