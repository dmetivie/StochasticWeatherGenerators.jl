```@meta
EditURL = "../../../examples/tuto_add_station_variable.jl"
```

````@example tuto_add_station_variable
using Markdown#hide
cd(@__DIR__)#hide
````

# Adding stations and weather variables

This short tutorial shows how to easily add weather stations given the hidden states sequence `z` obtained in the [previous tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/).
We will also explore how to add (very simplistic) mutlisite models for other weather variables such as daily Temperature minimum `TN`, maximum `TX`, solar irradiance `QQ` and evapotranspiration `ETPP`.
These new models will be trained with respect to the given hidden states and their parameters will be periodic and vary smoothly during a calendar year.
For now models will be trained at each site and correlated spatially with Gaussian Copulas.

The hope is that the hidden states and the seasonality will correlate enough the variables for the model to be realistic.

## Set up

### Package and functions

````@example tuto_add_station_variable
using CSV, DelimitedFiles, JLD# File Read/Load/Save/dwl
import Downloads
using DataFrames, DataFramesMeta # DataFrames
using Dates
using Random, Distributions
````

The two main packages for this tutorial are not yet registered in the official Julia registry, since they are not quite fully ready.
They can be either `add`ed through [my local Julia registry](https://github.com/dmetivie/LocalRegistry) with the [LocalRegistry.jl](https://github.com/GunnarFarneback/LocalRegistry.jl) package i.e.
```julia
using Pkg
pkg"registry add https://github.com/dmetivie/LocalRegistry"
Pkg.add("SmoothPeriodicStatsModels")
Pkg.add("StochasticWeatherGenerators")
```

````@example tuto_add_station_variable
using SmoothPeriodicStatsModels
using StochasticWeatherGenerators
````

## Data extraction and settings

````@example tuto_add_station_variable
path_INRAE_stations = "C:/Users/metivier/Dropbox/PC (2)/Documents/X_related/GenHack_2023/test_data_SAA"

local_order = 1
memory_order = 2^local_order
K = 4
degree = 1
T = 366
````

### Weather stations

Station French department number.

````@example tuto_add_station_variable
station_dep = [49, 80, 40, 63]
station_name = ["Montreuil-Bellay", "Mons-en-Chaussée", "Saint-Martin-de-Hinx", "Saint-Gènes-Champanelle"]

station_path = joinpath.(path_INRAE_stations, string.("Demandes_station_pour_GENHack_INRAE_STATION_", [49215002, 80557001, 40272002, 63345002], ".csv"))

station_ndep = string.(station_name, " (", station_dep, ")")

D = length(station_name)
````

### Load

Load the AutoRegressive Seasonal HMM computed in this [tutorial](https://dmetivie.github.io/StochasticWeatherGenerators.jl/dev/examples/tuto_paper/).

````@example tuto_add_station_variable
file_hmm = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/refs/heads/master/assets/tuto_1/hmm_fit_K_4_d_1_m_1.jld")
begin
    hmm_infos = load(file_hmm)
    hmm_fit_full = hmm_infos["hmm"]
    hist = hmm_infos["hist"]
    θq_fit = hmm_infos["Q_param"]
end
````

Load the sequence of estimated hidden states for the historical sequence.

````@example tuto_add_station_variable
file_df_z = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/refs/heads/master/assets/tuto_1/z_hat_K_4_d_1_m_1.csv")
df_z = CSV.File(file_df_z) |> DataFrame
df_z[1:10,:] # Show the first lines of the dataframes
````

Filter by date and valid data. There are a few missing values that we impute using `Impute.Interpolate`.

````@example tuto_add_station_variable
begin
    data_stations_full = collect_data_INRAE.(station_path; show_warning=[:RR, :TX], impute_missing=[:RR, :TX])
    for df in data_stations_full
        @transform!(df, :RO = onefy.(:RR))
    end
end

data_stations = [innerjoin(df, df_z; on=:DATE) for df in data_stations_full]
````

## Fitting
We now fit the observed weather variables to seasonal models w.r.t. the hidden variables i.e. the models we fit depend continuously on the day of the year $t\in [1,366]$ and on the provided hidden state $Z \in [1,K]$.

### Rain Occurrences
Fit the Rain Occurrences of INRAE stations with respect to the given hidden states sequence.
These distributions are then cast into the HMM emission distribution. The pre-trained transitons matrices Qₜ are kept the same.
We could have added these new emission distributions to the existing one, however here we focus only on these four stations.

````@example tuto_add_station_variable
θ = zeros(K, D, memory_order, 2degree + 1)
B = Array{Bernoulli}(undef, K, T, D, memory_order)
@time "Fit RO" for (j, df) in enumerate(data_stations)
    B[:, :, j, :], θ[:, j, :, :] = fit_mle_RO(df, local_order, degree)
end
θy_fit = θ # hmm_infos["Y_param"] for old stations
a = zeros(K)
a[data_stations[1].z[1]] = 1.0 # Initial state of the HMM
hmm_fit = ARPeriodicHMM(a, hmm_fit_full.A, B)
````

### Rain Amounts

Here for simplicity we select the double exponential model for the rain amount.
We aim for a generic interface where one could easily change this choice to other univariate distribution e.g. `mix₀ = MixtureModel([Exponential(1), Gamma(1,2)], [1/2,1/2])` or `mix₀ = Pareto(1)`.
For now this is not supported (because of the seasonal fit).

````@example tuto_add_station_variable
@time "Fit Rain amounts" mix_allE = fit_mle_RR.(data_stations, local_order; mix₀=StochasticWeatherGenerators.mix_ini(T));
nothing #hide
````

The Gaussian copula covariance matrices are then estimated.

````@example tuto_add_station_variable
Σ²RR = cov_RR(data_stations, K)
````

### Temperature

We first fit the daily maximal temperature (we could have started with minimal temperatures).
We use the same principle as for `RR` i.e. first fit each univariate distribution and then the copula.

#### Temperature Max

````@example tuto_add_station_variable
@time "Fit TX" ar1sTX = fit_AR1.(data_stations, :TX, degree)
Σ²TX = cov_ar1(data_stations, ar1sTX, :TX)
````

#### Minimal Temperature
To directly fit the `TN` one could do the following
```julia
ar1sTN = fit_AR1.(data_stations, :TN, degree)
Σ²TN = cov_ar1(dropmissing.(data_stations), ar1sTN, :TN)
```
However this produce `TN` independantly of `TX` (and we can have `TN>TX`).
To prevent that we fit the positive difference `ΔT = TX-TN` with a Gamma distribution.
We will then simulate the `TN` conditionally to the `TX`

````@example tuto_add_station_variable
@time "Fit TN residuals" θ_ΔT = fit_TN.(data_stations, 1, T; print_level=0) # 1 is the degree
θ_cor = cor_groupbyTXTN.(data_stations, T)
f(θ) = Gamma(θ[1], θ[2]) # other options e.g. MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
f(t, θ) = f([σₜ(t, θ[1:(2+1)]), σₜ(t, θ[(2+2):end])])
````

### Solar Irradiance (QQ)

`QQ` must be positive, we will truncate at simulation time negative instances.

````@example tuto_add_station_variable
@time "Fit QQ" ar1sQQ = fit_AR1.(data_stations, :QQ, degree)
Σ²QQ = cov_ar1(dropmissing.(data_stations), ar1sQQ, :QQ)
````

###  Evapotranspiration Penman (ETPP)

`ETPP` must be positive, we will truncate at simulation time negative instances.

````@example tuto_add_station_variable
@time "Fit ETPP" ar1sETPP = fit_AR1.(data_stations, :ETPP, degree)
Σ²ETPP = cov_ar1(dropmissing.(data_stations), ar1sETPP, :ETPP)
````

## Simulation

### Initial conditions

````@example tuto_add_station_variable
y_ini = [@subset(df, :DATE .== Date(2000) - Day(1)).RO[1] for df in data_stations]'
tx_ini = [@subset(df, :DATE .== Date(2000)).TX[1] for df in data_stations]
z_ini = @subset(data_stations[1], :DATE .== Date(2000)).z[1]

tn_ini = [@subset(df, :DATE .== Date(2000)).TN[1] for df in data_stations]
qq_ini = [@subset(df, :DATE .== Date(2000)).QQ[1] for df in data_stations]
et_ini = [@subset(df, :DATE .== Date(2000)).ETPP[1] for df in data_stations]
````

### Generation

````@example tuto_add_station_variable
indicatrix(x) = x > zero(x) ? x : zero(x) # used to manually suppress negative `QQ` and `ETPP`

Random.seed!(50000)
````

````@example tuto_add_station_variable
NSIMU = 1000
NYEAR = 34

year_start = 1986
date_range = Date(year_start):Day(1):Date(year_start + NYEAR - 1, 12, 31)
n2t = dayofyear_Leap.(date_range)
years = unique(year.(date_range))
N = length(n2t)
````

````@example tuto_add_station_variable
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
````

### Results

Cast the results into `NSIMU = 1000` $\times$ `D=4` `DataFrame`s of `NYEAR = 34` years each, with columns `[:DATE, :RR, :RN, :TX, :QQ, :ETPP, :STAID]` where `STAID` is the unique station identifier. Also cast everything into a `Dict` for convenience.

````@example tuto_add_station_variable
dicts_simu = Dict(:RR => rs, :TN => tns, :TX => txs, :QQ => qqs, :ETPP => ets)
dfs_simu = [[DataFrame(:DATE => date_range, :RR => rs[j, :, i], :TN => tns[j, :, i], :TX => txs[j, :, i], :QQ => qqs[j, :, i], :ETPP => ets[j, :, i], :STAID => fill(data_stations[j].STAID[1], length(n2t))) for j in 1:D] for i in 1:NSIMU];
nothing #hide
````

To save the data frames you can do something like
```julia
map(enumerate(station_dep)) do (i,dep)
    CSV.write(joinpath(save_tuto_path,"swg_simu_station_dep_$(dep)_years_$(length(years)).csv"), dfs_simu[i])
end
```

## Plots

### Settings for plotting

Some settings and packages to have nice plots.

````@example tuto_add_station_variable
using StatsPlots, LaTeXStrings
using StatsPlots.PlotMeasures # To play with margin in Plots

gr() # plotly() # for interactive plots
default(fontfamily="Computer Modern")
cur_colors = get_color_palette(:auto, 100);
my_palette(K) = palette(vcat(cur_colors[1], [cur_colors[c] for c in 3:4], cur_colors[2]), K)

file_for_plot_utilities = download("https://raw.githubusercontent.com/dmetivie/StochasticWeatherGenerators.jl/master/examples/utilities_plot.jl")
include(file_for_plot_utilities)
vars = [:RR, :TN, :TX, :QQ, :ETPP]
````

### Plot correlations

#### Multisite Correlation for each variables

````@example tuto_add_station_variable
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
````

#### Correlations between weather variables at each sites.

````@example tuto_add_station_variable
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
````

### Univariate distributions

#### Marginal Distributions

````@example tuto_add_station_variable
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
            XX == :RR ? ylims!(plt_dist_univ[XX][j], 1e-4, 0, yaxis = :log10) : nothing
            XX == :RR && j == 2 ? xlims!(plt_dist_univ[XX][j], -1, 100, yaxis = :log10) : nothing # some simulated RR are super extreme and messing with the xaxis
        end
        [xlabel!(plt_dist_univ[XX][j], xlabel_string[XX]) for j in [3, 4]]
        [ylabel!(plt_dist_univ[XX][j], "PDF") for j in [1, 3]]
station_ndep
        [title!(plt_dist_univ[XX][j], station_ndep[j]) for j = 1:D]

        plts_dist_univ[XX] = plot(plt_dist_univ[XX]..., size=(1000, 700), bottom_margin=11px, left_margin=15px)
    end
end
````

##### Rainfall `RR`

````@example tuto_add_station_variable
plts_dist_univ[:RR]
````

##### Temperature max `TX`

````@example tuto_add_station_variable
plts_dist_univ[:TX]
````

##### Temperature min `TN`

````@example tuto_add_station_variable
plts_dist_univ[:TN]
````

##### Solar irradiance `QQ`

````@example tuto_add_station_variable
plts_dist_univ[:QQ]
````

##### Evapotranspiration `ETPP`

````@example tuto_add_station_variable
plts_dist_univ[:ETPP]
````

#### Monthly statistics

````@example tuto_add_station_variable
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
    end
end
````

##### Rainfall `RR`

````@example tuto_add_station_variable
plot(plts_month[:RR])
````

##### Temperature max `TX`

````@example tuto_add_station_variable
plts_month[:TX]
````

##### Temperature min `TN`

````@example tuto_add_station_variable
plts_month[:TN]
````

##### Solar irradiance `QQ`

````@example tuto_add_station_variable
plts_month[:QQ]
````

##### Evapotranspiration `ETPP`

````@example tuto_add_station_variable
plts_month[:ETPP]
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
