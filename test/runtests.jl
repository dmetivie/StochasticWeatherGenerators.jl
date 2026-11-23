using StochasticWeatherGenerators
using Test

# The fitting routines live in a package extension of `SmoothPeriodicStatsModels`, which itself
# only defines them once Optimization and SimpleNonlinearSolve are loaded: that is what makes
# `fit_mle_RR`, `fit_AR1`, `fit_TN` and friends exist at all. Every fit takes its optimization
# algorithm explicitly, here `LBFGS` from OptimizationOptimJL.
using SmoothPeriodicStatsModels
using Optimization, SimpleNonlinearSolve
using OptimizationOptimJL
const SOLVER = LBFGS()

using CSV, DataFrames, DataFramesMeta, Dates
using Distributions, Random, Statistics

import Makie # loads the plot functions (MakieExt); no backend is needed to build the plots

const SWG = StochasticWeatherGenerators
const SPSM = SmoothPeriodicStatsModels

@testset "StochasticWeatherGenerators" begin

    @testset "the SmoothPeriodicStatsModels extension is loaded" begin
        @test !isnothing(Base.get_extension(SWG, :SmoothPeriodicStatsModelsExt))
    end

    @testset "Makie extension: errorline and errorlinehist" begin
        @test !isnothing(Base.get_extension(SWG, :MakieExt))
        error_message(f) =
            try
                f()
                ""
            catch e
                sprint(showerror, e)
            end

        # one row per x value; NaN are skipped and an all-NaN row gives NaN
        y = [1.0 2.0 3.0; 4.0 NaN 6.0; NaN NaN NaN]
        _, _, p = errorline(1:3, y; errortype=:percentile, percentiles=[0, 100], centertype=:median)
        @test p.y_center[][1:2] == [2.0, 5.0]
        @test p.y_lower[][1:2] == [1.0, 4.0]
        @test p.y_upper[][1:2] == [3.0, 6.0]
        @test all(isnan, (p.y_center[][3], p.y_lower[][3], p.y_upper[][3]))

        _, _, p = errorscatter(y[1:2, :]; errortype=:sem) # x defaults to 1:size(y, 1)
        @test p.x[] == 1:2
        @test p.y_upper[] - p.y_center[] ≈ [std([1, 2, 3]) / sqrt(3), std([4, 6]) / sqrt(2)]
        p.y[] = y[1:2, :] .+ 10 # the plot follows its data
        @test p.y_center[] ≈ [12.0, 15.0]

        # the samples share the bins; each column is the histogram of one sample
        _, _, h = errorlinehist([[1, 1, 2], [2, 3, 3, 3]]; bins=1:4, normalization=:probability)
        @test h.bin_x[] == [1, 2, 3]
        @test h.bin_y[] ≈ [2/3 0; 1/3 1/4; 0 3/4] atol = 1e-12
        rng = MersenneTwister(1)
        _, _, h = errorlinehist([randn(rng, 100) for _ in 1:3]; bins=10) # the bins cover every value
        bin_width = h.bin_x[][2] - h.bin_x[][1]
        @test size(h.bin_y[]) == (10, 3)
        @test vec(sum(h.bin_y[]; dims=1)) * bin_width ≈ ones(3) rtol = 1e-9

        @test occursin("one row per x value", error_message(() -> errorline(1:3, rand(2, 4))))
        @test occursin("errortype", error_message(() -> errorline(1:3, rand(3, 2); errortype=:foo)))
        @test occursin("errorstyle", error_message(() -> errorline(1:3, rand(3, 2); errorstyle=:foo)))
        @test occursin("sorted", error_message(() -> errorlinehist([rand(5)]; bins=[0, 2, 1])))
    end

    # ## Synthetic data: the fits, against parameters we know
    #
    # These are fast and deterministic. The station-data tests below check that the same
    # routines survive real inputs; these check that they get the right answer.

    @testset "mix_ini and fit_mle_RR" begin
        rng = MersenneTwister(2026)
        T, K = 60, 2

        mix₀ = mix_ini(T)
        @test mix₀ isa SPSM.SmoothSeasonalMixture
        @test ncomponents(mix₀) == 2
        @test length(parameters(mix₀).Y) == 6
        @test component_coefs(mix₀, 2) == [2.0, 0.0, 0.0]   # θσ20 default
        @test vec(parameters(mix₀).α) == [0.0, 0.0, 0.0]    # θα0 default
        @test length(mixtures(mix₀)) == T

        # the degree is read off the length of θσ10, not hard-coded
        mix2 = mix_ini(T; θσ10=zeros(5), θσ20=[2.0, 0, 0, 0, 0], θα0=zeros(5))
        @test length(component_coefs(mix2, 1)) == 5

        truth(t) = MixtureModel([Exponential(exp(0.2 + 0.4cos(2π * t))),
                Exponential(exp(2.0 - 0.3sin(2π * t)))], [0.35, 0.65])
        n2t_rain = [repeat(1:T, 40) for _ in 1:K]
        R_per_z = [[rand(rng, truth(t / T)) for t in n2t_rain[k]] for k in 1:K]

        mix_allE = fit_mle_RR(R_per_z, n2t_rain, mix₀; solver=SOLVER, maxiter=200, tol=1e-4)
        @test size(mix_allE) == (K, T)
        @test mix_allE[1, 1] isa MixtureModel
        @test all(sum(probs(m)) ≈ 1 for m in mix_allE)
        # both scales are recovered, in whichever order EM labelled them
        scales = sort([scale(c) for c in components(mix_allE[1, 1])])
        @test scales ≈ sort([exp(0.2 + 0.4cos(2π / T)), exp(2.0 - 0.3sin(2π / T))]) rtol = 0.2
    end

    @testset "fit_AR1" begin
        rng = MersenneTwister(2027)
        T, K = 60, 2
        μ = [0.5 + 0.3cos(2π * t / T) for t = 1:T]
        ρ = fill(0.4, T)
        σ = fill(0.5, T)
        y = rand(rng, SPSM.AR1(μ, ρ, σ), 6000; y₁=0.0)
        n2t = SPSM.n_to_t(6000, T)

        ar = fit_AR1(y, n2t, 1, T; solver=SOLVER)
        @test ar isa SPSM.AR1
        @test maximum(abs, ar.μ .- μ) < 0.1
        @test maximum(abs, ar.ρ .- ρ) < 0.1
        @test maximum(abs, ar.σ .- σ) < 0.1

        # one AR(1) per hidden state: the curves become K × T
        z = rand(rng, 1:K, 6000)
        arK = fit_AR1(y, z, n2t, 1, K, T; solver=SOLVER)
        @test size(arK.ρ) == (K, T)
        @test size(arK.μ) == (K, T)

        # The DataFrame entry point, which is what the tutorials call. It reads the period
        # index off the dates, so the period really is the year: `Tyear = 366`. The series is on
        # the scale of a temperature, a mean of about 15, which the fit must start near.
        Tyear = 366
        dates = Date(2000, 1, 1):Day(1):(Date(2000, 1, 1)+Day(5999))
        n2ty = dayofyear_Leap.(dates)
        μy = [4.5 + 2cos(2π * t / Tyear) for t = 1:Tyear]
        yy = rand(rng, SPSM.AR1(μy, fill(0.7, Tyear), fill(2.0, Tyear)), n2ty; y₁=15.0)
        zy = rand(rng, 1:K, length(dates))
        df = DataFrame(DATE=collect(dates), TX=yy, z=zy)

        arD = fit_AR1(df, :TX, 1, Tyear; solver=SOLVER)           # with `z`: one AR(1) per state
        @test size(arD.ρ) == (K, Tyear)
        @test maximum(abs, arD.ρ .- 0.7) < 0.1
        ar1 = fit_AR1(select(df, Not(:z)), :TX, 1, Tyear; solver=SOLVER)   # without: a single series
        @test ar1.ρ isa AbstractVector
        @test length(ar1.ρ) == Tyear
        @test maximum(abs, ar1.ρ .- 0.7) < 0.1
        @test maximum(abs, ar1.μ .- μy) < 0.5
        @test maximum(abs, ar1.σ .- 2.0) < 0.15

        # A period that does not match the index is caught, rather than surfacing as a
        # `BoundsError` from inside the sufficient statistics.
        @test_throws ArgumentError fit_AR1(df, :TX, 1, 60; solver=SOLVER)
    end

    @testset "fit_TN and rand_cond" begin
        rng = MersenneTwister(11)
        T, K = 366, 2
        dates = Date(2000, 1, 1):Day(1):Date(2019, 12, 31)
        n2t = dayofyear_Leap.(dates)
        z = rand(rng, 1:K, length(dates))

        shp(k, t) = exp((k == 1 ? 1.2 : 1.8) + 0.3cos(2π * t / T))
        scl(k, t) = exp((k == 1 ? 0.4 : 0.9) - 0.2sin(2π * t / T))
        ΔT = [rand(rng, Gamma(shp(z[n], t), scl(z[n], t))) for (n, t) in enumerate(n2t)]
        TX = 15 .+ 10sin.(2π .* n2t ./ T) .+ randn(rng, length(dates))
        df = DataFrame(DATE=collect(dates), TX=TX, TN=TX .- ΔT, z=z)

        # Both degrees. `rand_cond` used to rebuild the marginal from a coefficient vector it
        # sliced assuming degree 1, which silently mis-read anything else; it now reads the
        # marginal off the fit.
        for 𝐃𝐞𝐠 in (1, 2)
            fits = fit_TN(df, 𝐃𝐞𝐠, T; solver=SOLVER)
            @test length(fits) == K
            @test all(f -> f isa SPSM.TimeVaryingFit, fits)
            @test length(coef(fits[1])) == 2 * (2𝐃𝐞𝐠 + 1)
            @test fits[1](100) isa Gamma

            if 𝐃𝐞𝐠 == 1
                for k = 1:K
                    @test coef(fits[k], :α)[1] ≈ (k == 1 ? 1.2 : 1.8) atol = 0.1
                    @test coef(fits[k], :θ)[1] ≈ (k == 1 ? 0.4 : 0.9) atol = 0.1
                end
                @test curve(fits[1], :α, 1:T) ≈ [shp(1, t) for t = 1:T] rtol = 0.15
                @test curve(fits[2], :θ, 1:T) ≈ [scl(2, t) for t = 1:T] rtol = 0.15
            end

            θ_cor = [[0.5, 0.0, 0.0] for _ = 1:K]
            ϵ = randn(rng, length(dates))
            ΔT_sim = rand_cond(MersenneTwister(3), ϵ, z, fits, θ_cor, n2t, T)
            @test length(ΔT_sim) == length(dates)
            @test all(>(0), ΔT_sim)
            @test mean(ΔT_sim) ≈ mean(ΔT) rtol = 0.15
        end
    end

    # ## Real station data, reduced
    #
    # The same pipeline the tutorial runs, on the station files this package ships and the
    # hidden states it ships with them, so there is no download and no Baum-Welch. Three
    # stations, twelve years, a handful of simulations: enough to catch a shape or convention
    # error on real inputs without the tutorial's runtime.

    @testset "end to end on the bundled station data" begin
        weather = joinpath(pkgdir(SWG), "weather_files")
        zfile = joinpath(pkgdir(SWG), "assets", "tuto_1", "z_hat_K_4_d_1_m_1.csv")
        if !isdir(weather) || !isfile(zfile)
            @warn "station data not found, skipping the end-to-end testset"
        else
            T, K, 𝐃𝐞𝐠, local_order = 366, 4, 1, 1
            STAID = [32, 33, 34]
            date_start, date_end = Date(1980), Date(1991) - Day(1)
            date_start_w_memory = date_start - Day(local_order)

            data_stations = collect_data_ECA.(STAID, date_start_w_memory, date_end, weather,
                "RR"; portion_valid_data=1, skipto=22, header=21)
            @test all(df -> df isa DataFrame, data_stations)
            D = length(data_stations)
            for df in data_stations
                @transform!(df, :RO = onefy.(:RR))
            end

            # the hidden states this package ships, restricted to the window above
            df_z = CSV.read(zfile, DataFrame)
            data_stations_z = map(data_stations) do df
                innerjoin(df, df_z, on=:DATE)
            end
            @test all(df -> nrow(df) > 3000, data_stations_z)
            @test Set(data_stations_z[1].z) ⊆ Set(1:K)

            # --- rain occurrences, conditional on the states
            B, θ = fit_mle_RO(data_stations_z[1], local_order, 𝐃𝐞𝐠, K, T; solver=SOLVER)
            @test size(B) == (K, T, 1, 2^local_order)
            @test size(θ) == (K, 1, 2^local_order, 2𝐃𝐞𝐠 + 1)
            @test all(b -> 0 ≤ succprob(b) ≤ 1, B)

            # --- rain amounts: the seasonal mixture, per station and per state
            mix_allE = fit_mle_RR.(data_stations_z, local_order; solver=SOLVER, mix₀=mix_ini(T),
                maxiter=30, tol=1e-3)
            @test length(mix_allE) == D
            @test all(m -> size(m) == (K, T), mix_allE)
            @test all(m -> sum(probs(m)) ≈ 1, mix_allE[1])
            @test all(m -> all(>(0), scale.(components(m))), mix_allE[1])
            Σ²RR = cov_RR(data_stations_z, K)
            @test length(Σ²RR) == K
            @test all(S -> size(S) == (D, D), Σ²RR)

            # --- a short simulation off the fitted amounts
            n2t = dayofyear_Leap.(data_stations_z[1].DATE)
            N = length(n2t)
            zs = data_stations_z[1].z
            ys = BitMatrix(stack([df.RO for df in data_stations_z]; dims=1))
            rs = rand_RR(mix_allE, n2t, zs, ys, Σ²RR)
            @test size(rs) == (D, N)
            @test all(>=(0), rs)
            @test all(rs[ys.==0] .== 0)     # no rain on a dry day

            # --- temperature: an AR(1) per hidden state on a real series
            #
            # Only `RR` ships in the `ECA_blend_rr/` layout `collect_data_ECA` expects, so the
            # temperature series here is the wet-day rainfall itself on the log scale: not a
            # temperature, but a real, seasonal, autocorrelated series, which is what the fit
            # is being exercised on.
            df_ar = @chain copy(data_stations_z[1]) begin
                @transform(:X = log1p.(:RR * 0.1))
                select(:DATE, :X, :z)
            end
            ar1s = fit_AR1(df_ar, :X, 𝐃𝐞𝐠, T; solver=SOLVER)
            @test size(ar1s.ρ) == (K, T)
            @test all(-1 .< ar1s.ρ .< 1)      # the link keeps every state stationary
            @test all(>(0), ar1s.σ)
            @test all(isfinite, ar1s.μ)
        end
    end
end
