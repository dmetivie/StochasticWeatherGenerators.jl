@testset "fit and the class protocol" begin
    t = station_table(; dates = Date(2000):Day(1):Date(2001, 12, 31))
    data = WeatherData(t; verbose = false)
    w = Date(2001):Date(2001, 12, 31)
    g = fit(t, SeasonalClimatology(); window = w, seed = 1, verbose = false)

    @testset "fitted generator" begin
        @test g isa FittedGenerator{SeasonalClimatology}
        @test spec(g) == SeasonalClimatology() || spec(g) isa SeasonalClimatology
        @test dates(g) == collect(w) && stations(g) == [10, 20, 30] && variables(g) == [:TX, :TN, :RR]
        info = fitinfo(g)
        @test info.seed === UInt64(1) && info.rng_derivation == SWGC.RNG_DERIVATION
        @test info.solver == "default" && info.names == [:TX, :TN, :RR]
        @test haskey(info.versions, "julia") && haskey(info.versions, "StochasticWeatherGeneratorsCore")
        @test info.distances ≈ distances(data) && coords(g) == coords(data)
        @test SWGC.sites(g) == SWGC.sites(data) && SWGC.calendar(g) == Leap366()
        @test distances(g) === info.distances && distances(g; metric = (p, q) -> 2.0)[1, 2] == 2.0
        @test info.history.seed === UInt64(1)
        # days before the window reach the class as `past` only
        @test info.history.past_days == 366 && info.history.past_last == Date(2000, 12, 31)
        g0 = fit(t, SeasonalClimatology(); seed = 1, verbose = false)
        @test fitinfo(g0).history.past_days == 0 && dates(g0) == dates(data)
        @test fit(data, SeasonalClimatology(); window = w, seed = 1).params.components[1].μ == g.params.components[1].μ
        @test fitinfo(fit(t, SeasonalClimatology(); window = w, verbose = false)).seed isa UInt64
        @test fitinfo(fit(t, SeasonalClimatology(); window = w, seed = -1, verbose = false)).seed == typemax(UInt64)
        @test_throws "Bool" fit(t, SeasonalClimatology(); seed = true, verbose = false)
        @test sprint(show, MIME"text/plain"(), g) isa String && occursin("SeasonalClimatology", sprint(show, g))
        @test_throws "outside the data" fit(t, SeasonalClimatology(); window = Date(1999):Date(2000, 12, 31))
        @test_throws "window must be ≥ 0" fit(t, SeasonalClimatology(window = -1); verbose = false)
        @test_throws "joint variable QQ" fit(t, SeasonalClimatology(joint = ((:TX, :QQ),)); verbose = false)
    end

    @testset "likelihood statistics" begin
        ll, k, n = loglikelihood(g), dof(g), nobs(g)
        @test isfinite(ll) && k == 3 * 3 * 733 && n == 3 * 365 * 3
        @test bic(g) ≈ -2ll + k * log(n) && aic(g) ≈ -2ll + 2k
        @test_throws "AICc needs nobs > dof + 1 (nobs = 3285, dof = 6597)" aicc(g)
        ge = FittedGenerator(SeasonalClimatology(), nothing, SWGC.FitInfo(data; loglikelihood = -10, dof = 2, nobs = 100))
        @test aic(ge) == 24 && bic(ge) ≈ 20 + 2log(100) && aicc(ge) ≈ 24 + 12 / 97
        @test likelihood_kind(g) === :exact === likelihood_kind(SeasonalClimatology())
        @test_throws "by = :block" bic(g; by = :block)
        @test_throws "aic(g; by = :station)" aic(g; by = :station)
        gc = fit(t, CompositeToy(); window = w, verbose = false)
        @test_throws "CLIC is reserved" bic(gc)
        @test_throws "CLIC is reserved" aic(gc)
        @test loglikelihood(gc) == ll                                   # reported, not usable for AIC/BIC
        gm = fit(t, MissingTolerantToy(); verbose = false)
        @test_throws "have no likelihood" bic(gm)
        @test_throws "do not report a log-likelihood" loglikelihood(gm)
        @test_throws "do not report their degrees of freedom" dof(gm)
        @test_throws "number of observations" nobs(gm)
        tiny = fit(rows(t, findall(<=(Date(2000, 1, 3)), t.DATE)), SeasonalClimatology(); verbose = false)
        @test_throws "AICc needs nobs > dof + 1" aicc(tiny)
    end

    @testset "class hooks and errors" begin
        @test_throws "only daily" fit(t, WeeklyToy())
        @test_throws "likelihood_kind :weird" fit(t, WeirdKindToy())
        @test_throws r"returned a .*NamedTuple" fit(t, BrokenToy(:wrongtype); verbose = false)
        @test_throws "NoFitToy does not implement `fit_generator" fit(t, NoFitToy(); verbose = false)
        @test_throws "returned a Symbol" fit(t, BadResolveToy(); verbose = false)
        # requested_variables: the class sees TX only
        gp = @test_logs (:info, r"Dropping columns TN, RR: they are not in `variables`") fit(t, ProbeToy())
        @test variables(gp) == [:TX]
        @test variables(fit(data, ProbeToy())) == [:TX]
        @test_throws "no column for variable TX" fit((DATE = t.DATE, STAID = t.STAID, RR = t.RR), ProbeToy())
        @test_throws "available: RR" fit(WeatherData((DATE = t.DATE, STAID = t.STAID, RR = t.RR)), ProbeToy())
        # declared kinds keep a column that would be dropped
        gw = fit(merge(t, (WIND = abs.(t.TX),)), SeasonalClimatology(); window = w, verbose = false)
        @test SWGC.kinds(gw).WIND === :positive
        @test SWGC.capabilities(SeasonalClimatology(K = 2)) == (regimes = true,)
        @test SWGC.capabilities(NoFitToy()) == (;) && likelihood_kind(NoFitToy()) === :none
        @test SWGC.timestep(NoFitToy()) == Day(1) && SWGC.regime_source(NoFitToy()) === nothing
        @test SWGC.requested_variables(NoFitToy()) === nothing && SWGC.declared_kinds(NoFitToy()) == (;)
        @test SWGC.nregimes(gp) === nothing && SWGC.regime_source(g) === nothing
    end

    @testset "complete data" begin
        tx = Union{Missing,Float64}[t.TX...]
        tx[(t.STAID .== 20) .& (t.DATE .== Date(2001, 6, 1))] .= missing
        tg = merge(t, (TX = tx,))
        @test_throws "TX has 1 missing day at station 20" fit(tg, SeasonalClimatology(); verbose = false)
        @test fit(tg, SeasonalClimatology(); window = Date(2001, 6, 2):Date(2001, 12, 31), verbose = false) isa FittedGenerator
        gm = fit(tg, MissingTolerantToy(); verbose = false)                 # handles_missing skips the check
        @test params(gm)[:TX][2] ≈ mean(skipmissing(tx[t.STAID .== 20]))
        late = findall(i -> !(t.STAID[i] == 20 && t.DATE[i] < Date(2000, 7, 1)), eachindex(t.DATE))
        gl = @test_logs (:info, r"starts with station 20") fit(rows(t, late), SeasonalClimatology())
        @test first(dates(gl)) == Date(2000, 7, 1)
    end

    @testset "routing and read-outs" begin
        @test g[:TX] !== g[:TN] && g[:TX].vars == [:TX]
        gj = fit(t, SeasonalClimatology(joint = ((:TX, :TN),)); window = w, verbose = false)
        @test gj[:TX] === gj[:TN] && gj[:TX].vars == [:TX, :TN]
        @test gj[:TMAX] === gj[:TX] === gj["tasmin"]
        @test gj[:RR].vars == [:RR]
        @test_throws "available: TX, TN, RR" gj[:QQ]
        @test SWGC.simulated_variables(gj) == [:TX, :TN, :RR]
        ta = (date = t.DATE, station = t.STAID, TMAX = t.TX, PRCP = t.RR)
        ga = fit(ta, SeasonalClimatology(); window = w, verbose = false)
        @test ga[:TMAX] === ga[:TX] && ga[:PRCP] === ga[:RR]
        @test_throws "available: TMAX, PRCP" ga[:TN]
        gp = fit(t, ProbeToy(); verbose = false)
        @test_throws "do not expose components" gp[:TX]
        for f in (SWGC.transitions, SWGC.regime_model, decode)
            @test_throws "is not available for SeasonalClimatology generators" f(g)
        end
        @test_throws "`decode` is not available" decode(g, data; method = :viterbi)
        @test_throws "`correlation` is not available" correlation(g, :RR; regime = 1)
    end
end
