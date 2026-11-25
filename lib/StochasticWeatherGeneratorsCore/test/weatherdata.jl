@testset "WeatherData" begin
    t = station_table(; dates = Date(2000):Day(1):Date(2000, 3, 31))
    d = WeatherData(t; verbose = false)

    @testset "layout and accessors" begin
        @test stations(d) == [10, 20, 30]
        @test variables(d) == [:TX, :TN, :RR]
        @test SWGC.kinds(d) == (TX = :temperature, TN = :lower_temperature, RR = :rain)
        @test dates(d) == collect(Date(2000):Day(1):Date(2000, 3, 31))
        @test SWGC.calendar(d) == Leap366()
        @test size(d[:TX]) == (3, 91)
        @test d[:TX][2, 5] == t.TX[91 + 5]
        @test d["RR"] === d[:RR]
        @test_throws "available: TX, TN, RR" d[:WIND]
        @test coords(d) == [(2.5, 46.0), (3.0, 47.0), (3.5, 48.0)]
        @test SWGC.sites(d) == (station = [10, 20, 30], LAT = [46.0, 47.0, 48.0], LON = [2.5, 3.0, 3.5])
        @test sprint(show, MIME"text/plain"(), d) isa String
        cols = Tables.columns(d)
        @test Tables.columnnames(cols) == (:DATE, :STAID, :TX, :TN, :RR)
        @test isequal(DataFrame(d).TX, t.TX)
    end

    @testset "wide and long layouts agree" begin
        dl = WeatherData(long_table(t); verbose = false)
        @test dl == d
        @test_logs (:info, r"CO2") WeatherData(merge(long_table(t), (variable = replace(long_table(t).variable, "RR" => "CO2"),)))
    end

    @testset "aliases, kinds and unknown columns" begin
        ta = (date = t.DATE, station = t.STAID, TMAX = t.TX, PRCP = t.RR, tasmin = t.TN, CO2 = fill(330.0, length(t.TX)))
        da = @test_logs (:info, r"Dropping columns CO2.*kinds = \(CO2 = :continuous,\)") WeatherData(ta)
        @test variables(da) == [:TX, :RR, :TN]
        @test da[:TMAX] === da[:TX] === da[:tasmax]
        @test Tables.columnnames(Tables.columns(da)) == (:date, :station, :TMAX, :PRCP, :tasmin)
        @test_logs WeatherData(ta; verbose = false)
        dk = WeatherData(ta; kinds = (CO2 = :positive,), verbose = false)
        @test SWGC.kinds(dk).CO2 === :positive
        @test SWGC.kinds(WeatherData(ta; kinds = (TX = :continuous,), verbose = false)).TX === :continuous
        @test_throws "valid kinds are :rain" WeatherData(ta; kinds = (CO2 = :gas,))
        @test_throws "WIND, which is not a variable column" WeatherData(ta; kinds = (WIND = :positive,))
        # class-declared kinds for absent columns are ignored
        @test variables(WeatherData(ta; algkinds = (WIND = :positive, CO2 = :custom), verbose = false)) == [:TX, :RR, :TN, :CO2]
        @test_throws "TMAX and TX both map to TX" WeatherData(merge(ta, (TX = t.TX,)))
        @test variables(WeatherData(ta; variables = [:RR, :TMAX], verbose = false)) == [:RR, :TX]
        @test SWGC.kinds(WeatherData(ta; variables = [:CO2], verbose = false)).CO2 === :continuous
        @test_throws "no column for variable QQ" WeatherData(ta; variables = [:QQ])
        @test_throws "no weather variable found" WeatherData((DATE = t.DATE, CO2 = t.TX); verbose = false)
    end

    @testset "date and station columns" begin
        @test WeatherData((jour = t.DATE, id = t.STAID, TX = t.TX); verbose = false).datecol === :jour
        # `station` wins over `id`
        tp = (DATE = t.DATE, id = fill(1, length(t.TX)), station = t.STAID, TX = t.TX)
        @test stations(WeatherData(tp; verbose = false)) == [10, 20, 30]
        @test_throws "duplicate" WeatherData(tp; station = :id, verbose = false)
        one = (DATE = collect(Date(2000):Day(1):Date(2000, 1, 10)), TX = randn(10))
        d1 = WeatherData(one)
        @test stations(d1) == [1] && d1.stationcol === nothing
        @test Tables.columnnames(Tables.columns(d1)) == (:DATE, :TX)
        @test_throws "no date column" WeatherData((a = 1:3, TX = randn(3)))
        @test_throws "no column foo" WeatherData(one; date = :foo)
        @test_throws "not dates" WeatherData((DATE = string.(one.DATE), TX = one.TX); date = :DATE)
        mid = (DATE = DateTime.(one.DATE), TX = one.TX)
        @test dates(WeatherData(mid)) == one.DATE
        @test_throws "aggregate to daily" WeatherData((DATE = DateTime.(one.DATE) .+ Hour(12), TX = one.TX))
        @test_throws "missing values" WeatherData((DATE = [missing; one.DATE[2:end]], TX = one.TX))
        @test_throws ArgumentError WeatherData([1, 2, 3])
    end

    @testset "duplicate stations: time conflicts" begin
        @test_throws "duplicate (date, station) rows" WeatherData((DATE = t.DATE, TX = t.TX))
    end

    @testset "values" begin
        n = length(t.TX)
        msg = try
            WeatherData((DATE = [t.DATE; t.DATE[1:5]], STAID = [t.STAID; t.STAID[1:5]], TX = [t.TX; t.TX[1:5]]))
        catch e
            sprint(showerror, e)
        end
        @test occursin("5 duplicate (date, station) rows", msg)
        @test occursin("(2000-01-01, 10), (2000-01-02, 10), (2000-01-03, 10)", msg)
        @test !occursin("2000-01-04", msg)
        dupl = long_table(t)
        @test_throws "duplicate (date, station, variable)" WeatherData(rows(dupl, [1:length(dupl.DATE); 1]))

        tx = copy(t.TX); tx[3] = NaN; tx[7] = NaN
        dn = @test_logs (:warn, r"2 NaN values.*TX: 2") WeatherData((DATE = t.DATE, STAID = t.STAID, TX = tx))
        @test ismissing(dn[:TX][1, 3]) && count(ismissing, dn[:TX]) == 2
        tx[3] = Inf
        @test_throws "infinite value Inf in TX at station 10 on 2000-01-03" WeatherData((DATE = t.DATE, STAID = t.STAID, TX = tx))
        @test_throws "TX is not numeric" WeatherData((DATE = t.DATE, STAID = t.STAID, TX = string.(t.TX)))
        dm = WeatherData((DATE = t.DATE, STAID = t.STAID, TX = Union{Missing,Float64}[i == 2 ? missing : x for (i, x) in enumerate(t.TX)]))
        @test ismissing(dm[:TX][1, 2])
        @test eltype(WeatherData((DATE = t.DATE, STAID = t.STAID, RR = round.(Int, t.RR)))[:RR]) == Union{Missing,Float64}
        @test eltype(WeatherData((DATE = t.DATE, STAID = t.STAID, RR = Float32.(t.RR)))[:RR]) == Union{Missing,Float32}
        @test_throws "no rows" WeatherData((DATE = Date[], TX = Float64[]))
        @test_throws "no columns" WeatherData(NamedTuple())
        @test_logs (:warn, r"TX looks like Kelvin") WeatherData((DATE = t.DATE, STAID = t.STAID, TX = t.TX .+ 273.15))
        @test_logs WeatherData((DATE = t.DATE, STAID = t.STAID, RR = t.RR .+ 273.15))
    end

    @testset "gaps and calendars" begin
        keep = findall(dt -> dt != Date(2000, 1, 5), t.DATE)
        dg = WeatherData(rows(t, keep); verbose = false)
        @test length(dates(dg)) == 91 && all(ismissing, dg[:TX][:, 5])
        @test_throws "NoLeap365" WeatherData(t; calendar = NoLeap365(), verbose = false)
        nofeb29 = findall(dt -> dt != Date(2000, 2, 29), t.DATE)
        dn = WeatherData(rows(t, nofeb29); calendar = NoLeap365(), verbose = false)
        @test length(dates(dn)) == 90 && Date(2000, 2, 29) ∉ dates(dn) && !any(ismissing, dn[:TX])
        ds = d360_dates(2000, 2000)
        d3 = WeatherData((DATE = ds, TX = randn(360)); calendar = Day360())
        @test dates(d3) == ds && SWGC.calendar(d3) == Day360()
        @test_throws "gap-free" WeatherData((DATE = ds[[1:10; 12:360]], TX = randn(359)); calendar = Day360())
        @test_throws "30 days" WeatherData((DATE = collect(Date(2000):Day(1):Date(2000, 2, 1)), TX = randn(32)); calendar = Day360())
    end

    @testset "site metadata" begin
        @test_throws "LAT varies within station 10" WeatherData(merge(t, (LAT = [46.1; t.LAT[2:end]],)); verbose = false)
        lat = Union{Missing,Float64}[t.LAT...]
        lat[1] = missing
        @test SWGC.sites(WeatherData(merge(t, (LAT = lat,)); verbose = false)).LAT == [46.0, 47.0, 48.0]
        @test_throws "LAT is missing for station 10" WeatherData(merge(t, (LAT = [s == 10 ? missing : 1.0 for s in t.STAID],)); verbose = false)
        @test_throws "both give site metadata LAT" WeatherData(merge(t, (latitude = t.LAT,)); verbose = false)
        dp = WeatherData((DATE = t.DATE, STAID = t.STAID, TX = t.TX, X = t.LON .* 100, Y = t.LAT .* 100); crs = :projected)
        @test coords(dp) == [(250.0, 4600.0), (300.0, 4700.0), (350.0, 4800.0)]
        @test distances(dp)[1, 2] ≈ hypot(50, 100)
        @test_throws ":lonlat or :projected" WeatherData(t; crs = :utm)
        @test_throws "no station coordinates" distances(WeatherData(station_table(; coords = false); verbose = false))
        @test coords(WeatherData(station_table(; coords = false); verbose = false)) === nothing
    end

    @testset "Haversine" begin
        h = SWGC.Haversine()
        @test h((0.0, 0.0), (1.0, 0.0)) ≈ 6371 * π / 180
        @test h((5.0, 10.0), (5.0, 11.0)) ≈ 6371 * π / 180
        @test h((2.0, 45.0), (2.0, 45.0)) == 0
        @test h((0.0, 0.0), (180.0, 0.0)) ≈ 6371 * π
        D = distances(d)
        @test D ≈ D' && all(iszero, D[i, i] for i in 1:3) && D[1, 2] > 0
        @test distances(d; metric = (p, q) -> 1.0) == [0 1 1; 1 0 1; 1 1 0]
    end

    @testset "array constructors" begin
        A = cat(d[:TX], d[:TN], d[:RR]; dims = 3)
        dt = WeatherData((date = t.DATE, station = t.STAID, TX = t.TX, TN = t.TN, RR = t.RR, LAT = t.LAT, LON = t.LON))
        da = WeatherData(Float64.(A), dates(d), [10, 20, 30]; vars = [:TX, :TN, :RR], coords = (LAT = [46.0, 47, 48], LON = [2.5, 3, 3.5]))
        @test da == dt
        dn = WeatherData((TX = d[:TX], TN = d[:TN], RR = d[:RR]), dates(d); stations = [10, 20, 30],
            coords = (LON = [2.5, 3, 3.5], LAT = [46.0, 47, 48]))
        @test dn == dt
        @test stations(WeatherData((TX = randn(91),), dates(d))) == [1]
        @test_throws DimensionMismatch WeatherData((TX = randn(2, 90),), dates(d))
        @test_throws DimensionMismatch WeatherData(A, dates(d); vars = [:TX, :TN])
        @test_throws "followed by" WeatherData((TX = randn(1, 3),), [Date(2000), Date(2000, 1, 3), Date(2000, 1, 4)])
        @test_throws "unknown site metadata" WeatherData((TX = randn(1, 3),), dates(d)[1:3]; coords = (Z = [1.0],))
        @test_throws "TX and TMAX" WeatherData((TX = randn(1, 3), TMAX = randn(1, 3)), dates(d)[1:3])
    end

    @testset "subsets and merge" begin
        ds = d[Date(2000, 1, 10):Date(2000, 1, 19)]
        @test dates(ds) == collect(Date(2000, 1, 10):Day(1):Date(2000, 1, 19)) && ds[:TX] == d[:TX][:, 10:19]
        @test_throws "step of one day" d[Date(2000, 1, 10):Day(2):Date(2000, 1, 19)]
        @test_throws "outside the data" d[Date(1999, 12, 30):Date(2000, 1, 19)]
        @test_throws "empty" d[Date(2000, 1, 19):Date(2000, 1, 10)]
        @test_throws "index days by dates" d[1:10]

        t2 = station_table(; stations = [40], dates = Date(2000, 3, 1):Day(1):Date(2000, 4, 30), seed = 2)
        d2 = WeatherData((DATE = t2.DATE, STAID = t2.STAID, TX = t2.TX, QQ = t2.RR, LAT = t2.LAT, LON = t2.LON))
        m = merge(d, d2)
        @test stations(m) == [10, 20, 30, 40]
        @test dates(m) == collect(Date(2000):Day(1):Date(2000, 4, 30))
        @test variables(m) == [:TX, :TN, :RR, :QQ]
        @test m[:TX][1:3, 1:91] == d[:TX] && all(ismissing, m[:TX][4, 1:60]) && m[:TX][4:4, 61:end] == d2[:TX]
        @test all(ismissing, m[:QQ][1:3, :])
        @test SWGC.sites(m).LAT == [46.0, 47, 48, 46]
        @test merge(d, (DATE = t2.DATE, STAID = t2.STAID, TX = t2.TX, QQ = t2.RR, LAT = t2.LAT, LON = t2.LON); verbose = false) == m
        @test_throws "are in both" merge(d, d)
        @test_throws "cannot merge data on" merge(d, WeatherData(rows(t2, findall(!=(Date(2000, 2, 29)), t2.DATE)); calendar = NoLeap365(), verbose = false))
        d2b = WeatherData((DATE = t2.DATE, STAID = t2.STAID, TX = t2.TX); kinds = (TX = :continuous,))
        @test_throws "kind :temperature in the first" merge(d, d2b)
        mc = @test_logs (:info, r"Dropping site metadata LAT, LON") merge(d, WeatherData((DATE = t2.DATE, STAID = t2.STAID, TX = t2.TX)))
        @test isempty(mc.sitemeta)
    end
end
