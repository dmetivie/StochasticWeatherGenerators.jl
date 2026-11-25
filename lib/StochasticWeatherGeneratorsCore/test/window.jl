@testset "window and complete data" begin
    # station 20 starts later, station 30 ends earlier
    t = station_table(; dates = Date(2000):Day(1):Date(2000, 12, 31))
    late = findall(i -> !(t.STAID[i] == 20 && t.DATE[i] < Date(2000, 2, 1)), eachindex(t.DATE))
    early = findall(i -> !(t.STAID[i] == 30 && t.DATE[i] > Date(2000, 11, 30)), eachindex(t.DATE))
    d = WeatherData(rows(t, intersect(late, early)); verbose = false)

    w = @test_logs (:info, r"starts with station 20, ends with station 30") SWGC.default_window(d)
    @test w == Date(2000, 2, 1):Day(1):Date(2000, 11, 30)
    @test_logs SWGC.default_window(d; verbose = false)
    @test_logs SWGC.default_window(WeatherData(t; verbose = false))        # no line when nothing is cut
    @test check_complete(d, w) === nothing
    @test check_complete(d[w]) === nothing

    # a variable that starts later at one station limits the window too
    tn = Union{Missing,Float64}[t.TN...]
    tn[(t.STAID .== 10) .& (t.DATE .< Date(2000, 3, 1))] .= missing
    @test SWGC.default_window(WeatherData(merge(t, (TN = tn,)); verbose = false); verbose = false) ==
          Date(2000, 3, 1):Day(1):Date(2000, 12, 31)

    @testset "empty overlap" begin
        a = rows(t, findall(i -> t.STAID[i] != 10 || t.DATE[i] < Date(2000, 3, 1), eachindex(t.DATE)))
        b = rows(a, findall(i -> a.STAID[i] != 20 || a.DATE[i] > Date(2000, 6, 1), eachindex(a.DATE)))
        msg = try
            SWGC.default_window(WeatherData(b; verbose = false))
        catch e
            sprint(showerror, e)
        end
        @test occursin("share no period", msg)
        @test occursin("10: 2000-01-01 to 2000-02-29", msg) && occursin("20: 2000-06-02 to 2000-12-31", msg)
        rr = Union{Missing,Float64}[t.RR...]
        rr[t.STAID .== 30] .= missing
        @test_throws "30: no day with every variable" SWGC.default_window(WeatherData(merge(t, (RR = rr,)); verbose = false))
    end

    @testset "guided error" begin
        tx = Union{Missing,Float64}[t.TX...]
        gap = (t.STAID .== 20) .& (Date(2000, 5, 1) .<= t.DATE .<= Date(2000, 5, 7))
        tx[gap] .= missing
        rr = Union{Missing,Float64}[t.RR...]
        rr[(t.STAID .== 30) .& (t.DATE .== Date(2000, 8, 1))] .= missing
        dm = WeatherData(merge(t, (TX = tx, RR = rr)); verbose = false)
        msg = try
            check_complete(dm)
        catch e
            sprint(showerror, e)
        end
        @test occursin("TX has 7 missing days at station 20", msg)
        @test occursin("2000-05-01, 2000-05-02, 2000-05-03, 2000-05-04, 2000-05-05.", msg)
        @test !occursin("2000-05-06", msg)
        @test occursin("Also affected (variable, station): (RR, 30)", msg)
        @test occursin("impute = [:TX]", msg)
        @test occursin("window = Date(2000, 8, 2):Date(2000, 12, 31)", msg)
        @test check_complete(dm, Date(2000, 5, 8):Date(2000, 7, 31)) === nothing
        @test check_complete(dm, (Date(2000, 1, 1), Date(2000, 4, 30))) === nothing
        @test_throws "outside the data" check_complete(dm, Date(1999, 1, 1):Date(2000, 1, 1))
        @test_throws "step of one day" check_complete(dm, Date(2000, 1, 1):Day(2):Date(2000, 1, 9))
        @test_throws "a window is a date range" check_complete(dm, 1:10)
        allgaps = WeatherData((DATE = [Date(2000), Date(2000, 1, 2)], STAID = [1, 2], TX = [1.0, 2.0]); verbose = false)
        @test_throws "share no period" check_complete(allgaps)
        @test_throws "no day is complete" check_complete(allgaps, Date(2000):Date(2000, 1, 2))
    end
end
