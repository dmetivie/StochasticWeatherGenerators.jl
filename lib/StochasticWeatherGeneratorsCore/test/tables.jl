@testset "WeatherSimulation tables" begin
    t = station_table(; dates = Date(2000):Day(1):Date(2001, 12, 31))
    ta = (date = t.DATE, station = t.STAID, TMAX = t.TX, PRCP = t.RR, LAT = t.LAT, LON = t.LON)
    g = fit(ta, SeasonalClimatology(K = 2); seed = 1, verbose = false)
    sd = Date(2002):Date(2002, 1, 25)
    N, D, M = length(sd), 3, 3
    sims = simulate(g, sd; nsim = M, seed = 7)

    @testset "full table" begin
        cols = Tables.columns(sims)
        @test Tables.columnnames(cols) == (:member, :date, :station, :TMAX, :PRCP, :regime)
        @test all(c -> length(c) == M * N * D, cols)
        df = DataFrame(sims)
        @test size(df) == (M * N * D, 6)
        @test df.member == repeat(1:M, inner = D * N)
        @test df.station == repeat(repeat([10, 20, 30], inner = N), M)
        @test df.date == repeat(collect(sd), D * M)
        @test df.TMAX[(2 * D + 1) * N + 1:(2 * D + 2) * N] == sims[:TX][2, :, 3]
        @test df.PRCP == vec(permutedims(sims[:RR], (2, 1, 3)))
        @test df.regime == vec(repeat(reshape(sims[:regime], N, 1, M), 1, D, 1))
        @test sims[:TMAX] === sims[:TX] && sims["PRCP"] === sims[:RR]
        @test_throws "available: TMAX, PRCP" sims[:TN]
        @test_throws BoundsError cols.TMAX[M * N * D + 1]
        @test cols.member isa AbstractVector{Int} && cols.date isa AbstractVector{Date} && eltype(cols.TMAX) == Float64
        @test SWGC.kinds(sims) == (TX = :temperature, RR = :rain) && SWGC.sites(sims).LAT == [46.0, 47.0, 48.0]
        @test coords(sims) == coords(g) && SWGC.calendar(sims) == Leap366() && variables(sims) == [:TX, :RR]
        @test SWGC.regimes(sims) === sims[:regime]
        @test occursin("3 members", sprint(show, MIME"text/plain"(), sims)) && occursin("WeatherSimulation", sprint(show, sims))
    end

    @testset "partitions" begin
        df = DataFrame(sims)
        ps = collect(Tables.partitions(sims))
        @test length(ps) == M * D && all(p -> p isa NamedTuple && length(p.date) == N, ps)
        @test vcat(DataFrame.(ps)...) == df
        @test ps[2].station == fill(20, N) && ps[2].member == fill(1, N)
        for n in (1, 7, 25, 100)
            pc = collect(SWGC.partitions(sims; chunk = n))
            @test length(pc) == M * cld(N, n)
            @test length(pc[1].date) == D * min(n, N)
            @test length(pc[cld(N, n)].date) == D * (N - (cld(N, n) - 1) * n)            # the last chunk is short
            @test sort(vcat(DataFrame.(pc)...), [:member, :station, :date]) == df
        end
        @test_throws "positive number of days" SWGC.partitions(sims; chunk = 0)
    end

    @testset "sink" begin
        df = DataFrame(sims)
        got = NamedTuple[]
        prov = simulate(g, sd; nsim = M, seed = 7, sink = p -> push!(got, p))
        @test length(got) == M * D && vcat(DataFrame.(got)...) == df
        @test prov.seed === UInt64(7) && prov.members == 1:M && prov.dates == collect(sd) && prov.burnin == 365
        got2 = NamedTuple[]
        simulate(g, sd; nsim = M, seed = 7, sink = p -> push!(got2, p), chunk = 10, threads = false)
        @test length(got2) == M * cld(N, 10)
        @test sort(vcat(DataFrame.(got2)...), [:member, :station, :date]) == df
        got3 = NamedTuple[]
        simulate(g, sd; members = 2:3, seed = 7, sink = p -> push!(got3, p), eltype = Float32)
        @test eltype(got3[1].TMAX) == Float32 && unique(vcat((p.member for p in got3)...)) == [2, 3]
        @test_throws "`chunk` applies to `sink`" simulate(g, sd; chunk = 5)
    end

    @testset "members as data" begin
        m2 = SWGC.member(sims, 2)
        @test m2 isa WeatherData && m2[:TX] == sims[:TX][:, :, 2] && dates(m2) == collect(sd)
        @test Tables.columnnames(Tables.columns(m2)) == (:date, :station, :TMAX, :PRCP)
        @test SWGC.sites(m2) == SWGC.sites(sims)
        @test_throws "no member 9; members are 1:3" SWGC.member(sims, 9)
        # a simulated member can serve as `past` and as fit data
        nxt = Date(2002, 1, 26):Date(2002, 2, 5)
        @test size(simulate(g, nxt; seed = 1, past = m2)[:TX]) == (D, length(nxt), 1)
        @test fit(m2, SeasonalClimatology(); seed = 1) isa FittedGenerator
        one = (DATE = collect(Date(2000):Date(2000, 12, 31)), TX = randn(Xoshiro(2), 366))
        s1 = simulate(fit(one, SeasonalClimatology(); verbose = false), Date(2001):Date(2001, 1, 10); nsim = 2, seed = 1)
        @test Tables.columnnames(Tables.columns(s1)) == (:member, :DATE, :TX)
        @test SWGC.member(s1, 1).stationcol === nothing
    end
end
