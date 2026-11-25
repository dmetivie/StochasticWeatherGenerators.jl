@testset "simulate" begin
    t = station_table(; dates = Date(2000):Day(1):Date(2001, 12, 31))
    data = WeatherData(t; verbose = false)
    g = fit(data, SeasonalClimatology(); seed = 1)
    gk = fit(data, SeasonalClimatology(K = 3, joint = ((:TX, :TN),)); seed = 1)
    sd = Date(2002):Date(2002, 3, 31)
    N = length(sd)

    @testset "reproducibility" begin
        full = simulate(g, sd; nsim = 5, seed = 42)
        @test size(full[:TX]) == (3, N, 5) && members(full) == 1:5
        @test simulate(g, sd; nsim = 2, seed = 42)[:TX] == full[:TX][:, :, 1:2]
        @test simulate(g, sd; members = 3:4, seed = 42)[:RR] == full[:RR][:, :, 3:4]
        @test simulate(g, sd; nsim = 5, members = [5, 2], seed = 42)[:TN] == full[:TN][:, :, [5, 2]]
        @test simulate(g, sd; nsim = 5, seed = 42, threads = false)[:TX] == full[:TX]
        @test simulate(g, sd; members = 4, seed = UInt(42))[:TX] == full[:TX][:, :, 4:4]
        @test rand(g, sd; seed = 42)[:TX] == full[:TX][:, :, 1:1]
        @test rand(Xoshiro(1), g, sd)[:TX] == rand(Xoshiro(1), g, sd)[:TX]
        @test simulate(Xoshiro(3), g, sd; nsim = 2)[:TX] == simulate(g, sd; nsim = 2, seed = rand(Xoshiro(3), UInt64))[:TX]
        @test simulate(g, sd; nsim = 2, seed = 43)[:TX] != full[:TX][:, :, 1:2]
        f32 = simulate(g, sd; nsim = 3, seed = 42, eltype = Float32)
        @test f32[:TX] isa Array{Float32,3} && f32[:TX] == Float32.(full[:TX][:, :, 1:3])
        @test SWGC.provenance(full).seed === UInt64(42) && SWGC.provenance(full).burnin == 365
        @test SWGC.provenance(full).generator == "SeasonalClimatology" && SWGC.provenance(full).members == 1:5
        @test Threads.nthreads() == 1 || Threads.nthreads() >= 2   # run the suite with `-t 4` to exercise threads
    end

    @testset "dates and years" begin
        sy = simulate(g, 2002:2003; seed = 1)
        @test dates(sy) == collect(Date(2002):Day(1):Date(2003, 12, 31))
        @test dates(simulate(g, 2004; seed = 1)) == collect(Date(2004):Day(1):Date(2004, 12, 31))
        @test_throws "consecutive" simulate(g, [Date(2002), Date(2002, 1, 3)])
        @test_throws "range such as" simulate(g, [2002, 2004])
        @test_throws "no dates" simulate(g, Date[])
        @test_throws "consecutive dates or a range of years" simulate(g, "2002")
        @test dates(simulate(g, DateTime(2002):Day(1):DateTime(2002, 1, 5); seed = 1)) == collect(Date(2002):Date(2002, 1, 5))
        # NoLeap365: two years always have 730 steps
        tn = rows(t, findall(d -> !(month(d) == 2 && day(d) == 29), t.DATE))
        gn = fit(WeatherData(tn; calendar = NoLeap365(), verbose = false), SeasonalClimatology(); seed = 1)
        @test length(dates(simulate(gn, 2004:2005; seed = 1))) == 730
        @test_throws "NoLeap365" simulate(gn, Date(2004, 2, 28):Date(2004, 3, 1))
        # Day360 has no built-in dates
        ds = d360_dates(2000, 2001)
        d3 = WeatherData((DATE = ds, STAID = fill(1, length(ds)), TX = randn(Xoshiro(1), length(ds))); calendar = Day360())
        g3 = fit(d3, SeasonalClimatology(); seed = 1)
        @test_throws "CFTime" simulate(g3, 2002:2003)
        s3 = simulate(g3, d360_dates(2002, 2002); seed = 1, nsim = 2)
        @test size(s3[:TX]) == (1, 360, 2) && dates(s3) == d360_dates(2002, 2002)
    end

    @testset "members and nsim" begin
        @test_throws "positive integer" simulate(g, sd; nsim = 0)
        @test_throws "positive integer" simulate(g, sd; nsim = true)
        @test_throws "within 1:nsim = 1:3" simulate(g, sd; nsim = 3, members = 2:4)
        @test_throws "duplicates" simulate(g, sd; members = [1, 1])
        @test_throws "numbered from 1" simulate(g, sd; members = 0:2)
        @test_throws "empty" simulate(g, sd; members = Int[])
    end

    @testset "burn-in and past" begin
        gp = fit(data, ProbeToy(); seed = 1)
        sp = simulate(gp, sd; seed = 1, threads = false)
        ctx = LAST_CTX[]
        @test SWGC.nburnin(ctx) == 365 && SWGC.nsteps(ctx) == 365 + N && SWGC.member(ctx) == 1
        @test SWGC.period_indices(ctx) == [SWGC.period_index(Leap366(), d) for d in Date(2001, 1, 1):Date(2002, 3, 31)]
        @test sp[:TX][1, :, 1] == [SWGC.period_index(Leap366(), d) for d in sd]           # the burn-in is dropped
        @test sp[:regime] == ones(Int, N, 1) && dates(ctx) == collect(sd) && stations(ctx) == [10, 20, 30]
        @test SWGC.calendar(ctx) == Leap366() && SWGC.given(ctx) == (;) && SWGC.past(ctx) === nothing
        simulate(gp, sd; seed = 1, burnin = 0, threads = false)
        @test SWGC.nsteps(LAST_CTX[]) == N
        # past as data: stations are put in the generator's order
        last5 = data[Date(2001, 12, 27):Date(2001, 12, 31)]
        rev = SWGC._select_stations(last5, [3, 1, 2])
        simulate(gp, sd; seed = 1, past = rev, threads = false)
        @test stations(SWGC.past(LAST_CTX[])) == [10, 20, 30] && SWGC.nburnin(LAST_CTX[]) == 0
        @test SWGC.past(LAST_CTX[])[:TX] == last5[:TX]
        simulate(gp, sd; seed = 1, past = rows(t, findall(>=(Date(2001, 12, 1)), t.DATE)), threads = false)
        @test last(dates(SWGC.past(LAST_CTX[]))) == Date(2001, 12, 31)
        # past as a date is passed through to the class
        simulate(gp, sd; seed = 1, past = Date(2001, 12, 31), threads = false)
        @test SWGC.past(LAST_CTX[]) == Date(2001, 12, 31) && SWGC.nburnin(LAST_CTX[]) == 0
        @test_throws "takes `past` as data" simulate(g, sd; past = Date(2001, 12, 31))
        @test_throws "end the day before the first simulated date 2002-01-01" simulate(g, sd; past = data[Date(2001):Date(2001, 12, 30)])
        @test_throws "no data for stations 30" simulate(g, sd; past = SWGC._select_stations(last5, [1, 2]))
        @test_throws "`burnin` cannot be combined" simulate(g, sd; past = last5, burnin = 10)
        @test_throws "non-negative" simulate(g, sd; burnin = -1)
        @test_throws "must be a table" simulate(g, sd; past = 3)
        # past changes the start, not the stream
        s1 = simulate(g, sd; seed = 5, past = last5)
        @test s1[:TX] != simulate(g, sd; seed = 5)[:TX] && s1[:TX] == simulate(g, sd; seed = 5, past = last5)[:TX]
    end

    @testset "given" begin
        last5 = data[Date(2001, 12, 27):Date(2001, 12, 31)]
        z = repeat([1, 2, 3], cld(N, 3))[1:N]
        sg = simulate(gk, sd; nsim = 2, seed = 1, past = last5, given = (regime = z,))
        @test sg[:regime] == [z z]
        tx = fill(20.0, 3, N)
        sv = simulate(gk, sd; seed = 1, past = last5, given = (regime = z, TMAX = tx))
        @test sv[:TX][:, :, 1] == tx && sv[:RR] == sg[:RR][:, :, 1:1]
        free = simulate(gk, sd; nsim = 2, seed = 1)
        @test all(in(1:3), free[:regime]) && length(unique(free[:regime])) == 3
        @test_throws "`given` needs `past`" simulate(gk, sd; given = (regime = z,))
        @test_throws "needs `given.regime`" simulate(gk, sd; past = last5, given = (TX = tx,))
        @test_throws "has 10 values for $N dates" simulate(gk, sd; past = last5, given = (regime = z[1:10],))
        @test_throws "values must be in 1:3" simulate(gk, sd; past = last5, given = (regime = fill(4, N),))
        @test_throws "vector of integers" simulate(gk, sd; past = last5, given = (regime = fill(1.0, N),))
        @test_throws "needs a generator with regimes" simulate(g, sd; past = last5, given = (regime = z,))
        @test_throws "not a simulated variable" simulate(gk, sd; past = last5, given = (regime = z, QQ = tx))
        @test_throws DimensionMismatch simulate(gk, sd; past = last5, given = (regime = z, TX = tx[:, 1:10]))
        @test_throws "names TX twice" simulate(gk, sd; past = last5, given = (regime = z, TX = tx, TMAX = tx))
        @test_throws "`burnin` cannot be combined" simulate(gk, sd; past = last5, given = (regime = z,), burnin = 3)
        # a class without regimes may condition on a variable alone
        @test simulate(g, sd; seed = 1, past = last5, given = (TX = tx,))[:TX][:, :, 1] == tx
        @test_throws "this simulation has no regimes" simulate(g, sd; seed = 1)[:regime]
    end

    @testset "class errors" begin
        @test_throws "BrokenToy left NaN in TX for member 1" simulate(fit(data, BrokenToy(:nan)), sd; seed = 1)
        gt = fit(data, BrokenToy(:throw))
        @test_throws DomainError simulate(gt, sd; nsim = 8, seed = 1)
        @test_throws DomainError simulate(gt, sd; nsim = 8, seed = 1, threads = false)
        @test simulate(gt, sd; members = [1, 2, 4], seed = 1)[:TX] == zeros(3, N, 3)
        @test_throws "BrokenToy does not implement `simulate_member!" simulate(fit(data, BrokenToy(:nosim)), sd)
    end
end
