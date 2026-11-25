@testset "calendars" begin
    dayofyear_Leap(d) = dayofyear(d) + ((!isleapyear(d)) & (month(d) > 2))
    @test all(SWGC.period_index(Leap366(), d) == dayofyear_Leap(d) for d in Date(1999):Day(1):Date(2001, 12, 31))
    @test SWGC.period(Leap366()) == SWGC.period(NoLeap365()) == SWGC.period(Day360()) == 366

    @testset "29 February" begin
        @test SWGC.period_index(Leap366(), Date(2000, 2, 29)) == 60
        @test SWGC.period_index(Leap366(), Date(1999, 3, 1)) == SWGC.period_index(Leap366(), Date(2000, 3, 1)) == 61
        @test_throws "NoLeap365" SWGC.period_index(NoLeap365(), Date(2000, 2, 29))
        @test 60 ∉ [SWGC.period_index(NoLeap365(), d) for d in Date(1999):Day(1):Date(1999, 12, 31)]
        @test_throws ArgumentError SWGC.period_index(Leap366(), D360(2001, 2, 29))   # not a Gregorian day
        @test_throws ArgumentError SWGC.period_index(Leap366(), D360(2001, 13, 1))
    end

    @testset "consecutive steps" begin
        noleap = filter(d -> !(month(d) == 2 && day(d) == 29), Date(1999):Day(1):Date(2001, 12, 31))
        @test SWGC.check_consecutive(NoLeap365(), noleap) === nothing
        @test_throws "followed by" SWGC.check_consecutive(Leap366(), noleap)
        @test_throws ArgumentError SWGC.check_consecutive(Leap366(), Date[])
        @test_throws ArgumentError SWGC.check_consecutive(Leap366(), [Date(2000, 1, 2), Date(2000, 1, 1)])
        @test length(SWGC.calendar_dates(NoLeap365(), 2000:2001)) == 730
        @test length(SWGC.calendar_dates(Leap366(), 2000:2001)) == 731
        @test SWGC.calendar_dates(NoLeap365(), 2000:2000)[60] == Date(2000, 3, 1)
        @test_throws "CFTime" SWGC.calendar_dates(Day360(), 2000:2001)
        @test_throws ArgumentError SWGC.calendar_dates(Leap366(), 2001:2000)
    end

    @testset "Day360" begin
        idx = [SWGC.period_index(Day360(), d) for d in d360_dates(2000, 2000)]
        @test issorted(idx) && allunique(idx) && extrema(idx) == (1, 366)
        @test_throws "30 days" SWGC.period_index(Day360(), D360(2000, 1, 31))
        @test_throws "30 days" SWGC.period_index(Day360(), Date(2000, 1, 31))
        @test SWGC.check_consecutive(Day360(), d360_dates(1999, 2001)) === nothing
        @test_throws ArgumentError SWGC.check_consecutive(Day360(), [D360(2000, 1, 30), D360(2000, 2, 2)])
    end

    @testset "steps_before" begin
        pidx(cal, ds) = [SWGC.period_index(cal, d) for d in ds]
        @test SWGC.steps_before(Leap366(), Date(2000, 3, 2), 3) == [59, 60, 61]
        @test SWGC.steps_before(Leap366(), Date(2001, 3, 2), 3) == [58, 59, 61]
        @test SWGC.steps_before(NoLeap365(), Date(2001, 3, 2), 3) == [58, 59, 61]
        @test SWGC.steps_before(Day360(), D360(2000, 1, 2), 3) == pidx(Day360(), [D360(1999, 12, 29), D360(1999, 12, 30), D360(2000, 1, 1)])
        @test SWGC.steps_before(Leap366(), Date(2000, 1, 1), 0) == Int[]
        @test SWGC.steps_before(NoLeap365(), Date(2001, 1, 1), 730) == pidx(NoLeap365(), SWGC.calendar_dates(NoLeap365(), 1999:2000))
        @test SWGC.steps_before(Leap366(), Date(2001, 1, 1), 731) == pidx(Leap366(), SWGC.calendar_dates(Leap366(), 1999:2000))
    end
end
