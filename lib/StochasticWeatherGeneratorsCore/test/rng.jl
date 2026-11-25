@testset "RNG streams" begin
    draw(args...) = rand(SWGC.stream(args...), UInt64)

    # golden values pin the derivation (RNG_DERIVATION = 1)
    @test SWGC.RNG_DERIVATION == 1
    @test draw(50000, 1, 0) == 0x21c50d30fc63d82e
    @test draw(50000, :fit, 0) == 0x7a1cd59d33744b49
    @test SWGC.stream(1, 1) isa Xoshiro

    @testset "seed types" begin
        r = draw(1, 1)
        @test draw(UInt(1), 1) == draw(Int32(1), 1) == draw(big(1), 1) == draw(UInt8(1), 1) == r
        @test draw(big(2)^64 + 1, 1) == r                       # reduced modulo 2^64
        @test draw(-1, 1) == draw(typemax(UInt64), 1) == draw(Int8(-1), 1)
        @test draw(typemin(Int64), 1) == draw(UInt64(2)^63, 1)
        @test_throws "Bool" SWGC.stream(true, 1)
        @test_throws ArgumentError SWGC.stream(1, 1, true)
        @test_throws MethodError SWGC.stream(1.0, 1)
    end

    @testset "block ids and domains" begin
        @test draw(1, 1, :TX) == draw(1, 1, "TX") == draw(1, 1, SubString("TXX", 1, 2))
        @test draw(1, 1, :TX) != draw(1, 1, :TN)
        @test draw(1, :fit, :TX) == draw(1, :fit, "TX")
        @test draw(1, :fit, 0) != draw(1, 0, 0)                  # fit and member streams differ
        @test draw(1, 1, 0) != draw(1, 1, Symbol("0"))           # integer and name ids differ
        @test_throws ":sim" SWGC.stream(1, :sim)
        vals = [draw(s, m, b) for s in 0:4, m in 0:4, b in (0, 1, 2, :a, :b)]
        @test allunique(vals)
        @test allunique([draw(s, :fit, b) for s in 0:4, b in 0:4])
    end

    @testset "independent of thread and call order" begin
        seq = [draw(7, m, 0) for m in 1:16]
        par = zeros(UInt64, 16)
        Threads.@threads for m in 16:-1:1
            par[m] = draw(7, m, 0)
        end
        @test par == seq
    end
end
