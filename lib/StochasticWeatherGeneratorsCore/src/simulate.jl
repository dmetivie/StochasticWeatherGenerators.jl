# simulate: the ensemble driver.

"""
    simulate([rng,] g::FittedGenerator, dates | years; nsim = 1, seed = nothing, members = nothing,
             past = nothing, burnin = nothing, given = (;), sink = nothing, chunk = nothing,
             threads = true, eltype = Float64)

Simulate members of `g` over consecutive `dates`, or over whole `years` such as `1986:2035`, and return a
[`WeatherSimulation`](@ref).

- `seed` (any integer) fixes the output; `nothing` draws it from `rng`. Member `m` always gets the same stream,
  so the output does not depend on `nsim`, `members` or the number of threads.
- `members = a:b` simulates only those members (within `1:nsim` when `nsim` is given).
- `past` is the initial state: a table or [`WeatherData`](@ref) ending the day before the first date, or a date
  that the class interprets. Without `past` or `given`, `burnin` (default 365) steps are simulated and dropped.
- `given = (regime = z,)` imposes the regime path; a variable `given = (regime = z, TX = M)` needs the regime.
  `given` needs `past`.
- `sink` is called on each partition (see [`partitions`](@ref), `chunk` included) instead of keeping the arrays;
  members are simulated in batches of `Threads.nthreads()` and `simulate` returns the provenance.
- `eltype = Float32` halves the memory; classes still compute in `Float64`.
"""
simulate(g::FittedGenerator, when; kw...) = simulate(Random.default_rng(), g, when; kw...)

function simulate(rng::AbstractRNG, g::FittedGenerator, when; nsim = nothing, seed = nothing, members = nothing,
        past = nothing, burnin = nothing, given::NamedTuple = (;), sink = nothing, chunk = nothing,
        threads::Bool = true, eltype::Type{<:AbstractFloat} = Float64)
    T = eltype
    info = g.info
    cal = info.calendar
    simdates = _simdates(cal, when)
    nn = length(simdates)
    s = seed === nothing ? rand(rng, UInt64) : seedword(seed)
    mem = _members(nsim, members)
    pst = _sim_past(g, past, simdates)
    gvn = _sim_given(g, given, pst, nn)
    nb = _burnin(burnin, pst, gvn)
    chunk === nothing || sink !== nothing ||
        throw(ArgumentError("`chunk` applies to `sink`; for an in-memory result use `partitions(sims; chunk)`"))
    pidx = vcat(steps_before(cal, first(simdates), nb), Int[period_index(cal, d) for d in simdates])
    proto = SimContext(simdates, pidx, nb, 0, s, pst, gvn, copy(info.stations), cal)
    hasreg = _hascap(g, :regimes)
    prov = (seed = s, rng_derivation = RNG_DERIVATION, members = mem, dates = simdates,
        stations = copy(info.stations), burnin = nb, generator = string(_classname(g.spec)))
    if sink === nothing
        return _simulate_batch(T, g, proto, mem, hasreg, threads, prov)
    end
    batch = threads ? Threads.nthreads() : 1
    for part in Iterators.partition(mem, batch)
        sims = _simulate_batch(T, g, proto, collect(part), hasreg, threads, prov)
        for p in partitions(sims; chunk)
            sink(p)
        end
    end
    return prov
end

"""
    rand([rng,] g::FittedGenerator, dates | years; kw...)

Simulate one member of `g`; this is `simulate(rng, g, dates; nsim = 1, kw...)`.
"""
Base.rand(g::FittedGenerator, when::AbstractVector; kw...) = simulate(g, when; nsim = 1, kw...)
Base.rand(rng::AbstractRNG, g::FittedGenerator, when::AbstractVector; kw...) =
    simulate(rng, g, when; nsim = 1, kw...)

function _simdates(cal, when)
    when isa Integer && return calendar_dates(cal, when:when)
    when isa AbstractUnitRange{<:Integer} && return calendar_dates(cal, when)
    when isa AbstractVector{<:Integer} && throw(ArgumentError("years must be a range such as 2000:2010, got $when"))
    when isa AbstractVector || throw(ArgumentError("simulate takes consecutive dates or a range of years, got $(repr(when))"))
    isempty(when) && throw(ArgumentError("no dates to simulate"))
    d = _todates(when, :dates)
    check_consecutive(cal, d)
    return d
end

function _members(nsim, members)
    if nsim !== nothing
        (nsim isa Integer && !(nsim isa Bool) && nsim >= 1) ||
            throw(ArgumentError("`nsim` must be a positive integer, got $(repr(nsim))"))
    end
    members === nothing && return collect(1:Int(something(nsim, 1)))
    mem = members isa Integer ? [Int(members)] : collect(Int, members)
    isempty(mem) && throw(ArgumentError("`members` is empty"))
    allunique(mem) || throw(ArgumentError("`members` has duplicates"))
    all(>=(1), mem) || throw(ArgumentError("members are numbered from 1, got $(minimum(mem))"))
    nsim === nothing || all(<=(nsim), mem) ||
        throw(ArgumentError("`members` must be within 1:nsim = 1:$nsim, got $(_rangestring(mem))"))
    return mem
end

function _sim_past(g::FittedGenerator, past, simdates)
    past === nothing && return nothing
    info = g.info
    if past isa WeatherData
        pd = past
    elseif Tables.istable(past)
        algkinds = (; (v => k for (v, k) in zip(info.variables, info.kinds))...)
        pd = WeatherData(past; calendar = info.calendar, crs = info.crs, verbose = false, algkinds)
    elseif past isa Dates.AbstractTime
        return past
    else
        throw(ArgumentError("`past` must be a table, a WeatherData or a date, got a $(typeof(past))"))
    end
    typeof(pd.calendar) == typeof(info.calendar) ||
        throw(ArgumentError("`past` is on $(pd.calendar) but the generator on $(info.calendar)"))
    stepnumber(info.calendar, last(pd.dates)) == stepnumber(info.calendar, first(simdates)) - 1 ||
        throw(ArgumentError("`past` must end the day before the first simulated date $(first(simdates)); " *
                            "it ends on $(last(pd.dates))"))
    idx = [findfirst(==(s), pd.stations) for s in info.stations]
    absent = info.stations[findall(isnothing, idx)]
    isempty(absent) || throw(ArgumentError("`past` has no data for stations $(join(absent, ", "))"))
    return idx == eachindex(pd.stations) ? pd : _select_stations(pd, Int.(idx))
end

function _sim_given(g::FittedGenerator, given::NamedTuple, pst, nn)
    isempty(given) && return given
    pst === nothing && throw(ArgumentError(
        "`given` needs `past`: the conditioned simulation starts from an observed state; pass `past = ...`"))
    hasreg = _hascap(g, :regimes)
    vars, names = simulated_variables(g), _simnames(g)
    nd = length(g.info.stations)
    entries = Pair{Symbol,Any}[]
    for (k, v) in pairs(given)
        if k === :regime
            hasreg || throw(ArgumentError("`given.regime` needs a generator with regimes; $(_classname(g.spec)) has none"))
            v isa AbstractVector{<:Integer} || throw(ArgumentError("`given.regime` must be a vector of integers"))
            length(v) == nn || throw(DimensionMismatch("`given.regime` has $(length(v)) values for $nn dates"))
            K = nregimes(g)
            K === nothing || all(in(1:K), v) || throw(ArgumentError("`given.regime` values must be in 1:$K"))
            push!(entries, :regime => collect(Int, v))
        else
            i = _varindex(vars, names, k)
            i === nothing && throw(ArgumentError(
                "`given.$k` is not a simulated variable; they are $(join(names, ", "))"))
            v isa AbstractMatrix || throw(ArgumentError("`given.$k` must be a stations × days matrix"))
            size(v) == (nd, nn) || throw(DimensionMismatch("`given.$k` is $(join(size(v), "×")), expected $nd×$nn"))
            any(p -> p.first === vars[i], entries) && throw(ArgumentError("`given` names $(vars[i]) twice"))
            push!(entries, vars[i] => v)
        end
    end
    if hasreg && !any(p -> p.first === :regime, entries)
        throw(ArgumentError("conditioning $(_classname(g.spec)) on a variable needs `given.regime` in this version"))
    end
    return (; entries...)
end

function _burnin(burnin, pst, gvn)
    conditioned = pst !== nothing || !isempty(gvn)
    burnin === nothing && return conditioned ? 0 : 365
    (burnin isa Integer && !(burnin isa Bool) && burnin >= 0) ||
        throw(ArgumentError("`burnin` must be a non-negative integer, got $(repr(burnin))"))
    burnin > 0 && conditioned && throw(ArgumentError(
        "`burnin` cannot be combined with `past` or `given`, which set the initial state; drop `burnin`"))
    return Int(burnin)
end

function _simulate_batch(::Type{T}, g, proto, mem, hasreg, threads, prov) where {T}
    nd, nn, nm = length(proto.stations), length(proto.dates), length(mem)
    vars, names, kindv = simulated_variables(g), _simnames(g), _simkinds(g)
    arrays = [Array{T,3}(undef, nd, nn, nm) for _ in vars]
    reg = hasreg ? zeros(Int, nn, nm) : nothing
    nl = nsteps(proto)
    direct = T === Float64 && nl == nn
    work = (j, scratch) -> _simulate_member(arrays, reg, j, g, proto, mem[j], scratch, direct, vars, names)
    newscratch = () -> direct ? nothing :
        ([Matrix{Float64}(undef, nd, nl) for _ in vars], hasreg ? zeros(Int, nl) : nothing)
    nt = threads ? min(Threads.nthreads(), nm) : 1
    if nt == 1
        scratch = newscratch()
        for j in 1:nm
            work(j, scratch)
        end
    else
        next = Threads.Atomic{Int}(1)
        errors = Vector{Any}(nothing, nt)
        @sync for w in 1:nt
            Threads.@spawn begin
                scratch = newscratch()
                try
                    while true
                        j = Threads.atomic_add!(next, 1)
                        j > nm && break
                        work(j, scratch)
                    end
                catch e
                    errors[w] = e
                    Threads.atomic_xchg!(next, nm + 1)
                end
            end
        end
        i = findfirst(!isnothing, errors)
        i === nothing || throw(errors[i])
    end
    info = g.info
    sitemeta = (; (k => v for (k, v) in pairs(info.sites) if k !== :station)...)
    return WeatherSimulation(vars, names, kindv, arrays, reg, mem, copy(proto.dates), copy(proto.stations),
        info.datecol, info.stationcol, sitemeta, info.crs, info.calendar, prov)
end

function _simulate_member(arrays, reg, j, g, proto, m, scratch, direct, vars, names)
    ctx = _with_member(proto, m)
    nb = nburnin(ctx)
    if direct
        bufs = [fill!(view(A, :, :, j), NaN) for A in arrays]
        r = reg === nothing ? nothing : fill!(view(reg, :, j), 0)
        simulate_member!(stream(ctx.seed, m, 0), SimBuffer(vars, names, bufs, r), g, ctx)
    else
        bufs, r = scratch
        foreach(B -> fill!(B, NaN), bufs)
        r === nothing || fill!(r, 0)
        simulate_member!(stream(ctx.seed, m, 0), SimBuffer(vars, names, bufs, r), g, ctx)
        for (A, B) in zip(arrays, bufs)
            view(A, :, :, j) .= view(B, :, nb+1:size(B, 2))
        end
        r === nothing || (view(reg, :, j) .= view(r, nb+1:length(r)))
    end
    for (v, A) in enumerate(arrays)
        any(isnan, view(A, :, :, j)) && throw(ErrorException(
            "$(_classname(g.spec)) left NaN in $(names[v]) for member $m: `simulate_member!` must write every step"))
    end
    if reg !== nothing
        K = something(nregimes(g), typemax(Int))
        all(k -> 1 <= k <= K, view(reg, :, j)) || throw(ErrorException(
            "$(_classname(g.spec)) left regimes outside 1:$K for member $m: `simulate_member!` must write every step"))
    end
    return nothing
end
