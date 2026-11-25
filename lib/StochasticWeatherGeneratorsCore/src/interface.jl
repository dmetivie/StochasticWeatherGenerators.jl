# The generator-class protocol (T2): abstract types, traits and hooks.

"""
    AbstractGenerator

Supertype of generator classes, the unfitted model specifications passed to [`fit`](@ref).

A class implements [`fit_generator`](@ref) and [`simulate_member!`](@ref), and optionally [`resolve`](@ref),
[`requested_variables`](@ref), [`declared_kinds`](@ref), [`capabilities`](@ref), [`likelihood_kind`](@ref),
[`timestep`](@ref), [`regime_source`](@ref), [`nregimes`](@ref), [`component`](@ref),
[`simulated_variables`](@ref) and [`bic_by`](@ref).

Experimental: may change in minor releases.
"""
abstract type AbstractGenerator end

"""
    AbstractRegimeSource

Supertype of the regime sources a class can expose through [`regime_source`](@ref).

Experimental: may change in minor releases.
"""
abstract type AbstractRegimeSource end

const LIKELIHOOD_KINDS = (:exact, :conditional, :plugin, :composite, :mc_noisy, :none)

_classname(alg) = nameof(typeof(alg))

"""
    capabilities(alg)

Return the capabilities of class `alg` as a NamedTuple of `Bool`; an absent key means `false`.

Known keys: `regimes`, `handles_missing`, `sequential_only`, `approximate_conditional`.
Define it on the class instance or type.

Experimental: may change in minor releases.
"""
capabilities(alg::AbstractGenerator) = capabilities(typeof(alg))
capabilities(::Type{<:AbstractGenerator}) = (;)

_hascap(alg, key::Symbol) = get(capabilities(alg), key, false) === true

"""
    likelihood_kind(alg)

Return the kind of likelihood that class `alg` reports: `:exact`, `:conditional`, `:plugin`, `:composite`,
`:mc_noisy` or `:none` (the default).
"""
likelihood_kind(alg::AbstractGenerator) = likelihood_kind(typeof(alg))
likelihood_kind(::Type{<:AbstractGenerator}) = :none

"""
    timestep(alg)

Return the time step of class `alg`; only `Day(1)` (the default) is supported.

Experimental: may change in minor releases.
"""
timestep(alg::AbstractGenerator) = timestep(typeof(alg))
timestep(::Type{<:AbstractGenerator}) = Day(1)

"""
    regime_source(alg)

Return the [`AbstractRegimeSource`](@ref) of class `alg`, or `nothing` (the default).

Experimental: may change in minor releases.
"""
regime_source(alg::AbstractGenerator) = regime_source(typeof(alg))
regime_source(::Type{<:AbstractGenerator}) = nothing

"""
    requested_variables(alg)

Return the variables that class `alg` needs from the data, or `nothing` (the default) to take every known variable.

Experimental: may change in minor releases.
"""
requested_variables(alg::AbstractGenerator) = requested_variables(typeof(alg))
requested_variables(::Type{<:AbstractGenerator}) = nothing

"""
    declared_kinds(alg)

Return the variable kinds that class `alg` declares, as a NamedTuple such as `(WIND = :positive,)`.

Declared variables absent from the data are ignored. The default is `(;)`.

Experimental: may change in minor releases.
"""
declared_kinds(alg::AbstractGenerator) = declared_kinds(typeof(alg))
declared_kinds(::Type{<:AbstractGenerator}) = (;)

"""
    resolve(alg, data::WeatherData)

Return class `alg` with its defaults filled in from `data`; the default returns `alg`.

Experimental: may change in minor releases.
"""
resolve(alg::AbstractGenerator, data) = alg

"""
    fit_generator(data::WeatherData, alg; past, seed, rng, solver, verbose, kw...)

Fit class `alg` to the complete window `data` and return a [`FittedGenerator`](@ref).

`fit` calls it with `past` (the data before the window, or `nothing`), `seed::UInt64`,
`rng = stream(seed, :fit, 0)`, `solver` (or `nothing`) and `verbose`. Methods must accept `kw...`.

Experimental: may change in minor releases.
"""
function fit_generator(data, alg; kw...)
    throw(ArgumentError("$(_classname(alg)) does not implement `fit_generator(data, alg::$(_classname(alg)); kw...)`; " *
                        "class authors must define it (see `?StochasticWeatherGeneratorsCore.fit_generator`)"))
end

"""
    simulate_member!(rng, buf::SimBuffer, g::FittedGenerator, ctx::SimContext)

Simulate one member of `g` into `buf` over the `nsteps(ctx)` steps of `ctx`, burn-in included.

`rng` is `stream(seed, member, 0)`; other streams come from `stream(ctx, block)`.
Every value of `buf[var]` and, with regimes, of `regimes(buf)` must be written.

Experimental: may change in minor releases.
"""
function simulate_member!(rng, buf, g, ctx)
    throw(ArgumentError("$(_classname(spec(g))) does not implement `simulate_member!(rng, buf, g, ctx)`; " *
                        "class authors must define it (see `?StochasticWeatherGeneratorsCore.simulate_member!`)"))
end
