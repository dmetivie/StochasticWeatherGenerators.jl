#TODO: check that dropmissing (and potentially not contigous data) does not cause issue in MLE
"""
    fit_AR1(df_full::DataFrame, var, 𝐃𝐞𝐠, T = length(unique(dayofyear_Leap.(df_full.DATE))); solver, init, kwargs...)
    fit_AR1(y::AbstractVector, z::AbstractVector, n2t, 𝐃𝐞𝐠, K, T; solver, init, kwargs...)
    fit_AR1(y::AbstractVector, n2t, 𝐃𝐞𝐠, T; solver, init, kwargs...)

Fit a seasonal AR(1) of period `T`, one per hidden state when the states `z` are given.

``X_{n+1} = \\mu(t_n, z_n) + \\rho(t_n, z_n) X_n + \\sigma(t_n, z_n)\\xi``

with ``\\xi \\sim \\mathcal{N}(0,1)`` and each parameter a trigonometric polynomial of degree `𝐃𝐞𝐠`.
With a `DataFrame`, the series is the column `var`, the period index is the day of the year of
`df_full.DATE`, and the hidden states are the column `z` when there is one.

`init` is the starting point `(μ = …, ρ = …, σ = …)` of the fit, as in `SmoothAR1`. By default it
is read off the data: `ρ` is the lag-1 autocorrelation of the series minus its mean per period
index, `μ` follows the mean of the series per period index, and `σ` is its spread. `solver`, the
optimization algorithm, is required. It and the other keywords (`adtype`, `ensemble`, `solve_kwargs`) go to
`SmoothPeriodicStatsModels.fit_mle`, which fits a `SmoothAR1`. Returns the evaluated `AR1`;
build the model yourself if you want its coefficients or its fit history.
"""
function StochasticWeatherGenerators.fit_AR1(df_full::DataFrame, var, 𝐃𝐞𝐠, T = length(unique(dayofyear_Leap.(df_full.DATE))); kwargs...)
    if "z" ∈ names(df_full)
        df = dropmissing(df_full[:, [:DATE, var, :z]])
        z = df.z
        K = length(unique(z |> skipmissing))
        n2t = dayofyear_Leap.(df.DATE)
        y = df[:, var]
        return fit_AR1(y, z, n2t, 𝐃𝐞𝐠, K, T; kwargs...)
    else
        df = dropmissing(df_full[:, [:DATE, var]])
        n2t = dayofyear_Leap.(df.DATE)
        y = df[:, var]
        return fit_AR1(y, n2t, 𝐃𝐞𝐠, T; kwargs...)
    end
end

function StochasticWeatherGenerators.fit_AR1(y::AbstractVector, z::AbstractVector, n2t, 𝐃𝐞𝐠, K, T; init=_ar1_init(y, n2t, 𝐃𝐞𝐠, T), kwargs...)
    # `fit_mle` reduces the series to its sufficient statistics per (day, state) and solves the
    # `K` independent problems as one `EnsembleProblem`. Since v4 the model carries its own
    # coefficients, so `AR1` of the fit is the whole reconstruction.
    fitted, _ = fit_mle(SmoothAR1(𝐃𝐞𝐠, T; K, init), y, z; n2t, kwargs...)
    return AR1(fitted)
end

function StochasticWeatherGenerators.fit_AR1(y::AbstractVector, n2t, 𝐃𝐞𝐠, T; init=_ar1_init(y, n2t, 𝐃𝐞𝐠, T), kwargs...)
    fitted, _ = fit_mle(SmoothAR1(𝐃𝐞𝐠, T; init), y; n2t, kwargs...)
    return AR1(fitted)
end

# A starting point near the data. From the zero start (`μ = 0`, `ρ = 0`, `σ = 1`), a series far
# from zero, such as a temperature, makes the optimizer push `ρ` to 1, where it cannot move any
# more. `ρ` is measured once the mean of each period index is removed, since the seasonal cycle
# alone would read as a lag-1 autocorrelation near 1. `μ` is the seasonal mean of the series times
# `1 - ρ`, projected on the trigonometric basis: from a constant `μ`, a series with a strong
# seasonal cycle ends at a poor optimum where `ρ(t)` carries the cycle. `σ` gives the stationary
# spread of the series.
function _ar1_init(y::AbstractVector, n2t::AbstractVector, 𝐃𝐞𝐠::Integer, T::Integer)
    length(y) < 2 && return (μ=0.0, ρ=0.0, σ=1.0)
    sums, counts = Dict{eltype(n2t),Float64}(), Dict{eltype(n2t),Int}()
    for (yₙ, t) in zip(y, n2t)
        sums[t] = get(sums, t, 0.0) + yₙ
        counts[t] = get(counts, t, 0) + 1
    end
    a = [yₙ - sums[t] / counts[t] for (yₙ, t) in zip(y, n2t)]
    ρ = length(a) > 2 ? cor(view(a, 2:length(a)), view(a, 1:length(a)-1)) : 0.0
    ρ = isfinite(ρ) ? clamp(ρ, -0.9, 0.9) : 0.0   # NaN when every period index is seen once
    s = std(a) > 0 ? std(a) : std(y) > 0 ? std(y) : 1.0
    ts = collect(keys(counts))
    # the basis is `1, cos(2π t/T), sin(2π t/T), cos(4π t/T), ...`
    X = [j == 1 ? 1.0 : iseven(j) ? cos(2π * (j ÷ 2) * t / T) : sin(2π * (j ÷ 2) * t / T) for t in ts, j in 1:(2𝐃𝐞𝐠+1)]
    μ = X \ [(1 - ρ) * sums[t] / counts[t] for t in ts]
    return (μ=μ, ρ=ρ, σ=s * sqrt(1 - ρ^2))
end
