#TODO: check that dropmissing (and potentially not contigous data) does not cause issue in MLE
"""
    fit_AR1(df_full::DataFrame, var, 𝐃𝐞𝐠, K = length(unique(df_full.z)), T = length(unique(n2t)))
    fit_AR1(y::AbstractArray, z::AbstractArray, n2t, 𝐃𝐞𝐠, K = length(unique(z |> skipmissing)), T = length(unique(n2t)))
Fit a Seasonal AR(1) model of period `T` and with `K` hidden states for the variable `X` of the DataFrame `df_full`.
The hidden states must be given in a the column `z` of i.e. `df_full.z`.
The correspondance between day of the year `t` and index in the time series `n` must be given in the column `n2t` i.e. `df_full.n2t`.

``X_{n+1} = \\mu(t_n, z_n) + \\phi(t_n, z_n) X_n + \\sigma(t_n, z_n)\\xi``

with ``\\xi \\sim \\mathcal{N}(0,1)``.

Extra keywords (`solver`, `adtype`, `ensemble`, `solve_kwargs`) are forwarded to
`SmoothPeriodicStatsModels.fit_mle_AR1`.
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

function StochasticWeatherGenerators.fit_AR1(y::AbstractVector, z::AbstractVector, n2t, 𝐃𝐞𝐠, K, T; kwargs...)
    # `fit_mle_AR1` reduces the series to its sufficient statistics per (day, state) and solves the
    # `K` independent problems as one `EnsembleProblem`; it replaces the per-observation JuMP model
    # and the `observable` dictionaries this function used to assemble by hand.
    θ_μ, θ_ρ, θ_σ = fit_mle_AR1(y, z, 𝐃𝐞𝐠, T; n2t=n2t, K=K, kwargs...)

    return AR1([μₜ(t / T, θ_μ[:, k]) for k in 1:K, t in 1:T], [ρₜ(t / T, θ_ρ[:, k]) for k in 1:K, t in 1:T], [σₜ(t / T, θ_σ[:, k]) for k in 1:K, t in 1:T])
end

function StochasticWeatherGenerators.fit_AR1(y::AbstractVector, n2t, 𝐃𝐞𝐠, T; kwargs...)
    θ_μ, θ_ρ, θ_σ = fit_mle_AR1(y, 𝐃𝐞𝐠, T; n2t=n2t, kwargs...)

    return AR1([μₜ(t / T, θ_μ) for t in 1:T], [ρₜ(t / T, θ_ρ) for t in 1:T], [σₜ(t / T, θ_σ) for t in 1:T])
end
