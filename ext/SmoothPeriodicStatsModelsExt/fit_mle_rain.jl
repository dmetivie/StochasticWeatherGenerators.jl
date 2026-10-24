#TODO docstring + test + exemple
#TODO @assert col = :DATE exists

#* Rain occurrences (Bernoulli)

"""
    fit_mle_RO(df::DataFrame, K, T, degree, local_order)
Given a DataFrame `df` with known hidden states column `z ∈ 1:K`. The rain occurrences of the new station are fitted conditionally to the hidden state.
For `local_order>0` the model is also autoregressive with its past.
"""
function fit_mle_RO(df::DataFrame, local_order, degree, K=length(unique(df.z)), T=length(unique(dayofyear_Leap.(df.DATE))), kwargs...)
    z = df.z[1+local_order:end]
    n2t = dayofyear_Leap.(df.DATE)[1+local_order:end]

    # D × N, the emission convention of SmoothPeriodicStatsModels; here a single station, D = 1
    𝐘all = BitArray(undef, 1, length(df.RO))
    𝐘all[1, :] = df.RO
    𝐘_past = BitMatrix(𝐘all[:, 1:local_order]) # rand(Bool, D, local_order)
    𝐘 = 𝐘all[:, 1+local_order:end]

    return fit_mle_RO(𝐘, 𝐘_past, z, n2t, degree, K, T; kwargs...)
end

function fit_mle_RO(df::DataFrame, df_z::DataFrame, local_order, degree, K=length(unique(df.z)), T=length(unique(dayofyear_Leap.(df.DATE))), kwargs...)
    df_f = innerjoin(df, df_z, on=:DATE)
    return fit_mle_RO(df_f, local_order, degree, K, T; kwargs...)
end

#* Rainfall amounts (positive)

"""
    fit_mle_RR(df::DataFrame, local_order,  K = length(unique(df.z)); maxiter=5000, tol=2e-4, robust=true, display=:none, mix₀=mix_ini(length(unique(dayofyear_Leap.(df.DATE)))), kwargs...)
Fit the strictly positive rainfall amounts `RR>0` distribution ``g_{k,t}(r)`` w.r.t. to each hidden states `k∈[1,K]` (provided in a column of `df.z`). 
The fitted model could in principle be any seasonal model. For now by default it is a double exponential model,

``g_{k,t}(r) = \\alpha(t,k)\\exp(-r/\\theta_1(t,k))/\\theta_1(t,k) + (1-\\alpha(t,k))\\exp(-r/\\theta_2(t,k))/\\theta_2(t,k).``
"""
function StochasticWeatherGenerators.fit_mle_RR(df::DataFrame, local_order::Integer, K=length(unique(df.z |> skipmissing)); maxiter=5000, tol=2e-4, robust=true, display=:none, mix₀=mix_ini(length(unique(dayofyear_Leap.(df.DATE)))), kwargs...)
    dfs = df[1+local_order:end, :]

    n2t_rain = dayofyear_Leap.([@subset(dfs, :z .== k, :RR .> 0).DATE for k in 1:K]) # un peu bourrin mais bon...
    ẑ_per_cat = [findall(dfs.z .== k) for k in 1:K]
    R_per_z = [dfs[!, :RR][ẑ_per_cat[k]] for k = 1:K] # rain per category
    return fit_mle_RR(R_per_z, n2t_rain, mix₀; maxiter=maxiter, tol=tol, robust=robust, display=display, kwargs...)
end

function StochasticWeatherGenerators.fit_mle_RR(R_per_z, n2t_rain, mix_ini; maxiter=5000, tol=2e-4, robust=true, display=:none, kwargs...)
    mix0, θ_al0, θ_Y0 = mix_ini
    K = length(R_per_z)
    T = length(mix0)
    talE = [similar(θ_al0) for k in 1:K]
    tYE = [similar(θ_Y0) for k in 1:K]
    hE = Array{Dict}(undef, K)
    mix_allE = Array{MixtureModel}(undef, K, T)
    for k in 1:K
        mix_allE[k, :], talE[k], tYE[k], hE[k] = fit_mle(mix0, θ_al0, θ_Y0, filter(!iszero, R_per_z[k]), n2t_rain[k]; maxiter=maxiter, tol=tol, robust=robust, display=display, kwargs...)
    end
    return mix_allE
end

#TODO add degree option! Here degree is forced to 1
"""
    mix_ini(T::Integer; θσ10 = [0, 0, 0], θσ20 = [2, 0, 0.0], θα0 = [0.0, 0.0, 0])
Initializes a seasonal mixture model for rainfall amount with two exponential distributions, one of seasonal parameters `θσ10` and `θσ20` with a seasonal portion `θα0` of class 1.
"""
function StochasticWeatherGenerators.mix_ini(T::Integer; θσ10 = [0, 0, 0], θσ20 = [2, 0, 0.0], θα0 = [0.0, 0.0, 0])
    𝕗(θ) = MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
    
    θ_Y0 = [θσ10, θσ20]
    θ_al0 = θα0'
    return [𝕗([σₜ(t / T, θ_Y0[1]), σₜ(t / T, θ_Y0[2]), αₜ(t / T, θ_al0)]) for t in 1:T], θ_al0, θ_Y0
end