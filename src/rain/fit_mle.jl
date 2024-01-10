#TODO docstring + test + exemple
#TODO @assert col = :DATE exists

#* Rain occurrences (Bernoulli)

"""
    fit_mle_stations(df::DataFrame, K, T, degree, local_order)
Given a
"""
function fit_mle_stations(df::DataFrame, K, T, degree, local_order)
    z = df.z[1+local_order:end]
    n2t = dayofyear_Leap.(df.DATE)[1+local_order:end]
    N = length(n2t)

    𝐘all = BitArray(undef, N + 1, 1)
    𝐘all[:, 1] = df.bin
    𝐘_past = BitMatrix(𝐘all[1:local_order, :]) # rand(Bool, local_order, D)
    𝐘 = 𝐘all[1+local_order:end, :]

    B, θ = fit_mle_stations(𝐘, 𝐘_past, z, n2t, K, T, degree)
    return B, θ
end

function fit_mle_stations(df::DataFrame, df_z::DataFrame, K, T, degree, local_order)
    df_f = innerjoin(df, df_z, on=:DATE)
    return fit_mle_stations(df_f, K, T, degree, local_order)
end

#* Rain amount

#TODO add degree option! Here degree is forced to 1
function mix_ini(T::Integer)
    𝕗(θ) = MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
    θσ10 = [0, 0, 0]
    θσ20 = [2, 0, 0.0]
    θα0 = [0.0, 0.0, 0]
    # θ0 = vec(hcat(θσ10, θσ20, θα0))
    θ_Y0 = [θσ10, θσ20]
    θ_al0 = θα0'
    return [𝕗([σₜ(t / T, θ_Y0[1]), σₜ(t / T, θ_Y0[2]), αₜ(t / T, θ_al0)]) for t in 1:T], θ_al0, θ_Y0
end

function fit_mle_RR(𝐑_per_z, n2t_rain, mix_ini; maxiter=5000, tol=2e-4, robust=true, silence=true, warm_start=true, display=:none)
    mix0, θ_al0, θ_Y0 = mix_ini
    D, K = size(𝐑_per_z)
    T = length(mix0)
    talE = [similar(θ_al0) for k in 1:K, j in 1:D]
    tYE = [similar(θ_Y0) for k in 1:K, j in 1:D]
    hE = Array{Dict}(undef, K, D)
    mix_allE = Array{MixtureModel}(undef, K, T, D)
    @time for k in 1:K, j in 1:D
        mix_allE[k, :, j], talE[k, j], tYE[k, j], hE[k, j] = fit_mle(mix0, θ_al0, θ_Y0, filter(!iszero, 𝐑_per_z[k][:, j]), n2t_rain[k, j], maxiter=maxiter, tol=tol, robust=robust, silence=silence, warm_start=warm_start, display=display)
    end
    return mix_allE
end
#TODO add degree option! Here degree is forced to 1
"""
    fit_mle_RR(df::DataFrame, K, local_order; maxiter=5000, tol=2e-4, robust=true, silence=true, warm_start=true, display=:none, mix₀=mix_ini(T))
```julia
mix_allE = fit_mle_RR.(data_stations, K, local_order)
```

"""
function fit_mle_RR(df::DataFrame, K, local_order; maxiter=5000, tol=2e-4, robust=true, silence=true, warm_start=true, display=:none, mix₀=mix_ini(length(unique(dayofyear_Leap.(df.DATE)))))
    mix0, θ_al0, θ_Y0 = mix₀
    dfs = df[1+local_order:end, :]

    n2t_rain = dayofyear_Leap.([@subset(dfs, :z .== k, :RR .> 0).DATE for k in 1:K]) # un peu bourrin mais bon...
    ẑ_per_cat = [findall(dfs.z .== k) for k in 1:K]
    𝐑_per_z = [dfs[!, :RR][ẑ_per_cat[k]] for k = 1:K] # rain per category

    T = length(mix0)
    talE = [similar(θ_al0) for k in 1:K]
    tYE = [similar(θ_Y0) for k in 1:K]
    hE = Array{Dict}(undef, K)
    mix_allE = Array{MixtureModel}(undef, K, T)
    for k in 1:K
        mix_allE[k, :], talE[k], tYE[k], hE[k] = fit_mle(mix0, θ_al0, θ_Y0, filter(!iszero, 𝐑_per_z[k]), n2t_rain[k], maxiter=maxiter, tol=tol, robust=robust, silence=silence, warm_start=warm_start, display=display)
    end
    return mix_allE
end
