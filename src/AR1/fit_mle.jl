
#TODO: check that dropmissing (and potentially not contigous data) does not cause issue in MLE
"""
    fit_AR1(df_full::DataFrame, X, 𝐃𝐞𝐠, T, K)
Fit a Seasonal AR(1) model of period `T` and with `K` hidden states for the variable `X` of the DataFrame `df_full`.
``X_{n+1} = \\mu(t_n) + \\phi(t_n) X_t + \\sigma(t_n)\\xi``
"""
function fit_AR1(df_full::DataFrame, var, 𝐃𝐞𝐠, T, K)
    df = dropmissing(df_full[:, [:DATE, var, :z]])
    z = df.z
    n2t = dayofyear_Leap.(df.DATE)
    n_in_t_k = [setdiff(findall(.&(n2t .== t, z .== k)), 1) for t in 1:T, k in 1:K]


    model_AR1_JuMP = model_for_loglikelihood_AR1(𝐃𝐞𝐠, T, silence=true)
    θ_μ, θ_ρ, θ_σ = zeros(K, 2𝐃𝐞𝐠 + 1), zeros(K, 2𝐃𝐞𝐠 + 1), zeros(K, 2𝐃𝐞𝐠 + 1)

    y = df[:, var]

    observable = [Dict{Symbol,Vector}() for k in 1:K]
    for k in 1:K
        observable[k][:N] = [length(n_in_t_k[t, k]) for t in 1:T]
        observable[k][:∑yₜ²] = [sum(y[i]^2 for i in n_in_t_k[t, k]; init=0) for t in 1:T]
        observable[k][:∑yₜ₋₁²] = [sum(y[i-1]^2 for i in n_in_t_k[t, k]; init=0) for t in 1:T]
        observable[k][:∑yₜyₜ₋₁] = [sum(y[i] * y[i-1] for i in n_in_t_k[t, k]; init=0) for t in 1:T]
        observable[k][:∑yₜ] = [sum(y[i] for i in n_in_t_k[t, k]; init=0) for t in 1:T]
        observable[k][:∑yₜ₋₁] = [sum(y[i-1] for i in n_in_t_k[t, k]; init=0) for t in 1:T]
    end

    for k in 1:K
        initialvalue_optimize!(@view(θ_μ[k, :]), @view(θ_ρ[k, :]), @view(θ_σ[k, :]), model_AR1_JuMP, observable[k]; warm_start=true)
    end

    return AR1([μₜ(t, θ_μ[k, :], T) for k in 1:K, t in 1:T], [ρₜ(t, θ_ρ[k, :], T) for k in 1:K, t in 1:T], [σₜ(t, θ_σ[k, :], T) for k in 1:K, t in 1:T])
end

"""
Fit residual to constant (in time) cov matrices for each weather regime
Example:
```julia
cov_ar1(data_stations, ar1sTX, :TX, K)
```
"""
function cov_ar1(dfs::AbstractArray{<:DataFrame}, ar1s, var, K)
    #TODO buggy when missing
    date_start = maximum([df.DATE[1] for df in dfs])
    date_end = minimum([df.DATE[end] for df in dfs])
    df_i = DataFrame(:DATE => date_start:Day(1):date_end)
    for df in dfs
        leftjoin!(df_i, df, on=:DATE, makeunique=true)
    end
    dropmissing!(df_i)
    𝕿 = Matrix(df_i[:, findall([startswith(c, string(var)) for c in names(df_i)])])
    ẑ = df_i.z
    n_in_k = [findall(ẑ .== k) for k in 1:K]
    n2t = dayofyear_Leap.(df_i.DATE)
    N = length(n2t)

    ϵ = zeros(size(𝕿))
    for (j, ar1) in enumerate(ar1s)
        for n in 2:N
            ϵ[n, j] = (𝕿[n] - ar1.ρ[ẑ[n], n2t[n]] * 𝕿[n-1] - ar1.μ[ẑ[n], n2t[n]]) / ar1.σ[ẑ[n], n2t[n]]
        end
    end
    return [cov(ϵ[n_in_k[k], :]) for k in 1:K]
end