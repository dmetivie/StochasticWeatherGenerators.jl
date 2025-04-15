
#TODO: check that dropmissing (and potentially not contigous data) does not cause issue in MLE
"""
    fit_AR1(df_full::DataFrame, var, 𝐃𝐞𝐠, K = length(unique(df_full.z)), T = length(unique(n2t)))
    fit_AR1(y::AbstractArray, z::AbstractArray, n2t, 𝐃𝐞𝐠, K = length(unique(z |> skipmissing)), T = length(unique(n2t)))
Fit a Seasonal AR(1) model of period `T` and with `K` hidden states for the variable `X` of the DataFrame `df_full`.
The hidden states must be given in a the column `z` of i.e. `df_full.z`.
The correspondance between day of the year `t` and index in the time series `n` must be given in the column `n2t` i.e. `df_full.n2t`.

``X_{n+1} = \\mu(t_n, z_n) + \\phi(t_n, z_n) X_n + \\sigma(t_n, z_n)\\xi``

with ``\\xi \\sim \\mathcal{N}(0,1)``.
"""
function fit_AR1(df_full::DataFrame, var, 𝐃𝐞𝐠, T = length(unique(dayofyear_Leap.(df_full.DATE))))
    if "z" ∈ names(df_full)
        df = dropmissing(df_full[:, [:DATE, var, :z]])
        z = df.z
        K = length(unique(z |> skipmissing))
        n2t = dayofyear_Leap.(df.DATE)
        y = df[:, var]
        return fit_AR1(y, z, n2t, 𝐃𝐞𝐠, K, T)
    else
        df = dropmissing(df_full[:, [:DATE, var]])
        n2t = dayofyear_Leap.(df.DATE)
        y = df[:, var]
        n2t = dayofyear_Leap.(df.DATE)
        y = df[:, var]
        return fit_AR1(y, n2t, 𝐃𝐞𝐠, T)
    end
end

function fit_AR1(y::AbstractVector, z::AbstractVector, n2t, 𝐃𝐞𝐠, K, T)
    n_in_t_k = [setdiff(findall(.&(n2t .== t, z .== k)), 1) for t in 1:T, k in 1:K]

    model_AR1_JuMP = model_for_loglikelihood_AR1(𝐃𝐞𝐠, T, silence=true)
    θ_μ, θ_ρ, θ_σ = zeros(K, 2𝐃𝐞𝐠 + 1), zeros(K, 2𝐃𝐞𝐠 + 1), zeros(K, 2𝐃𝐞𝐠 + 1)

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

function fit_AR1(y::AbstractVector, n2t, 𝐃𝐞𝐠, T)
    n_in_t = [setdiff(findall(.&(n2t .== t)), 1) for t in 1:T]

    model_AR1_JuMP = model_for_loglikelihood_AR1(𝐃𝐞𝐠, T, silence=true)
    θ_μ, θ_ρ, θ_σ = zeros(2𝐃𝐞𝐠 + 1), zeros(2𝐃𝐞𝐠 + 1), zeros(2𝐃𝐞𝐠 + 1)

    observable = Dict{Symbol,Vector}()

    observable[:N] = [length(n_in_t[t]) for t in 1:T]
    observable[:∑yₜ²] = [sum(y[i]^2 for i in n_in_t[t]; init=0) for t in 1:T]
    observable[:∑yₜ₋₁²] = [sum(y[i-1]^2 for i in n_in_t[t]; init=0) for t in 1:T]
    observable[:∑yₜyₜ₋₁] = [sum(y[i] * y[i-1] for i in n_in_t[t]; init=0) for t in 1:T]
    observable[:∑yₜ] = [sum(y[i] for i in n_in_t[t]; init=0) for t in 1:T]
    observable[:∑yₜ₋₁] = [sum(y[i-1] for i in n_in_t[t]; init=0) for t in 1:T]

    initialvalue_optimize!(θ_μ, θ_ρ, θ_σ, model_AR1_JuMP, observable; warm_start=true)

    return AR1([μₜ(t, θ_μ, T) for t in 1:T], [ρₜ(t, θ_ρ, T) for t in 1:T], [σₜ(t, θ_σ, T) for t in 1:T])
end

"""
    cov_ar1(dfs::AbstractArray{<:DataFrame}, ar1s, var, K = length(unique(dfs[1].z)))
Fit the covariance matrix of the residual `ϵ` of several AR(1) models `ar1s`. One matrix is fitted per hidden state. 
The hidden state `z` must be given in `df.z`. Note that we consider constant in time the covariance matrices.
"""
function cov_ar1(dfs::AbstractArray{<:DataFrame}, ar1s, var, K = length(unique(dfs[1].z |> skipmissing)))
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