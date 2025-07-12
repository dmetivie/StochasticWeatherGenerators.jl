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

function fit_AR1 end