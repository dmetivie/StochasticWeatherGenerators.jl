##* Rain correlations *#

"""
    joint_rain(M::AbstractMatrix, j1::Integer, j2::Integer, r = 0)
Select all the rows of `M` with values for (two) columns above some value `r`.
"""
joint_rain(M::AbstractMatrix, j1::Integer, j2::Integer, r=0) = M[(M[:, j1].>r).&(M[:, j2].>r), [j1, j2]]

"""
    Σ_Spearman2Pearson(M::AbstractMatrix)
Compute the Pearson correlation coefficient i.e. the classic one from the Spearman correlation
#TODO Add ref
"""
function Σ_Spearman2Pearson(M::AbstractMatrix)
    n, d = size(M)
    if n > 1
        return 2sin.(corspearman(M) .* (π / 6))
    else
        return [j == 1 ? 1 : missing for i in 1:d, j in 1:d]
    end
end

"""
    Σ_Kendall2Pearson(M::AbstractMatrix)
Compute the Pearson correlation coefficient i.e. the classic one from the Kendall correlation
#TODO Add ref
"""
function Σ_Kendall2Pearson(M::AbstractMatrix)
    n, d = size(M)
    if n > 1
        return sin.(corkendall(M) .* (π / 2))
    else
        return [j == 1 ? 1 : missing for i in 1:d, j in 1:d]
    end
end

"""
    corTail(x::AbstractMatrix, q = 0.95)
Compute the (symmetric averaged) tail index matrix `M` of a vector x, i.e. M[i, j] = (ℙ(x[:,j] > Fxⱼ(q) ∣ x[:,i] > Fxᵢ(q)) + ℙ(x[:,i] > Fxᵢ(q) ∣ x[:,j] > Fxⱼ(q)))/2 where Fx(q) is the CDF of x.
Note it uses the same convention as `cor` function i.e. observations in rows and features in column.
"""
function corTail(x, q=0.95)
    n, k = size(x)
    c = zeros(k, k)
    for i in 1:k
        ind_i = findall(x[:, i] .>= quantile(x[:, i], q))
        for j in 1:k
            c[i, j] = count(x[ind_i, j] .>= quantile(x[:, j], q)) / length(ind_i)
        end
    end

    return (c + c') / 2
end

#TODO: is NearestCorrelationMatrix.jl the best way to enforce positivity? Uncertainty in estimation could be interesting (coefficients we trust or not).
"""
    cor_RR(dfs::AbstractArray{<:DataFrame}[, K]; cor_method=Σ_Spearman2Pearson, force_PosDef = true)
Compute the (strictly positive) rain pair correlations `cor(Rs₁ > 0, Rs₂ > 0)` between each pair of stations `s₁, s₂` for each hidden state `Z = k`.

Input: a array `dfs` of `df::DataFrame` of length `S` (number of station) where each `df` have :DATE, :RR, :z (same :z for each df).

Output: `K` correlation matrix of size `S×S`

Options:

- `force_PosDef` will enforce Positive Definite matrix with [NearestCorrelationMatrix.jl](https://github.com/adknudson/NearestCorrelationMatrix.jl).
- `cor_method`: typically `Σ_Spearman2Pearson` or `Σ_Kendall2Pearson`
- `impute_missing`: if `nothing`, `missing` will be outputted when two stations do not have at least two rain days in common. Otherwise the value `impute_missing` will be set.

```julia
ΣRR = cor_RR(data_stations, K)
```
"""
function cor_RR(dfs::AbstractArray{<:DataFrame}, K; cor_method=Σ_Spearman2Pearson, force_PosDef=true, impute_missing=:nothing)
    D = length(dfs)
    ΣS_k = Matrix{Union{Float64,Missing}}[zeros(D, D) + StochasticWeatherGenerators.I for k in 1:K]
    for j1 in 2:D
        for j2 in 1:j1-1
            R1R2 = @subset(innerjoin(dfs[j1], dfs[j2], on=:DATE, makeunique=true), :RR .> 0, :RR_1 .> 0)
            for k in 1:K
                R1R2_k = @subset(R1R2, :z .== k)[!, [:RR, :RR_1]] |> Matrix
                ΣS_k[k][j1, j2] = ΣS_k[k][j2, j1] = ifelse(!isnothing(impute_missing) && size(R1R2_k, 1) < 1, impute_missing, cor_method(R1R2_k)[1, 2])
            end
        end
    end
    if all([all((!ismissing).(S)) for S in ΣS_k]) # are they no missing?
        ΣS_k = convert.(Matrix{Float64}, ΣS_k)
        if force_PosDef
            ΣS_k = nearest_cor!.(ΣS_k)
        end
    else
        for k in 1:K
            aremissing = findall(ismissing, ΣS_k[k])
            if length(aremissing) > 0
                @warn "ΣS_k[$(k)], $(findall(ismissing, ΣS_k[k])) are missing"
            end
        end
    end
    return ΣS_k
end

"""
    cov_RR(dfs::AbstractArray{<:DataFrame}[, K]; cor_method=Σ_Spearman2Pearson, force_PosDef = true)
Compute the (strictly positive) rain pair covariance `cov(Rs₁ > 0, Rs₂ > 0)` between each pair of stations `s₁, s₂` for each hidden state `Z = k`.

Input: a array `dfs` of `df::DataFrame` of length `S` (number of station) where each `df` have :DATE, :RR, :z (same :z for each df).

Output: `K` covariance matrix of size `S×S`

Options:

- `force_PosDef` will enforce Positive Definite matrix with [NearestCorrelationMatrix.jl](https://github.com/adknudson/NearestCorrelationMatrix.jl).
- `cor_method`: typically `Σ_Spearman2Pearson` or `Σ_Kendall2Pearson`
- `impute_missing`: if `nothing`, `missing` will be outputted when two stations do not have at least two rain days in common. Otherwise the value `impute_missing` will be set.

```julia
ΣRR = cov_RR(data_stations, K)
```
"""
function cov_RR(dfs::AbstractArray{<:DataFrame}, K; cor_method=Σ_Spearman2Pearson, force_PosDef=true, impute_missing=:nothing)
    ΣS_k = cor_RR(dfs, K; cor_method=cor_method, force_PosDef=force_PosDef, impute_missing=impute_missing)
    return [cor2cov(Σ, [std(@subset(df, :z .== k, :RR .> 0).RR) for df in dfs]) for (k, Σ) in enumerate(ΣS_k)]
end

function cor_RR(dfs::AbstractArray{<:DataFrame}; cor_method=Σ_Spearman2Pearson, force_PosDef=true, impute_missing=:nothing)
    D = length(dfs)
    ΣS = zeros(D, D) + StochasticWeatherGenerators.I
    for j1 in 2:D
        for j2 in 1:j1-1
            R1R2 = @subset(innerjoin(dfs[j1], dfs[j2], on=:DATE, makeunique=true), :RR .> 0, :RR_1 .> 0)
            R1R2 = R1R2[!, [:RR, :RR_1]] |> Matrix
            ΣS[j1, j2] = ΣS[j2, j1] = ifelse(!isnothing(impute_missing) && size(R1R2, 1) < 1, impute_missing, cor_method(R1R2)[1, 2])
        end
    end
    if all((!ismissing).(ΣS)) # are they no missing?
        ΣS = convert(Matrix{Float64}, ΣS)
        if force_PosDef
            nearest_cor!(ΣS)
        end
    else
        aremissing = findall(ismissing, ΣS)
        if length(aremissing) > 0
            @warn "ΣS, $(aremissing) are missing"
        end
    end
    return ΣS
end

function cov_RR(dfs::AbstractArray{<:DataFrame}; cor_method=Σ_Spearman2Pearson, force_PosDef=true, impute_missing=:nothing)
    ΣS = cor_RR(dfs; cor_method=cor_method, force_PosDef=force_PosDef, impute_missing=impute_missing)
    return cor2cov(ΣS, [std(@subset(df, :RR .> 0).RR) for df in dfs])
end