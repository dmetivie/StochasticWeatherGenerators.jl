##* Rain correlations *#

# These four mirror `Statistics.cor` / `cov`, so they take the same `dims` keyword with the same
# meaning: `dims = 1` for observations in rows (the `cor` default), `dims = 2` for observations in
# columns -- the `D x N` layout used everywhere else in this package, which is why every call site
# here passes `dims = 2`.
_check_dims(dims::Integer) = dims in (1, 2) ? dims :
    throw(ArgumentError("dims must be 1 (observations in rows) or 2 (observations in columns), got $dims"))

# the variables of `M`, as a lazy vector of vectors
_variables(M::AbstractMatrix, dims::Integer) = _check_dims(dims) == 1 ? eachcol(M) : eachrow(M)

# `M` rearranged so observations are in rows, which is what StatsBase's `cor*` want
_obs_in_rows(M::AbstractMatrix, dims::Integer) = _check_dims(dims) == 1 ? M : permutedims(M)

"""
    joint_rain(M::AbstractMatrix, j1::Integer, j2::Integer, r = 0; dims = 1)
Select the observations of `M` where variables `j1` and `j2` are both above some value `r`, keeping
only those two variables. With `dims = 1` (the `cor` convention) variables are columns and the
result is `n x 2`; with `dims = 2` they are rows and the result is `2 x n`.
"""
function joint_rain(M::AbstractMatrix, j1::Integer, j2::Integer, r=0; dims=1)
    v = _variables(M, dims)
    keep = (v[j1] .> r) .& (v[j2] .> r)
    return dims == 1 ? M[keep, [j1, j2]] : M[[j1, j2], keep]
end

"""
    Σ_Spearman2Pearson(M::AbstractMatrix; dims = 1)
Compute the Pearson correlation coefficient i.e. the classic one from the Spearman correlation.
`dims` follows `Statistics.cor`: `1` for observations in rows, `2` for observations in columns.
#TODO Add ref
"""
function Σ_Spearman2Pearson(M::AbstractMatrix; dims=1)
    X = _obs_in_rows(M, dims)
    n, d = size(X)
    if n > 1
        return 2sin.(corspearman(X) .* (π / 6))
    else
        return [j == 1 ? 1 : missing for i in 1:d, j in 1:d]
    end
end

"""
    Σ_Kendall2Pearson(M::AbstractMatrix; dims = 1)
Compute the Pearson correlation coefficient i.e. the classic one from the Kendall correlation.
`dims` follows `Statistics.cor`: `1` for observations in rows, `2` for observations in columns.
#TODO Add ref
"""
function Σ_Kendall2Pearson(M::AbstractMatrix; dims=1)
    X = _obs_in_rows(M, dims)
    n, d = size(X)
    if n > 1
        return sin.(corkendall(X) .* (π / 2))
    else
        return [j == 1 ? 1 : missing for i in 1:d, j in 1:d]
    end
end

"""
    corTail(x::AbstractMatrix, q = 0.95; dims = 1)
Compute the (symmetric averaged) tail index matrix `M` of `x`, i.e. M[i, j] = (ℙ(xⱼ > Fxⱼ(q) ∣ xᵢ > Fxᵢ(q)) + ℙ(xᵢ > Fxᵢ(q) ∣ xⱼ > Fxⱼ(q)))/2 where Fx(q) is the CDF of x.
`dims` follows `Statistics.cor`: `1` for observations in rows, `2` for observations in columns.
"""
function corTail(x, q=0.95; dims=1)
    vars = _variables(x, dims)
    k = length(vars)
    c = zeros(k, k)
    for i in 1:k
        xi = vars[i]
        ind_i = findall(xi .>= quantile(xi, q))
        for j in 1:k
            xj = vars[j]
            c[i, j] = count(xj[ind_i] .>= quantile(xj, q)) / length(ind_i)
        end
    end

    return (c + c') / 2
end

#TODO: is NearestCorrelationMatrix.jl the best way to enforce positivity? Uncertainty in estimation could be interesting (coefficients we trust or not).
#Todo: proper rewrite  dealing with missing, NaN, impute

"""
    cor_RR(dfs::AbstractArray{<:DataFrame}[, K]; cor_method=Σ_Spearman2Pearson, force_PosDef = true)
Compute the (strictly positive) rain pair correlations `cor(Rs₁ > 0, Rs₂ > 0)` between each pair of stations `s₁, s₂` for each hidden state `Z = k`.

Input: a array `dfs` of `df::DataFrame` of length `S` (number of station) where each `df` have :DATE, :RR, :z (same :z for each df).

Output: `K` correlation matrix of size `S×S`

Options:

- `force_PosDef` will enforce Positive Definite matrix with [NearestCorrelationMatrix.jl](https://github.com/adknudson/NearestCorrelationMatrix.jl).
- `cor_method`: typically `Σ_Spearman2Pearson` or `Σ_Kendall2Pearson`; it is called as `cor_method(M; dims = 2)`, i.e. it must accept `dims` like `Statistics.cor` does
- `impute_missing`: if `nothing`, `missing` will be outputted when two stations do not have at least two rain days in common. Otherwise the value `impute_missing` will be set.

```julia
ΣRR = cor_RR(data_stations, K)
```
"""
function cor_RR(dfs::AbstractArray{<:DataFrame}, K; cor_method=Σ_Spearman2Pearson, force_PosDef=true, impute_missing=nothing)
    D = length(dfs)
    ΣS_k = Matrix{Union{Float64,Missing}}[zeros(D, D) + LinearAlgebra.I for k in 1:K]
    for j1 in 2:D
        for j2 in 1:j1-1
            R1R2 = @subset(innerjoin(dfs[j1], dfs[j2], on=:DATE, makeunique=true), :RR .> 0, :RR_1 .> 0)
            for k in 1:K
                R1R2_k = permutedims(@subset(R1R2, :z .== k)[!, [:RR, :RR_1]] |> Matrix) # 2 × n
                # if size(R1R2_k, 2) ≤ 2
                    ΣS_k[k][j1, j2] = ΣS_k[k][j2, j1] = ifelse(!isnothing(impute_missing) && size(R1R2_k, 2) < 2, impute_missing, cor_method(R1R2_k; dims=2)[1, 2])
                # else
                    # ΣS_k[k][j1, j2] = ΣS_k[k][j2, j1] = cor_method(R1R2_k)[1, 2]
                # end
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
- `cor_method`: typically `Σ_Spearman2Pearson` or `Σ_Kendall2Pearson`; it is called as `cor_method(M; dims = 2)`, i.e. it must accept `dims` like `Statistics.cor` does
- `impute_missing`: if `nothing`, `missing` will be outputted when two stations do not have at least two rain days in common. Otherwise the value `impute_missing` will be set.

```julia
ΣRR = cov_RR(data_stations, K)
```
"""
function cov_RR(dfs::AbstractArray{<:DataFrame}, K; cor_method=Σ_Spearman2Pearson, force_PosDef=true, impute_missing=nothing)
    ΣS_k = cor_RR(dfs, K; cor_method=cor_method, force_PosDef=force_PosDef, impute_missing=impute_missing)
    return [cor2cov(Σ, [std(@subset(df, :z .== k, :RR .> 0).RR) for df in dfs]) for (k, Σ) in enumerate(ΣS_k)]
end

function cor_RR(dfs::AbstractArray{<:DataFrame}; cor_method=Σ_Spearman2Pearson, force_PosDef=true, impute_missing=nothing)
    D = length(dfs)
    ΣS = zeros(D, D) + StochasticWeatherGenerators.I
    for j1 in 2:D
        for j2 in 1:j1-1
            R1R2 = @subset(innerjoin(dfs[j1], dfs[j2], on=:DATE, makeunique=true), :RR .> 0, :RR_1 .> 0)
            R1R2 = permutedims(R1R2[!, [:RR, :RR_1]] |> Matrix) # 2 × n
            ΣS[j1, j2] = ΣS[j2, j1] = ifelse(!isnothing(impute_missing) && size(R1R2, 2) < 1, impute_missing, cor_method(R1R2; dims=2)[1, 2])
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

function cov_RR(dfs::AbstractArray{<:DataFrame}; cor_method=Σ_Spearman2Pearson, force_PosDef=true, impute_missing=nothing)
    ΣS = cor_RR(dfs; cor_method=cor_method, force_PosDef=force_PosDef, impute_missing=impute_missing)
    return cor2cov(ΣS, [std(@subset(df, :RR .> 0).RR) for df in dfs])
end