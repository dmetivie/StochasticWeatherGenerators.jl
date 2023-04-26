##* Rain correlations *#

"""
    joint_rain(M::AbstractMatrix, j1::Integer, j2::Integer, r = 0)
Select all the rows of `M` with values for (two) colums above some value `r`.
    
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
Note it uses the same convention as `cor` function i.e. observations in rows and features in colums.
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