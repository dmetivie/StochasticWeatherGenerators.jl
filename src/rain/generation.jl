"""
    rand_rain(mixs::AbstractArray{<:MixtureModel}, n2t::AbstractVector, z::AbstractVector, y::AbstractMatrix, Σk::AbstractArray)
Generate a (nonhomegenous) sequence of length `length(n2t)` of rain amounts conditionally to a given dry/wet matrix `y` and (hidden) state sequence `z`.
Univariate distribution are given by `mixs` while correlations are given by covariance matrix Σk.
"""
function rand_rain(mixs::AbstractArray{<:MixtureModel}, n2t::AbstractVector, z::AbstractVector, y::AbstractArray, Σk::AbstractArray)
    D, N = size(y)

    r = zeros(D, N) # could be specific to `eltype` of mix
	
    for (n, t) in enumerate(n2t)
        nz_r = findall(!iszero, y[:,n]) # Station j with precipitation
        if length(nz_r) == 0
            continue
        elseif length(nz_r) == 1 #|| z[n] == K (in a old version I did that because dry weather regime z = K was expected to be uncorrelated regime)
            for j in nz_r
                r[j, n] = rand(mixs[z[n], t, j])
            end
        else
			C = GaussianCopula(Σk[z[n]][nz_r, nz_r]) #! use cor2cov if you just have correlations matrix !!! (#? in practice does it make a difference?)
			D = SklarDist(C, tuple(mixs[z[n], t, nz_r]...)) # The final distribution
            r[nz_r, n] = rand(D)
        end
    end
    return r
end