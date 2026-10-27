"""
    rand_RR([rng::AbstractRNG, ]mixs::AbstractArray{<:MixtureModel}, n2t::AbstractVector, z::AbstractVector, y::AbstractMatrix, Σk::AbstractArray)
Generate a (nonhomegenous) sequence of length `length(n2t)` of rainfall amounts conditionally to a given dry/wet matrix `y` and (hidden) state sequence `z`.
Univariate distribution are given by `mixs` while correlations are given by covariance matrix Σk.
"""
function rand_RR(rng::AbstractRNG, mixs::AbstractArray{<:MixtureModel}, n2t::AbstractVector, z::AbstractVector, y::AbstractArray, Σk::AbstractArray)
    # My experience is that many correlations matrix choices works with same result in my example i.e. cor, cov, for rain per categorie with zeros included or not etc.
    D, N = size(y)
    K = size(mixs, 1)
    r = zeros(D, N) # could be specific to `eltype` of mix
    if K < 2
        for (n, t) in enumerate(n2t)
            nz_r = findall(!iszero, y[:, n]) # Station j with precipitation
            if length(nz_r) == 0
                continue
            elseif length(nz_r) == 1
                for j in nz_r
                    r[j, n] = rand(rng, mixs[z[n], t, j])
                end
            else
                C = GaussianCopula(Σk[z[n]][nz_r, nz_r]) #! use cor2cov if you just have correlations matrix !!! (#? in practice does it make a difference?)
                𝔇 = SklarDist(C, tuple(mixs[z[n], t, nz_r]...)) # The final distribution
                @views r[nz_r, n] = rand(rng, 𝔇)
            end
        end
    else
        for (n, t) in enumerate(n2t)
            nz_r = findall(!iszero, y[:, n]) # Station j with precipitation
            if length(nz_r) == 0
                continue
            elseif length(nz_r) == 1 || (z[n] == K) # I did that because dry weather regime z = K was expected to be uncorrelated regime, pragmatically without that because Σ₄ is not semi definite postive generation might fail
                for j in nz_r
                    r[j, n] = rand(rng, mixs[z[n], t, j])
                end
            else
                C = GaussianCopula(Σk[z[n]][nz_r, nz_r]) #! use cor2cov if you just have correlations matrix !!! (#? in practice does it make a difference?)
                𝔇 = SklarDist(C, tuple(mixs[z[n], t, nz_r]...)) # The final distribution
                @views r[nz_r, n] = rand(rng, 𝔇)
            end
        end
    end
    return r
end

rand_RR(mixs::AbstractArray{<:MixtureModel}, n2t::AbstractVector, z::AbstractVector, y::AbstractArray, Σk::AbstractArray) = rand_RR(GLOBAL_RNG, mixs, n2t, z, y, Σk)

#TODO clarify input shape (array of array or big array?)
function rand_RR(rng::AbstractRNG, mixs::AbstractArray{M}, n2t::AbstractVector, z::AbstractVector, y::AbstractArray, Σk::AbstractArray) where {M<:AbstractMatrix{<:MixtureModel}}
    # My experience is that many correlations matrix choices works with same result in my example i.e. cor, cov, for rain per categorie with zeros included or not etc.
    K, T = size(mixs[1])
    D = length(mixs)
    reshape_mixs = Array{MixtureModel}(undef, K, T, D)
    for j in 1:D
        reshape_mixs[:, :, j] = mixs[j]
    end
    return rand_RR(rng, reshape_mixs, n2t, z, y, Σk)
end

rand_RR(mixs::AbstractArray{M}, n2t::AbstractVector, z::AbstractVector, y::AbstractArray, Σk::AbstractArray) where {M<:AbstractMatrix{<:MixtureModel}} = rand_RR(GLOBAL_RNG, mixs, n2t, z, y, Σk)

function rand_cond end