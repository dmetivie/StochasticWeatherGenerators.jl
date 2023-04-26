"""
    rand_rain(mixs::AbstractArray{<:MixtureModel}, n2t::AbstractVector, z::AbstractVector, y::AbstractMatrix, Σk::AbstractArray)
Generate a (nonhomegenous) sequence of length `length(n2t)` of rain amounts conditionally to a given dry/wet matrix `y` and (hidden) state sequence `z`.
Univariate distribution are given by `mixs` while correlations are given by Σk.
"""
function rand_rain(mixs::AbstractArray{<:MixtureModel}, n2t::AbstractVector, z::AbstractVector, y::AbstractMatrix, Σk::AbstractArray)
    D, N = size(y)

    r = zeros(D, N)

    for n in 1:N
        nz_r = findall(!iszero, y[:, n]) # Station j with precipitation
        if length(nz_r) == 0
            continue
        elseif length(nz_r) == 1 #|| z[n] == K
            for j in nz_r
                r[j, n] = rand(mixs[z[n], n2t[n], j])
            end
        else
            Σnz = Σk[z[n]][nz_r, nz_r]
            r[nz_r, n] = rand(GaussianCopula(Σnz))
        end
    end
    return r
end