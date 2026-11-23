hist_attributes() = Makie.@DocumentedAttributes begin
    "Number of bins, or a sorted vector of bin edges (bins are closed on the left)."
    bins = 15
    "`:pdf`, `:density`, `:probability` or `:none`, as in `StatsBase.normalize`."
    normalization = :pdf
end

"""
    errorlinehist(data; kwargs...)

Keyword arguments accepted by `errorlinehist` and `errorlinehist!`.
"""
@recipe ErrorLineHist (data,) begin
    Makie.documented_attributes(ErrorLine)...
    hist_attributes()...
end

"""
    errorscatterhist(data; kwargs...)

Keyword arguments accepted by `errorscatterhist` and `errorscatterhist!`.
"""
@recipe ErrorScatterHist (data,) begin
    Makie.documented_attributes(ErrorScatter)...
    hist_attributes()...
end

function hist_edges(samples, bins::Integer)
    bins ≥ 1 || throw(ArgumentError("bins must be positive, got $bins"))
    lo, hi = float.(extrema(Iterators.flatten(samples)))
    lo == hi && return [lo - 0.5, hi + 0.5]
    return range(lo, nextfloat(hi); length=bins + 1) # bins are closed on the left: keep the maximum
end

function hist_edges(_, bins::AbstractVector)
    (length(bins) ≥ 2 && issorted(bins)) || throw(ArgumentError("bin edges must be a sorted vector of at least two values"))
    return bins
end

# Left bin edges and the matrix of histograms (one column per sample) on shared bins.
function hist_matrix(data, bins, normalization)
    samples = [filter(isfinite, sample) for sample in data]
    all(isempty, samples) && throw(ArgumentError("no finite value to bin"))
    edges = hist_edges(samples, bins)
    y = zeros(length(edges) - 1, length(samples))
    for (i, sample) in enumerate(samples)
        h = fit(Histogram, sample, edges; closed=:left)
        y[:, i] = normalize(float(h); mode=normalization).weights
    end
    y .+= eps() # empty bins stay finite on a log axis
    return collect(edges[1:end-1]), y
end

function Makie.plot!(p::ErrorLineHist)
    map!(hist_matrix, p.attributes, [:data, :bins, :normalization], [:bin_x, :bin_y])
    errorline!(p, Makie.shared_attributes(p, ErrorLine), p.bin_x, p.bin_y)
    return p
end

function Makie.plot!(p::ErrorScatterHist)
    map!(hist_matrix, p.attributes, [:data, :bins, :normalization], [:bin_x, :bin_y])
    errorscatter!(p, Makie.shared_attributes(p, ErrorScatter), p.bin_x, p.bin_y)
    return p
end
