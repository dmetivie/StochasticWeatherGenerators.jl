error_attributes() = Makie.@DocumentedAttributes begin
    "Style of the spread: `:ribbon`, `:stick` or `:plume`."
    errorstyle = :ribbon
    "Center of each row of `y`: `:mean` or `:median`."
    centertype = :mean
    "Spread of each row of `y`: `:std`, `:sem` or `:percentile`."
    errortype = :std
    "Lower and upper percentiles used when `errortype = :percentile`."
    percentiles = [25, 75]
    "Color of the ribbon, sticks or plume; `automatic` uses `color`."
    secondarycolor = Makie.automatic
    "Transparency of the ribbon, sticks or plume."
    secondaryalpha = 0.1
    "Number of plume lines (the first columns of `y`), or a vector of column indices."
    secondarylines = 100
    "Width of the sticks and plume lines."
    secondarylinewidth = 1
    "Width of the stick whiskers."
    whiskerwidth = 10
end

"""
    errorline(x, y; kwargs...)

Keyword arguments accepted by `errorline` and `errorline!`.
"""
@recipe ErrorLine (x, y) begin
    Makie.documented_attributes(Lines)...
    error_attributes()...
end

"""
    errorscatter(x, y; kwargs...)

Keyword arguments accepted by `errorscatter` and `errorscatter!`.
"""
@recipe ErrorScatter (x, y) begin
    Makie.documented_attributes(Scatter)...
    error_attributes()...
end

for P in (ErrorLine, ErrorScatter)
    @eval Makie.convert_arguments(::Type{<:$P}, y::AbstractMatrix) = (1:size(y, 1), y)
    @eval Makie.convert_arguments(::Type{<:$P}, x::AbstractVector, y::AbstractVector) = (x, reshape(y, :, 1))
end

# Center and spread of each row of `y`, skipping NaN.
function center_and_bounds(x, y::AbstractMatrix, centertype::Symbol, errortype::Symbol, percentiles)
    size(y, 1) == length(x) || throw(DimensionMismatch(
        "y must have one row per x value, got $(size(y, 1)) rows for $(length(x)) x values"))
    centertype in (:mean, :median) || throw(ArgumentError("centertype must be :mean or :median, got :$centertype"))
    errortype in (:std, :sem, :percentile) || throw(ArgumentError("errortype must be :std, :sem or :percentile, got :$errortype"))
    if errortype === :percentile && !(length(percentiles) == 2 && 0 ≤ percentiles[1] ≤ percentiles[2] ≤ 100)
        throw(ArgumentError("percentiles must be two increasing values in [0, 100], got $percentiles"))
    end
    n = size(y, 1)
    center, lower, upper = fill(NaN, n), fill(NaN, n), fill(NaN, n)
    for i in 1:n
        r = [float(v) for v in view(y, i, :) if !isnan(v)]
        isempty(r) && continue
        center[i] = centertype === :mean ? mean(r) : median(r)
        if errortype === :percentile
            lower[i], upper[i] = quantile(r, percentiles[1] / 100), quantile(r, percentiles[2] / 100)
        else
            e = std(r) # NaN for a single value
            errortype === :sem && (e /= sqrt(length(r)))
            lower[i], upper[i] = center[i] - e, center[i] + e
        end
    end
    return center, lower, upper
end

# Columns of `y` as one NaN-separated line.
function plume_lines(x, y::AbstractMatrix, secondarylines)
    columns = secondarylines isa Integer ? (1:min(secondarylines, size(y, 2))) : secondarylines
    px, py = Float64[], Float64[]
    for j in columns
        append!(px, x)
        push!(px, NaN)
        append!(py, view(y, :, j))
        push!(py, NaN)
    end
    return px, py
end

function plot_spread!(p)
    map!(center_and_bounds, p.attributes, [:x, :y, :centertype, :errortype, :percentiles],
        [:y_center, :y_lower, :y_upper])
    map!(p.attributes, [:color, :secondarycolor], :error_color) do color, secondarycolor
        return secondarycolor === Makie.automatic ? color : secondarycolor
    end
    errorstyle = p.errorstyle[]
    if errorstyle === :ribbon
        band!(p, p.x, p.y_lower, p.y_upper; color=p.error_color, alpha=p.secondaryalpha)
    elseif errorstyle === :stick
        map!(p.attributes, [:y_center, :y_lower, :y_upper], [:error_low, :error_high]) do center, lower, upper
            return center .- lower, upper .- center
        end
        errorbars!(p, p.x, p.y_center, p.error_low, p.error_high; color=p.error_color,
            alpha=p.secondaryalpha, linewidth=p.secondarylinewidth, whiskerwidth=p.whiskerwidth)
    elseif errorstyle === :plume
        map!(plume_lines, p.attributes, [:x, :y, :secondarylines], [:plume_x, :plume_y])
        lines!(p, p.plume_x, p.plume_y; color=p.error_color, alpha=p.secondaryalpha,
            linewidth=p.secondarylinewidth)
    else
        throw(ArgumentError("errorstyle must be :ribbon, :stick or :plume, got :$errorstyle"))
    end
    return p
end

function Makie.plot!(p::ErrorLine)
    plot_spread!(p)
    lines!(p, Makie.shared_attributes(p, Lines), p.x, p.y_center)
    return p
end

function Makie.plot!(p::ErrorScatter)
    plot_spread!(p)
    scatter!(p, Makie.shared_attributes(p, Scatter), p.x, p.y_center)
    return p
end
