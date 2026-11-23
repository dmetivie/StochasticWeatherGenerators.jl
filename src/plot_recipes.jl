# Plotting functions. They have no methods until a Makie backend is loaded (`MakieExt`), and
# `map_with_stations` until GeoMakie and NaturalEarth are loaded (`GeoMakieExt`).

"""
    errorline(x, y; kwargs...)
    errorline(y; kwargs...)
    errorline!(ax, x, y; kwargs...)

Plot the center of each row of the matrix `y` against `x`, with the spread across its columns as a ribbon, error sticks or a plume.
`y` has one row per `x` value and one column per realization; `NaN` values are skipped. Requires Makie (e.g. `using CairoMakie`).

# Keyword arguments
- `errorstyle = :ribbon`: `:ribbon`, `:stick` or `:plume`.
- `centertype = :mean`: `:mean` or `:median`.
- `errortype = :std`: `:std`, `:sem` (standard error of the mean) or `:percentile`.
- `percentiles = [25, 75]`: lower and upper percentiles when `errortype = :percentile`.
- `secondarycolor = automatic`: color of the ribbon, sticks or plume; defaults to `color`.
- `secondaryalpha = 0.1`: transparency of the ribbon, sticks or plume.
- `secondarylines = 100`: number of plume lines (the first columns), or a vector of column indices.
- `secondarylinewidth = 1`: width of the sticks and plume lines.
- `whiskerwidth = 10`: width of the stick whiskers.

The center line accepts every `lines` attribute (`color`, `linewidth`, `linestyle`, `label`, ...).

# Example
```julia
using CairoMakie, StochasticWeatherGenerators
y = cumsum(randn(50, 200); dims=1)
fig, ax, p = errorline(1:50, y; errortype=:percentile, percentiles=[5, 95], centertype=:median)
errorline!(ax, 1:50, y .+ 10; errorstyle=:plume, secondarylines=30)
fig
```
"""
function errorline end
function errorline! end

"""
    errorscatter(x, y; kwargs...)
    errorscatter(y; kwargs...)
    errorscatter!(ax, x, y; kwargs...)

Same as [`errorline`](@ref) with the center drawn as markers instead of a line.
The center accepts every `scatter` attribute (`color`, `marker`, `markersize`, `label`, ...). Requires Makie.
"""
function errorscatter end
function errorscatter! end

"""
    errorlinehist(data; bins=15, normalization=:pdf, kwargs...)
    errorlinehist!(ax, data; bins=15, normalization=:pdf, kwargs...)

Plot the spread of the histograms of the samples `data[1]`, `data[2]`, ... with [`errorline`](@ref).
All samples share the same bins, so the spread at each bin is computed across samples; pass `[sample]` to plot a single histogram.
Each histogram is drawn at the left edges of its bins. Requires Makie.

# Keyword arguments
- `bins = 15`: number of bins, or a sorted vector of bin edges (bins are closed on the left).
- `normalization = :pdf`: `:pdf`, `:density`, `:probability` or `:none` (counts), as in `StatsBase.normalize`.

All other keywords are those of [`errorline`](@ref).

# Example
```julia
using CairoMakie, StochasticWeatherGenerators, Distributions
samples = [rand(Exponential(), 100) for _ in 1:1000]
fig, ax, p = errorlinehist(samples; bins=0:0.25:8, errortype=:percentile, percentiles=[0, 100], axis=(yscale=log10,))
errorlinehist!(ax, [rand(Exponential(), 100)]; bins=0:0.25:8, errortype=:percentile, color=:black)
fig
```
"""
function errorlinehist end
function errorlinehist! end

"""
    errorscatterhist(data; bins=15, normalization=:pdf, kwargs...)
    errorscatterhist!(ax, data; bins=15, normalization=:pdf, kwargs...)

Same as [`errorlinehist`](@ref) with the center drawn as markers, see [`errorscatter`](@ref). Requires Makie.
"""
function errorscatterhist end
function errorscatterhist! end

"""
    map_with_stations(LON, LAT, value=ones(length(LON)); kwargs...)
    map_with_stations(LON, LAT, values::AbstractVector{<:AbstractVector}; kwargs...)

Return a map of France with the stations at longitudes `LON` and latitudes `LAT` (decimal degrees) colored by `value`.
With a vector of value vectors, draw one map per vector side by side, sharing one color range (e.g. one map per hidden state).
Requires `using CairoMakie, GeoMakie, NaturalEarth`.

# Keyword arguments
- `station_name = nothing`: names written above the stations (single map only).
- `show_value = false`: write the values right of the stations (single map only).
- `labelsize`: font size of the names and values, by default the theme `fontsize`.
- `colorbar_show = false`, `colorbar_label = ""`: add a colorbar and its label.
- `colormap`, `colorrange`, `markersize`: marker coloring; the color range defaults to the padded range of `value`, or `(0, 1)` for several maps.
- `titles = [L"Z = k" for k in 1:K]`: titles of the maps (several maps only).
- `precision_scale = 50`: Natural Earth scale of the coastlines, borders and rivers: `10`, `50` or `110` (1:10m, 1:50m or 1:110m).
- `LON_min = -5`, `LON_max = 10.5`, `LAT_min = 41`, `LAT_max = 52`: map limits.
"""
function map_with_stations end

const _MAKIE_FUNCTIONS = (errorline, errorline!, errorscatter, errorscatter!, errorlinehist, errorlinehist!,
    errorscatterhist, errorscatterhist!)

function _plot_error_hint(io, exc, _argtypes, _kwargs)
    if exc.f in _MAKIE_FUNCTIONS && isnothing(Base.get_extension(@__MODULE__, :MakieExt))
        print(io, "\n`$(nameof(exc.f))` is defined once a Makie backend is loaded, e.g. `using CairoMakie`.")
    elseif exc.f === map_with_stations && isnothing(Base.get_extension(@__MODULE__, :GeoMakieExt))
        print(io, "\n`map_with_stations` is defined once GeoMakie and NaturalEarth are loaded, e.g. `using CairoMakie, GeoMakie, NaturalEarth`.")
    end
end
