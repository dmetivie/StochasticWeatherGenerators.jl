module GeoMakieExt

using StochasticWeatherGenerators
import StochasticWeatherGenerators: map_with_stations

using GeoMakie # re-exports Makie
using GeoMakie: Makie
using NaturalEarth: naturalearth

# Natural Earth layers drawn under the stations.
function map_features(precision_scale)
    return (
        ocean=naturalearth("ocean", precision_scale),
        borders=naturalearth("admin_0_boundary_lines_land", precision_scale),
        coastlines=naturalearth("coastline", precision_scale),
        rivers=naturalearth("rivers_lake_centerlines_scale_rank", precision_scale),
    )
end

function map_axis(position, features; LON_min, LON_max, LAT_min, LAT_max, kwargs...)
    ax = GeoAxis(position; dest="+proj=merc", xgridvisible=false, ygridvisible=false,
        xticksvisible=false, yticksvisible=false, kwargs...)
    xlims!(ax, LON_min, LON_max)
    ylims!(ax, LAT_min, LAT_max)
    poly!(ax, features.ocean.geometry)
    lines!(ax, GeoMakie.to_multilinestring.(features.borders.geometry); color=:black, linestyle=:dash, linewidth=0.75)
    lines!(ax, GeoMakie.to_multilinestring.(features.coastlines.geometry); color=:gray, linewidth=0.75)
    lines!(ax, GeoMakie.to_multilinestring.(features.rivers.geometry); linewidth=0.5)
    return ax
end

# Value range padded by 1/8 of its magnitude on each side.
function padded_range(value)
    lo, hi = float.(extrema(value))
    lo, hi = lo - abs(lo) / 8, hi + abs(hi) / 8
    return lo == hi ? (lo - 1, hi + 1) : (lo, hi)
end

# Width in pixels of each label, written with the theme font `font` (e.g. `:bold`).
function label_widths(texts, font, fontsize)
    f = Makie.to_font(theme(:fonts), font)
    return [Makie.widths(Makie.text_bb(s, f, fontsize))[1] for s in texts]
end

# Name centered above its station at `x` pixels from the west border of a map `W` pixels wide,
# or starting (ending) at the station when a name `w` pixels wide would cross the west (east) border.
name_align(x, w, W) = (x - w / 2 < 0 ? :left : x + w / 2 > W ? :right : :center, :bottom)

# Value right of its station, or left of it when it would cross the east border.
value_align(x, w, W, gap) = (x + gap + w > W ? :right : :left, :center)

# Black labels over white halos, readable on top of coastlines, borders and rivers.
# Every halo is drawn before any text, so that no halo hides another label.
function halo_text!(ax, positions, labels, fontsize)
    for l in labels
        text!(ax, positions; l..., fontsize, color=:white, strokecolor=:white, strokewidth=3)
    end
    for l in labels
        text!(ax, positions; l..., fontsize, color=:black)
    end
end

function map_with_stations(LON_idx, LAT_idx, value=ones(length(LON_idx)); station_name=nothing, show_value=false,
    colorbar_show=false, colorbar_label="", colormap=:plasma, colorrange=padded_range(value), markersize=15,
    labelsize=theme(:fontsize)[], precision_scale=50, LON_min=-5, LON_max=10.5, LAT_min=41, LAT_max=52)
    length(LON_idx) == length(LAT_idx) == length(value) || throw(DimensionMismatch(
        "LON, LAT and value must have the same length, got $(length(LON_idx)), $(length(LAT_idx)) and $(length(value))"))
    features = map_features(precision_scale)
    fig = Figure()
    ax = map_axis(fig[1, 1], features; LON_min, LON_max, LAT_min, LAT_max)
    sc = scatter!(ax, LON_idx, LAT_idx; color=value, markersize, colormap, colorrange)
    colorbar_show && Colorbar(fig[1, 2], sc; label=colorbar_label)
    colsize!(fig.layout, 1, Aspect(1, 1.0)) # no gap between map and colorbar
    resize_to_layout!(fig)
    # Labels are aligned once the layout is final, from the map width in pixels.
    W = Makie.widths(Makie.viewport(ax.scene)[])[1]
    x = (LON_idx .- LON_min) ./ (LON_max - LON_min) .* W # Mercator x is linear in longitude
    gap = markersize / 2 + 2 # pixels between the marker and its labels
    labels = NamedTuple[]
    if station_name !== nothing
        text = string.(station_name)
        align = name_align.(x, label_widths(text, :bold, labelsize), W)
        push!(labels, (; text, align, offset=(0, gap), font=:bold))
    end
    if show_value
        text = string.(value)
        align = value_align.(x, label_widths(text, :regular, labelsize), W, gap)
        offset = [(h === :right ? -gap : gap, 0) for (h, _) in align]
        push!(labels, (; text, align, offset, font=:regular))
    end
    halo_text!(ax, Point2.(LON_idx, LAT_idx), labels, labelsize)
    return fig
end

function map_with_stations(LON_idx, LAT_idx, value::AbstractArray{V}; colorbar_show=false, colorbar_label="",
    colormap=Reverse(:plasma), colorrange=(0, 1), markersize=11, titles=[L"Z = %$k" for k in eachindex(value)],
    precision_scale=50, LON_min=-5, LON_max=10.5, LAT_min=41, LAT_max=52) where {V<:AbstractArray}
    all(length(LON_idx) == length(LAT_idx) == length(v) for v in value) || throw(DimensionMismatch(
        "LON, LAT and every value vector must have the same length"))
    K = length(value)
    features = map_features(precision_scale)
    fig = Figure(size=(800, 250))
    local sc
    for (k, v) in enumerate(value)
        ax = map_axis(fig[1, k], features; LON_min, LON_max, LAT_min, LAT_max, title=titles[k])
        sc = scatter!(ax, LON_idx, LAT_idx; color=v, markersize, colormap, colorrange)
        k > 1 && hideydecorations!(ax; ticks=false)
    end
    if colorbar_show
        Colorbar(fig[1, K+1], sc; height=Relative(1), label=colorbar_label)
        colgap!(fig.layout, -7)
    end
    return fig
end

end # module
