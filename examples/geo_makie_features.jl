# Packages

using HTTP, JSON3
import GeoMakie, CairoMakie
import GeoMakie.GeometryBasics

# Functions from [NaturalEarth.jl](https://github.com/JuliaGeo/NaturalEarth.jl)
# https://github.com/JuliaGeo/NaturalEarth.jl/blob/master/gen/geojson_files.jl
geojson_files = Base.download("https://raw.githubusercontent.com/JuliaGeo/NaturalEarth.jl/master/gen/geojson_files.jl")
include(geojson_files)

metadata = get_naturalearth_geojson_metadata()

dataset_info(name) = metadata[findfirst(x -> x.name == name, metadata)]

function fetch_dataset(name)
    return get!(GeoMakie.LOAD_CACHE, name) do  # comment out this if you don't want to save output to cache
        ds = dataset_info(name)
        path = HTTP.download(ds.url, joinpath(mktempdir(), "$(ds.name).geojson"))
        geometry = GeoMakie.GeoJSON.read(read(path))
        # Vector of LineString (and possibly MultiLineString)
        basic = GeoMakie.geo2basic(geometry)
        eltype(basic) <: Union{GeometryBasics.MultiPolygon,GeometryBasics.Polygon} && return basic

        # This code fixes an issue for certain coastlines that contain
        # both LineString and MultiLineString elements.
        # These are converted to a Vector{LineString} for plotting.
        lstrings = GeometryBasics.LineString[]
        for lstring in basic
            if lstring isa GeometryBasics.LineString
                push!(lstrings, lstring)
            elseif lstring isa GeometryBasics.MultiLineString
                append!(lstrings, lstring)
            end
        end
        return lstrings
    end
end

# Adding Features

abstract type AbstractFeatures end

struct LineFeature <: AbstractFeatures
    name::String
end

struct PolyFeature <: AbstractFeatures
    name::String
end

function add_feature!(ax, feature::LineFeature; kwargs...)
    lstrings = fetch_dataset(feature.name)
    GeoMakie.lines!(ax, lstrings; kwargs...)
end

function add_feature!(ax, feature::PolyFeature; kwargs...)
    lstrings = fetch_dataset(feature.name)
    GeoMakie.poly!(ax, lstrings; kwargs...)
end

borders(scale="50m") = LineFeature(string("ne_", scale, "_", "admin_0_boundary_lines_land"))
states(scale="50m") = LineFeature(string("ne_", scale, "_", "admin_1_states_provinces"))
coastline(scale="50m") = LineFeature(string("ne_", scale, "_", "coastline"))
rivers(scale="50m") = LineFeature(string("ne_", scale, "_", "rivers_lake_centerlines_scale_rank"))
"""
    land(scale = "50m")
Land polygons including major islands
Available scales: `"10m"`, `"50m"`, `"110m"`.
"""
land(scale="50m") = LineFeature(string("ne_", scale, "_", "land"))
"""
    ocean(scale = "50m")
Ocean polygon split into contiguous pieces.
Available scales: `"10m"`, `"50m"`, `"110m"`.
"""
ocean(scale="50m") = PolyFeature(string("ne_", scale, "_", "ocean"))

"""
    map_with_stations(LON_idx, LAT_idx; station_name = nothing, value=nothing, show_value=false, fontsize=15,
    precision_scale="50m", colorbar_title=nothing, colorbar_show=false
    LON_min=-5, # West
    LON_max=10, # Est
    LAT_min=41, # South
    LAT_max=52 # North
    )
Plot a [cartopy](https://scitools.org.uk/cartopy/docs/latest/) map (by default France) with point at the given Longitude and Latitude in  Decimal Degrees.
One can add station names and a value display to it.
"""
function map_with_stations(LON_idx, LAT_idx, value=:coral; station_name=nothing, show_value=false, fontsize=15,
    precision_scale="50m", colorbar_title=:none, colorbar_show=false, colorbar_limits=:none,
    LON_min=-5, # West
    LON_max=10.5, # Est
    LAT_min=41, # South
    LAT_max=52 # North
)
    fig = GeoMakie.with_theme(GeoMakie.theme_latexfonts(), fontsize=fontsize) do
        fig = GeoMakie.Figure()
        ax = GeoMakie.GeoAxis(fig[1, 1], dest="+proj=merc", xgridvisible=false, ygridvisible=false, xticklabelsvisible=true, yticklabelsvisible=true, xticksvisible=false, yticksvisible=false)
        GeoMakie.xlims!(ax, LON_min, LON_max)
        GeoMakie.ylims!(ax, LAT_min, LAT_max)
        add_feature!(ax, ocean(precision_scale))
        add_feature!(ax, rivers("10m"), linewidth=0.5)
        add_feature!(ax, coastline(precision_scale), color=:black, linewidth=0.75)
        add_feature!(ax, borders(precision_scale), linestyle=:dash, color=:black, linewidth=0.75)

        sc = GeoMakie.scatter!(ax, LON_idx, LAT_idx; color=value, markersize=15, colormap=:plasma, colorrange=(30, 80))
        if show_value == true
            GeoMakie.text!(ax, LON_idx, LAT_idx; text=string.(value), color=:black, font=:bold, offset=(8, 10), align=(:left, :top), fontsize=18)
            # [annotate!(ax, "$(value[j])", (LON_idx[j] - lon_shift[j], LAT_idx[j] - lat_shift[j]), fontsize=fontsize, weight="bold") for j = 1:size(station_name, 1)]
        end
        if colorbar_show == true
            GeoMakie.Colorbar(fig[1, 2], sc)
        end

        if station_name !== nothing
            GeoMakie.text!(ax, LON_idx, LAT_idx; text=station_name, color=:black, font=:bold, offset=(-7, 4), align=(:center, :bottom))
            # [ax.annotate(station_name[j], (LON_idx[j] - 30lon_shift[j], LAT_idx[j] + 2.5lat_shift[j]), fontsize=fontsize / 1.3, weight="bold") for j in 1:N]
        end

        GeoMakie.colsize!(fig.layout, 1, GeoMakie.Aspect(1, 1.0)) # remove white gap in between colorbar/map
        GeoMakie.resize_to_layout!(fig) # remove white gap around figure
        fig
    end
    return fig
end

function map_with_stations(LON_idx, LAT_idx, value::AbstractArray{V}; station_name=nothing, show_value=false, fontsize=15,
    precision_scale="50m", colorbar_title=:none, colorbar_show=false, colorbar_limits=:none,
    LON_min=-5, # West
    LON_max=10.5, # Est
    LAT_min=41, # South
    LAT_max=52 # North
) where {V<:AbstractArray}
    K = length(value)
    fig = GeoMakie.with_theme(GeoMakie.theme_latexfonts(), fontsize=fontsize) do
        fig = GeoMakie.Figure()
        ax = [GeoMakie.GeoAxis(fig[1, k], dest="+proj=merc", xgridvisible=false, ygridvisible=false, xticklabelsvisible=true, yticklabelsvisible=true, xticksvisible=false, yticksvisible=false) for k in 1:K]
        for k in 1:K
            GeoMakie.xlims!(ax[k], LON_min, LON_max)
            GeoMakie.ylims!(ax[k], LAT_min, LAT_max)
            add_feature!(ax[k], ocean(precision_scale))
            add_feature!(ax[k], rivers("10m"), linewidth=0.5)
            add_feature!(ax[k], coastline(precision_scale), color=:black, linewidth=0.75)
            add_feature!(ax[k], borders(precision_scale), linestyle=:dash, color=:black, linewidth=0.75)

            sc = GeoMakie.scatter!(ax[k], LON_idx, LAT_idx; color=value[k], markersize=15, colormap=:plasma, colorrange=(0, 1))
            if show_value == true
                GeoMakie.text!(ax[k], LON_idx, LAT_idx; text=string.(value), color=:black, font=:bold, offset=(8, 10), align=(:left, :top), fontsize=18)
                # [annotate!(ax, "$(value[j])", (LON_idx[j] - lon_shift[j], LAT_idx[j] - lat_shift[j]), fontsize=fontsize, weight="bold") for j = 1:size(station_name, 1)]

                if station_name !== nothing
                    GeoMakie.text!(ax, LON_idx, LAT_idx; text=station_name, color=:black, font=:bold, offset=(-7, 4), align=(:center, :bottom))
                    # [ax.annotate(station_name[j], (LON_idx[j] - 30lon_shift[j], LAT_idx[j] + 2.5lat_shift[j]), fontsize=fontsize / 1.3, weight="bold") for j in 1:N]
                end
            end
            if colorbar_show == true && k == K
                cb = GeoMakie.Colorbar(fig[1, K+1], sc)
                cb.alignmode = GeoMakie.Mixed(right=0)
            end
        end

        # GeoMakie.colsize!(fig.layout, 1, GeoMakie.Aspect(1, 1.0)) # remove white gap in between colorbar/map
        # GeoMakie.resize_to_layout!(fig) # remove white gap around figure
        fig
    end
    return fig
end
# function map_with_stations(LON_idx, LAT_idx, K::Integer; station_name=nothing, value=nothing, show_value=false, fontsize=15,
#     precision_scale="50m", colorbar_title=nothing, colorbar_show=false, vmin=0, vmax=nothing,
#     LON_min=-5, # West
#     LON_max=10, # Est
#     LAT_min=41, # South
#     LAT_max=52 # North
# )
#     N = length(LON_idx)
#     if K == 1
#         return map_with_stations(LON_idx, LAT_idx; station_name=station_name, value=value[1], show_value=show_value, fontsize=fontsize,
#             precision_scale=precision_scale, colorbar_title=colorbar_title, colorbar_show=colorbar_show, LON_min=LON_min, LON_max=LON_max, LAT_min=LAT_min, LAT_max=LAT_max)
#     else
#         fig, ax = PyPlot.subplots(ncols=K, figsize=(20 * 4 / 5, 12.5 * 4 / 5 / 2), subplot_kw=Dict("projection" => ccrs.PlateCarree()))
#         for k in 1:K
#             ax[k].set_extent([LON_min, LON_max, LAT_min, LAT_max]) # LON_min, LON_max, LAT_min, LAT_max
#             ax[k].add_feature(cfeature.COASTLINE.with_scale(precision_scale))
#             ax[k].add_feature(cfeature.OCEAN.with_scale(precision_scale))
#             ax[k].add_feature(cfeature.RIVERS.with_scale("10m"))
#             ax[k].add_feature(cfeature.BORDERS.with_scale(precision_scale), linestyle="--")
#         end

#         # Value to display next to station 
#         lat_shift = fill(-0.15, N)
#         lon_shift = fill(0.02, N)

#         colo_bar = plt.cm.plasma.reversed()
#         sc = [ax[k].scatter(LON_idx, LAT_idx, c=value[k], s=80, cmap=colo_bar, vmin=vmin, vmax=vmax) for k in 1:K]
#         if show_value == true
#             [ax.annotate("$(value[j])", (LON_idx[j] - lon_shift[j], LAT_idx[j] - lat_shift[j]), fontsize=fontsize, weight="bold") for j = 1:size(station_name, 1)]
#         end
#         PyPlot.subplots_adjust(wspace=0.08) # Space between subplot
#         fig.subplots_adjust(right=0.85)

#         if colorbar_show == true
#             cbar_ax = fig.add_axes([0.86, 0.14, 0.02, 0.74])
#             clb = fig.colorbar(sc[1], cax=cbar_ax)
#             if colorbar_title !== nothing
#                 clb.ax.set_title(colorbar_title)
#             end
#         end

#         if station_name !== nothing
#             [ax.annotate(station_name[j], (LON_idx[j] - 30lon_shift[j], LAT_idx[j] + 2.5lat_shift[j]), fontsize=fontsize / 1.3, weight="bold") for j in 1:N]
#         end
#         [ax[k].set_aspect("auto") for k in 1:K]
#         [ax[k].set_title(L"Z = %$(k)", fontsize=16) for k in 1:K]

#         return fig
#     end
# end

