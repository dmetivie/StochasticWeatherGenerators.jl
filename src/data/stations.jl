# * Station Coordinates * #
"""
    dms_to_dd(l)
Convert Degrees Minutes Seconds to Decimal Degrees. Inputs are strings of the form
* LAT    : Latitude in degrees:minutes:seconds (+: North, -: South)
* LON    : Longitude in degrees:minutes:seconds (+: East, -: West)
"""
function dms_to_dd(l)
    deg, minutes, seconds = parse.(Float64, split(l, ":"))
    (abs(deg) + minutes / 60 + seconds / (60 * 60)) * sign(deg)
end

#! Type piracy
Geodesy.LLA(x::String, y::String, z) = LLA(dms_to_dd(x), dms_to_dd(y), z)

"""
    distance_x_to_y
Distance in km between two stations.
"""
function distance_x_to_y(station_x, station_y)
    coord_station_x = LLA(station_x.LAT, station_y.LON, 0.0)
    coord_station_y = LLA(station_y.LAT, station_y.LON, 0)
    return Geodesy.distance(coord_station_x, coord_station_y) / 1000 # distance in km
end

function distance_x_to_all_stations(central_row, station)
    coord_central = LLA(central_row.LAT, central_row.LON, 0.0)
    coord_stations = [LLA(station[i, :].LAT, station[i, :].LON, 0) for i in 1:nrow(station)]
    return [Geodesy.distance(coord_central, pos) for pos in coord_stations] / 1000 # distance in km
end

# * Plots * #

## * Plot map using PyPlot * ##
#TODO: remove function because cause to much dependency 
# """
#     cartopy_map_with_stations(LON_idx, LAT_idx; station_name = nothing, value = nothing, fontsize=15, 
#     precision_scale="50m", title_bar=nothing,
#     LON_min=-5, # West
#     LON_max=10, # Est
#     LAT_min=41, # South
#     LAT_max=52 # North
#     )
# Plot a [cartopy](https://scitools.org.uk/cartopy/docs/latest/) map (by default France) with point at the given Longitude and Latitude in  Decimal Degrees.
# One can add station names and a value display to it.
# """
# function cartopy_map_with_stations(LON_idx, LAT_idx; station_name=nothing, value=nothing, fontsize=15,
#     precision_scale="50m", title_bar=nothing,
#     LON_min=-5, # West
#     LON_max=10, # Est
#     LAT_min=41, # South
#     LAT_max=52 # North
# )
#     N = length(LON_idx)

#     fig, ax = PyPlot.subplots(figsize=(7, 7), subplot_kw=Dict("projection" => ccrs.PlateCarree()))
#     ax.set_extent([LON_min, LON_max, LAT_min, LAT_max]) # LON_min, LON_max, LAT_min, LAT_max
#     ax.add_feature(cfeature.COASTLINE.with_scale(precision_scale))
#     ax.add_feature(cfeature.OCEAN.with_scale(precision_scale))
#     ax.add_feature(cfeature.RIVERS.with_scale("10m"))
#     ax.add_feature(cfeature.BORDERS.with_scale(precision_scale), linestyle="--")

#     # Value to display next to station 
#     lat_shift = fill(-0.15, N)

#     lon_shift = fill(0.02, N)

#     if value !== nothing
#         sc = ax.scatter(LON_idx, LAT_idx, c=value, s=80, cmap=plt.cm.plasma, vmin=0)
#         [ax.annotate("$(value[j])", (LON_idx[j] - lon_shift[j], LAT_idx[j] - lat_shift[j]), fontsize=fontsize, weight="bold") for j = 1:size(station_name, 1)]
#         clb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
#     else
#         sc = ax.scatter(LON_idx, LAT_idx, s=80, cmap=plt.cm.plasma, vmin=0)
#     end

#     if title_bar !== nothing
#         clb.ax.set_title(title_bar)
#     end

#     if station_name !== nothing
#         [ax.annotate(station_name[j], (LON_idx[j] - 30lon_shift[j], LAT_idx[j] + 2.5lat_shift[j]), fontsize=fontsize / 1.3, weight="bold") for j in 1:N]
#     end

#     ax.set_aspect("auto")
#     PyPlot.tight_layout(h_pad=1)
#     return fig
# end
