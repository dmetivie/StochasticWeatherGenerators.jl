# * Station Coordinates * #
"""
    dms_to_dd(l)
Convert `l` in Degrees:Minutes:Seconds to Decimal Degrees. Inputs are strings of the form
* LAT    : Latitude in degrees:minutes:seconds (+: North, -: South)
* LON    : Longitude in degrees:minutes:seconds (+: East, -: West)
"""
function dms_to_dd(l)
    deg, minutes, seconds = parse.(Float64, split(l, ":"))
    (abs(deg) + minutes / 60 + seconds / (60 * 60)) * sign(deg)
end

"""
    haversine(lat1, long1, lat2, long2, r = 6372.8)

Compute the haversine distance between two points on a sphere of radius `r`, where the points are given by the latitude/longitude pairs `lat1/long1` and `lat2/long2` (in degrees).
From this [JuMP.jl tutorial](https://jump.dev/JuMP.jl/stable/tutorials/linear/geographic_clustering/#Obtaining-the-distances-between-each-city).
"""
function haversine(lat1, long1, lat2, long2, r=6372.8; round_Int = true)
    lat1, long1 = deg2rad(lat1), deg2rad(long1)
    lat2, long2 = deg2rad(lat2), deg2rad(long2)
    hav(a, b) = sin((b - a) / 2)^2
    inner_term = hav(lat1, lat2) + cos(lat1) * cos(lat2) * hav(long1, long2)
    d = 2 * r * asin(sqrt(inner_term))
    # Round distance to nearest kilometer.
    return round_Int ? round(Int, d) : d
end


"""
    LLA{T<:Real}
Coordinate structure inspired by Geodesy.jl
"""
struct LLA{T<:Real}
    LAT::T
    LON::T
    ALT::T
end

"""
    LLA(x::String, y::String, z)
Constructor converting coordinate strings Degrees:Minutes:Seconds to Decimal Degrees.
"""
LLA(x::String, y::String, z) = LLA(dms_to_dd(x), dms_to_dd(y), z)

haversine(LLA_x::LLA, LLA_y::LLA, r=6372.8; kwargs...) = haversine(LLA_x.LAT, LLA_x.LON, LLA_y.LAT, LLA_y.LON, r; kwargs...)

"""
    distance_x_to_y(station_x, station_y)
Distance in km between two stations. Does not take into account altitude. `station` must have a field `LAT` and `LON` in Decimal Degree.
"""
function distance_x_to_y(station_x, station_y)
    coord_station_x = LLA(station_x.LAT, station_y.LON, 0.0)
    coord_station_y = LLA(station_y.LAT, station_y.LON, 0)
    return haversine(coord_station_x, coord_station_y) # distance in km
end

function distance_x_to_all_stations(central_row, station)
    coord_central = LLA(central_row.LAT, central_row.LON, 0.0)
    coord_stations = [LLA(station[i, :].LAT, station[i, :].LON, 0) for i in 1:nrow(station)]
    return [haversine(coord_central, pos) for pos in coord_stations] # distance in km
end