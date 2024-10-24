# * Station Coordinates * #
"""
    dms_to_dd(l)
Convert `l` in Degrees Minutes Seconds to Decimal Degrees. Inputs are strings of the form
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
    distance_x_to_y(station_x, station_y)
Distance in km between two stations. Does not take into account altitude. `station` must have a field `LAT` and `LON` in Decimal Degree.
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