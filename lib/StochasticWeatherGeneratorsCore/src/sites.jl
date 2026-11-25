# Distances between stations.

"""
    Haversine(radius = 6371.0)

Great-circle distance in km on a sphere of `radius` km; it is called as `h(p, q)` on (lon, lat) pairs in degrees.

Experimental: may change in minor releases.
"""
struct Haversine
    radius::Float64
end
Haversine() = Haversine(6371.0)

function (h::Haversine)(p, q)
    λ1, φ1 = deg2rad(p[1]), deg2rad(p[2])
    λ2, φ2 = deg2rad(q[1]), deg2rad(q[2])
    a = sin((φ2 - φ1) / 2)^2 + cos(φ1) * cos(φ2) * sin((λ2 - λ1) / 2)^2
    return 2 * h.radius * asin(min(1.0, sqrt(a)))
end

_euclidean(p, q) = hypot(p[1] - q[1], p[2] - q[2])

_default_metric(crs::Symbol) = crs === :lonlat ? Haversine() : _euclidean

"""
    distances(data; metric = Haversine())

Return the D×D matrix of distances between stations: km with `Haversine()` (R = 6371 km, no rounding),
or the Euclidean distance on (X, Y) with `crs = :projected`.

`metric(p, q)` takes two (lon, lat) pairs, so Distances.jl metrics work too.
"""
function distances(d::WeatherData; metric = _default_metric(d.crs))
    return _distances(coords(d), metric)
end

function _distances(c, metric)
    c === nothing && throw(ArgumentError(
        "no station coordinates: add LAT and LON columns (or X and Y with `crs = :projected`)"))
    n = length(c)
    M = zeros(Float64, n, n)
    for j in 1:n, i in 1:j-1
        M[i, j] = M[j, i] = metric(c[i], c[j])
    end
    return M
end
