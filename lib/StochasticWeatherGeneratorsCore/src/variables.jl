# Variable aliases, kinds and site-metadata column names.

const VARIABLE_ALIASES = Dict{String,Symbol}(
    "RR" => :RR, "PRCP" => :RR, "PR" => :RR,
    "TX" => :TX, "TMAX" => :TX, "TASMAX" => :TX,
    "TN" => :TN, "TMIN" => :TN, "TASMIN" => :TN,
    "QQ" => :QQ, "RSDS" => :QQ,
    "ETPP" => :ETPP, "ET0" => :ETPP, "PET" => :ETPP,
    "FG" => :FG,
)

const ALIAS_KINDS = Dict{Symbol,Symbol}(
    :RR => :rain, :TX => :temperature, :TN => :lower_temperature,
    :QQ => :radiation, :ETPP => :evapotranspiration, :FG => :wind,
)

const VALID_KINDS = (:rain, :temperature, :lower_temperature, :radiation, :evapotranspiration, :wind,
    :positive, :continuous, :custom)

const SITE_ALIASES = Dict{String,Symbol}(
    "LAT" => :LAT, "LATITUDE" => :LAT,
    "LON" => :LON, "LONG" => :LON, "LONGITUDE" => :LON, "LNG" => :LON,
    "ALT" => :ALT, "ALTITUDE" => :ALT, "ELEVATION" => :ALT, "ELEV" => :ALT,
    "X" => :X, "Y" => :Y,
)
const SITE_KEYS = Dict(:lonlat => (:LAT, :LON, :ALT), :projected => (:X, :Y, :ALT))

const DATE_NAMES = ("DATE", "TIME")
const STATION_NAMES = ("STATION", "STAID", "SITE", "ID")

_upper(x) = uppercase(String(x))

# canonical name of a variable column name, or `nothing` when it is not an alias
_alias(name) = get(VARIABLE_ALIASES, _upper(name), nothing)

function _checkkind(name, k)
    k isa Symbol && k in VALID_KINDS && return k
    throw(ArgumentError("invalid kind $(repr(k)) for $name; valid kinds are " * join(repr.(VALID_KINDS), ", ")))
end

# index of variable `v` in `vars` (canonical) or `names` (original), alias-aware
function _varindex(vars::AbstractVector{Symbol}, names::AbstractVector{Symbol}, v)
    s = Symbol(v)
    i = findfirst(==(s), vars)
    i === nothing || return i
    i = findfirst(==(s), names)
    i === nothing || return i
    a = _alias(s)
    a === nothing || (i = findfirst(==(a), vars))
    return i
end

function _varindex_or_error(vars, names, v; what = "data")
    i = _varindex(vars, names, v)
    i === nothing && throw(ArgumentError("no variable $(repr(Symbol(v))) in the $what; available: " *
                                         join(string.(names), ", ")))
    return i
end
