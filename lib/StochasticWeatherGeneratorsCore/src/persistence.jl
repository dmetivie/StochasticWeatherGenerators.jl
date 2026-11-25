# JSON persistence: a plain tree with type tags, decoded through a registry. Never `eval`.

const FORMAT = "StochasticWeatherGenerators.FittedGenerator"
const FORMAT_VERSION = 1

struct Registration
    type::Any
    package::String
    version::VersionNumber
end

const REGISTRY = Dict{String,Registration}()
const TAGS = IdDict{Any,String}()
const REGISTRY_LOCK = ReentrantLock()

_wrapper(T::Type) = Base.typename(T).wrapper
function _pkgmodule(T::Type)
    m = parentmodule(T)
    r = Base.moduleroot(m)
    return r === Main ? m : r
end

"""
    register_type(T; package = <T's package>, version = <its version>)

Allow values of struct type `T` to be saved with [`save_generator`](@ref) and loaded back.

Packages call it in their `__init__`. Saved values carry the tag `"package.T"`, the package and its version.

Experimental: may change in minor releases.
"""
function register_type(T::Type; package::AbstractString = string(nameof(_pkgmodule(T))),
        version = something(pkgversion(_pkgmodule(T)), v"0.0.0"))
    W = _wrapper(T)
    tag = string(package, ".", nameof(W))
    lock(REGISTRY_LOCK) do
        REGISTRY[tag] = Registration(W, String(package), VersionNumber(version))
        TAGS[W] = tag
    end
    return tag
end

function _registration(T::Type)
    W = _wrapper(T)
    tag = lock(() -> get(TAGS, W, nothing), REGISTRY_LOCK)
    tag === nothing && throw(ArgumentError(
        "cannot save a $(nameof(W)): its package must call " *
        "`StochasticWeatherGeneratorsCore.register_type($(nameof(W)))` in its `__init__`"))
    return tag, lock(() -> REGISTRY[tag], REGISTRY_LOCK)
end

_obj(ps::Pair...) = JSON.Object{String,Any}(ps...)
_tagged(tag, value) = _obj("_type" => tag, "value" => value)

const PLAIN_INTS = (Int8, Int16, Int32, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
const COMPACT_ELTYPES = (Float64, Float32, Int64, PLAIN_INTS..., Bool, String, Symbol, Date, DateTime)
const ELTYPE_BY_TAG = Dict{String,Type}(string(T) => T for T in
    (COMPACT_ELTYPES..., (Union{Missing,T} for T in COMPACT_ELTYPES)...))
const INT_BY_TAG = Dict{String,Type}(string(T) => T for T in PLAIN_INTS)

_nonfinite(x) = isnan(x) ? "NaN" : x > 0 ? "Inf" : "-Inf"
_floatplain(x::Float64) = isfinite(x) ? x : _nonfinite(x)

"""
    to_plain(x)
    to_plain(x, fields::NamedTuple)

Return the JSON-ready tree of `x`: numbers, strings, `nothing`, and tagged objects for other values.

Registered structs are written field by field; a package customises the fields of its type `T` with
`to_plain(x::T) = to_plain(x, (a = x.a, b = ...))` and reads them back in [`from_plain`](@ref).

Experimental: may change in minor releases.
"""
to_plain(::Nothing) = nothing
to_plain(x::Bool) = x
to_plain(x::Int64) = x
to_plain(x::AbstractString) = String(x)
to_plain(x::Float64) = isfinite(x) ? x : _tagged("Float64", _nonfinite(x))
to_plain(x::Float32) = _tagged("Float32", _floatplain(Float64(x)))
to_plain(x::Integer) = _tagged(string(typeof(x)), string(x))
to_plain(x::Symbol) = _tagged("Symbol", String(x))
to_plain(x::Date) = _tagged("Date", string(x))
to_plain(x::DateTime) = _tagged("DateTime", string(x))
to_plain(x::VersionNumber) = _tagged("VersionNumber", string(x))
to_plain(::Missing) = _obj("_type" => "Missing")
to_plain(x::Tuple) = _obj("_type" => "Tuple", "data" => Any[to_plain(e) for e in x])
to_plain(x::NamedTuple) = _obj("_type" => "NamedTuple", "names" => Any[String(k) for k in keys(x)],
    "data" => Any[to_plain(v) for v in values(x)])
to_plain(x::AbstractDict) = _obj("_type" => "Dict", "keytype" => _eltag(keytype(x)), "valtype" => _eltag(valtype(x)),
    "data" => Any[Any[to_plain(k), to_plain(v)] for (k, v) in x])
to_plain(x::UnitRange{Int}) = _obj("_type" => "UnitRange", "start" => first(x), "stop" => last(x))
to_plain(x::StepRange{Date,Day}) = _obj("_type" => "DateRange", "start" => string(first(x)),
    "step" => Dates.value(step(x)), "stop" => string(last(x)))
to_plain(::Function) = throw(ArgumentError("functions cannot be saved; store data instead"))

_eltag(T) = haskey(ELTYPE_BY_TAG, string(T)) && ELTYPE_BY_TAG[string(T)] == T ? string(T) : T === Any ? "Any" : "*"

_compact(x::Float64) = _floatplain(x)
_compact(x::Float32) = _floatplain(Float64(x))
_compact(x::Union{Int64,Bool,String}) = x
_compact(x::Integer) = string(x)
_compact(x::Union{Symbol,Date,DateTime}) = string(x)
_compact(::Missing) = nothing

function to_plain(A::AbstractArray)
    tag = _eltag(eltype(A))
    data = tag in ("Any", "*") ? Any[to_plain(x) for x in A] : Any[_compact(x) for x in A]
    return _obj("_type" => "Array", "eltype" => tag, "size" => collect(size(A)), "data" => data)
end

function to_plain(x::T) where {T}
    isstructtype(T) || throw(ArgumentError("cannot save a value of type $T"))
    return to_plain(x, NamedTuple{fieldnames(T)}(ntuple(i -> getfield(x, i), fieldcount(T))))
end

function to_plain(x, fields::NamedTuple)
    tag, reg = _registration(typeof(x))
    plain = _obj((String(k) => to_plain(v) for (k, v) in pairs(fields))...)
    return _obj("_type" => tag, "_package" => reg.package, "_version" => string(reg.version), "fields" => plain)
end

_malformed(what) = throw(ArgumentError("malformed generator file: $what"))

"""
    from_plain(tree)
    from_plain(T::Type, fields::NamedTuple, version::VersionNumber)

Return the value encoded by the plain `tree` of [`to_plain`](@ref).

The three-argument method builds a registered struct `T` from its decoded `fields`, saved by version `version` of
its package; the default calls the positional constructor in field order. Packages may extend it.

Experimental: may change in minor releases.
"""
from_plain(x::Union{Nothing,Integer,Float64,String}) = x
from_plain(x::AbstractVector) = _malformed("a bare array")

function from_plain(x::AbstractDict)
    tag = get(x, "_type", nothing)
    tag isa String || _malformed("an object without `_type`")
    if tag == "Float64"
        return _float(x["value"])
    elseif tag == "Float32"
        return Float32(_float(x["value"]))
    elseif haskey(INT_BY_TAG, tag)
        return parse(INT_BY_TAG[tag], x["value"])
    elseif tag == "Symbol"
        return Symbol(x["value"])
    elseif tag == "Date"
        return Date(x["value"])
    elseif tag == "DateTime"
        return DateTime(x["value"])
    elseif tag == "VersionNumber"
        return VersionNumber(x["value"])
    elseif tag == "Missing"
        return missing
    elseif tag == "Tuple"
        return Tuple(from_plain(e) for e in x["data"])
    elseif tag == "NamedTuple"
        return NamedTuple{Tuple(Symbol.(x["names"]))}(Tuple(from_plain(e) for e in x["data"]))
    elseif tag == "Dict"
        return _decode_dict(x)
    elseif tag == "UnitRange"
        return x["start"]:x["stop"]
    elseif tag == "DateRange"
        return Date(x["start"]):Day(x["step"]):Date(x["stop"])
    elseif tag == "Array"
        return _decode_array(x)
    end
    return _decode_struct(tag, x)
end

function _float(v)
    v isa Real && return Float64(v)
    v == "NaN" && return NaN
    v == "Inf" && return Inf
    v == "-Inf" && return -Inf
    _malformed("$(repr(v)) is not a number")
end

_elem(::Type{Float64}, v) = _float(v)
_elem(::Type{Float32}, v) = Float32(_float(v))
_elem(::Type{T}, v) where {T<:Integer} = v isa Integer ? T(v) : parse(T, v)
_elem(::Type{Bool}, v) = v::Bool
_elem(::Type{String}, v) = v::String
_elem(::Type{Symbol}, v) = Symbol(v)
_elem(::Type{Date}, v) = Date(v)
_elem(::Type{DateTime}, v) = DateTime(v)

function _decode_array(x)
    tag = x["eltype"]
    dims = Tuple(Int.(x["size"]))
    data = x["data"]
    length(data) == prod(dims) || _malformed("array data of length $(length(data)) for size $dims")
    if tag == "Any"
        return reshape(Any[from_plain(e) for e in data], dims)
    elseif tag == "*"
        return reshape(identity.(Any[from_plain(e) for e in data]), dims)
    end
    T = get(ELTYPE_BY_TAG, tag, nothing)
    T === nothing && _malformed("unknown array eltype $tag")
    T0 = nonmissingtype(T)
    A = Array{T}(undef, dims)
    for i in eachindex(A, data)
        v = data[i]
        A[i] = (v === nothing && T !== T0) ? missing : _elem(T0, v)
    end
    return A
end

function _decode_dict(x)
    ps = [from_plain(p[1]) => from_plain(p[2]) for p in x["data"]]
    K = get(ELTYPE_BY_TAG, x["keytype"], nothing)
    V = get(ELTYPE_BY_TAG, x["valtype"], nothing)
    (K === nothing || V === nothing) && return isempty(ps) ? Dict{Any,Any}() : Dict(ps)
    return Dict{K,V}(ps)
end

function _decode_struct(tag, x)
    reg = lock(() -> get(REGISTRY, tag, nothing), REGISTRY_LOCK)
    pkg = get(x, "_package", first(split(tag, ".")))
    reg === nothing && throw(ArgumentError("unknown type $tag in the generator file; load package $pkg first (`using $pkg`)"))
    saved = VersionNumber(get(x, "_version", "0.0.0"))
    saved > reg.version && throw(ArgumentError(
        "$tag was saved by $(reg.package) v$saved but v$(reg.version) is loaded; update $(reg.package)"))
    plain = x["fields"]
    fields = NamedTuple{Tuple(Symbol.(collect(keys(plain))))}(Tuple(from_plain(v) for v in values(plain)))
    return from_plain(reg.type, fields, saved)
end

function from_plain(T::Type, fields::NamedTuple, version::VersionNumber)
    names = fieldnames(T)
    absent = [n for n in names if !haskey(fields, n)]
    isempty(absent) || throw(ArgumentError(
        "the saved $(nameof(T)) (v$version) lacks fields $(join(absent, ", ")); update its package or refit"))
    return T((fields[n] for n in names)...)
end

"""
    save_generator(path_or_io, g::FittedGenerator)

Write `g` as JSON: the class, the parameters and the fit record, with type tags and package versions.
"""
function save_generator(io::IO, g::FittedGenerator)
    versions = _obj((k => v for (k, v) in _versions(g.spec))...)
    root = _obj("format" => FORMAT, "format_version" => FORMAT_VERSION, "rng_derivation" => RNG_DERIVATION,
        "versions" => versions, "generator" => to_plain(g))
    JSON.json(io, root)
    return nothing
end
save_generator(path::AbstractString, g::FittedGenerator) = open(io -> save_generator(io, g), path, "w")

"""
    load_generator(path_or_io)

Read a generator written by [`save_generator`](@ref); the packages of its classes must be loaded.
"""
function load_generator(io::IO)
    root = JSON.parse(read(io, String))
    (root isa AbstractDict && get(root, "format", nothing) == FORMAT) ||
        throw(ArgumentError("not a generator file written by `save_generator`"))
    fv = get(root, "format_version", nothing)
    fv isa Integer || _malformed("no format_version")
    fv > FORMAT_VERSION && throw(ArgumentError(
        "the file has format version $fv but this StochasticWeatherGeneratorsCore reads up to $FORMAT_VERSION; " *
        "update StochasticWeatherGeneratorsCore"))
    rd = get(root, "rng_derivation", RNG_DERIVATION)
    rd == RNG_DERIVATION || @warn "The file was saved with RNG derivation $rd; simulations now use derivation " *
                                  "$RNG_DERIVATION and differ from those of the saving version."
    g = from_plain(root["generator"])
    g isa FittedGenerator || _malformed("the generator is a $(typeof(g))")
    return g
end
load_generator(path::AbstractString) = open(load_generator, path)

function _register_core_types()
    for T in (FittedGenerator, FitInfo, Leap366, NoLeap365, Day360, Haversine)
        register_type(T; package = "StochasticWeatherGeneratorsCore", version = pkgversion(@__MODULE__))
    end
    return nothing
end
