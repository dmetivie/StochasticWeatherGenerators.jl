# Counter-based RNG streams: one Xoshiro per (seed, member, block id), derived with SplitMix64.

"""
    RNG_DERIVATION

Version of the stream derivation used by [`stream`](@ref); it is stamped into fits, simulations and saved files.

Experimental: may change in minor releases.
"""
const RNG_DERIVATION = 1

const GOLDEN = 0x9e3779b97f4a7c15
const TAG_SIM = 0x5357472d73696d75   # "SWG-simu"
const TAG_FIT = 0x5357472d66697421   # "SWG-fit!"
const TAG_INT = 0x5357472d696e7469   # "SWG-inti"
const TAG_STR = 0x5357472d73747269   # "SWG-stri"

function splitmix64(x::UInt64)
    z = x + GOLDEN
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

function fnv1a64(bytes)
    h = 0xcbf29ce484222325
    for b in bytes
        h = (h ⊻ b) * 0x00000100000001b3
    end
    return h
end

seedword(::Bool) = throw(ArgumentError("a seed must be an integer, not a Bool"))
seedword(seed::Integer) = seed % UInt64
seedword(seed) = throw(ArgumentError("a seed must be an integer, got $(repr(seed))"))

blockid(::Bool) = throw(ArgumentError("a block id must be an integer, a Symbol or a String, not a Bool"))
blockid(b::Integer) = splitmix64((b % UInt64) ⊻ TAG_INT)
blockid(b::Union{Symbol,AbstractString}) = fnv1a64(codeunits(String(b))) ⊻ TAG_STR

function _xoshiro(key::UInt64)
    w0 = splitmix64(key)
    w1 = splitmix64(key + GOLDEN)
    w2 = splitmix64(key + 2GOLDEN)
    w3 = splitmix64(key + 3GOLDEN)
    return Random.Xoshiro(w0, w1, w2, w3)
end

_streamkey(seed, tag::UInt64, m::UInt64, block) =
    splitmix64(splitmix64(splitmix64(seedword(seed) ⊻ tag) ⊻ m) ⊻ blockid(block))

"""
    stream(seed, member::Integer, block = 0)
    stream(seed, :fit, block = 0)

Return the `Xoshiro` stream of `(seed, member, block)` for simulation, or of `(seed, block)` for fitting.

`block` is an integer, a `Symbol` or a `String`; a `Symbol` and the equal `String` give the same stream.
The streams are the same on every machine, thread count and process (see [`RNG_DERIVATION`](@ref)).

Experimental: may change in minor releases.
"""
stream(seed::Integer, member::Integer, block = 0) =
    _xoshiro(_streamkey(seed, TAG_SIM, seedword(member), block))
function stream(seed::Integer, s::Symbol, block = 0)
    s === :fit || throw(ArgumentError("`stream(seed, s, block)` takes a member number or `:fit`, got :$s"))
    return _xoshiro(_streamkey(seed, TAG_FIT, zero(UInt64), block))
end
