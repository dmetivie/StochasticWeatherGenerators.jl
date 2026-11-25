# Calendars map dates to period indices 1:366 and number consecutive model steps.

"""
    AbstractCalendar

Supertype of the calendars that map dates to period indices `1:period(cal)`.

Experimental: may change in minor releases.
"""
abstract type AbstractCalendar end

"""
    Leap366()

Gregorian calendar indexed on 366 days: 1 March is always index 61 and 29 February is index 60.
"""
struct Leap366 <: AbstractCalendar end

"""
    NoLeap365()

365-day calendar on the 366-day index: index 60 (29 February) is never produced and 29 February is an error.
"""
struct NoLeap365 <: AbstractCalendar end

"""
    Day360()

360-day calendar (twelve 30-day months) mapped on 366 indices by `round(Int, (30(m - 1) + d) * 366 / 360)`.
It takes any date-like value with `year`, `month` and `day` methods, e.g. CFTime `DateTime360Day`.
"""
struct Day360 <: AbstractCalendar end

"""
    period(cal::AbstractCalendar)

Return the number of period indices of `cal` (366 for every calendar).

Experimental: may change in minor releases.
"""
period(::AbstractCalendar) = 366

# cumulative day counts before each month in a leap year
const CUMDAYS_LEAP = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)

_ymd(d) = (Dates.year(d), Dates.month(d), Dates.day(d))

function _check_gregorian(d)
    y, m, dd = _ymd(d)
    1 <= m <= 12 || throw(ArgumentError("invalid month $m in date $d"))
    1 <= dd <= Dates.daysinmonth(y, m) || throw(ArgumentError("invalid day $dd in date $d"))
    return y, m, dd
end

"""
    period_index(cal::AbstractCalendar, d)

Return the period index in `1:366` of date `d` under `cal`.

Experimental: may change in minor releases.
"""
function period_index(::Leap366, d)
    _, m, dd = _check_gregorian(d)
    return CUMDAYS_LEAP[m] + dd
end

function period_index(::NoLeap365, d)
    _, m, dd = _check_gregorian(d)
    m == 2 && dd == 29 &&
        throw(ArgumentError("29 February ($d) does not exist under NoLeap365(); drop it or use `calendar = Leap366()`"))
    return CUMDAYS_LEAP[m] + dd
end

function period_index(::Day360, d)
    _, m, dd = _ymd(d)
    _day360_index(_day360_doy(m, dd, d))
end

function _day360_doy(m, dd, d)
    1 <= m <= 12 || throw(ArgumentError("invalid month $m in date $d"))
    1 <= dd <= 30 || throw(ArgumentError("day $dd of $d does not exist under Day360(): months have 30 days"))
    return 30 * (m - 1) + dd
end
_day360_index(x::Integer) = round(Int, (366 * x) // 360)

# Consecutive model steps have consecutive step numbers.
stepnumber(::Leap366, d::Date) = Dates.value(d)
stepnumber(::Leap366, d) = (_check_gregorian(d); Dates.value(Date(_ymd(d)...)))
function stepnumber(cal::NoLeap365, d)
    y = Dates.year(d)
    idx = period_index(cal, d)
    return 365 * y + idx - (idx > 60)
end
function stepnumber(::Day360, d)
    y, m, dd = _ymd(d)
    return 360 * y + _day360_doy(m, dd, d)
end

# period index of a step number
function _index_of_step(::Leap366, s::Integer)
    d = Date(Dates.UTD(s))
    return CUMDAYS_LEAP[Dates.month(d)] + Dates.day(d)
end
function _index_of_step(::NoLeap365, s::Integer)
    doy = s - 365 * fld(s - 1, 365)
    return doy + (doy >= 60)
end
_index_of_step(::Day360, s::Integer) = _day360_index(s - 360 * fld(s - 1, 360))

# date of a step number, when the calendar can build it
_date_of_step(::Leap366, s::Integer, ::Type{Date}) = Date(Dates.UTD(s))
function _date_of_step(::NoLeap365, s::Integer, ::Type{Date})
    y = fld(s - 1, 365)
    idx = _index_of_step(NoLeap365(), s)
    m = findlast(<(idx), CUMDAYS_LEAP)
    return Date(y, m, idx - CUMDAYS_LEAP[m])
end
_date_of_step(cal, s, ::Type) = nothing

"""
    calendar_dates(cal, years)

Return the consecutive dates of `years` under `cal`.
"""
function calendar_dates(cal::AbstractCalendar, years::AbstractUnitRange{<:Integer})
    isempty(years) && throw(ArgumentError("`years` is empty"))
    s1 = stepnumber(cal, Date(first(years), 1, 1))
    s2 = stepnumber(cal, Date(last(years), 12, 31))
    return [_date_of_step(cal, s, Date) for s in s1:s2]
end
calendar_dates(::Day360, years::AbstractUnitRange{<:Integer}) = throw(ArgumentError(
    "Day360() has no built-in date type: pass the dates (e.g. CFTime.DateTime360Day values) instead of years"))

"""
    check_consecutive(cal, dates)

Throw an error unless `dates` are consecutive steps of `cal`.
"""
function check_consecutive(cal::AbstractCalendar, dates::AbstractVector)
    isempty(dates) && throw(ArgumentError("no dates given"))
    prev = stepnumber(cal, first(dates))
    for i in 2:length(dates)
        s = stepnumber(cal, dates[i])
        s == prev + 1 || throw(ArgumentError(
            "dates must be consecutive days under $(cal)(): $(dates[i-1]) is followed by $(dates[i])"))
        prev = s
    end
    return nothing
end

"""
    steps_before(cal, first, b)

Return the period indices of the `b` steps that precede date `first` under `cal`.
"""
function steps_before(cal::AbstractCalendar, first, b::Integer)
    s1 = stepnumber(cal, first)
    return Int[_index_of_step(cal, s) for s in (s1 - b):(s1 - 1)]
end

Base.show(io::IO, cal::AbstractCalendar) = print(io, nameof(typeof(cal)), "()")
