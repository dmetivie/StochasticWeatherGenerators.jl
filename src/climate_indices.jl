"""
    longuest_spell(y::AbstractArray; value=0)
Compute the length of the longuest consecutive sequence of `value` in `y`
"""
function longuest_spell(y::AbstractArray; value=0)
    maxx = zero(eltype(y))
    for (val, len) in eachrow(hcat(rle(y)...))
        if len > maxx && val == value # 0 dry || 1 rain
            maxx = len
        end
    end
    return maxx
end

"""
    pmf_spell(y::AbstractVector, value)
Return the distribution of spells (consecutive sequence of with the same value) length of `value` in `y`
"""
function pmf_spell(y::AbstractVector, value)
    spells = rle(y)
    idx_spells = findall(spells[1] .== value)
    return spells[2][idx_spells] 
end

#TODO rewrite all index with @chain from DataFramesMeta!
"""
    VCX3(df; y_col, nb = 3)
Yearly Max of `nb = 3` days sliding mean for `y` for every year. 
By default, `y_col` is the first column not with a `Date` type

```julia
using DataFrames, Dates, RollingFunctions
time_range = Date(1956):Day(1):Date(2019,12,31)
df = DataFrame(:DATE => time_range, :Temperature => 20 .+ 5*randn(length(time_range)))
VCX3(df)
```
"""
function VCX3(df::DataFrame; y_col = findfirst(eltype.(eachcol(df)) .!= Date), nb = 3) 
   types = eltype.(eachcol(df))
   if count(types .== Date) > 1
    @warn "More than one `Date` column. The first one is used"
   end
   idx_date_col = findfirst(types .== Date)
   year_range = unique(year.(df[:,idx_date_col]))
   idx_year = [findall(x-> year.(x) == m, df[:, idx_date_col]) for m in year_range]
   return VCX3(df[:,y_col], idx_year; nb = nb)
end

"""
    VCX3(y, idxs; nb = 3)
Yearly Max of `nb = 3` days sliding mean for `y`.
Here `idxs` can be a vector of vector (or range) corresponds to the index of every year.

```julia
using DataFrames, Dates, RollingFunctions
time_range = Date(1956):Day(1):Date(2019,12,31)
year_range = unique(year.(time_range))
df = DataFrame(:DATE => time_range, :Temperature => 20 .+ 5*randn(length(time_range)))
idx_year = [findall(x-> year.(x) == m, df[:, :DATE]) for m in year_range]
VCX3(df.Temperature, idx_year)
```
"""
VCX3(y::AbstractVector, idxs, nb = 3) = [maximum(rollmean(y[idx], nb)) for idx in idxs]
# VC(y::AbstractArray, idxs, nb = 3, outer_op = maximum, inner_op = rollmean(x, nb)) = [outer_op(inner_op(y[idx], nb)) for idx in idxs]

"""
    monthly_agg(y::AbstractArray, idxs)
```julia
using DataFrames, Dates
time_range = Date(1956):Day(1):Date(2019,12,31)
year_range = unique(year.(time_range))
df = DataFrame(:DATE => time_range, :Temperature => 20 .+ 5*randn(length(time_range)))
monthly_agg(df, :Temperature) 
monthly_agg(df, :Temperature, mean) 
# or
idx_year = [findall(x-> year.(x) == m, df[:, :DATE]) for m in year_range]
idx_month = [findall(x-> month.(x) == m, df[:, :DATE]) for m in 1:12]
idx_all = [intersect(yea, mon) for yea in idx_year, mon in idx_month]
monthly_agg(df.Temperature, idx_all)
```
"""
monthly_agg(y::AbstractArray, idxs, aggfunc = sum) = [aggfunc(y[idx]) for idx in idxs]


function monthly_agg(df::DataFrame, y_col, aggfunc = sum; maxtrixform = true) 
    dfs_agg = @chain df begin 
        @transform(:MONTH = month.(:DATE), :YEAR = year.(:DATE))
        @by([:MONTH, :YEAR], :agg = aggfunc($y_col))
        groupby(:MONTH)
    end
    if maxtrixform == true
        return reduce(hcat, [df_agg.agg for df_agg in dfs_agg])
    else 
        return [df_agg.agg for df_agg in dfs_agg]
    end
end