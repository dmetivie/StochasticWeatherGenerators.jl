# * Data station extraction and treatment * #

# maybe DataFramesMeta would be appropriate
""" 
    select_in_range_df(datas, start_Date, interval_Date, [portion]) 
Select station with some data availability in dates and quality (portion of valid data). 
Input is a `vector` (array) of `DataFrame` (one for each station for example) or a `Dict` of `DataFrame`.
If `0 < portion ≤ 1` is specified, it will authorize some portion of data to be missing.
"""
select_in_range_df(data, start_Date, interval_Date) = findall(a -> a[a.Q_RR.==0, :].DATE[1] ≤ start_Date && a[a.Q_RR.==0, :].DATE[end] ≥ start_Date + interval_Date, data)

select_in_range_df(data, start_Date, interval_Date, portion) = sort(findall(a -> a[a.Q_RR.==0, :].DATE[1] ≤ start_Date && a[a.Q_RR.==0, :].DATE[end] ≥ start_Date + interval_Date && size(a[.&(a.Q_RR .== 0, start_Date .≤ a.DATE .≤ start_Date + interval_Date), :])[1] ≥ portion * Day(start_Date + interval_Date - start_Date).value, data))

"""
    collect_data_ECA(STAID::AbstractArray{<:Integer}, path::String, var::String="RR")
`path` gives the path where all data files are stored in a vector
"""
function collect_data_ECA(STAID::AbstractArray{<:Integer}, path::String, var::String="RR")
    return [CSV.read(joinpath(path, string("ECA_blend_$(lowercase(var))/$(uppercase(var))_", @sprintf("STAID%06.d.txt", i))),
        DataFrame, comment="#", normalizenames=true, dateformat="yyyymmdd", types=Dict(:DATE => Date))
            for i in STAID]
end

function collect_data_ECA(STAID::AbstractArray{<:Integer}, date_start::Date, date_end::Date, path::String, var::String="RR"; portion_valid_data::Real=1)
    data = collect_data_ECA(STAID::AbstractArray{<:Integer}, path::String, var::String)
    total_time = length(date_start:Day(1):date_end)
    for i in eachindex(data)
        @subset!(data[i], date_start .≤ :DATE .≤ date_end)
        @assert nrow(data[i]) ≥ total_time * portion_valid_data
    end

    return data
end

function collect_data_ECA!(data, path::String, var)
    STAID = [data[j][!, :STAID][1] for j in eachindex(data)]
    for (j, i) in enumerate(STAID)
        new_var = CSV.read(joinpath(path, string("ECA_blend_$(lowercase(var))/$(uppercase(var))_", @sprintf("STAID%06.d.txt", i))),
            DataFrame, comment="#", normalizenames=true, dateformat="yyyymmdd", types=Dict(:DATE => Date))

        insertcols!(data[j], Symbol(var) => new_var[data[j].DATE[1].<=new_var.DATE.<=data[j].DATE[end], Symbol(var)])
    end
end

function shortname(name::String)
    n = uppercase(name)
    if startswith(n, "LA ") || startswith(n, "LE ") || startswith(n, "LES ") || startswith(n, "ST ") || startswith(n, "BELLE ")
        m = join(split(name)[1:2], " ")
    elseif startswith(n, "PTE DE LA ")
        m = join(split(name)[3:4], " ")
    elseif startswith(n, "PTE DE ")
        m = split(name)[3]
    else
        m = split(n)[1]
    end
    return split(m, "-")[1]
end
