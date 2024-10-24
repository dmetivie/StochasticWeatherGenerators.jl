# * Data station extraction and treatment * #

# maybe DataFramesMeta would be appropriate
""" 
    select_in_range_df(datas, start_Date, interval_Date, [portion]) 
Select station with some data availability in dates and quality (portion of valid data). 
Input is a `vector` (array) of `DataFrame` (one for each station for example) or a `Dict` of `DataFrame`.
If `0 < portion ≤ 1` is specified, it will authorize some portion of data to be missing.
"""
select_in_range_df(data, start_Date, interval_Date) = findall(df -> df[df.Q_RR.==0, :].DATE[1] ≤ start_Date && df[df.Q_RR.==0, :].DATE[end] ≥ start_Date + interval_Date, data)

select_in_range_df(data, start_Date, interval_Date, portion) = sort(findall(df -> df[df.Q_RR.==0, :].DATE[1] ≤ start_Date && df[df.Q_RR.==0, :].DATE[end] ≥ start_Date + interval_Date && size(df[.&(df.Q_RR .== 0, start_Date .≤ df.DATE .≤ start_Date + interval_Date), :])[1] ≥ portion * Day(start_Date + interval_Date - start_Date).value, data))

#TODO: remove ECA_blend_RR in path, people can have other folder name

"""
    collect_data_ECA(STAID::Integer, path::String, var::String="RR"; skipto=19, header = 18) 
`path` gives the path where all data files are stored in a vector
"""
function collect_data_ECA(STAID::Integer, path::String, var::String="RR"; skipto=19, header = 18, url = false) 
    file = url ? Base.download(string(path, @sprintf("STAID%06.d.txt", STAID))) : joinpath(path, string("ECA_blend_$(lowercase(var))/$(uppercase(var))_", @sprintf("STAID%06.d.txt", STAID)))
    if isfile(file)
        return CSV.read(file, DataFrame, skipto=skipto, header = header, comment="#", normalizenames=true, dateformat="yyyymmdd", types=Dict(:DATE => Date))
    else
       return @warn "STAID $STAID File does not exists $(file)"
    end
end

"""
    collect_data_ECA(STAID, date_start::Date, date_end::Date, path::String, var::String="RR"; portion_valid_data=1, skipto=19, header = 18, return_nothing = true)
- `path` gives the path where all data files are stored in a vector
- Filter the `DataFrame` s.t. `date_start ≤ :DATE ≤ date_end`
- var = "RR", "TX" etc.
- `portion_valid_data` is the portion of valid data we are ok with. If we don't want any missing, fix it to `1`.
- `skipto` and `header` for `csv` files with meta informations/comments at the beginning of files. See `CSV.jl`.
- `return_nothing` if `true` it will return `nothing` is the file does not exists or does not have enough valid data.
"""
function collect_data_ECA(STAID, date_start::Date, date_end::Date, path::String, var::String="RR"; portion_valid_data=1, skipto=19, header = 18, return_nothing = true, url = false)
    @assert 0 ≤ portion_valid_data ≤ 1
    data = collect_data_ECA(STAID, path, var; skipto=skipto, header = header, url = url)
    total_time = length(date_start:Day(1):date_end)
    if data isa DataFrame
        @subset!(data, date_start .≤ :DATE .≤ date_end)
        enougrow = nrow(data) ≥ total_time * portion_valid_data
        NONvalidrow = count(!iszero, data[:, Symbol(string("Q_",var))])
        enougrow ? nothing : @warn "STAID $(data.STAID[1]) nrow = $(nrow(data)) < total_time = $(total_time * portion_valid_data)."
        NONvalidrow == 0 ? nothing : @warn "STAID $(data.STAID[1]) There are $NONvalidrow missing rows."
        if return_nothing
            if NONvalidrow > 0 || !(enougrow)
                return nothing
            else
                return data
            end
        else
            return data
        end
    else
        return data
    end
end

function collect_data_ECA!(data, path::String, var; validity = false)
    STAID = [data[j][!, :STAID][1] for j in eachindex(data)]
    col = validity ? [:DATE, Symbol(var), Symbol(string("Q_", uppercase(var)))] : [:DATE, Symbol(var)]
    for (j, i) in enumerate(STAID)
        file_name = joinpath(path, string("ECA_blend_$(lowercase(var))/$(uppercase(var))_", @sprintf("STAID%06.d.txt", i)))
        if !isfile(file_name)
            @warn "$(file_name) is not a valid file or doesn't exist"
            continue
        end
        new_var = CSV.read(joinpath(path, string("ECA_blend_$(lowercase(var))/$(uppercase(var))_", @sprintf("STAID%06.d.txt", i))),
            DataFrame, comment="#", normalizenames=true, dateformat="yyyymmdd", types=Dict(:DATE => Date))
        leftjoin!(data[j], new_var[:, col], on=:DATE)
        # insertcols!(data[j], Symbol(var) => new_var[data[j].DATE[1].<=new_var.DATE.<=data[j].DATE[end], Symbol(var)])
    end
end

"""
    shortname(name::String)
Experimental function that returns only the most relevant part of a station name.
```julia
long_name = "TOULOUSE-BLAGNAC"
shortname(long_name) # "TOULOUSE"
```
"""
function shortname(name::AbstractString)
    n = uppercase(name)
    if startswith(n, "LA ") || startswith(n, "LE ") || startswith(n, "LES ") || startswith(n, "ST ") || startswith(n, "BELLE ")
        m = join(split(name)[1:2], " ")
    elseif startswith(n, "PTE DE LA ")
        m = join(split(name)[3:4], " ")
    elseif startswith(n, "PTE DE ")
        m = split(name)[3]
    elseif startswith(n, "ST-") || startswith(n, "MONT-")
        return rstrip(n)
    else
        m = split(n)[1]
    end
    return split(m, "-")[1]
end

"""
    collect_data_INRAE(station_path::String; show_warning=false, impute_missing=[])
    collect_data_INRAE(df_full::DataFrame; show_warning=false, impute_missing=[])
Read from a file an INRAE formatted weather station data and transform it to match ECA standard naming conventions.
Possibility to impute missing with `Impute.Interpolate` for specified columns e.g. `impute_missing=[:TX]`.
"""
function collect_data_INRAE(station_path::String; show_warning=false, impute_missing=[])
    df_full = CSV.read(station_path, DataFrame, header=13)
    collect_data_INRAE(df_full; show_warning=show_warning, impute_missing=impute_missing)
end

function collect_data_INRAE(df_full::DataFrame; show_warning=false, impute_missing=[])
    df = @chain df_full begin
        @transform(:STAID = :NUM_POSTE)
        @transform(:DATE = Date.(:AN, :MOIS, :JOUR))
        @transform(:CO2 = fill(330.0, length(:AN)))
        @transform(:QQ = :RG / 100)
        DataFramesMeta.@select :STAID :DATE :TN :TX :QQ :ETPP :RR :CO2
    end
    if show_warning != false
        show_warning = show_warning == true ? names(df)[3:ncol(df)] : show_warning
        for c in show_warning
            tof = findall(ismissing, df[:, c])
            if length(tof) > 0
                @warn "Missing col $c rows $tof"
            end
        end
    end
    if length(impute_missing) > 0
        for c in impute_missing
            impute!(df[!, c], Interpolate())
            # NOTE:  Interpolate does not interpolate first values so I remove row by row (I could use `dropmissing!` but there are plenty of missing in :QQ, etc and I want to keep them)
            while df[1, c] isa Missing
                deleteat!(df, 1)
            end
            disallowmissing!(df, c)
        end
    end
    return df
end

