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
function collect_data_ECA(STAID::Integer, path::String, var::String="RR"; skipto=19, header=18, url=false)
    file = url ? Base.download(string(path, @sprintf("STAID%06.d.txt", STAID))) : joinpath(path, string("ECA_blend_$(lowercase(var))/$(uppercase(var))_", @sprintf("STAID%06.d.txt", STAID)))
    if isfile(file)
        return CSV.read(file, DataFrame, skipto=skipto, header=header, comment="#", normalizenames=true, dateformat="yyyymmdd", types=Dict(:DATE => Date))
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
function collect_data_ECA(STAID, date_start::Date, date_end::Date, path::String, var::String="RR"; portion_valid_data=1, skipto=19, header=18, return_nothing=true, url=false)
    @assert 0 ≤ portion_valid_data ≤ 1
    data = collect_data_ECA(STAID, path, var; skipto=skipto, header=header, url=url)
    total_time = length(date_start:Day(1):date_end)
    if data isa DataFrame
        @subset!(data, date_start .≤ :DATE .≤ date_end)
        enougrow = nrow(data) ≥ total_time * portion_valid_data
        NONvalidrow = count(!iszero, data[:, Symbol(string("Q_", var))])
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

function collect_data_ECA!(data, path::String, var; validity=false)
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

function isvalid_data_ECA(STAID::Integer, date_start, date_end, path::String, var::String; skipto=19, header=18, url=false, portion_valid_data=1)
    file = url ? Base.download(string(path, @sprintf("STAID%06.d.txt", STAID))) : joinpath(path, string("ECA_blend_$(lowercase(var))/$(uppercase(var))_", @sprintf("STAID%06.d.txt", STAID)))
    if isfile(file)
        data = CSV.read(file, DataFrame, skipto=skipto, header=header, comment="#", normalizenames=true, dateformat="yyyymmdd", types=Dict(:DATE => Date))
        @assert 0 ≤ portion_valid_data ≤ 1
        total_time = length(date_start:Day(1):date_end)
        @subset!(data, date_start .≤ :DATE .≤ date_end)
        if date_start != data[1, :DATE]
            @warn "Starting date of data is $(data[1,:DATE])"
            return false
        elseif date_end != data[end, :DATE]
            @warn "Ending date of data is $(data[end,:DATE])"
            return false
        end
        Q = Symbol("Q_", var)
        valid = count(data[:, Q] .!= 9)
        if valid / nrow(data) < portion_valid_data
            @warn "Too much missing data -> $(nrow(data)-valid) out of $(nrow(data))"
            return false
        end
        # @subset!(data, Q .== 0)
        return true
    else
        return false
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
Read from a file an INRAE formatted weather station data and transform it to match ECA standard naming conventions.
- `impute_missing` expects a vector of column name(s) where to impute missing with `Impute.Interpolate` e.g. `impute_missing=[:TX]`.
- `show_warning` in case of missing data. `false` for no column, `true` for all variables columns and for selected columns e.g. `show_warning = [:TX]`.
"""
function collect_data_INRAE(station_path::String; show_warning=false, impute_missing=[])
    df_full = CSV.read(station_path, DataFrame, header=13)
    df = @chain df_full begin
        @transform(:STAID = :NUM_POSTE)
        @transform(:DATE = Date.(:AN, :MOIS, :JOUR))
        @transform(:CO2 = fill(330.0, length(:AN)))
        @transform(:QQ = :RG / 100)
        DataFramesMeta.@select :STAID :DATE :TN :TX :QQ :ETPP :RR :CO2
    end
    return clean_data(df; show_warning=show_warning, impute_missing=impute_missing)
end

"""
    clean_data(df::DataFrame; show_warning=false, impute_missing=[])
Impute missing and show warning for missings.
It assumes that the first two columns are not numeric.
- `impute_missing` expects a vector of column name(s) where to impute missing with `Impute.Interpolate` e.g. `impute_missing=[:TX]`.
- `show_warning` in case of missing data. `false` for no column, `true` for all variables columns and for selected columns e.g. `show_warning = [:TX]`.
"""
function clean_data(df::DataFrame; show_warning=false, impute_missing=[])
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

const meteofrance_id_to_url = "https://www.data.gouv.fr/fr/datasets/r/966dcd99-2856-44d0-8a2d-0b1a1c5ee737"

df_api_url = CSV.read(Base.download(meteofrance_id_to_url), DataFrame)

"""
    download_data_MeteoFrance(STAID, period = "1950-2021", variables = "all")
- Option for `period` are "1846-1949", "1950-2021", "2022-2023"
- Option for `variables` are `all`, "RR-T-Wind", "others"
"""
function download_data_MeteoFrance(STAID; period="1950-2021", variables="all")
    dep = string(STAID)[1:2]
    if variables != "all"
        if variables == "R-T-Wind"
            varFR = "RR-T-Vent"
        elseif variables == "others"
            varFR = "autres-parametres"
        else
            return "Check spellin"
        end
        id_name = string("QUOT_departement_", dep, "_periode_", period, "_", varFR)
        id = @subset(df_api_url, :title .== id_name)[1, :id]
        path = string("https://tabular-api.data.gouv.fr/api/resources/$(id)/data/csv/?NUM_POSTE__exact=", STAID)
        return df = CSV.read(Base.download(path), DataFrame)
    else
        dfRTW = download_data_MeteoFrance(STAID; period=period, variables="R-T-Wind")
        dfoth = download_data_MeteoFrance(STAID; period=period, variables="others")
        return leftjoin!(dfRTW, dfoth[:, 7:end], on=:AAAAMMJJ)
    end
end

"""
    collect_data_MeteoFrance(STAID; show_warning=false, impute_missing=[], period="1950-2021", variables = "all")
Given a `STAID` (station ID given by Météo France), it returns a `DataFrame` with data in `period` and for the `variables`.
- `STAID` can be an integer or string.
- Option for `period` are "1846-1949", "1950-2021", "2022-2023"
- Option for `variables` are `all`, "RR-T-Wind", "others"
- `impute_missing` expects a vector of column name(s) where to impute missing with `Impute.Interpolate` e.g. `impute_missing=[:TX]`.
- `show_warning` in case of missing data. `false` for no column, `true` for all variables columns and for selected columns e.g. `show_warning = [:TX]`.
The data is available through the French [Data.gouv.fr](https://www.data.gouv.fr/en/) website api. Data may be updated without notice.
See the following two links to get informations on the "RR-T-Wind" and "others" variables (in French)
- https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_descriptif_champs_RR-T-Vent.csv
- https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_descriptif_champs_autres-parametres.csv
Or the the [SICLIMA website](https://agroclim.inrae.fr/siclima/help/references.html) with information (in French) about computation and conversion for some weather variables/index.
"""
function collect_data_MeteoFrance(STAID; show_warning=false, impute_missing=[], kwargs...)
    df_full = download_data_MeteoFrance(STAID; kwargs...)
    rename!(df_full, :NUM_POSTE => :STAID, :NOM_USUEL => :STANAME, :AAAAMMJJ => :DATE)#, :GLOT => :QQ)
    return clean_data(df_full; show_warning=show_warning, impute_missing=impute_missing)
end
