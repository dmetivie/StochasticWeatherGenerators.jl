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

