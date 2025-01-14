"""
    run_stics_mod_yield(df::DataFrame; infos=0, output="climat.txt", stics_path, work_path, all_years = unique(year.(df.DATE))
Run the model STICS for the input weather data frame `df` for each year of the `df`.
- `stics_path` path to the `stics_modulo` executable something like `[...]/JavaSTICS-1.5.1-STICS-10.0.0/JavaSTICS-1.5.1-STICS-10.0.0/bin/stics_modulo.exe` or without the `.exe` on Linux.
- `work_path` where you have the maize variety files. 
"""
function run_stics_mod_yield(df::DataFrame; infos=0, output="climat.txt", stics_path, work_path)
    all_years = unique(year.(df.DATE))
    stics_exe = `$stics_path`
    cd(work_path)
    crop_yields = DataFrame(YEAR=Int[], BIOMASS=Union{Float64,Missing}[], YIELD=Union{Float64,Missing}[], YIELD40=Union{Float64,Missing}[])
    for (i, ye) in enumerate(all_years)
        write_simu2stamais(df, ye, work_path; output=output)
        infos_stics = @capture_out run(stics_exe)
        if split(infos_stics)[7] != "successful."
            @warn "Error on year $(ye)"
            push!(crop_yields, [ye; missing; missing; missing])
        else
            crop_yield = get_yield(work_path)
            if infos != false && (i % infos) == 0
                println("STAID = $(df.STAID[1]) Year = ", ye, " yield = ", crop_yield)
                flush(stdout)
            end
            push!(crop_yields, [ye; crop_yield])
        end
    end
    return crop_yields
end

function simu2stamais(df_full, ye)
    df = @chain df_full begin
        @subset(year.(:DATE) .== ye)
        @transform(:STANAME = fill("climais", length(:DATE)))
        @transform(:YEAR = year.(:DATE), :MONTH = month.(:DATE), :DAYINMONTH = dayofmonth.(:DATE))
        @transform(:DAY = dayofyear.(:DATE))
        @transform(:AA = fill(missing, length(:DATE)))
        @transform(:BB = fill(missing, length(:DATE)))
        @transform(:CO2 = fill(330.0, length(:DATE)))
        DataFramesMeta.@select :STANAME :YEAR :MONTH :DAYINMONTH :DAY :TN :TX :QQ :ETPP :RR :AA :BB :CO2
    end
    for c in 6:10
        tof = findall(ismissing, df[:, c])
        if length(tof) > 0
            @warn "$(year) Missing col $c $(names(df)[c]) rows $tof"
        end
    end

    return df
end

function simu2stamais_fast(df_full, ye)
    df = @chain df_full begin
        @subset(year.(:DATE) .== ye)
        @transform(:STANAME = fill("climais", length(:DATE)))
        @transform(:YEAR = year.(:DATE), :MONTH = month.(:DATE), :DAYINMONTH = dayofmonth.(:DATE))
        @transform(:DAY = dayofyear.(:DATE))
        @transform(:AA = fill(-999.9, length(:DATE)))
        @transform(:BB = fill(-999.9, length(:DATE)))
        @transform(:CO2 = fill(330.0, length(:DATE)))
        DataFramesMeta.@select :STANAME :YEAR :MONTH :DAYINMONTH :DAY :TN :TX :QQ :ETPP :RR :AA :BB :CO2
    end
    return df
end

"""
write_inrae2stamais(df_full, year, workspace_path; output = "climaisj.1996", warning = false)
Format and write into `workspace_path` an CLIMATIK weather station data to the STICS dataframe for `year`.
"""
function write_simu2stamais(df_full, year, workspace_path; output="climaisj.1996", show_warning = false)
    if !show_warning
        # GC.gc();
        df = simu2stamais_fast(df_full, year)
        try
            open(joinpath(workspace_path, output), "w") do io
                writedlm(io, Matrix(df))
            end
        catch
            GC.gc();GC.gc();
            open(joinpath(workspace_path, output), "w") do io
                writedlm(io, Matrix(df))
            end
        end
    else
        df = simu2stamais(df_full, year)
        CSV.write(joinpath(workspace_path, output), df, missingstring="-999.9", writeheader=false, delim=" ")
    end
end

"""
	get_yield(workspace_path)
Extract the 3 "yields" values from the `mod_bmaize.sti` files

- Aerial biomass at harvest (0% water) (t/ha)
- Grain or fruit Yield   (0% water) (t/ha)
- Grain or fruit Yield (40.% water) (t/ha)
"""
function get_yield(workspace_path)
    yield = Float64[]
    open(joinpath(workspace_path, "mod_bmaize.sti")) do file
        for line in eachline(file)
            # Process each line
            if startswith(line, "   Grain or fruit Yield")
                s = split(line)
                push!(yield, parse(Float64, s[end-1]))
            elseif startswith(line, "   Aerial biomass at harvest")
                s = split(line)
                push!(yield, parse(Float64, s[end-1]))
            end
        end
    end
    return yield
end

chunky(n, k) = [[(n÷k)*j+1:(n÷k)*(j+1) for j in 0:k-2]; [((n÷k)*(k-1)+1):n]]

function stats_fortnightly(df, week_date, years)
    groups = chunky(Dates.value(week_date.date_end - week_date.date_begin) + 1, week_date.N_period)
    dfchain = @chain df begin
        @transform(:YEAR = year.(:DATE))
        @subset(Date.(:YEAR, monthday(week_date.date_begin)...) .≤ :DATE .≤ Date.(:YEAR, monthday(week_date.date_end)...))
        @subset(years[1] .≤ :YEAR .≤ years[end])
        groupby([:YEAR])
        @transform(:FORTNIGHT = reduce(vcat, [fill(i, length(group)) for (i, group) in enumerate(groups)]))
        @by([:YEAR, :FORTNIGHT], :MEAN_TX = mean(skipmissing(:TX)), :MEAN_RR = mean(skipmissing(:RR)))
        groupby([:FORTNIGHT])
    end

    df_wider = @combine(df, :YEAR = years)
    for (i, df) in enumerate(dfchain)
        insertcols!(df_wider, Symbol(string("MEAN_TX_$(i)")) => df[:, :MEAN_TX])
    end
    for (i, df) in enumerate(dfchain)
        insertcols!(df_wider, Symbol(string("MEAN_RR_$(i)")) => df[:, :MEAN_RR])
    end
    return df_wider
end

dada = Date(1996):Day(1):Date(1996,12,31)
period_range_func(group) = [dada[dayofyear(week_date.date_begin) .+ group[1] .- 1], dada[dayofyear(week_date.date_begin) .+ group[end] .- 1]]
