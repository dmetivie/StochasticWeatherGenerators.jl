# Packages

import CairoMakie, GeoMakie
using CairoMakie.Makie: GeometryBasics
using NaturalEarth

"""
    map_with_stations(LON_idx, LAT_idx; station_name = nothing, value=nothing, show_value=false, fontsize=15,
    precision_scale="50m", colorbar_title=nothing, colorbar_show=false
    LON_min=-5, # West
    LON_max=10, # Est
    LAT_min=41, # South
    LAT_max=52 # North
    )
Plot a map (by default France) with point at the given Longitude and Latitude in  Decimal Degrees.
One can add station names and a value display to it.
"""
function map_with_stations(LON_idx, LAT_idx, value=:coral; station_name=nothing, show_value=false, fontsize=15,
    precision_scale=50, colorbar_title=:none, colorbar_show=false, colorbar_limits=:none,
    LON_min=-5, # West
    LON_max=10.5, # Est
    LAT_min=41, # South
    LAT_max=52 # North
)

    borders_ne = naturalearth("admin_0_boundary_lines_land", precision_scale)
    coastlines_ne = naturalearth("coastline", precision_scale)
    rivers_ne = naturalearth("rivers_lake_centerlines_scale_rank", precision_scale)
    ocean_ne = naturalearth("ocean", precision_scale)

    fig = GeoMakie.with_theme(GeoMakie.theme_latexfonts(), fontsize=fontsize) do
        fig = GeoMakie.Figure()
        ax = GeoMakie.GeoAxis(fig[1, 1], dest="+proj=merc", xgridvisible=false, ygridvisible=false, xticklabelsvisible=true, yticklabelsvisible=true, xticksvisible=false, yticksvisible=false)
        GeoMakie.xlims!(ax, LON_min, LON_max)
        GeoMakie.ylims!(ax, LAT_min, LAT_max)
        GeoMakie.poly!(ax, ocean_ne.geometry)
        GeoMakie.lines!(ax, GeoMakie.to_multilinestring.(borders_ne.geometry); color=:black, linestyle=:dash, linewidth=0.75)
        GeoMakie.lines!(ax, GeoMakie.to_multilinestring.(coastlines_ne.geometry); color=:gray, linewidth=0.75)
        GeoMakie.lines!(ax, GeoMakie.to_multilinestring.(rivers_ne.geometry); linewidth=0.5)

        sc = GeoMakie.scatter!(ax, LON_idx, LAT_idx; color=value, markersize=15, colormap=:plasma, colorrange=((3.5minimum(value)) ÷ 4, (4.5maximum(value)) ÷ 4))
        if show_value == true
            for i in eachindex(station_name)
                if i == 7
                    GeoMakie.text!(ax, [LON_idx[i]], [LAT_idx[i]]; text=string.([value[i]]), color=:black, font=:bold, offset=(-2, 0), align=(:left, :top), fontsize=19)
                else
                    GeoMakie.text!(ax, [LON_idx[i]], [LAT_idx[i]]; text=string.([value[i]]), color=:black, font=:bold, offset=(8, 10), align=(:left, :top), fontsize=19)
                end
            end
            ## GeoMakie.text!(ax, LON_idx, LAT_idx; text=string.(value), color=:black, font=:bold, offset=(8, 10), align=(:left, :top), fontsize=18)
        end
        if colorbar_show == true
            GeoMakie.Colorbar(fig[1, 2], sc, label="Days")
        end

        if station_name !== nothing
            ## GeoMakie.text!(ax, LON_idx, LAT_idx; text=station_name, color=:black, font=:bold, offset=(-7, 4), align=(:center, :bottom))
            for i in eachindex(station_name)
                if i ∈ [7, 2]
                    GeoMakie.text!(ax, [LON_idx[i]], [LAT_idx[i]]; text=station_name[i], color=:black, font=:bold, offset=(-14, 4), align=(:center, :bottom))
                elseif i == 3
                    GeoMakie.text!(ax, [LON_idx[i]], [LAT_idx[i]]; text=station_name[i], color=:black, font=:bold, offset=(0, 4), align=(:center, :bottom))
                else
                    GeoMakie.text!(ax, [LON_idx[i]], [LAT_idx[i]]; text=station_name[i], color=:black, font=:bold, offset=(-7, 4), align=(:center, :bottom))
                end
            end
        end

        GeoMakie.colsize!(fig.layout, 1, GeoMakie.Aspect(1, 1.0)) # remove white gap in between colorbar/map
        GeoMakie.resize_to_layout!(fig) # remove white gap around figure
        fig
    end
    return fig
end

function map_with_stations(LON_idx, LAT_idx, value::AbstractArray{V}; station_name=nothing, show_value=false, fontsize=15,
    precision_scale=50, colorbar_title=:none, colorbar_show=false, colorbar_limits=:none,
    LON_min=-5, # West
    LON_max=10.5, # Est
    LAT_min=41, # South
    LAT_max=52 # North
) where {V<:AbstractArray}
    K = length(value)

    borders_ne = naturalearth("admin_0_boundary_lines_land", precision_scale)
    coastlines_ne = naturalearth("coastline", precision_scale)
    rivers_ne = naturalearth("rivers_lake_centerlines_scale_rank", precision_scale)
    ocean_ne = naturalearth("ocean", precision_scale)

    fig = GeoMakie.with_theme(GeoMakie.theme_latexfonts(), fontsize=fontsize) do
        fig = GeoMakie.Figure(size=(800,250))
        ax = [GeoMakie.GeoAxis(fig[1:3, k], dest="+proj=merc", xgridvisible=false, ygridvisible=false, xticklabelsvisible=true, yticklabelsvisible=true, xticksvisible=false, yticksvisible=false, title = L"Z = %$k") for k in 1:K]
        for k in 1:K
            GeoMakie.xlims!(ax[k], LON_min, LON_max)
            GeoMakie.ylims!(ax[k], LAT_min, LAT_max)
            GeoMakie.poly!(ax[k], ocean_ne.geometry)
            GeoMakie.lines!(ax[k], GeoMakie.to_multilinestring.(borders_ne.geometry); color=:black, linestyle=:dash, linewidth=0.75)
            GeoMakie.lines!(ax[k], GeoMakie.to_multilinestring.(coastlines_ne.geometry); color=:gray, linewidth=0.75)
            GeoMakie.lines!(ax[k], GeoMakie.to_multilinestring.(rivers_ne.geometry); linewidth=0.5)

            sc = GeoMakie.scatter!(ax[k], LON_idx, LAT_idx; color=value[k], markersize=11, colormap=GeoMakie.Reverse(:plasma), colorrange=(0, 1))
            if show_value == true
                for i in eachindex(station_name)
                    if i == 7
                        GeoMakie.text!(ax[k], [LON_idx[i]], [LAT_idx[i]]; text=string.([value[i]]), color=:black, font=:bold, offset=(-2, 0), align=(:left, :top), fontsize=19)
                    else
                        GeoMakie.text!(ax[k], [LON_idx[i]], [LAT_idx[i]]; text=string.([value[i]]), color=:black, font=:bold, offset=(8, 10), align=(:left, :top), fontsize=19)
                    end
                end
                ## GeoMakie.text!(ax, LON_idx, LAT_idx; text=string.(value), color=:black, font=:bold, offset=(8, 10), align=(:left, :top), fontsize=18)

                if station_name !== nothing
                    ## GeoMakie.text!(ax, LON_idx, LAT_idx; text=station_name, color=:black, font=:bold, offset=(-7, 4), align=(:center, :bottom))
                    for i in eachindex(station_name)
                        if i ∈ [7, 2]
                            GeoMakie.text!(ax[k], [LON_idx[i]], [LAT_idx[i]]; text=station_name[i], color=:black, font=:bold, offset=(-14, 4), align=(:center, :bottom))
                        elseif i == 3
                            GeoMakie.text!(ax[k], [LON_idx[i]], [LAT_idx[i]]; text=station_name[i], color=:black, font=:bold, offset=(0, 4), align=(:center, :bottom))
                        else
                            GeoMakie.text!(ax[k], [LON_idx[i]], [LAT_idx[i]]; text=station_name[i], color=:black, font=:bold, offset=(-7, 4), align=(:center, :bottom))
                        end
                    end
                end
            end
            if colorbar_show == true && k == K
                GeoMakie.Colorbar(fig[:, K+1], sc, height=GeoMakie.Relative(1), label = colorbar_title)
                GeoMakie.colgap!(fig.layout, -7)
            end
            k > 1 && GeoMakie.hideydecorations!(ax[k], ticks = false)

        end
        fig
    end
    return fig
end

function savefigcrop(plt::CairoMakie.Figure, save_name)
    CairoMakie.save(string(save_name, ".pdf"), plt)
    run(`pdfcrop $(string(save_name,".pdf"))`) # Petit délire pour croper proprement la figure 
    mv(string(save_name, "-crop", ".pdf"), string(save_name, ".pdf"), force=true)
end