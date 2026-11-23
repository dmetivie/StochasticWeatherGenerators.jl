# Helpers to save the tutorial figures (used in `#src` lines only).
# The plot functions (`errorline`, `errorlinehist`, `map_with_stations`, ...) are in StochasticWeatherGenerators.jl,
# once Makie (and GeoMakie + NaturalEarth for maps) is loaded.

"""
    savefigcrop(fig, save_name)
    savefigcrop(fig, save_name, path)

Save the figure as `save_name.pdf` (in the folder `path`) and crop its white margins with `pdfcrop`.
"""
function savefigcrop(fig, save_name::AbstractString)
    endswith(save_name, ".pdf") || (save_name = string(save_name, ".pdf"))
    save(save_name, fig)
    run(`pdfcrop $(save_name)`)
    mv(replace(save_name, r"\.pdf$" => "-crop.pdf"), save_name, force=true)
end

# On Windows, `pdfcrop` has permission issues outside of the current folder.
function savefigcrop(fig, save_name, path)
    current = pwd()
    cd(path)
    try
        savefigcrop(fig, save_name)
    finally
        cd(current)
    end
end
