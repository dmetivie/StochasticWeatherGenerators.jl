"""
# errorlinehist(y; kwargs):
    Function for parsing inputs to easily make a [`ribbons`] (https://ggplot2.tidyverse.org/reference/geom_ribbon.html),
    stick errorbar (https://www.mathworks.com/help/matlab/ref/errorbar.html), or plume
    (https://stackoverflow.com/questions/65510619/how-to-prepare-my-data-for-plume-plots) with several histograms plot.

# Inputs: default values are indicated with *s

y is a Vector of vector

    bins (*:auto*, AbstractVector)

    norm (`Symbol` - *:false*, `:pdf`, `:probability`)

    error_style (`Symbol` - *:ribbon*, :stick, :plume) - determines whether to use a ribbon style or stick style error
     representation.

    centertype (symbol - *:mean* or :median) - which approach to use to represent the central value of y at each x-value.

    errortype (symbol - *:std*, :sem, :percentile) - which error metric to use to show the distribution of y at each x-value.

    percentiles (Vector{Int64} *[25, 75]*) - if using errortype === :percentile then which percentiles to use as bounds.

    secondarycolor (`Symbol`, `RGB`, `:matched` - *:Gray60*) - When using stick mode this will allow for the setting of the stick color.
        If `:matched` is given then the color of the sticks with match that of the main line.

    secondarylinealpha (float *.1*) - alpha value of plume lines.

    numsecondarylines (int *100*) - number of plume lines to plot behind central line.

    stickwidth (Float64 *.01*) - How much of the x-axis the horizontal aspect of the error stick should take up.

# Example
```julia
using Distributions

dist = Exponential()
N = 100 # number of sample used in each histogram
N_hist = 1000 # number of histogram
yc = [rand(dist, N) for _ in 1:N_hist]

edges = 0:0.05:11.1

errorlinehist(yc, errortype = :percentile, percentiles = [0,100], fillalpha = 0.1, normalize = :pdf, bins = edges)
```
"""
@userplot ErrorLineHist

@recipe function f(p::ErrorLineHist)
    v = p.args[1]
    bins = get(plotattributes, :bins, :auto)
    normed = get(plotattributes, :normalize, false)
    weights = get(plotattributes, :weights, nothing)

    vs = filter.(isfinite, (reduce(vcat,v),))
    edges = Plots._hist_edges(vs, bins)
    nbins = length(edges[1]) .- 1
    x = edges[1][1:end-1]

    ngroups = length(v)

    # compute weights (frequencies) by group using those edges
    y = zeros(nbins, ngroups)
    for i in 1:ngroups
        v_i = filter(isfinite, v[i])
        w_i = weights
        h_i = Plots._make_hist((v_i,), edges; normed = normed, weights = w_i)
        y[:, i] += h_i.weights .+ eps() # for numerical stability when in log-scale
    end

    StatsPlots.ErrorLine((x, y))
end

"""
    islabel(j, j_target, label)
Utilities to only display label on target subplots
```julia
using Plots
plt = [plot(1:10, label = islabel(j, [5,4], "label")) for j in 1:4]
plot(plt...)
```    
"""
islabel(j, j_target, label) = j ∈ j_target ? label : :none

islabel(j, j_target, label, other_option) = j ∈ j_target ? label : :other_option

"""
    savefigcrop(plt, path)
Save the figure and crop it to avoid white spaces
"""
function savefigcrop(plt, save_name)
    # save_name = replace(save_name, "." => "p")
    isnothing(plt) ? savefig(string(save_name, ".pdf")) : savefig(plt, string(save_name, ".pdf"))
    run(`pdfcrop $(string(save_name,".pdf"))`) # Petit délire pour croper proprement la figure 
    mv(string(save_name, "-crop", ".pdf"), string(save_name, ".pdf"), force=true)
end

savefigcrop(save_name) = savefigcrop(nothing, save_name)