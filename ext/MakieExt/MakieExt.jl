module MakieExt

using StochasticWeatherGenerators
import StochasticWeatherGenerators: errorline, errorline!, errorscatter, errorscatter!,
    errorlinehist, errorlinehist!, errorscatterhist, errorscatterhist!

using Makie
using StatsBase: Histogram, fit, normalize, mean, median, std, quantile

include("errorline.jl")
include("errorlinehist.jl")

end # module
