module StochasticWeatherGeneratorsCoreOptimizationOptimJLExt

using StochasticWeatherGeneratorsCore: StochasticWeatherGeneratorsCore, OptimJLBackend
using OptimizationOptimJL: LBFGS

StochasticWeatherGeneratorsCore._default_solver(::OptimJLBackend) = LBFGS()

end
