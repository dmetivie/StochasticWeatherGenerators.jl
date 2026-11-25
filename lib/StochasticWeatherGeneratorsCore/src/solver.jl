# Default optimizer, provided by the OptimizationOptimJL extension.

struct OptimJLBackend end

"""
    default_solver()

Return the optimizer that classes use when `fit` gets `solver = nothing`: `LBFGS()` from OptimizationOptimJL.

It needs `using OptimizationOptimJL` (or `using StochasticWeatherGenerators`).

Experimental: may change in minor releases.
"""
default_solver() = _default_solver(OptimJLBackend())

_default_solver(backend) = throw(ArgumentError(
    "no default solver: load OptimizationOptimJL (`using OptimizationOptimJL`) or StochasticWeatherGenerators, " *
    "or pass `solver =` to `fit`"))
