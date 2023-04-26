struct OptimMLE{T,V} <: AbstractMLE where {T, V<:AbstractVector}
    ℓ::Function # -loglikelihood
    solver::T # <:AbstractOptimizationAlgorithm
    θ₀::V
end

function Distributions.fit_mle(Opt::OptimMLE, y::AbstractArray; solvekwargs...)
    OptFunc = OptimizationFunction(Opt.ℓ, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(OptFunc, Opt.θ₀, y)

    return solve(prob, Opt.solver; solvekwargs...)
end