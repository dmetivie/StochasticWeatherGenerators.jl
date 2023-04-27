function fit_mle(mix::AbstractArray{<:MixtureModel}, θ_α::AbstractArray, θ_Y::AbstractArray, y::AbstractVecOrMat, n2t::AbstractVector;
    display=:none, maxiter=100, tol=1e-3, robust=false, silence=true, warm_start=true)
    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K, T = size_sample(y), ncomponents(mix[1]), size(mix, 1)
    # TODO deal with K=1 case
    @argcheck K > 1
    d = (length(θ_α[1, :]) - 1) ÷ 2 #
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))
    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    s_γ = zeros(T, K)

    c = zeros(N)

    # Initial parameters
    θ_α = copy(θ_α)
    θ_Y = copy(θ_Y)
    n_in_t = [findall(n2t .== t) for t = 1:T]

    α = copy(hcat(probs.(mix)...))
    dists = copy(hcat(components.(mix)...))
    types = typeof.(dists[:, 1])
    dists = [convert(Vector{types[k, 1]}, dists[k, :]) for k = 1:K] # I have to separate the array to preserve types

    model_α = model_for_α(θ_α, s_γ; silence=silence)
    model_dist = [model_for_dist(dists[k][1], θ_Y[k], T; silence=silence) for k = 1:K]

    # E-step
    # evaluate likelihood for each type k
    E_step!(LL, c, γ, dists, α, y, n2t; robust=robust)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Method = $(method)\nIteration 0: Loglikelihood = ", logtot)

    for it = 1:maxiter

        # M-step
        # with γ in hand, maximize (update) the parameters
        s_γα!(s_γ, γ, n_in_t)
        update_α!(α, θ_α, s_γ, model_α; warm_start=warm_start)
        for k = 1:K
            dists[k][:], θ_Y[k] = update_dist(dists[k], θ_Y[k], γ[:, k], y, n_in_t, model_dist[k], warm_start=warm_start)
        end

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y, n2t; robust=robust)

        # Loglikelihood
        logtotp = sum(c)
        (display == :iter) && println("Iteration $(it): loglikelihood = ", logtotp)

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in ", it, " iterations, final loglikelihood = ", logtotp)
            history["converged"] = true
            break
        end

        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println(
                "EM has not converged after $(history["iterations"]) iterations, final loglikelihood = $logtot",
            )
        end
    end

    return [MixtureModel([dists[k][t] for k = 1:K], [α[k, t] for k = 1:K]) for t = 1:T], θ_α, θ_Y, history
end

# update distributions Exponential
function update_dist(dist::AbstractVector{F} where {F<:Exponential}, θ_Y::AbstractArray, γ::AbstractVector, observations::AbstractArray, n_in_t, model::Model; warm_start=true)
    @argcheck size(γ, 1) == size(observations, 1)
    dist = copy(dist)
    θ_Y = copy(θ_Y)

    N = size(γ, 1)
    T = size(dist, 1)
    θ_jump = model[:θ_jump]
    s_π = model[:s_π]
    s_π_R = model[:s_π_R]

    ## Update the smoothing parameters in the JuMP model
    for t = 1:T
        set_value(s_π[t], sum(γ[n] for n in n_in_t[t]; init=0))
        set_value(s_π_R[t], sum(γ[n] * observations[n] for n in n_in_t[t]; init=0))
    end

    warm_start && set_start_value.(θ_jump, θ_Y[:])

    optimize!(model)
    θ_Y[:] = value.(θ_jump)

    p = [exp(polynomial_trigo(t/T, θ_Y[:])) for t = 1:T]
    dist[:] = Exponential.(p)
    return dist, θ_Y
end

# JuMP model for Exponential
"""
    model_for_dist(dist::Exponential, θ_Y::AbstractArray, T::Int; silence = true)
Create a JuMP model for the Loglikelihood of Seasonal `Exponential` model
"""
function model_for_dist(dist::Exponential, θ_Y::AbstractArray, T::Int; silence=true)
    d = (size(θ_Y, 1) - 1) ÷ 2

    model = Model(Ipopt.Optimizer)
    silence && set_silent(model)
    f = 2π / T

    cos_nj = [cos(f * j * t) for t = 1:T, j = 1:d]
    sin_nj = [sin(f * j * t) for t = 1:T, j = 1:d]
    trig = [[1; interleave2(cos_nj[t, :], sin_nj[t, :])] for t = 1:T]

    @variable(model, θ_jump[j=1:(2d+1)])

    # Polynomial P
    @NLexpression(model, Pθ[t=1:T], sum(trig[t][j] * θ_jump[j] for j = 1:(2d+1)))

    @NLparameter(model, s_π[t=1:T] == 1)
    @NLparameter(model, s_π_R[t=1:T] == 1) # same initialization but will be changed later

    # -sum(πk[n]*(Pθ[n]+R[n]*exp(-Pθ[n])) for n in 1:N)
    @NLobjective(
        model, Max,
        -sum(s_π[t] * Pθ[t] + s_π_R[t] * exp(-Pθ[t]) for t = 1:T)
    )

    # I don't know if it is the best but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:s_π] = s_π
    model[:s_π_R] = s_π_R

    return model
end

# Uppdate γ
"""
    s_γα!(s_ξ, ξ, n_in_t)
Give an aggregated weigth to each category `k` for each "day" `t` which is the sum of the weigth `ξ`
"""
function s_γα!(s_ξ, ξ, n_in_t)
    T, K = size(s_ξ, 1), size(s_ξ, 2)
    for t = 1:T
        for k = 1:K
            s_ξ[t, k] = sum(ξ[n, k] for n in n_in_t[t]; init=0)
        end
    end
end

# Uppdate α
"""
    model_for_α(θ_α::AbstractMatrix, s_γ::AbstractMatrix; silence=true) 
Create a JuMP model to update the weights of each mixture components `α`

TODO -> to clarify, I don't remember exactly what I did!!!
"""
function model_for_α(θ_α::AbstractMatrix, s_γ::AbstractMatrix; silence=true)
    T, K = size(s_γ)
    d = (size(θ_α, 2) - 1) ÷ 2
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_cpu_time", 60.0)
    set_optimizer_attribute(model, "max_iter", 100)
    silence && set_silent(model)
    f = 2π / T
    cos_nj = [cos(f * j * t) for t = 1:T, j = 1:d]
    sin_nj = [sin(f * j * t) for t = 1:T, j = 1:d]
    trig = [[1; interleave2(cos_nj[t, :], sin_nj[t, :])] for t = 1:T]

    @variable(model, θ_jump[k=1:K-1, j=1:(2d+1)])
    # Polynomial P
    @NLexpression(model, P[t=1:T, k=1:K-1], sum(trig[t][j] * θ_jump[k, j] for j = 1:(2d+1)))

    @NLparameter(model, s_π[t=1:T, k=1:K] == s_γ[t, k])
    @NLobjective(
        model,
        Max,
        sum(sum(s_π[t, k] * P[t, k] for k = 1:K-1) - sum(s_π[t, k] for k = 1:K) * log1p(sum(exp(P[t, k]) for k = 1:K-1)) for t = 1:T) #Σ_k πk[n,k] = 1
    )
    # I don't know if it is the best but https://discourse.julialang.org/t/jump-updating-nlparameter-of-a-model-in-a-loop/35081/3
    model[:s_π] = s_π
    return model
end

function update_α!(
    α::AbstractArray{F},
    θ::AbstractArray{F},
    s_γ::AbstractMatrix,
    model::Model;
    warm_start=true
) where {F}
    T, K = size(s_γ)

    # ξ are the filtering probabilities
    θ_jump = model[:θ_jump]
    s_π = model[:s_π]
    ## Update the smoothing parameters in the JuMP model
    for t = 1:T, k = 1:K
        set_value(s_π[t, k], s_γ[t, k])
    end
    warm_start && set_start_value.(θ_jump, θ)
    # Optimize the updated model
    optimize!(model)
    # Obtained the new parameters
    θ[:, :] = value.(θ_jump)
    [α[k, t] = exp(polynomial_trigo(t/T, θ[k, :])) for k = 1:K-1, t = 1:T]
    [α[K, t] = 1 for t = 1:T] # last column is 1/normalization (one could do otherwise)
    normalization_polynomial = [1 + sum(α[k, t] for k = 1:K-1) for t = 1:T]
    [α[k, t] /= normalization_polynomial[t] for k = 1:K, t = 1:T]
end

function E_step!(
    LL::AbstractMatrix{T},
    c::AbstractVector{T},
    γ::AbstractMatrix{T},
    dists::AbstractArray,
    α::AbstractMatrix,
    y::AbstractVector{<:Real}, n2t::AbstractVector;
    robust=false
) where {T<:AbstractFloat}
    # evaluate likelihood for each type k
    for k in eachindex(dists)
        for (n, t) in enumerate(n2t)
            LL[n, k] = log(α[k, t]) + logpdf(dists[k][t], y[n])
        end
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    logsumexp!(c, LL) # c[:] = logsumexp(LL, dims=2)
    γ[:, :] .= exp.(LL .- c)
end

function E_step!(
    LL::AbstractMatrix{T},
    c::AbstractVector{T},
    γ::AbstractMatrix{T},
    dists::AbstractArray,
    α::AbstractMatrix,
    y::AbstractMatrix, n2t::AbstractVector;
    robust=false
) where {T<:AbstractFloat}
    # evaluate likelihood for each type k
    for k in eachindex(dists)
        for (n, t) in enumerate(n2t)
            LL[n, k] = log(α[k, t]) + logpdf(dists[k][t], y[:, n])
        end
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    c[:] = logsumexp(LL, dims=2)
    γ[:, :] = exp.(LL .- c)
end

# Utilities 

size_sample(y::AbstractMatrix) = size(y, 2)
size_sample(y::AbstractVector) = length(y)
