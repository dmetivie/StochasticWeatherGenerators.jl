#TODO: Julify this code with fit_mle/fit, rand
#TODO: better the code e.g. a lot of time when simulating is spent `getindex` 
#TODO: Maybe more effective way to store the states than ordered dic?
#TODO: work on the wgen struct
""" 
    wgen
    wgen(p::Integer, D::Integer, ncat::Integer=2)
    wgen(p::Integer, D::Integer, idxs::AbstractArray, ncat::Integer=2)
Structure for a multisite rain occurrence model as described in (Wilks 1998) and generalized to higher order by Srikanthan et al. (2009). 
It contains Markov chain transition probabilities of order `p` at each sites `j ∈ 1:D`, and a correlation matrix for the unobserved Multivariate Gaussian variable (used to generate correlated rain occurrences). When `idxs` is provided it is an `Vector` of size 12 (typically) with index of days corresponding to each months.
`ncat` can be larger than 2 to model generic categorical variables (experimental feature).

## Example

```julia
p = 4
date_range = Date(1956):Day(1):Date(2019,12,31)
D = 5
Y = rand(Bool, length(date_range), D)
idx_months = [findall(x -> month.(x) == m, date_range) for m in 1:12]
# Fit returning a `wgen` struct
wgen4_model = fit_wgen(Y, idx_months, wgen_order)
# Simulate from `wgen` struct
Y_simu = rand(wgen4_model, 1956:2019; Y_ini=rand(Bool, p, D))
```

## References
- Wilks, D. S. "Multisite generalization of a daily stochastic precipitation generation model". Journal of Hydrology, (1998). https://doi.org/10.1016/S0022-1694(98)00186-3.
- Srikanthan, Ratnasingham, et Geoffrey G. S. Pegram. "A nested multisite daily rainfall stochastic generation model". Journal of Hydrology 2009. https://doi.org/10.1016/j.jhydrol.2009.03.025.
"""
struct wgen
    transition_probs
    Ω
end

function wgen(p::Integer, D::Integer, ncat::Integer=2)
    state_space = MarkovStateSpace(p, ncat)
    transition_probs = [OrderedDict(s => OrderedDict(s′ => 0.0 for s′ in state_space) for s in state_space) for j in D]
    Ω = Diagonal(fill(1, D))
    return wgen(transition_probs, Ω)
end

function wgen(p::Integer, D::Integer, idxs::AbstractArray, ncat::Integer=2)
    state_space = MarkovStateSpace(p, ncat)
    transition_probs = [OrderedDict(s => OrderedDict(s′ => 0.0 for s′ in state_space) for s in state_space) for j in D, m in eachindex(idxs)]
    Ω = [Diagonal(fill(1, D)) for m in eachindex(idxs)]
    return wgen(transition_probs, Ω)
end

function MarkovStateSpace(p::Integer, ncat=2)
    if ncat == 2
        state_space = Iterators.product(fill(0:1, p)...) |> collect |> vec |> sort .|> collect
    elseif ncat > 2
        state_space = Iterators.product(fill(1:ncat, p)...) |> collect |> vec |> sort .|> collect
    end
end

function fit_wgen(Y::AbstractMatrix, idxs, p::Integer; kwargs...)
    transition_probs = fit_markov_chain(Y, idxs, p)
    Ω = fit_Ω(transition_probs, Y, idxs; kwargs...)
    return wgen(transition_probs, Ω)
end

"""
    fit_markov_chain(Y::AbstractVector{<:Integer}, p::Integer)
    fit_markov_chain(Y::AbstractMatrix{<:Integer}, p::Integer)
    fit_markov_chain(Y::AbstractMatrix{<:Integer}, idxs::AbstractArray, p::Integer)
```julia
y = rand(Bool, 1000)
markov = fit_markov_chain(y, 2)
markov[[0,1]]
```    
TODO: add doc
"""
function fit_markov_chain(Y::AbstractVector{<:Integer}, p::Integer, state_space=MarkovStateSpace(p))
    if sort(unique([Y[i:i+p-1] for i in 1:length(Y)-p])) != MarkovStateSpace(p)
        @warn "Unobserved $(setdiff(MarkovStateSpace(p), sort(unique([Y[i:i+p-1] for i in 1:length(Y)-p]))))"
    end
    transition_counts = OrderedDict(s => OrderedDict(s′ => 0 for s′ in state_space) for s in state_space)
    transition_probs = OrderedDict(s => OrderedDict(s′ => 0.0 for s′ in state_space) for s in state_space)

    for i in 1:length(Y)-p
        current_state = Y[i:i+p-1]
        next_state = Y[i+1:i+p]
        transition_counts[current_state][next_state] += 1
    end

    for s in state_space
        total_transitions = sum(values(transition_counts[s]))
        if total_transitions > 0
            for s′ in state_space
                transition_probs[s][s′] = transition_counts[s][s′] / total_transitions
            end
        end
    end

    return transition_probs
end

fit_markov_chain(Y::AbstractMatrix{<:Integer}, p::Integer) = [fit_markov_chain(c, p) for c in eachcol(Y)]
fit_markov_chain(Y::AbstractMatrix{<:Integer}, idxs::AbstractArray, p::Integer) = @views [fit_markov_chain(Y[idx, j], p) for j in axes(Y, 2), idx in idxs]

"""
    simulate_markov_gaussian(N::Integer, ΩX::AbstractMatrix, transition_probs; Y_ini)
    simulate_markov_gaussian(years::AbstractArray{<:Integer}, ΩX, transition_probs; Y_ini)
Function to simulate `Xt` given a correlation matrix `ΩX` and transition probabilities.

- If `N` is an integer then it return a sequence of size `N` with fixed parameters.
- If not then it 
"""
function simulate_markov_gaussian(N::Integer, ΩU::AbstractMatrix, transition_probs; Y_ini)
    p, d = size(Y_ini)  # Number of locations, order of the chain
    U = rand(MvNormal(zeros(d), ΩU), N)  # Generate multivariate normal samples
    X = BitArray(undef, N, d)
    states = [Y_ini[:, i] for i in axes(Y_ini, 2)]
    for n in 1:N
        for (j, s) in enumerate(states)
            pr = transition_probs[j][s][[s[2:p]; 1]] # rain proba
            Φ⁻¹ = quantile(Normal(), pr)   # Compute Φ⁻¹(pr)
            X[n, j] = U[j, n] ≤ Φ⁻¹  # Apply thresholding to get binary states
            states[j] = [s[2:end]; X[n, j]]
        end
    end
    return X
end

function simulate_markov_gaussian(years::AbstractArray{<:Integer}, ΩX::AbstractArray, transition_probs; Y_ini)
    p, d = size(Y_ini)
    @assert d == size(transition_probs, 1)
    @assert size(transition_probs, 2) == length(ΩX)

    X = BitArray(undef, length(Date(years[1]):Day(1):Date(years[end], 12, 31)), d)
    id = 0
    for y in years
        for m in eachindex(ΩX) # typically 1:12 for all months
            N = daysinmonth(y, m)
            idend = (id + N)
            X[(id+1):idend, :] .= simulate_markov_gaussian(N, ΩX[m], transition_probs[:, m]; Y_ini=Y_ini)
            Y_ini .= X[(idend-(p-1)):idend]
            id = idend
        end
    end
    return X
end

Base.rand(model::wgen, years::AbstractArray{<:Integer}; Y_ini) = simulate_markov_gaussian(years, model.Ω, model.transition_probs; Y_ini)

"""
    fit_Ω(transition_probs, Y; ω0=(-0.999, 0.999))
    fit_Ω(transition_probs::AbstractMatrix, Y, idxs; ω0=(-0.999, 0.999))    
Fit the mutlisite mutlivariate Gaussian variable.
"""
function fit_Ω(transition_probs, Y; ω0=(-0.999, 0.999))
    @assert size(Y, 2) == 2
    ρ = cor(Y)[1, 2]
    p_order = Int(log2(length(transition_probs[1])))
    f(ω, p) = abs(ω) < 1 ? (cor(simulate_markov_gaussian(2000, [1.0 ω; ω 1.0], transition_probs; Y_ini=zeros(Int, p_order, 2)))[1, 2] - ρ) : Inf

    prob = IntervalNonlinearProblem(f, ω0)
    sol = solve(prob)
    if !SciMLBase.successful_retcode(sol.retcode)
        @warn "Fail with ρ = $(ρ)"
    end
    return sol.u
end

function fit_Ω(transition_probs::AbstractMatrix, Y, idxs; force_PosDef=true, kwargs...)
    D, M = size(transition_probs)
    @assert M == length(idxs)
    @assert D == size(Y, 2)
    Ω = [ones(D, D) for m in 1:M]
    for m in 1:M
        for i in 1:D-1
            for j in i+1:D
                Ω[m][i, j] = Ω[m][j, i] = fit_Ω(transition_probs[[i, j], m], Y[idxs[m], [i, j]]; kwargs...)
            end
        end
    end
    for m in 1:M
        if !isposdef(Ω[m])
            if force_PosDef
                Ω[m] .= nearest_cor(Ω[m])
            end
            @warn "Ω[m] not Positive Definite for m = $m -> force_PosDef = $(force_PosDef)"
        end
    end
    return Ω
end


# owm
# @views mark_model = [fit_markov_chain(Yall[2:end, :][idx_months[m], j], 1) for j in 1:D, m in 1:12]
# using LinearAlgebra
# ΩX = Diagonal(ones(D))
# mark_model[1:2, 1]
# simulate_markov_gaussian(31, ΩX, mark_model[:, 1]; Y_ini=Y_past)
# cor(Yall)

# y = rand(Bool) #generate_markov_chain(a, 200000)
# b = fit_markov_chain(y, 2)
# D = 10
# ΩXs = fill(Diagonal(ones(D)), 12)
# yy = [simulate_markov_gaussian(1956:2019, ΩXs, mark_model; Y_ini=Y_past) for i in 1:1000]

# using LinearAlgebra
# using Dates
# D = 10
# day_range = Date(1900):Day(1):Date(2000)
# y = rand(Bool, length(day_range), D)
# idxs = [findall(month.(day_range) .== m) for m in 1:12]
# p = 2
# ms = fit_markov_chain(y, idxs, p)
# ΩXs = fit_Ω(ms, y, idxs)
# yy = [simulate_markov_gaussian(1956:2019, ΩXs, ms; Y_ini=zeros(Bool, p, D)) for i in 1:1000]
