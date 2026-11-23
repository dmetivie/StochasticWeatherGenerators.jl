"""
    fit_TN(df_full::DataFrame, 𝐃𝐞𝐠, T; solver, kwargs...)
Fit the variable `TN` (daily minimum temperature). In fact it fits the difference `ΔT = TX - TN` to ensure a positive difference between `TX` and `TN`

``\\Delta T \\mid z \\sim \\mathrm{Gamma}(\\alpha(t), \\theta(t))``, with both the shape and the
scale seasonal: a trigonometric polynomial of degree `𝐃𝐞𝐠` pushed through `exp`, one fit per
hidden state `z`.

Returns one `TimeVaryingFit` per hidden state. Each carries the parametrization with it, so
`fit(t)` is the `Gamma` on day `t`, `coef(fit, :α)` the shape's coefficients and
`curve(fit, :θ, 1:T)` the scale over the year — nothing downstream has to know the layout.
`solver`, the optimization algorithm, is required.
"""
function StochasticWeatherGenerators.fit_TN(df_full::DataFrame, 𝐃𝐞𝐠, T; kwargs...)
    #TODO check that dropmissing (and potentially not contiguous data) does not cause issue in MLE
    df = dropmissing(df_full[:, [:DATE, :TX, :TN, :z]])

    # `seasonal_family(Gamma, …)` is `ParamSpec(α = basis => LogLink(), θ = basis => LogLink())`,
    # which is exactly the pair of `σₜ` polynomials this used to slice out of one flat vector by
    # hand. Declaring it once means `rand_cond` can no longer disagree about the layout.
    fam = seasonal_family(Gamma, TrigBasis(T, 𝐃𝐞𝐠))
    design = PeriodicDesign(fam.spec, 1:T)
    θ0 = zeros(ncoef(fam.spec))

    dfk = @chain df begin
        @transform(:n2t = dayofyear_Leap.(:DATE))
        @transform(:y = :TX - :TN)
        groupby([:z])
    end
    # A comprehension, not `map`: `map` over a `GroupedDataFrame` is reserved by DataFrames.
    return [_fit_seasonal_gamma(fam, design, θ0, dfkⱼ, T; kwargs...) for dfkⱼ in dfk]
end

function _fit_seasonal_gamma(fam, design, θ0, dfkⱼ, T; kwargs...)
    # `PeriodicNLL` builds `T` distributions per iteration instead of one per observation, and
    # the objective is the *average* rather than the sum: that leaves the maximiser where it is
    # and keeps the first gradient O(1), so the line search cannot throw the `exp` link into
    # underflow on its first step.
    nll = PeriodicNLL(fam, design, dfkⱼ.y, group_by_time(dfkⱼ.n2t, T))
    N = length(dfkⱼ.y)
    ℓ(θ, _x) = nll(θ) / N
    sol = fit_loss_optim(ℓ, dfkⱼ.y, θ0; kwargs...)
    return TimeVaryingFit(fam, sol.u, sol)
end

#TODO: robustify

"""
    cor_groupby(df::DataFrame, var1, var2, T::Integer; solver, θ0 = [0, 0.0, 0.0], kwargs...)
Compute and fit the `cor` between two `var` with a smooth function for each `z`. `solver`, the
optimization algorithm, is required; it and the other keywords go to `fit_loss_optim`.
"""
function StochasticWeatherGenerators.cor_groupby(df::DataFrame, var1, var2, T::Integer; θ0 = [0, 0.0, 0.0], kwargs...)
    df_groups = @chain df begin
        dropmissing
        @transform(:DAY = dayofyear_Leap.(:DATE))
        groupby([:DAY, :z])
        @combine(:COR = cor($(var1), $(var2)))
        @subset(@byrow !isnan(:COR))
        groupby(:z)
    end
    ρs = map(enumerate(df_groups)) do (k, df)
        ℓₖ(θ, x) = sum(abs2, ρₜ(t / T, θ) - x[n] for (n, t) in enumerate(df.DAY))
        fit_loss_optim(ℓₖ, df.COR, θ0; kwargs...)
    end
    return ρs
end

"""
    cor_groupbyTXTN(df::DataFrame, T::Integer; solver, θ0 = [0, 0.0, 0.0], kwargs...)
Compute and fit the `cor` between `:TX` and `:TX-:TN` with a smooth function for each `z`.
`solver`, the optimization algorithm, is required; it and the other keywords go to
`fit_loss_optim`.
"""
function StochasticWeatherGenerators.cor_groupbyTXTN(df::DataFrame, T::Integer; θ0 = [0, 0.0, 0.0], kwargs...)
    df_groups = @chain df begin
        dropmissing
        @transform(:DAY = dayofyear_Leap.(:DATE))
        groupby([:DAY, :z])
        @combine(:COR = cor(:TX, :TX-:TN))
        @subset(@byrow !isnan(:COR))
        groupby(:z)
    end
    ρs = map(enumerate(df_groups)) do (k, df)
        ℓₖ(θ, x) = sum(abs2, ρₜ(t / T, θ) - x[n] for (n, t) in enumerate(df.DAY))
        fit_loss_optim(ℓₖ, df.COR, θ0; kwargs...)
    end
    return ρs
end

#TODO: do that functions just for residu `ϵ₁` `ϵ₂` leave the transformation back to a marginal for another function
"""
    rand_cond([rng::AbstractRNG, ]ϵ, z, θ_uni, θ_cor, n2t, T)
Generate a random variable conditionally to another one assuming a Gaussian copula dependance with correlation `ρₜ(t / T, θ_cor)` (depending on the day of the year).
`ϵ` is assumed `Normal(0,1)`.

`θ_uni[k]` is hidden state `k`'s marginal: anything callable at a day, which is what
[`fit_TN`](@ref) returns. The distribution family is therefore the caller's choice and no
longer fixed here.
"""
function StochasticWeatherGenerators.rand_cond(rng::AbstractRNG, ϵ, z, θ_uni, θ_cor, n2t, T)
    # given ϵ, the other standard normal is N(ρϵ, 1 - ρ²): `Normal` takes the standard deviation
    q(t, ϵ, θ_cor) = cdf(Normal(), rand(rng, Normal(ρₜ(t / T, θ_cor) * ϵ, sqrt(1 - ρₜ(t / T, θ_cor)^2))))
    return map(enumerate(n2t)) do (n, t)
        k = z[n]
        # `θ_uni[k](t)` is the fitted marginal on day `t`. This used to rebuild a `Gamma` from a
        # hand-sliced coefficient vector with the degree hard-coded to 1 — a silent mis-slice
        # for any other `𝐃𝐞𝐠` passed to `fit_TN`.
        quantile(θ_uni[k](t), q(t, ϵ[n], θ_cor[k]))
    end
end

StochasticWeatherGenerators.rand_cond(ϵ, z, θ_uni, θ_cor, n2t, T) = StochasticWeatherGenerators.rand_cond(GLOBAL_RNG, ϵ, z, θ_uni, θ_cor, n2t, T)

# ```math
# X_1 \mid (X_2 = ϵ) \sim \mathcal{N}\left(\mu_1 + \dfrac{\sigma_1}{\sigma_2}\rho (a - \mu_2), (1-\rho^2)\sigma_1^2 \right)
# ```
# For two random gaussian variable
# ```math
# X_1 \mid X_2 = ϵ \sim \mathcal{N}\left(\rho a , (1-\rho^2) \right), \quad for X_1,\, X_2 \sim \mathcal{N}(0,1)
# ```