#TODO: rebase that (with a part inside SmoothPeriodicStatsModels maybe), allow other distribution
"""
    fit_TN(df_full::DataFrame, 𝐃𝐞𝐠, T; kwargs...)
Fit the variable `TN` (daily minimum temperature). In fact it fits the difference `ΔT = TX - TN` to ensure a positive difference between `TX` and `TN`
"""
function fit_TN(df_full::DataFrame, 𝐃𝐞𝐠, T; kwargs...)
    #TODO check that dropmissing (and potentially not contiguous data) does not cause issue in MLE
    df = dropmissing(df_full[:, [:DATE, :TX, :TN, :z]])

    f(θ) = Gamma(θ[1], θ[2], check_args = false)# MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
    f(t, θ) = f([σₜ(t, θ[1:(2𝐃𝐞𝐠+1)]), σₜ(t, θ[(2𝐃𝐞𝐠+2):end])])

    θσ10 = zeros(2𝐃𝐞𝐠 + 1)
    θσ20 = zeros(2𝐃𝐞𝐠 + 1)
    θ0 = hcat(θσ10, θσ20)
    dfk = @chain df begin
        @transform(:n2t = dayofyear_Leap.(:DATE))
        @transform(:y = :TX - :TN)
        groupby([:z])
    end
    sol_Ipopt = map(enumerate(dfk)) do (k, dfkⱼ)
        n2t = dfkⱼ.n2t
        ℓ(θ, x) = -sum(logpdf(f(t / T, θ), x[n]) for (n, t) in enumerate(n2t)) # = -loglikelihood
        y = dfkⱼ.y
        return fit_loss_optim(ℓ, y, θ0; kwargs...)# fit_mle(OptimMLE(ℓ, Ipopt.Optimizer(), vec(θ0)), y)
    end
    return sol_Ipopt#[t->f(t,θ_fit) for θ_fit in sol_Ipopt]
end