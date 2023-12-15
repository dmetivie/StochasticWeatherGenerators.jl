function fit_TN(df_full, 𝐃𝐞𝐠)
    #TODO check that dropmissing (and potentially not contigous data) does not cause issue in MLE
    df = dropmissing(df_full[:, [:DATE, :TX, :TN, :z]])

    f(θ) = Gamma(θ[1], θ[2])# MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
    f(t, θ) = f([σₜ(t, θ[1:(2𝐃𝐞𝐠+1)]), σₜ(t, θ[(2𝐃𝐞𝐠+2):end])])

    θσ10 = zeros(2𝐃𝐞𝐠 + 1)
    θσ20 = zeros(2𝐃𝐞𝐠 + 1)
    θ0 = hcat(θσ10, θσ20)
    dfk = @chain df begin
        @transform(:n2t = dayofyear_Leap.(:DATE))
        @transform(:y = :TX - :TN)
        groupby([:z])
    end
    sol_Ipopt = map(dfk) do dfkⱼ
        n2t = dfkⱼ.n2t
        ℓ(θ, x) = -sum(logpdf(f(t / T, θ), x[n]) for (n, t) in enumerate(n2t)) # = -loglikelihood
        y = dfkⱼ.y
        return SmoothPeriodicStatsModels.fit_mle(SmoothPeriodicStatsModels.OptimMLE(ℓ, SmoothPeriodicStatsModels.Ipopt.Optimizer(), vec(θ0)), y)
    end
    return sol_Ipopt#[t->f(t,θ_fit) for θ_fit in sol_Ipopt]
end
