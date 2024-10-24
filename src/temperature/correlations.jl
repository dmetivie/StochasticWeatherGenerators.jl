#TODO: robustify

"""
    cor_groupby(df::DataFrame, var1, var2, T::Integer; θ0 = [0, 0.0, 0.0])
Compute and fit the `cor` between two `var` with a smooth function for each `z`. 
"""
function cor_groupby(df::DataFrame, var1, var2, T::Integer; θ0 = [0, 0.0, 0.0])
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
        fit_loss_optim(ℓₖ, df.COR, θ0, :Ipopt, print_level=0)
    end
    return ρs
end

"""
    cor_groupbyTXTN(df::DataFrame, T::Integer; θ0 = [0, 0.0, 0.0])
Compute and fit the `cor` between `:TX` and `:TX-:TN` with a smooth function for each `z`. 
"""
function cor_groupbyTXTN(df::DataFrame, T::Integer; θ0 = [0, 0.0, 0.0])
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
        fit_loss_optim(ℓₖ, df.COR, θ0, :Ipopt, print_level=0)
    end
    return ρs
end

#TODO: add flexibility in distribution + simplify arguments if possible (maybe dataframe?)
#TODO: do that functions just for residu `ϵ₁` `ϵ₂` leave the transformation back to a marginal for another function
"""
    rand_cond(ϵ, z, θ_uni, θ_cor, n2t, T)
Generate a random variable conditionally to another one assuming a Gaussian copula dependance with correlation `ρₜ(t / T, θ_cor)` (depending on the day of the year).
`ϵ` is assumed `Normal(0,1)`.
"""
function rand_cond(ϵ, z, θ_uni, θ_cor, n2t, T)
    f(θ_uni) = Gamma(θ_uni[1], θ_uni[2])# MixtureModel([Exponential(θ[1]), Exponential(θ[2])], [θ[3], 1 - θ[3]])
    f(t, θ_uni) = f([σₜ(t/T, θ_uni[1:(2+1)]), σₜ(t/T, θ_uni[(2+2):end])])
    q(t, ϵ, θ_cor) = cdf(Normal(), rand(Normal(ρₜ(t / T, θ_cor) * ϵ, (1 - ρₜ(t / T, θ_cor)^2))))
    return map(enumerate(n2t)) do (n, t)
        k = z[n]
        quantile(f(t, θ_uni[k]), q(t, ϵ[n], θ_cor[k]))
    end
end

# ```math
# X_1 \mid (X_2 = ϵ) \sim \mathcal{N}\left(\mu_1 + \dfrac{\sigma_1}{\sigma_2}\rho (a - \mu_2), (1-\rho^2)\sigma_1^2 \right)
# ```
# For two random gaussian variable
# ```math
# X_1 \mid X_2 = ϵ \sim \mathcal{N}\left(\rho a , (1-\rho^2) \right), \quad for X_1,\, X_2 \sim \mathcal{N}(0,1)
# ```