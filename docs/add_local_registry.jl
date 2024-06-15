# Add PackageNursery registry for CI
# https://github.com/j-fu/LiquidElectrolytes.jl/blob/main/docs/addnursery.jl
# https://discourse.julialang.org/t/hosting-docs-with-documenter-jl-with-dependencies-github-actions-with-unregistered-packages/83364/9?u=dmetivie
using Pkg
Pkg.Registry.add("General")
Pkg.Registry.add(RegistrySpec(url = "https://github.com/dmetivie/LocalRegistry"))