# 🌐 Other SWG Packages

Many other SWG packages, Git repositories, and codebases exist — though detailed documentation is often lacking.
Below is a **far from exhaustive list**[^1] that we (the authors and contributors) will aim to complete over time.

[^1]: Focusing on open source code only.

!!! note
    If you know of a useful model with code not listed here, feel free to suggest it or contribute via Pull Request.

```@raw html
<embed type="text/html" src="../other_pkg/table_pkg.html" width="100vw" height="50vh">
```

!!! tip "Two Languages Problem"
    Having two languages inside a software makes it difficult to read, maintain, and compose with other packages.
    Some of the packages listed here use a "friendly" but slower language for most of the program (e.g. Matlab, R, Python), along with a "less-friendly" but faster language (e.g. C++, Fortran) for the core functions.
    This is known (mostly in the Julia world) as the [Two Languages Problem](https://juliadatascience.io/julia_accomplish), which Julia is designed to solve. One advantage of this approach is that the source code is "easy" to read and write while still being very fast. Moreover, it can integrate with state-of-the-art DataFrame libraries, optimization solvers, and statistical packages. All of this makes writing a Stochastic Weather Generator in Julia highly relevant.

```@bibliography
Pages = []
Canonical = false
1981_richardson_StochasticSimulationDaily
2012_chen_WeaGETSMatlabbasedDaily
2017_sommer_GloballyCalibratedScheme
2017_peleg_AdvancedStochasticWeather
2024_obakrim_MultivariateSpacetimeStochastic
```
