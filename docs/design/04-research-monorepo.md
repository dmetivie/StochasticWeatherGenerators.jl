# SWG 2.0 monorepo: should the libraries share a `[workspace]`, and how do testing, docs and registration work?

## Short answer

1. **SciML does not use `[workspace]` in OrdinaryDiffEq.jl.** There is no `[workspace]` in the root, in `lib/OrdinaryDiffEqTsit5` or in `lib/OrdinaryDiffEqCore`. Each package points to its siblings with its own `[sources]` table. The roots of NonlinearSolve.jl, Optimization.jl, ModelingToolkit.jl and Enzyme.jl also have `[sources]` and no `[workspace]`. Makie.jl has no workspace either.
2. **A root `[workspace]` that lists `lib/*` does exist, but it is not the standard.** JuliaGPU uses it in CUDA.jl and GPUArrays.jl. In both, the subpackages are released in lockstep with exact compat (`"=6.5.1"`, `"= 0.2.2"`).
3. **Lux.jl uses exactly the layout you recommend.** Each `lib/X` has its own `[sources]` and its own `[workspace] projects = ["test"]`. The root workspace contains only `["test", "docs"]`.
4. **Your rule matches practice for libraries that are versioned independently, which SWG wants.** Keep it, with two edits listed at the end.
5. **The critic's concern is real but narrow.** Pkg intersects `[compat]`, weak dependencies included, across every project in a workspace. A legacy root that says `SmoothPeriodicStatsModels = "4.1"` would therefore block SPSM 5 for every library in the same workspace. It only matters if the legacy code and the libraries share one workspace when SPSM 5.0 lands.
6. **Docs, tests and registration all work with this layout.** Costs: one registration comment per package per release, a 3-day wait for each new package, and a `stable` docs site that moves only on umbrella releases.

---

## 1. What OrdinaryDiffEq.jl actually does

| Topic | What I found | Source |
|---|---|---|
| Root `Project.toml` | No `[workspace]`. `[sources]` at line 20, e.g. `OrdinaryDiffEqCore = {path = "lib/OrdinaryDiffEqCore"}` (l.25). `julia = "1.10"` (l.76). Tests use the legacy `[extras]`/`[targets]` (l.78, l.142). | [ode-root] |
| `lib/OrdinaryDiffEqTsit5` | No `[workspace]`. `[sources]` `OrdinaryDiffEqCore = {path = "../OrdinaryDiffEqCore"}` (l.23). Sibling compat `OrdinaryDiffEqCore = "4.15.1"` (l.36). The test env is `[targets] test = ["ADTypes", "DiffEqDevTools", …]` (l.62), **not** a separate `test/Project.toml`. | [ode-tsit5] |
| `lib/OrdinaryDiffEqCore` | No `[workspace]`. `[sources]` at l.123–137 also points to **test-only** siblings (`[sources.OrdinaryDiffEqTsit5] path = "../OrdinaryDiffEqTsit5"`), which are listed in `[extras]` (l.109–111). | [ode-core] |
| Per-library test groups | `const TEST_GROUP = get(ENV, "GROUP", "ALL")`, plus `test_groups.toml` with `[Core] versions = ["lts", "1", "pre"]`. | [ode-tsit5-run], [ode-tsit5-groups] |
| CI | `SublibraryCI.yml` calls `SciML/.github/.github/workflows/sublibrary-project-tests.yml@v1` with `group-env-name: GROUP`. That workflow tests "a changed sublibrary plus every sublibrary that transitively depends on it", each with `project: lib/<name>`. Its header notes that "tests.yml transitively git-develops the sublibrary's in-repo [sources], so no root runtests.jl GROUP dispatcher is needed". | [ode-subci], [sciml-subproj] |
| Julia 1.10 workaround | tests.yml l.200–221 develops the `[sources]` paths by hand, because "On Julia < 1.11 the [sources] section is not auto-resolved … No-op on >= 1.11". | [sciml-tests], [sciml-devsrc] |
| TagBot | Calls `monorepo-tagbot.yml@v1`. It reads the subdir from the registry notification and runs a matrix over `subdir` with `max-parallel: 1`, each job using `JuliaRegistries/TagBot@v1` with `subdir:` set. | [ode-tagbot], [sciml-monotag], [sciml-tagbot] |
| Dependency bumps | There is no CompatHelper workflow. `dependabot.yml` uses `package-ecosystem: "julia"` with `directories: "/", "/docs", "/lib/*", …` and `group-by: dependency-name`. | [ode-dependabot] |
| Docs | Built in-repo under `docs/`. `docs/Project.toml` has `[sources]` entries: `OrdinaryDiffEq = {path = ".."}` and every `../lib/X`. `make.jl` lists all the modules, sets `checkdocs = :public`, and calls `deploydocs(repo = "github.com/SciML/OrdinaryDiffEq.jl"; push_preview = true)`. A comment there reads "Keep pages.jl separate for the DiffEqDocs.jl build", so the same pages are also pulled into an aggregate site. The SciMLDocs aggregate uses MultiDocumenter. | [ode-docs-proj], [ode-make], [scimldocs] |
| Docs CI | The reusable workflow runs `julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'`. | [sciml-docs] |
| Registration | One commit comment per package, e.g. `@JuliaRegistrator register() subdir=lib/DiffEqBase`. Registrator's reply suggests the tag `DiffEqBase-v7.21.6`. | [ode-regcomment] |
| Registry entry | `subdir = "lib/OrdinaryDiffEqTsit5"`. Version 2.1.5, whose `Project.toml` carries `[sources]`, is in General. | [general-tsit5] |

**Inference:** SciML avoids `[workspace]` because it still supports Julia 1.10. Workspaces need Julia 1.12+ ([pkg-toml] "Workspaces require Julia 1.12+"), and SciML maintains `develop_sources.jl` only to emulate `[sources]` on 1.10. SWG and SPSM already require `julia = "1.12"` (SWG `Project.toml:57`, SPSM `Project.toml:51`), so that constraint does not apply to you.

## 2. Other monorepos

| Repo | Root `[workspace]` | Sibling resolution | Versioning | Tests | Tags and registration | Docs |
|---|---|---|---|---|---|---|
| SciML/NonlinearSolve.jl | none | root `[sources.X] path = "lib/X"`; libs use `../X` | independent | `sublibrary-project-tests.yml`, `group-env-name: NONLINEARSOLVE_TEST_GROUP` | `monorepo-tagbot.yml` | `docs/Project.toml` `[sources]` for `..` and `../lib/*`; uses DocumenterInterLinks ([nls-root], [nls-subci], [nls-docs], [nls-make]) |
| SciML/Optimization.jl | none | root `[sources.OptimizationBase]`; `OptimizationOptimJL` has `OptimizationBase = {path = "../OptimizationBase"}` and `[targets] test` | independent (OptimJL 0.4.23, root 5.9.1) | same SciML reusable workflow | same | ([opt-root], [opt-optimjl]) |
| LuxDL/Lux.jl | `projects = ["test", "docs"]` (l.153–154) | root `[sources]` (l.58–62); `LuxLib` has `[sources]` (l.48–50) **and** its own `[workspace] projects = ["test"]` (l.111–112); `lib/LuxCore/test/Project.toml` has `LuxCore = {path = ".."}` | independent | one workflow per lib → `CommonCI.yml` with `project: lib/X`; develops siblings by hand only on 1.10 ("For 1.11 and beyond we can use sources") | TagBot matrix `subdir: "lib/${{ matrix.package }}"`; `Register.yml` posts `@JuliaRegistrator register subdir=…` automatically when a lib's version is bumped | `docs/Project.toml` `[sources]` for every lib ([lux-root], [lux-luxlib], [lux-luxcore-test], [lux-commonci], [lux-tagbot], [lux-register]) |
| JuliaGPU/CUDA.jl | `projects = ["test", "docs", "perf", "CUDACore", …, "lib/cublas", …]` | root `[sources]`; `lib/cublas` has **no** `[sources]` (the workspace locates siblings) plus a nested `[workspace] projects = ["test"]` | **lockstep** `CUDACore = "=6.5.1"` | Buildkite | TagBot matrix with `release: false`: "Subpackages are versioned with the main package" | ([cuda-root], [cuda-cublas], [cuda-tagbot]) |
| JuliaGPU/GPUArrays.jl | `projects = ["lib/GPUArraysCore", "lib/JLArrays", "test", "docs"]` | workspace only; `JLArrays` has no `[sources]`; CI still runs `Pkg.develop` on the lib paths, since it supports 1.10 | lockstep-ish `GPUArraysCore = "= 0.2.2"` | one job | TagBot `subdir` matrix, `release: false` | docs job develops the paths ([gpuarrays-root], [jlarrays], [gpuarrays-test], [gpuarrays-tagbot]) |
| MakieOrg/Makie.jl | none; there is no root package (root `Project.toml` returns 404) | `CairoMakie` has `[sources.Makie] path = "../Makie"` | **lockstep** `Makie = "=0.24.17"` | `pkg"dev ./Makie ./CairoMakie …"` into a temporary `monorepo` env, then `Pkg.test("CairoMakie")` | TagBot `subdir: Makie`, `tag_prefix: NO_PREFIX`; `CompatHelper.main(subdirs=["Makie","GLMakie",…])` | `docs/Project.toml` `[sources]` for all packages ([makie-cairo], [makie-tagbot], [makie-compat], [makie-ci], [makie-docs]) |
| EnzymeAD/Enzyme.jl, SciML/ModelingToolkit.jl | none | root `[sources]` for `lib/EnzymeCore` and `lib/ModelingToolkitBase` | — | — | — | ([enzyme-root], [mtk-root]) |

**Inference:** shared root workspaces go with lockstep versioning, where everything is always resolved and released together. Independently versioned monorepos (SciML, Lux) give each library its own manifest.

## 3. Facts about Julia, Pkg, the registry and the bots

**`[workspace]`**
- "the requirements of all the projects in the workspace … are recorded in a single manifest file located next to the base project file" ([pkg-toml] l.305).
- Workspaces can be nested, and are then merged into one manifest at the root (l.320–321). They need Julia 1.12+ (l.324).
- Pkg now calls `[extras]`/`[targets]` "legacy". For 1.13+ it recommends a workspace plus `test/Project.toml` ([pkg-toml] l.329–330, [pkg-create] l.297–330).
- In code, `get_compat_workspace` intersects compat over every project: `compat = intersect(compat, get_compat(project, name))` ([pkg-ops] l.956–959).
- `load_workspace_weak_deps` merges the `weakdeps` of all projects ([pkg-types] l.682–689).
- `[compat]` "applies to packages listed under [deps], [weakdeps], and [extras]" ([pkg-toml], "The [compat] section").
- Member packages are located automatically: "the workspace already locates them" ([pkg-types] l.1602–1604).

**Known Pkg bugs when mixing a workspace with per-library `[sources]`**
- In the Pkg v1.14 changelog: #4237, a workspace no longer copies other projects' `[sources]`; and #4830, `Pkg.test` of a member "no longer fails … when the member declares relative path [sources] and its test dependencies come from [extras]/[targets]" ([pkg-changelog] l.50–58).
- The #4830 report says the bug was "Reproduced on Julia 1.12.6, 1.12.7, 1.13.0" ([pkg-4830]).

**`[sources]`**
- Requires Julia 1.11+.
- "If a package is used as a dependency in another project, its [sources] section is **not** consulted" ([pkg-toml] l.203–210). End users never see it.
- My search of `AutoMerge/src` in RegistryCI and of RegistryTools `src` found no reference to `sources` or `workspace` ([am-src], [registrytools]).
- Packages that carry path `[sources]` are registered routinely (OrdinaryDiffEqTsit5 2.1.5, see §1).

**AutoMerge and the General registry**
- The repo-URL rule "Repo URL ends with /PackageName.jl.git" is skipped for subdir packages: `return true, "" # we do not apply this check if the package is a subdirectory package` ([am-src] l.744–749).
- "A copy (not a symbolic link) of the license file needs to be in the package subdirectory" ([general-readme] l.293).
- A new subdir package "will be handled by AutoMerge like any other package" (l.294).
- Waiting periods: "New Julia packages: 3 days", "New versions of existing packages: 15 minutes" (l.76–77). The 3 days count "from the PR being opened" (l.267).
- Guidelines ([am-guidelines]):
  - the package must be installable with `Pkg.add`;
  - versions must be sequential (2.0.0 after 1.x is allowed);
  - every dependency needs an upper-bounded `[compat]` entry;
  - a breaking release needs release notes that mention "breaking" or "changelog".
- General today holds SWG `1.3.1` only ([general-swg]).

**Registrator**
- `@JuliaRegistrator register branch=name-of-your-branch` ([registrator-readme] l.48) and `@JuliaRegistrator register subdir=path/to/my/package` (l.109).
- Keyword arguments are parsed into a `Dict` and duplicates are rejected ([registrator-parse]). So one comment registers one subdir, and `branch=next subdir=lib/X` together is valid.

**TagBot**
- Subdir tags look like `{PACKAGE}-v{VERSION}`, e.g. `SubpackageName-v0.2.3`; the root tag stays `v{VERSION}` ([tagbot-readme] l.444).
- `NO_PREFIX` is only recommended with a single package (l.514–515).
- You need an SSH key if the tag should trigger the Documenter workflow (l.27, l.80–85).

**CompatHelper and Dependabot**
- `CompatHelper.main(; subdirs=["", "Subdir1", …])`, where `""` means the root ([compathelper-options] l.9–13).
- Dependabot has supported Julia since 2025-12-16 ([dependabot-julia]).

**Documenter and DocumenterInterLinks**
- With the default `tag_prefix = ""`, "only version tags will trigger deployment" ([documenter-deploy] l.171–175).
- The monorepo recipe for separate per-package sites uses `dirname` and `tag_prefix` ([documenter-hosting] l.677–745).
- DocumenterInterLinks: pass `InterLinks("Name" => url)` in `plugins` ([interlinks-readme] l.38–46). It needs an `objects.inv`, written by Documenter ≥ 1.3 (l.66).

**Your repos today (local files)**
- SWG `master` already has `[workspace] projects = ["docs", "test"]` (`Project.toml:6-7`).
- The version is already `1.4.0` (l.3), not 3.0.0.
- SPSM is a weak dependency with compat `"4.1"` (l.30, l.55). `docs/Project.toml:55` and `test/Project.toml:40` require SPSM `"4.1"` as a hard dependency.
- `.gitignore` covers `/Manifest.toml` and `docs/Manifest.toml`.
- TagBot is configured for the root only, with `ssh: ${{ secrets.DOCUMENTER_KEY }}`.
- `docs/make.jl:89` calls `deploydocs(; repo="github.com/dmetivie/StochasticWeatherGenerators.jl", devbranch="master")`.
- SPSM `docs/make.jl:111` sets `devbranch = "main"`.

---

## 4(a). Workspace or not: what to keep and what to change

- **Keep "no shared `[workspace]` across the libraries."** It is SciML's practice: `[sources]` only ([ode-root], [ode-tsit5]). It is also Lux's practice: `[sources]` plus a workspace per library ([lux-luxlib]).
- **A root workspace that lists `lib/*` is a legitimate alternative, but a poor fit for SWG (inference).** It is the lockstep pattern (CUDA.jl, GPUArrays.jl). For SWG it would:
  - test every library against one manifest that is the compat intersection of all libraries, the umbrella and the docs ([pkg-ops] l.956–959). CairoMakie or Documenter could then hold back versions that a user installing only `…WGEN` would get;
  - make the per-library Downgrade lane in your plan awkward;
  - expose you on Julia 1.12/1.13 to the workspace + relative `[sources]` + `[targets]` bug fixed only in Pkg 1.14 ([pkg-4830]).
- **The critic's concern is real, but only for the case it describes.** Mechanism: compat intersection plus weak dependencies ([pkg-ops], [pkg-types]). Under caret semantics `"4.1"` means `[4.1.0, 5.0.0)` ([pkg-compat] l.81), so SPSM 4.2 resolves fine and only SPSM 5 is blocked. That happens only if the **legacy** root (`Project.toml:55`, `docs`/`test` `"4.1"`) sits in the same workspace as libraries that need SPSM 5. On your `next` branch the root is already the 2.0 umbrella, so the conflict does not arise as long as `test/reference/` (pinned to SPSM 4.1) stays outside the workspace, as the plan already says.
- **Adjust the test environment.** OrdinaryDiffEqTsit5 uses `[extras]`/`[targets]`, which Pkg now calls legacy. Since SWG requires Julia ≥ 1.12, copy Lux instead: `lib/X/test/Project.toml` plus `[workspace] projects = ["test"]` inside `lib/X`. This also avoids #4830.
- **No dev-path scripts are needed (inference).** On Julia ≥ 1.11, `[sources]` is honoured natively ([sciml-devsrc] header), so `julia --project=lib/X -e 'using Pkg; Pkg.test()'` just works.

## 4(b). Recipes to copy

**Root `Project.toml` (umbrella, on `next`)**
```toml
name = "StochasticWeatherGenerators"
uuid = "3eadb0b9-5057-49d2-9b8f-52a8c334ebf6"
version = "2.0.0"
authors = ["David Métivier <david.metivier@inrae.fr>"]

[workspace]
projects = ["test", "docs"]          # NOT lib/*

[deps]
StochasticWeatherGeneratorsCore = "<uuid>"
StochasticWeatherGeneratorsBlocks = "<uuid>"
StochasticWeatherGeneratorsRegimeHMM = "<uuid>"
StochasticWeatherGeneratorsWGEN = "<uuid>"
StochasticWeatherGeneratorsCensoredGaussian = "<uuid>"
# + CSV, Downloads, Optimization, OptimizationOptimJL, SimpleNonlinearSolve, ForwardDiff, Distributions …

[weakdeps]
Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
GeoMakie = "db073c08-6b98-4ee5-b6a4-5efafb3259c6"
NaturalEarth = "436b0209-26ab-4e65-94a9-6526d86fea76"

[extensions]
MakieExt = "Makie"
GeoMakieExt = ["GeoMakie", "NaturalEarth"]

[sources]
StochasticWeatherGeneratorsCore = {path = "lib/StochasticWeatherGeneratorsCore"}
StochasticWeatherGeneratorsBlocks = {path = "lib/StochasticWeatherGeneratorsBlocks"}
StochasticWeatherGeneratorsRegimeHMM = {path = "lib/StochasticWeatherGeneratorsRegimeHMM"}
StochasticWeatherGeneratorsWGEN = {path = "lib/StochasticWeatherGeneratorsWGEN"}
StochasticWeatherGeneratorsCensoredGaussian = {path = "lib/StochasticWeatherGeneratorsCensoredGaussian"}

[compat]
StochasticWeatherGeneratorsCore = "1"
StochasticWeatherGeneratorsBlocks = "1"
StochasticWeatherGeneratorsRegimeHMM = "1"
StochasticWeatherGeneratorsWGEN = "1"
StochasticWeatherGeneratorsCensoredGaussian = "1"
julia = "1.12"
```
`test/Project.toml` keeps `StochasticWeatherGenerators = {path = ".."}` in `[sources]`, as today.

**One library: `lib/StochasticWeatherGeneratorsBlocks/Project.toml`**
```toml
name = "StochasticWeatherGeneratorsBlocks"
uuid = "<uuid4()>"
version = "1.0.0"
authors = ["David Métivier <david.metivier@inrae.fr>"]

[workspace]
projects = ["test"]                   # its own manifest: lib/…Blocks/Manifest.toml (gitignored)

[deps]
StochasticWeatherGeneratorsCore = "<uuid>"
SmoothPeriodicStatsModels = "3d8e1505-165e-443f-a184-83a230b5a9e5"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
NearestCorrelationMatrix = "59ddf330-608c-4938-8bc9-a4ee97bbbea6"

[sources]
StochasticWeatherGeneratorsCore = {path = "../StochasticWeatherGeneratorsCore"}

[compat]
StochasticWeatherGeneratorsCore = "1"   # raise the lower bound when you start using new Core API
SmoothPeriodicStatsModels = "4.2"       # becomes "4.2, 5" on the cutover branch
Distributions = "0.25"
NearestCorrelationMatrix = "1"
julia = "1.12"
```

`lib/StochasticWeatherGeneratorsBlocks/test/Project.toml`, following the Lux pattern ([lux-luxcore-test]):
```toml
[deps]
StochasticWeatherGeneratorsBlocks = "<uuid>"
StochasticWeatherGeneratorsCore = "<uuid>"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[sources]
StochasticWeatherGeneratorsBlocks = {path = ".."}
StochasticWeatherGeneratorsCore = {path = "../../StochasticWeatherGeneratorsCore"}

[compat]
Test = "1"
```

Also add to each `lib/X`:
- a copy of `LICENSE` (required by [general-readme] l.293);
- a `README.md`.

**`.gitignore`**
```
Manifest*.toml
!test/reference/Manifest.toml
```
The exception is my inference: the legacy-fixture environment must stay pinned.

**`.github/workflows/CI.yml`**
```yaml
name: CI
on:
  push: {branches: [master, next]}
  pull_request:
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ startsWith(github.ref, 'refs/pull/') }}
jobs:
  test:
    name: ${{ matrix.project }} / Julia ${{ matrix.version }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        version: ['1.12', '1']
        project:
          - lib/StochasticWeatherGeneratorsCore
          - lib/StochasticWeatherGeneratorsBlocks
          - lib/StochasticWeatherGeneratorsRegimeHMM
          - lib/StochasticWeatherGeneratorsWGEN
          - lib/StochasticWeatherGeneratorsCensoredGaussian
          - .                                   # umbrella
    steps:
      - uses: actions/checkout@v7
      - uses: julia-actions/setup-julia@v3
        with: {version: "${{ matrix.version }}"}
      - uses: julia-actions/cache@v3
      - uses: julia-actions/julia-buildpkg@v1
        with: {project: "${{ matrix.project }}"}
      - uses: julia-actions/julia-runtest@v1
        with: {project: "${{ matrix.project }}"}
        env: {GROUP: All}
      - uses: julia-actions/julia-processcoverage@v1
        with: {directories: "${{ matrix.project }}/src"}   # it errors on missing dirs such as ext/
      - uses: codecov/codecov-action@v7
        with: {files: lcov.info, token: "${{ secrets.CODECOV_TOKEN }}"}
```
- The `project:` inputs of `julia-buildpkg` and `julia-runtest` are used the same way in [sciml-tests] l.232 and l.244–246.
- The warning about missing coverage directories comes from [sciml-docs] ("julia-processcoverage hard-errors on directories that don't exist").
- **Alternative:** `uses: SciML/.github/.github/workflows/sublibrary-project-tests.yml@v1`. It gives you affected-library detection. However, libraries without a `test_groups.toml` default to `[lts,1,pre]` ([sciml-subproj] header), which is incompatible with `julia = "1.12"`. You would need a `test_groups.toml` with `versions = ["1.12", "1"]` in every library. Being able to call it from another org is an inference.

**Docs**

`docs/Project.toml` (in the root workspace):
```toml
[deps]
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
DocumenterInterLinks = "<uuid>"          # fill with ]add
StochasticWeatherGenerators = "3eadb0b9-5057-49d2-9b8f-52a8c334ebf6"
StochasticWeatherGeneratorsCore = "<uuid>"
# … the other 4 libs, CairoMakie, Literate, DocumenterCitations …

[sources]
StochasticWeatherGenerators = {path = ".."}
StochasticWeatherGeneratorsCore = {path = "../lib/StochasticWeatherGeneratorsCore"}
# … the other 4 libs

[compat]
Documenter = "1"
DocumenterInterLinks = "1"
```

`docs/make.jl`:
```julia
using Documenter, DocumenterInterLinks
using StochasticWeatherGenerators, StochasticWeatherGeneratorsCore, StochasticWeatherGeneratorsBlocks,
      StochasticWeatherGeneratorsRegimeHMM, StochasticWeatherGeneratorsWGEN,
      StochasticWeatherGeneratorsCensoredGaussian
links = InterLinks("SmoothPeriodicStatsModels" => "https://dmetivie.github.io/SmoothPeriodicStatsModels.jl/stable/")
makedocs(; sitename = "StochasticWeatherGenerators.jl",
    modules = [StochasticWeatherGenerators, StochasticWeatherGeneratorsCore, StochasticWeatherGeneratorsBlocks,
               StochasticWeatherGeneratorsRegimeHMM, StochasticWeatherGeneratorsWGEN,
               StochasticWeatherGeneratorsCensoredGaussian],
    checkdocs = :public, plugins = [links], pages = [...])
deploydocs(; repo = "github.com/dmetivie/StochasticWeatherGenerators.jl", devbranch = "master", push_preview = true)
```
The SPSM URL is an inference from the default GitHub Pages pattern ([documenter-hosting] l.571–575) and SPSM `docs/make.jl:111`.

`Documentation.yml`: keep today's file, with two changes:
- `on: push: tags: ['v*']`, so only umbrella tags trigger a build; prefixed library tags cannot deploy anyway ([documenter-deploy] l.171–175);
- the install step becomes `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`. With no manifest it resolves from the project files ([pkg-main] l.703–706), and `[sources]` already points to `..` and `../lib/*`.

**`TagBot.yml`** (CUDA/Lux layout: one job for the root, one matrix job for the libraries; permissions block as in [ode-tagbot])
```yaml
name: TagBot
on:
  issue_comment: {types: [created]}
  workflow_dispatch: {inputs: {lookback: {default: "3"}}}
permissions: {contents: write, issues: read, pull-requests: read, actions: read, checks: read, statuses: read}
jobs:
  TagBot:
    if: github.event_name == 'workflow_dispatch' || github.actor == 'JuliaTagBot'
    runs-on: ubuntu-latest
    steps:
      - uses: JuliaRegistries/TagBot@v1
        with: {token: "${{ secrets.GITHUB_TOKEN }}", ssh: "${{ secrets.DOCUMENTER_KEY }}"}
  TagBot-libs:
    if: github.event_name == 'workflow_dispatch' || github.actor == 'JuliaTagBot'
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        subdir: [lib/StochasticWeatherGeneratorsCore, lib/StochasticWeatherGeneratorsBlocks,
                 lib/StochasticWeatherGeneratorsRegimeHMM, lib/StochasticWeatherGeneratorsWGEN,
                 lib/StochasticWeatherGeneratorsCensoredGaussian]
    steps:
      - uses: JuliaRegistries/TagBot@v1
        with: {token: "${{ secrets.GITHUB_TOKEN }}", ssh: "${{ secrets.DOCUMENTER_KEY }}", subdir: "${{ matrix.subdir }}"}
```
Resulting tags: `v2.0.0` for the umbrella and `StochasticWeatherGeneratorsCore-v1.0.0` for a library ([tagbot-readme] l.444). Add `release: false` if you want tags without GitHub releases, as CUDA and GPUArrays do.

**Dependency bumps.** Choose one of the two.

`CompatHelper.yml` (Makie layout, [makie-compat]):
```julia
CompatHelper.main(; subdirs = ["", "docs", "test",
  "lib/StochasticWeatherGeneratorsCore", "lib/StochasticWeatherGeneratorsBlocks",
  "lib/StochasticWeatherGeneratorsRegimeHMM", "lib/StochasticWeatherGeneratorsWGEN",
  "lib/StochasticWeatherGeneratorsCensoredGaussian"])
```

Or `.github/dependabot.yml` (current SciML choice, [ode-dependabot]):
```yaml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule: {interval: "weekly"}
  - package-ecosystem: "julia"
    directories: ["/", "/docs", "/test", "/lib/*", "/lib/*/test"]
    schedule: {interval: "weekly"}
    groups: {all-julia-packages: {group-by: dependency-name, patterns: ["*"]}}
```

**Registration, first release.** Each line is a separate comment on a commit of `next`, or on an issue ([registrator-readme], [registrator-parse]).
1. In the SPSM repo: `@JuliaRegistrator register` for 4.2.0. AutoMerge in about 15 minutes.
2. `@JuliaRegistrator register branch=next subdir=lib/StochasticWeatherGeneratorsCore`. New package, 3 days.
3. After Core is merged: `@JuliaRegistrator register branch=next subdir=lib/StochasticWeatherGeneratorsBlocks`. 3 days.
4. After Blocks is merged, three comments:
   - `… subdir=lib/StochasticWeatherGeneratorsRegimeHMM`
   - `… subdir=lib/StochasticWeatherGeneratorsWGEN`
   - `… subdir=lib/StochasticWeatherGeneratorsCensoredGaussian`

   These run in parallel, 3 days.
5. Release day:
   - SPSM: `@JuliaRegistrator register` with `Release notes:` mentioning "breaking";
   - a Blocks patch that widens compat to `"4.2, 5"`, then `@JuliaRegistrator register subdir=lib/StochasticWeatherGeneratorsBlocks`. Per the §1 table, only Blocks lists SPSM;
   - merge `next` into `master`, then on the merge commit:
     ```
     @JuliaRegistrator register

     Release notes:

     ## Breaking changes
     - …
     ```
     2.0.0 directly after 1.x is allowed ([am-guidelines]).

The ordering in steps 2–4 is my inference from AutoMerge's "installable" check: a library cannot install until its siblings are in General. You can shorten it by opening all the pull requests on day 0 and re-invoking Registrator once each dependency is merged. The 3-day timer runs "from the PR being opened" ([general-readme] l.267), and re-invoking Registrator updates the existing pull request ([ode-regcomment] reply text).

**Registration, a later release of one library**
1. Bump `version` in `lib/X/Project.toml` (and sibling lower bounds if you use new sibling API), then merge.
2. Comment `@JuliaRegistrator register subdir=lib/X` on the merge commit.
3. AutoMerge in about 15 minutes, then TagBot creates `X-v1.1.0`.

Lux automates step 2 with a workflow that comments when a library's version changes ([lux-register]).

## 4(c). Operational cost

| Task | Cost |
|---|---|
| First release | About 9 comments: SPSM 4.2, 5 new libraries, SPSM 5.0, the Blocks widening patch, umbrella 2.0.0. Each new library waits 3 days; a strict sequence takes about 9 days, which fits the plan's "≥4 weeks before" ([general-readme] l.76). |
| One-library release | 1 version bump, 1 comment, about 15 minutes, automatic tag. If the release is breaking: release notes mentioning "breaking", plus an umbrella release to widen its compat (2 comments). |
| Docs | One site. `dev` redeploys on every push to master. `stable` moves only on umbrella `v*` tags, so a library-only release does not refresh `stable` (inference from [documenter-deploy] l.171–175). Separate per-library sites with `dirname`/`tag_prefix` are possible ([documenter-hosting] l.677–745) but are not worth it at launch. |
| CI | 6 projects × 2 Julia versions = 12 jobs, plus docs. No dev scripts. |
| One-time setup | A `LICENSE` copy per library. One `DOCUMENTER_KEY` deploy key, already in use, so that TagBot tags trigger docs builds ([tagbot-readme] l.27). |

## Two edits to the plan's §1
1. Replace "its own test environment, as OrdinaryDiffEqTsit5 does" with: "its own `test/Project.toml` and a library-local `[workspace] projects = ["test"]`, as Lux.jl does (OrdinaryDiffEqTsit5 uses the legacy `[targets]`)".
2. Replace "A workspace exists only for the umbrella's test/ and docs/, and only after the cutover" with: "The root workspace is `["test", "docs"]`. It already exists on master (`Project.toml:6-7`) and is valid on `next` from day one. `lib/*` and `test/reference` are never members."

Side note: the plan's "Versions" paragraph says master is labelled 3.0.0, but `Project.toml:3` is already `1.4.0`.

[ode-root]: https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/Project.toml
[ode-tsit5]: https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/lib/OrdinaryDiffEqTsit5/Project.toml
[ode-core]: https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/lib/OrdinaryDiffEqCore/Project.toml
[ode-tsit5-run]: https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/lib/OrdinaryDiffEqTsit5/test/runtests.jl
[ode-tsit5-groups]: https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/lib/OrdinaryDiffEqTsit5/test/test_groups.toml
[ode-subci]: https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/.github/workflows/SublibraryCI.yml
[ode-tagbot]: https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/.github/workflows/TagBot.yml
[ode-dependabot]: https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/.github/dependabot.yml
[ode-docs-proj]: https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/docs/Project.toml
[ode-make]: https://raw.githubusercontent.com/SciML/OrdinaryDiffEq.jl/master/docs/make.jl
[ode-regcomment]: https://github.com/SciML/OrdinaryDiffEq.jl/commit/8609fa92241e7e4368b92d9212ceb9d3cf37ad4d#commitcomment-208006709
[sciml-subproj]: https://raw.githubusercontent.com/SciML/.github/v1/.github/workflows/sublibrary-project-tests.yml
[sciml-tests]: https://raw.githubusercontent.com/SciML/.github/v1/.github/workflows/tests.yml
[sciml-devsrc]: https://raw.githubusercontent.com/SciML/.github/v1/scripts/develop_sources.jl
[sciml-monotag]: https://raw.githubusercontent.com/SciML/.github/v1/.github/workflows/monorepo-tagbot.yml
[sciml-tagbot]: https://raw.githubusercontent.com/SciML/.github/v1/.github/workflows/tagbot.yml
[sciml-docs]: https://raw.githubusercontent.com/SciML/.github/v1/.github/workflows/documentation.yml
[scimldocs]: https://raw.githubusercontent.com/SciML/SciMLDocs/main/docs/Project.toml
[nls-root]: https://raw.githubusercontent.com/SciML/NonlinearSolve.jl/master/Project.toml
[nls-subci]: https://raw.githubusercontent.com/SciML/NonlinearSolve.jl/master/.github/workflows/SublibraryCI.yml
[nls-docs]: https://raw.githubusercontent.com/SciML/NonlinearSolve.jl/master/docs/Project.toml
[nls-make]: https://raw.githubusercontent.com/SciML/NonlinearSolve.jl/master/docs/make.jl
[opt-root]: https://raw.githubusercontent.com/SciML/Optimization.jl/master/Project.toml
[opt-optimjl]: https://raw.githubusercontent.com/SciML/Optimization.jl/master/lib/OptimizationOptimJL/Project.toml
[lux-root]: https://raw.githubusercontent.com/LuxDL/Lux.jl/main/Project.toml
[lux-luxlib]: https://raw.githubusercontent.com/LuxDL/Lux.jl/main/lib/LuxLib/Project.toml
[lux-luxcore-test]: https://raw.githubusercontent.com/LuxDL/Lux.jl/main/lib/LuxCore/test/Project.toml
[lux-commonci]: https://raw.githubusercontent.com/LuxDL/Lux.jl/main/.github/workflows/CommonCI.yml
[lux-tagbot]: https://raw.githubusercontent.com/LuxDL/Lux.jl/main/.github/workflows/TagBot.yml
[lux-register]: https://raw.githubusercontent.com/LuxDL/Lux.jl/main/.github/workflows/Register.yml
[cuda-root]: https://raw.githubusercontent.com/JuliaGPU/CUDA.jl/master/Project.toml
[cuda-cublas]: https://raw.githubusercontent.com/JuliaGPU/CUDA.jl/master/lib/cublas/Project.toml
[cuda-tagbot]: https://raw.githubusercontent.com/JuliaGPU/CUDA.jl/master/.github/workflows/TagBot.yml
[gpuarrays-root]: https://raw.githubusercontent.com/JuliaGPU/GPUArrays.jl/master/Project.toml
[jlarrays]: https://raw.githubusercontent.com/JuliaGPU/GPUArrays.jl/master/lib/JLArrays/Project.toml
[gpuarrays-test]: https://raw.githubusercontent.com/JuliaGPU/GPUArrays.jl/master/.github/workflows/Test.yml
[gpuarrays-tagbot]: https://raw.githubusercontent.com/JuliaGPU/GPUArrays.jl/master/.github/workflows/TagBot.yml
[makie-cairo]: https://raw.githubusercontent.com/MakieOrg/Makie.jl/master/CairoMakie/Project.toml
[makie-tagbot]: https://raw.githubusercontent.com/MakieOrg/Makie.jl/master/.github/workflows/TagBot.yml
[makie-compat]: https://raw.githubusercontent.com/MakieOrg/Makie.jl/master/.github/workflows/CompatHelper.yml
[makie-ci]: https://raw.githubusercontent.com/MakieOrg/Makie.jl/master/.github/workflows/_cairomakie.yml
[makie-docs]: https://raw.githubusercontent.com/MakieOrg/Makie.jl/master/docs/Project.toml
[enzyme-root]: https://raw.githubusercontent.com/EnzymeAD/Enzyme.jl/main/Project.toml
[mtk-root]: https://raw.githubusercontent.com/SciML/ModelingToolkit.jl/master/Project.toml
[pkg-toml]: https://raw.githubusercontent.com/JuliaLang/Pkg.jl/master/docs/src/toml-files.md
[pkg-create]: https://raw.githubusercontent.com/JuliaLang/Pkg.jl/master/docs/src/creating-packages.md
[pkg-compat]: https://raw.githubusercontent.com/JuliaLang/Pkg.jl/master/docs/src/compatibility.md
[pkg-ops]: https://raw.githubusercontent.com/JuliaLang/Pkg.jl/master/src/Operations.jl
[pkg-types]: https://raw.githubusercontent.com/JuliaLang/Pkg.jl/master/src/Types.jl
[pkg-main]: https://raw.githubusercontent.com/JuliaLang/Pkg.jl/master/src/Pkg.jl
[pkg-changelog]: https://raw.githubusercontent.com/JuliaLang/Pkg.jl/master/CHANGELOG.md
[pkg-4830]: https://github.com/JuliaLang/Pkg.jl/pull/4830
[am-guidelines]: https://juliaregistries.github.io/RegistryCI.jl/dev/guidelines/
[am-src]: https://raw.githubusercontent.com/JuliaRegistries/RegistryCI.jl/master/AutoMerge/src/guidelines.jl
[registrytools]: https://github.com/JuliaRegistries/RegistryTools.jl
[general-readme]: https://raw.githubusercontent.com/JuliaRegistries/General/master/README.md
[general-tsit5]: https://raw.githubusercontent.com/JuliaRegistries/General/master/O/OrdinaryDiffEqTsit5/Package.toml
[general-swg]: https://raw.githubusercontent.com/JuliaRegistries/General/master/S/StochasticWeatherGenerators/Versions.toml
[registrator-readme]: https://raw.githubusercontent.com/JuliaRegistries/Registrator.jl/master/README.md
[registrator-parse]: https://raw.githubusercontent.com/JuliaRegistries/Registrator.jl/master/src/commentbot/parse_comment.jl
[tagbot-readme]: https://raw.githubusercontent.com/JuliaRegistries/TagBot/master/README.md
[compathelper-options]: https://raw.githubusercontent.com/JuliaRegistries/CompatHelper.jl/master/docs/src/options.md
[dependabot-julia]: https://github.blog/changelog/2025-12-16-dependabot-version-updates-now-support-julia/
[documenter-deploy]: https://raw.githubusercontent.com/JuliaDocs/Documenter.jl/master/src/deploydocs.jl
[documenter-hosting]: https://raw.githubusercontent.com/JuliaDocs/Documenter.jl/master/docs/src/man/hosting.md
[interlinks-readme]: https://raw.githubusercontent.com/JuliaDocs/DocumenterInterLinks.jl/master/README.md

### Critical Files for Implementation
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\Project.toml
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\docs\Project.toml
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\docs\make.jl
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\.github\workflows\TagBot.yml
- c:\Users\metivier\Dropbox\PC (2)\Documents\dev\StochasticWeatherGenerators\.gitignore