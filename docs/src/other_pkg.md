# 🌐 Other SWG Packages

Many other SWG packages, Git repositories, and codebases exist — though detailed documentation is often lacking.
Below is a **far from exhaustive list**[^1] that we (the authors and contributors) will aim to complete over time.

[^1]: Focusing on open source code only.

!!! note
    If you know of a useful model with code not listed here, feel free to suggest it or contribute via Pull Request.

```@raw html
<table>
  <thead>
    <tr>
      <th>Logo</th>
      <th>Name</th>
      <th>Language</th>
      <th>Type</th>
      <th>Description</th>
      <th>Paper</th>
      <th>Comments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="https://sobakrim.github.io/MSTWeatherGen/reference/figures/MSTWeatherGen.png" alt="MSTWeatherGenLogo" width="40px"/></td>
      <td><a href="https://sobakrim.github.io/MSTWeatherGen/index.html">MSTWeatherGen</a></td>
      <td>
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/r/r-original.svg" alt="R" width="40px"/>
      </td>
      <td>Package</td>
      <td>
        <div class="foldable-wrapper">
          <div class="foldable">Spatial, multivariate</div>
          <button class="toggle-button" onclick="toggleFold(this)">Show more</button>
        </div>
      </td>
      <td><a href="https://hal.science/hal-04534990">Obakrim et al. (2024)</a></td>
      <td>
        <div class="foldable-wrapper">
          <div class="foldable">Documentation</div>
          <button class="toggle-button" onclick="toggleFold(this)">Show more</button>
        </div>
      </td>
    </tr>

    <tr>
      <td>—</td>
      <td><a href="https://github.com/guillaumeevin/GWEX">GWEX</a></td>
      <td>
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/r/r-original.svg" alt="R" width="40px"/>
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/c/c-original.svg" alt="C++" width="40px"/>
      </td>
      <td>Package</td>
      <td>
        <div class="foldable-wrapper">
          <div class="foldable">Multisite Precipitation and Temperature</div>
          <button class="toggle-button" onclick="toggleFold(this)">Show more</button>
        </div>
      </td>
      <td><a href="https://hess.copernicus.org/articles/22/655/2018/">Evin et al. (2018)</a></td>
      <td>—</td>
    </tr>

    <tr>
      <td>—</td>
      <td><a href="https://arve-research.github.io/gwgen/getting_started.html">GWGEN</a></td>
      <td>
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg" alt="Python" width="40px"/>
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/fortran/fortran-original.svg" alt="Fortran" width="40px"/>
      </td>
      <td>Package</td>
      <td>
        <div class="foldable-wrapper">
          <div class="foldable">Globally applicable weather generator inspired by the original Richardson model (1981).</div>
          <button class="toggle-button" onclick="toggleFold(this)">Show more</button>
        </div>
      </td>
      <td><a href="https://gmd.copernicus.org/articles/10/3771/2017/">Sommer et al. (2017)</a></td>
      <td>
        <div class="foldable-wrapper">
          <div class="foldable">Sort of Documentation. Not sure what "globally applicable" means.</div>
          <button class="toggle-button" onclick="toggleFold(this)">Show more</button>
        </div>
      </td>
    </tr>

    <tr>
      <td><img src="https://fr.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/29136/versions/7/screenshot.jpg" alt="WeaGETS" width="40px"/></td>
      <td><a href="https://fr.mathworks.com/matlabcentral/fileexchange/29136-stochastic-weather-generator-weagets">WeaGETS</a></td>
      <td><img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/matlab/matlab-original.svg" alt="Matlab" width="40px"/></td>
      <td>Package</td>
      <td>
        <div class="foldable-wrapper">
          <div class="foldable">Multivariate Markov/AR based SWG for `RR`, `TX` and `TN`</div>
          <button class="toggle-button" onclick="toggleFold(this)">Show more</button>
        </div>
      </td>
      <td><a href="https://www.sciencedirect.com/science/article/pii/S1878029612002125">Feng et al. (2012)</a></td>
      <td>—</td>
    </tr>

    <tr>
      <td><img src="https://hyd.ifu.ethz.ch/research-data-models/awe-gen-2d/_jcr_content/par/fullwidthimage/image.imageformat.fullwidthwidepage.829715501.jpg" alt="awegen2d" width="40px"/></td>
      <td><a href="https://zenodo.org/records/10525023">AWE-GEN-2d</a></td>
      <td><img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/matlab/matlab-original.svg" alt="Matlab" width="40px"/></td>
      <td>Code</td>
      <td>
        <div class="foldable-wrapper">
          <div class="foldable">AWE-GEN-2d is an advanced stochastic weather generator that combines physical and stochastic approaches to simulate multiple variables like precipitation, temperature, radiation, and cloud cover at high spatial (2 km) and temporal (5 min) resolution. It has been developed to support complex hydrological simulations under changing climate scenarios.</div>
          <button class="toggle-button" onclick="toggleFold(this)">Show more</button>
        </div>
      </td>
      <td><a href="https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016MS000854">Peleg et al. (2024)</a></td>
      <td>
        <div class="foldable-wrapper">
          <div class="foldable">Other models available from the author <a href="https://wp.unil.ch/hydmet/team/peleg/">Nadav Peleg</a>.</div>
          <button class="toggle-button" onclick="toggleFold(this)">Show more</button>
        </div>
      </td>
    </tr>
  </tbody>
</table>

<script>
  function toggleFold(button) {
    const foldable = button.previousElementSibling;
    foldable.classList.toggle("expanded");
    button.textContent = foldable.classList.contains("expanded") ? "Show less" : "Show more";
  }

  function checkOverflow(el) {
    return el.scrollHeight > el.clientHeight;
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".foldable-wrapper").forEach(wrapper => {
      const foldable = wrapper.querySelector(".foldable");
      const button = wrapper.querySelector(".toggle-button");
      if (checkOverflow(foldable)) {
        wrapper.classList.add("show-toggle");
      } else {
        button.style.display = "none";
      }
    });
  });
</script>
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
