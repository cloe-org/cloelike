<p align="center">
  <img src="cloelike-banner.png" alt="cloelike logo" width="650">
</p>

**cloelike** is the likelihood module for _Euclid_ primary observables, interfacing with `cloelib`.

## ✨ Features

🔹 **loglike calculation for Euclid primary probes** – currently supporting 3×2pt and spectroscopic galaxy clustering full-shape and BAOs

🔹 **Modular design** – Mix-and-match photometric probes (WL, GCph, GGL) using composable mixin classes

---

## 📋 Available Likelihoods

| Class                            | Observable                                                            |
| -------------------------------- | --------------------------------------------------------------------- |
| `EuclidLikelihood_WL`            | Weak Lensing (WL) angular power spectra                               |
| `EuclidLikelihood_GCph`          | Photometric Galaxy Clustering (GCph) angular power spectra            |
| `EuclidLikelihood_GGL`           | Galaxy–Galaxy Lensing (GGL) angular power spectra                     |
| `EuclidLikelihood_3x2pt`         | 3×2pt (WL + GCph + GGL) combined                                      |
| `EuclidLikelihood_2x2pt`         | 2×2pt (WL + GGL) combined                                             |
| `EuclidLikelihood_GCspectro_Pls` | Spectroscopic Galaxy Clustering (GCspectro) power spectrum multipoles |
| `EuclidLikelihood_GCspectro_BAO` | Baryon Acoustic Oscillations (BAO)                                    |

---

--8<-- "README.md:contributors"

---

## ⚖️ License

This project is licensed under the [MIT License](https://github.com/cloe-org/cloelib?tab=MIT-1-ov-file).
