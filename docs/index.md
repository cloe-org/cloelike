# cloelike

**cloelike** is the likelihood module for _Euclid_ primary observables, interfacing with `cloelib`.

[![CI](https://github.com/cloe-org/cloelike/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cloe-org/cloelike/actions/workflows/ci.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-blue?logo=pytest)](https://docs.pytest.org/)
[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-purple?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)

---

## ✨ Features

🔹 **Intuitive & User-Friendly** – General description of the likelihood options by classes

🔹 **loglike calculation for Euclid primary probes** – currently supporting 3×2pt and spectroscopic galaxy clustering full-shape

🔹 **Modular design** – Mix-and-match photometric probes (WL, GCph, GGL) using composable mixin classes

---

## 🚀 Quick Start

Install `cloelike` from source:

```sh
pip install git+https://github.com/cloe-org/cloelike.git
```

Then compute a log-likelihood:

```python
from cloelike import EuclidLikelihood_BAO

likelihood = EuclidLikelihood_BAO(data=data, Background=BackgroundClass)
log_like = likelihood.loglike(parameters)
```

See the [Getting Started](getting_started/installation.md) guide for full installation instructions, and the [API Reference](api/index.md) for detailed documentation.

---

## 📋 Available Likelihoods

| Class | Observable |
|---|---|
| `EuclidLikelihood_WL` | Weak Lensing (WL) angular power spectra |
| `EuclidLikelihood_GCph` | Photometric Galaxy Clustering (GCph) angular power spectra |
| `EuclidLikelihood_GGL` | Galaxy–Galaxy Lensing (GGL) angular power spectra |
| `EuclidLikelihood_3x2pt` | 3×2pt (WL + GCph + GGL) combined |
| `EuclidLikelihood_2x2pt` | 2×2pt (WL + GGL) combined |
| `EuclidLikelihood_GCspectro_Pls` | Spectroscopic Galaxy Clustering (GCspectro) power spectrum multipoles |
| `EuclidLikelihood_BAO` | Baryon Acoustic Oscillations (BAO) |

---

## 🙏 Acknowledgements

👩‍💻🧑‍💻 Authored by M. Bonici, G. Cañas-Herrera, P. Carrilho, S. Casas, C. Moretti, and A. Pezzotta (listed in alphabetical order).
