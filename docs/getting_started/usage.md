# Usage

Explore the **tutorials** in the [`cloe-org/playground/tutorials/likelihood`](https://github.com/cloe-org/playground/tutorials/likelihood) repository for examples on how to compute cosmological observables and use `cloelike`.

## Basic example: BAO likelihood

```python
import numpy as np
from cloelike.EuclidLikelihood_GCspectro_BAO import EuclidLikelihood_GCspectro_BAO
from cloelib.cosmology.CAMBcosmology import CAMBBackground  # or your preferred Background class

# Load your data dictionary (e.g. from a .pkl or .fits file)
data = {...}  # Check out the expected format in notebooks

# Instantiate the likelihood
likelihood = EuclidLikelihood_GCspectro_BAO(data=data, Background=CAMBBackground)

# Evaluate the log-likelihood for a set of cosmological parameters
parameters = {
    "H0": 67.32,
    "Omega_cdm0": 0.264,
    "Omega_b0": 0.049,
    "Omega_k0": 0.0,
    "w0": -1.0,
    "wa": 0.0,
    "ns": 0.966,
    "sigma8": 0.816,
}
log_like = likelihood.loglike(parameters)
```

## Basic example: Photometric 3×2pt likelihood

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations
from cloelike.EuclidLikelihood_photo_Cls import EuclidLikelihood_3x2pt

# Load data and settings dictionaries
data = {...}
settings = {...}

# Provide Background, linear and non-linear perturbation classes
likelihood = EuclidLikelihood_3x2pt(
    data=data,
    settings=settings,
    Background=CAMBBackground,
    LinPerturbations=HMemuLinearPerturbations,
    NonLinPerturbations=HMemuNonLinearPerturbations,
)

log_like = likelihood.loglike(parameters)
```

## Basic example: Spectroscopic GCspectro likelihood

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.CometEFT_spectro import CometEFT_SpectroPower
from cloelike.EuclidLikelihood_GCspectro_Pls import EuclidLikelihood_GCspectro_Pls

likelihood = EuclidLikelihood_GCspectro_Pls(
    data=data,
    settings=settings,
    Background=CAMBBackground,
    SpectroPower=CometEFT_SpectroPower,
)

log_like = likelihood.loglike(parameters)
```
