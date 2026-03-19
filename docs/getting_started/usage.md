# Usage

Explore the **tutorials** in the [`cloe-org/playground`](https://github.com/cloe-org/playground) repository for examples on how to compute cosmological observables and use `cloelike`.

## Basic example: BAO likelihood

```python
import numpy as np
from cloelike import EuclidLikelihood_BAO
from cloelib.cosmology.cosmology import Background  # or your preferred Background class

# Load your data dictionary (e.g. from a .pkl or .fits file)
data = {...}  # must contain a "BAO" key with the BAO measurements

# Instantiate the likelihood
likelihood = EuclidLikelihood_BAO(data=data, Background=Background)

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
from cloelike import EuclidLikelihood_3x2pt

# Load data and settings dictionaries
data = {...}
settings = {...}

# Provide Background, linear and non-linear perturbation classes
likelihood = EuclidLikelihood_3x2pt(
    data=data,
    settings=settings,
    Background=Background,
    LinPerturbations=LinPerturbations,
    NonLinPerturbations=NonLinPerturbations,
)

log_like = likelihood.loglike(parameters)
```

## Basic example: Spectroscopic GCspectro likelihood

```python
from cloelike import EuclidLikelihood_GCspectro_Pls

likelihood = EuclidLikelihood_GCspectro_Pls(
    data=data,
    settings=settings,
    Background=Background,
    SpectroPower=SpectroPower,
)

log_like = likelihood.loglike(parameters)
```
