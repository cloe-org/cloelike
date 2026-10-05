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

### Selecting the photometric theory model

The intrinsic-alignment model, galaxy bias model and nonlinear/baryonic options are chosen in `settings`. All keys are optional, and the photometric Cls and 2PCF likelihoods accept the same ones.

| Key                     | Options                                                       | Parameters read from `parameters`                                                                                      |
| ----------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `ia_model`              | `"NLA"` (default), `"TATT"`, `None`                           | NLA: `AIA`, `EtaIA`, `CIA`. TATT: `AIA`, `A2IA`, `bTA`, optionally `EtaIA`, `Eta2IA`, `CIA`, `z0IA` (default 0.62)     |
| `galaxy_bias_model`     | `"poly"` (default), `"per_bin"`, `"per_bin_int"`, `"nonlinear"` | poly: `b1_photo_poly0..3`. per_bin: `b1_photo_bin{i}`. nonlinear: `b1_photo_nl_bin{i}`, optionally `b2_`, `bs2_`, `b3nl_`, `bk2_photo_nl_bin{i}` (`i` starts at 0) |
| `include_rsd`           | `False` (default), `True`                                     | Not supported with `galaxy_bias_model="nonlinear"`                                                                     |
| `nonlinear_param_keys`  | default `("log10TAGN",)`                                      | Passed as keyword arguments to `NonLinPerturbations`. Use `()` for backends without baryonic feedback (e.g. EE2)       |
| `baryon_param_keys`     | default `()`                                                  | Passed as `baryon_kwargs` to a class built with `cloelib.cosmology.cosmology.with_baryon_boost`                        |
| `nonlinear_kwargs`      | default `{}`                                                  | Fixed keyword arguments for `NonLinPerturbations`, e.g. `{"nonlinear_model": "mead2020"}`                              |

TATT and the nonlinear galaxy bias compute their one-loop kernels with FAST-PT (`pip install fast-pt`) from the linear perturbations. You can supply your own kernels with `settings["tatt_loop_computer"]` / `settings["nl_bias_loop_computer"]`: each is a callable that takes the linear perturbations and returns an object with a `.compute(name)` method.

For example, TATT with nonlinear galaxy bias and the FLAMINGO baryonic response on top of HMcode2020Emu:

```python
from cloelib.cosmology.cosmology import with_baryon_boost
from cloelib.cosmology.FlamingoBaryonResponseEmulator_cosmology import FlamingoBaryonBoostMixin

settings = {
    **settings,
    "ia_model": "TATT",
    "galaxy_bias_model": "nonlinear",
    "nonlinear_param_keys": (),
    "baryon_param_keys": ("fgas_sigma", "Mstar_sigma", "jet_fraction"),
}
likelihood = EuclidLikelihood_3x2pt(
    data=data,
    settings=settings,
    Background=CAMBBackground,
    LinPerturbations=HMemuLinearPerturbations,
    NonLinPerturbations=with_baryon_boost(HMemuNonLinearPerturbations, FlamingoBaryonBoostMixin),
)
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
