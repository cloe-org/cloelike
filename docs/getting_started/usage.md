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

The intrinsic-alignment model, galaxy bias model and nonlinear/baryonic options are chosen in `settings`. All keys are optional. The photometric Cls, 2PCF and COSEBIs likelihoods accept the same ones (COSEBIs is shear only, so it ignores the galaxy-bias keys).

| Key                    | Options                                                         | Parameters read from `parameters`                                                                                                                                  |
| ---------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ia_model`             | `"NLA"` (default), `"TATT"`, `None`                             | NLA: `AIA`, `EtaIA`, `CIA`. TATT: `AIA`, `A2IA`, `bTA`, optionally `EtaIA`, `Eta2IA`, `CIA`, `z0IA` (default 0.62)                                                 |
| `galaxy_bias_model`    | `"poly"` (default), `"per_bin"`, `"per_bin_int"`, `"nonlinear"`, `"hod"` | hod: none. poly: `b1_photo_poly0..3`. per*bin: `b1_photo_bin{i}`. nonlinear: `b1_photo_nl_bin{i}`, optionally `b2*`, `bs2*`, `b3nl*`, `bk2_photo_nl_bin{i}` (`i` starts at 0) |
| `include_rsd`          | `False` (default), `True`                                       | Not supported with `galaxy_bias_model="nonlinear"`                                                                                                                 |
| `nonlinear_param_keys` | default `("log10TAGN",)`                                        | Passed as keyword arguments to `NonLinPerturbations`. Use `()` for backends without baryonic feedback (e.g. EE2)                                                   |
| `baryon_param_keys`    | default `()`                                                    | Passed as `baryon_kwargs` to a class built with `cloelib.cosmology.cosmology.with_baryon_boost`                                                                    |
| `nonlinear_kwargs`     | default `{}`                                                    | Fixed keyword arguments for `NonLinPerturbations`, e.g. `{"nonlinear_model": "mead2020"}`                                                                          |
| `shear_tracer_kwargs`, `positions_tracer_kwargs` | default `{}`                          | Extra keyword arguments for `ShearTracer` / `PositionsTracer`, e.g. `{"spin2_prefactor": False}` |

`galaxy_bias_model="hod"` takes the bias from halo-model galaxy spectra and reads no bias parameters; it needs a halo-model `NonLinPerturbations` (GODMAX, see below).

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

## tSZ × photometric likelihoods (GODMAX)

`cloelike.EuclidLikelihood_tsz` provides the thermal Sunyaev-Zel'dovich (Compton-y) blocks and their crosses with the Euclid photometric probes, computed with the GODMAX halo model of cloelib (`GODMAXNonLinearPerturbations`):

| Class                         | Blocks (data-vector order)                   |
| ----------------------------- | -------------------------------------------- |
| `EuclidLikelihood_tSZxtSZ`    | Y×Y                                          |
| `EuclidLikelihood_tSZxGC`     | POS×Y                                        |
| `EuclidLikelihood_tSZxWL`     | SHE×Y (E-mode)                               |
| `EuclidLikelihood_tSZx`       | SHE×Y, POS×Y, Y×Y                            |
| `EuclidLikelihood_3x2pt_tSZx` | SHE×Y, POS×Y, Y×Y, WL, GGL, GCph             |
| `EuclidLikelihood_3x2pt_GODMAX` | WL, GGL, GCph (3x2pt on the halo model)    |

The data keys are `("Y", "Y", 1, 1)`, `("POS", "Y", i, 1)` and `("SHE", "Y", i, 1)` (shape `(2, n_ells)`, E-mode first). The tSZ blocks run only with `mode="decoupled"` for now: cloelib cannot apply mixing matrices to Compton-y pairs yet, so the data must be decoupled (mask-corrected or full-sky) Cls at `data["ells"]`.

Install GODMAX from GitHub (it is not on PyPI):

```sh
pip install "godmax @ git+https://github.com/vilasinits/GODMAX.git@cloe-integration"
```

```python
import numpy as np
from cloelib.cosmology.jax_cosmology import JAXBackground, JAXLinearPerturbations
from cloelib.cosmology.godmax_cosmology import GODMAXNonLinearPerturbations
from cloelike import EuclidLikelihood_3x2pt_tSZx

settings = {
    "scale_cuts": scale_cuts,  # one [ell_min, ell_max] per data key
    # Sampled GODMAX halo-model parameters (the default ("log10TAGN",) is not one).
    "nonlinear_param_keys": ("theta_ej_0", "log10_Mc0"),
    "nonlinear_kwargs": {"model_tSZ": True, "galaxies": {"nbar": 5e-4}},
    # Sampled HOD parameters, merged into nonlinear_kwargs["galaxies"]["hod"].
    "hod_param_keys": ("log10M1_fshmr",),
    # Halo-model redshift nodes (default: data["z_arr"]); about 50 up to z = 4 is enough.
    "nonlinear_redshifts": np.linspace(0.05, 3.0, 50),
    # "godmax" (HOD, pivot NLA, no magnification/RSD/photo-z, convergence, 1.4' y beam)
    # or "baseline"; overrides as {"preset": "godmax", "rsd": True}.
    "kernel_convention": "godmax",
}
likelihood = EuclidLikelihood_3x2pt_tSZx(
    data=data,
    settings=settings,
    Background=JAXBackground,
    LinPerturbations=JAXLinearPerturbations,
    NonLinPerturbations=GODMAXNonLinearPerturbations,
    mode="decoupled",
)
```

The kernel convention sets `ia_model`, `galaxy_bias_model`, `include_rsd`, the shear `spin2_prefactor` and `tsz_beam_fwhm_arcmin`; giving one of them with a different value raises. Without a convention these settings are used as given. Notes:

- GODMAX is flat w0CDM without massive neutrinos: use `Omega_k0 = 0`, `wa = 0`, `mnu = 0` (and `N_mnu = 0`).
- TATT and `galaxy_bias_model="nonlinear"` are not available on the halo model and raise at construction.
- The `alpha_*` 1-halo/2-halo transition parameters cannot be JAX-traced.
- BNT is not available with the tSZ blocks.

## Basic example: Spectroscopic GCspectro likelihood

```python
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.spectro.CometEFT_spectro import CometEFT_SpectroPower
from cloelike.EuclidLikelihood_GCspectro_Pls import EuclidLikelihood_GCspectro_Pls

likelihood = EuclidLikelihood_GCspectro_Pls(
    data=data,
    settings=settings,
    Background=CAMBBackground,
    SpectroPower=CometEFT_SpectroPower,
)

log_like = likelihood.loglike(parameters)
```
