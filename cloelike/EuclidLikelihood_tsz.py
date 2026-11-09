"""
tSZ (Compton-y) x Euclid photometric likelihoods on the GODMAX halo model.

The Compton-y spectra come from cloelib's
`cloelib.cosmology.godmax_cosmology.GODMAXNonLinearPerturbations`, so every class
here needs it as `NonLinPerturbations`, with `JAXBackground` and
`JAXLinearPerturbations`. GODMAX is flat w0CDM without massive neutrinos: the
parameters must have `Omega_k0 = 0`, `wa = 0` and `mnu = 0`.

The tSZ blocks only support `mode="decoupled"`: cloelib's `get_pseudo_Cl` has no
mixing-matrix rule for Compton-y yet, so the data must be Cls with the mask
effect removed (or full-sky), at `data["ells"]`.

Data keys (in `data["cells"]` and `settings["scale_cuts"]`):

- `("Y", "Y", 1, 1)`: array over ells;
- `("POS", "Y", i, 1)`: array over ells, i = 1..n_pos_bins;
- `("SHE", "Y", i, 1)`: shape `(2, n_ells)`, E-mode first, i = 1..n_she_bins.

Settings read by `GODMAXMixin`, on top of those of `PhotoLikelihoodBase`:

- `nonlinear_param_keys`: sampled GODMAX halo-model parameters, e.g.
  `("theta_ej_0", "log10_Mc0")`. The base default `("log10TAGN",)` is not a
  GODMAX parameter, so set it (use `()` to sample none).
- `nonlinear_kwargs`: fixed GODMAX options, e.g.
  `{"model_tSZ": True, "galaxies": {"nbar": 1e-3, "hod": {...}}}`. `galaxies`
  is needed for `galaxy_bias_model="hod"`.
- `hod_param_keys`: sampled HOD parameters (scalar names from cloelib's
  `godmax_cosmology.HOD_PARAMETERS`), merged into
  `nonlinear_kwargs["galaxies"]["hod"]`. Default `()`.
- `nonlinear_redshifts`: halo-model redshift nodes. Default `data["z_arr"]`.
  GODMAX's cost grows with the number of nodes (about 50 up to z = 4 is
  enough); they must cover `data["z_arr"]`.
- `kernel_convention`: `"godmax"`, `"baseline"`, or
  `{"preset": ..., **overrides}` (see cloelib's
  `cloelib.observables.kernel_conventions`). It sets `ia_model`,
  `galaxy_bias_model`, `include_rsd`, the shear `spin2_prefactor` and the y
  beam, and switches off the systematics the convention leaves out. Default
  None (the settings are used as given).
- `tsz_beam_fwhm_arcmin`: FWHM of the y-map beam in arcmin. Default 0 (none).
"""

import dataclasses
import functools

import numpy as np
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint

from cloelike.EuclidLikelihood_photo_base import PhotoLikelihoodBase
from cloelike.EuclidLikelihood_photo_Cls import GCphMixin, GGLMixin, WLMixin

# Settings that a kernel convention sets. Each is either taken from the
# convention or must agree with it.
_CONVENTION_SETTINGS = ("ia_model", "galaxy_bias_model", "include_rsd")


def _build_kernel_convention(spec):
    """Build a cloelib `KernelConvention` from a preset name or a dict."""
    # Imported here so cloelike still imports with cloelib versions without GODMAX.
    from cloelib.observables.kernel_conventions import kernel_convention

    if isinstance(spec, str):
        return kernel_convention(spec)
    spec = dict(spec)
    preset = spec.pop("preset", "godmax")
    return kernel_convention(preset, **spec)


def _set_from_convention(settings, key, value):
    """Set `settings[key] = value`, raising if it is already set differently."""
    if key in settings and settings[key] != value:
        raise ValueError(
            f"settings[{key!r}] = {settings[key]!r} disagrees with the kernel "
            f"convention ({value!r}); remove it or change the convention."
        )
    settings[key] = value


class GODMAXMixin:
    """
    Mixin running the photometric likelihoods on the GODMAX halo model.

    It sits just left of `PhotoLikelihoodBase` in the MRO and provides:

    - the kernel convention (`settings["kernel_convention"]`);
    - sampled HOD parameters (`settings["hod_param_keys"]`) and halo-model
      redshift nodes (`settings["nonlinear_redshifts"]`);
    - early checks that the backend is a halo model and supports the selected
      IA and galaxy bias models (e.g. no TATT, no nonlinear bias);
    - `_get_tsz_tracer`, the cached `tSZTracer` used by the tSZ mixins.

    See the module docstring for the settings.
    """

    def __init__(self, data, settings, *args, **kwargs):
        settings = dict(settings)
        convention_spec = settings.get("kernel_convention")
        self._convention = (
            None
            if convention_spec is None
            else _build_kernel_convention(convention_spec)
        )
        if self._convention is not None:
            conv = self._convention
            values = {
                "ia_model": conv.shear_options()["ia_model"],
                "galaxy_bias_model": conv.galaxy_bias_model,
                "include_rsd": conv.rsd,
            }
            for key in _CONVENTION_SETTINGS:
                _set_from_convention(settings, key, values[key])
            shear_kwargs = dict(settings.get("shear_tracer_kwargs", {}))
            _set_from_convention(shear_kwargs, "spin2_prefactor", conv.spin2_prefactor)
            settings["shear_tracer_kwargs"] = shear_kwargs
            _set_from_convention(
                settings, "tsz_beam_fwhm_arcmin", conv.y_beam_fwhm_arcmin
            )

        super().__init__(data, settings, *args, **kwargs)

        self.tsz_beam_fwhm_arcmin = settings.get("tsz_beam_fwhm_arcmin", 0.0)
        self.hod_param_keys = tuple(settings.get("hod_param_keys", ()))
        nonlinear_redshifts = settings.get("nonlinear_redshifts")
        self.nonlinear_redshifts = (
            None if nonlinear_redshifts is None else np.asarray(nonlinear_redshifts)
        )
        self._check_halo_model_settings()

    def _check_halo_model_settings(self):
        """Fail at construction for settings the halo model cannot run."""
        nlp_class = self.NonLinPerturbations
        # Unwrap functools.partial so a partially applied backend also works.
        while isinstance(nlp_class, functools.partial):
            nlp_class = nlp_class.func
        supported_ia = getattr(nlp_class, "supported_ia_models", None)
        supported_bias = getattr(nlp_class, "supported_galaxy_bias_models", None)
        if supported_ia is None or supported_bias is None:
            raise ValueError(
                f"{type(self).__name__} needs a halo-model NonLinPerturbations "
                "(cloelib.cosmology.godmax_cosmology.GODMAXNonLinearPerturbations), "
                f"got {getattr(nlp_class, '__name__', nlp_class)!r}."
            )
        if self.ia_model not in supported_ia:
            raise NotImplementedError(
                f"ia_model={self.ia_model!r} is not available on the halo model; "
                f"use one of {list(supported_ia)}."
            )
        if self.galaxy_bias_model not in supported_bias:
            raise NotImplementedError(
                f"galaxy_bias_model={self.galaxy_bias_model!r} is not available on "
                f"the halo model; use one of {list(supported_bias)}."
            )

        if self.hod_param_keys:
            # Imported here so cloelike still imports with cloelib versions
            # without GODMAX.
            from cloelib.cosmology.godmax_cosmology import HOD_PARAMETERS

            unknown = set(self.hod_param_keys) - set(HOD_PARAMETERS)
            if unknown:
                raise ValueError(
                    f"Unknown hod_param_keys {sorted(unknown)}; expected names "
                    f"from {HOD_PARAMETERS}."
                )
            if "galaxies" not in self.settings.get("nonlinear_kwargs", {}):
                raise ValueError(
                    "hod_param_keys needs settings['nonlinear_kwargs']['galaxies'] "
                    "(the galaxy sample of the HOD)."
                )

    def _require_decoupled(self, block):
        """Raise for `mode="coupled"`, which the tSZ blocks do not support yet."""
        if self.mode == "coupled":
            raise NotImplementedError(
                f"The {block} block supports only mode='decoupled' until cloelib's "
                "get_pseudo_Cl handles tSZ (Compton-y) pairs."
            )

    def _nonlinear_kwargs(self, parameters):
        """GODMAX keyword arguments: adds the halo-model redshift nodes and the
        sampled HOD parameters to those of the base class."""
        nl_kwargs = super()._nonlinear_kwargs(parameters)
        if self.nonlinear_redshifts is not None:
            nl_kwargs["redshifts"] = self.nonlinear_redshifts
        if self.hod_param_keys:
            galaxies = dict(nl_kwargs["galaxies"])
            hod = dict(galaxies.get("hod") or {})
            hod.update({k: parameters[k] for k in self.hod_param_keys})
            galaxies["hod"] = hod
            nl_kwargs["galaxies"] = galaxies
        return nl_kwargs

    def _cosmo_key(self, parameters):
        """Cache key of the halo model: also changes with the HOD parameters."""
        return super()._cosmo_key(parameters) + tuple(
            parameters[k] for k in self.hod_param_keys
        )

    def _convention_nuisance(self, parameters, with_ia=True):
        """Parameters with the systematics the kernel convention leaves out
        switched off (and, with `with_ia`, AIA rescaled for its pivot NLA)."""
        if self._convention is None:
            return parameters
        convention = self._convention
        if not with_ia:
            # The galaxy kernel has no IA, so do not require the IA parameters.
            convention = dataclasses.replace(convention, ia_model=None)
        return convention.nuisance(parameters)

    def _get_she_tracer(self, parameters, nlp):
        return super()._get_she_tracer(self._convention_nuisance(parameters), nlp)

    def _get_pos_tracer(self, parameters, nlp):
        return super()._get_pos_tracer(
            self._convention_nuisance(parameters, with_ia=False), nlp
        )

    def _get_tsz_tracer(self, parameters, nlp):
        """Build (and cache) the tSZTracer for the given parameters/nlp, so the
        tSZ blocks do not each rebuild it within the same call."""
        # Imported here so cloelike still imports with cloelib versions without it.
        from cloelib.observables.cmb import tSZTracer

        key = self._cosmo_key(parameters)
        cached = getattr(self, "_tsz_tracer_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        tracer = tSZTracer(nlp, self.zs, beam_fwhm_arcmin=self.tsz_beam_fwhm_arcmin)
        self._tsz_tracer_cache = (key, tracer)
        return tracer


class tSZxtSZMixin:
    """
    Mixin class providing the tSZ auto-spectrum (Compton-y x Compton-y) block.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the next class in the MRO
        self._init_tszxtsz()

    def _init_tszxtsz(self):
        self._require_decoupled("tSZxtSZ")
        self.tSZxtSZ_keys = [("Y", "Y", 1, 1)]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.tSZxtSZ_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array([self.data["cells"][key] for key in self.tSZxtSZ_keys]).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        _, _, nlp = self._get_perturbations(parameters)
        tsz = self._get_tsz_tracer(parameters, nlp)
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(tsz, tsz).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key] for key in self.tSZxtSZ_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(tsz, tsz).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key] for key in self.tSZxtSZ_keys]).flatten()
        self.theory_prediction.update(cell_all_th)
        return np.concatenate([v, vec])


class tSZxGCMixin:
    """
    Mixin class providing the tSZ x photometric galaxy clustering (POS x Y) block.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the next class in the MRO
        self._init_tszxgc()

    def _init_tszxgc(self):
        self._require_decoupled("tSZxGC")
        self.n_pos_bins = self.data["dndz_pos"].shape[0]
        self._init_pos_keys()
        self.tSZxGC_keys = [("POS", "Y", i, 1) for i in range(1, self.n_pos_bins + 1)]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.tSZxGC_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array([self.data["cells"][key] for key in self.tSZxGC_keys]).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        _, _, nlp = self._get_perturbations(parameters)
        pos = self._get_pos_tracer(parameters, nlp)
        tsz = self._get_tsz_tracer(parameters, nlp)
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(pos, tsz).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key] for key in self.tSZxGC_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(pos, tsz).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key] for key in self.tSZxGC_keys]).flatten()
        self.theory_prediction.update(cell_all_th)
        return np.concatenate([v, vec])


class tSZxWLMixin:
    """
    Mixin class providing the tSZ x weak lensing (SHE x Y) block (E-mode part).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the next class in the MRO
        self._init_tszxwl()

    def _init_tszxwl(self):
        self._require_decoupled("tSZxWL")
        self.n_she_bins = self.data["dndz_she"].shape[0]
        self._init_she_keys()
        self.tSZxWL_keys = [("SHE", "Y", i, 1) for i in range(1, self.n_she_bins + 1)]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.tSZxWL_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array(
            [self.data["cells"][key][0] for key in self.tSZxWL_keys]
        ).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        _, _, nlp = self._get_perturbations(parameters)
        she = self._get_she_tracer(parameters, nlp)
        tsz = self._get_tsz_tracer(parameters, nlp)
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(she, tsz).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key][0] for key in self.tSZxWL_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(she, tsz).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key][0] for key in self.tSZxWL_keys]).flatten()
        self.theory_prediction.update(cell_all_th)
        return np.concatenate([v, vec])


class EuclidLikelihood_tSZxtSZ(tSZxtSZMixin, GODMAXMixin, PhotoLikelihoodBase):
    """
    EuclidLikelihood_tSZxtSZ computes the tSZ auto-spectrum (Compton-y x Compton-y)
    likelihood on the GODMAX halo model.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GODMAXMixin: Mixin running the likelihood on the GODMAX halo model.
        tSZxtSZMixin: Mixin providing the tSZ auto-spectrum block.

    Parameters
    ----------
    data : dict
        Input data, including `cells[("Y", "Y", 1, 1)]`, `ells`, `z_arr`, `cov`.
    settings : dict
        Configuration settings (see the module docstring).
    Background : object
        `JAXBackground`.
    LinPerturbations : object
        `JAXLinearPerturbations`.
    NonLinPerturbations : object
        `GODMAXNonLinearPerturbations`.
    mode : str
        Must be "decoupled".
    """

    pass


class EuclidLikelihood_tSZxGC(tSZxGCMixin, GODMAXMixin, PhotoLikelihoodBase):
    """
    EuclidLikelihood_tSZxGC computes the tSZ x photometric galaxy clustering
    likelihood on the GODMAX halo model.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GODMAXMixin: Mixin running the likelihood on the GODMAX halo model.
        tSZxGCMixin: Mixin providing the tSZ x GCph block.

    Parameters
    ----------
    data : dict
        Input data, including `cells[("POS", "Y", i, 1)]`, `ells`, `z_arr`,
        `dndz_pos`, `cov`.
    settings : dict
        Configuration settings (see the module docstring).
    Background : object
        `JAXBackground`.
    LinPerturbations : object
        `JAXLinearPerturbations`.
    NonLinPerturbations : object
        `GODMAXNonLinearPerturbations`.
    mode : str
        Must be "decoupled".
    """

    pass


class EuclidLikelihood_tSZxWL(tSZxWLMixin, GODMAXMixin, PhotoLikelihoodBase):
    """
    EuclidLikelihood_tSZxWL computes the tSZ x weak lensing likelihood on the
    GODMAX halo model.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GODMAXMixin: Mixin running the likelihood on the GODMAX halo model.
        tSZxWLMixin: Mixin providing the tSZ x WL block.

    Parameters
    ----------
    data : dict
        Input data, including `cells[("SHE", "Y", i, 1)]`, `ells`, `z_arr`,
        `dndz_she`, `cov`.
    settings : dict
        Configuration settings (see the module docstring).
    Background : object
        `JAXBackground`.
    LinPerturbations : object
        `JAXLinearPerturbations`.
    NonLinPerturbations : object
        `GODMAXNonLinearPerturbations`.
    mode : str
        Must be "decoupled".
    """

    pass


class EuclidLikelihood_tSZx(
    tSZxtSZMixin, tSZxGCMixin, tSZxWLMixin, GODMAXMixin, PhotoLikelihoodBase
):
    """
    EuclidLikelihood_tSZx combines the tSZ x weak lensing, tSZ x GCph and tSZ
    auto-spectrum likelihoods on the GODMAX halo model.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GODMAXMixin: Mixin running the likelihood on the GODMAX halo model.
        tSZxWLMixin, tSZxGCMixin, tSZxtSZMixin: the tSZ blocks.

    Note: The order of inheritance matters due to the method resolution order (MRO) in Python
    and how mixins extend the base class functionality. Each mixin appends its block after
    the blocks of the mixins that follow it in the MRO, so the data vector, and therefore
    the covariance matrix blocks, are ordered tSZxWL, tSZxGC and tSZxtSZ.

    Parameters
    ----------
    data : dict
        Input data, including the tSZ cells, `ells`, `z_arr`, `dndz_pos`,
        `dndz_she`, `cov`.
    settings : dict
        Configuration settings (see the module docstring).
    Background : object
        `JAXBackground`.
    LinPerturbations : object
        `JAXLinearPerturbations`.
    NonLinPerturbations : object
        `GODMAXNonLinearPerturbations`.
    mode : str
        Must be "decoupled".
    """

    pass


class EuclidLikelihood_3x2pt_GODMAX(
    GCphMixin, GGLMixin, WLMixin, GODMAXMixin, PhotoLikelihoodBase
):
    """
    EuclidLikelihood_3x2pt_GODMAX is the 3x2pt likelihood (WL, GGL, GCph) on the
    GODMAX halo model, with the HOD galaxy bias and the kernel convention
    available. It has no tSZ block, so it also runs in coupled mode.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GODMAXMixin: Mixin running the likelihood on the GODMAX halo model.
        WLMixin, GGLMixin, GCphMixin: the 3x2pt blocks.

    Note: The data vector, and therefore the covariance matrix blocks, are
    ordered WL, GGL and GCph, as in EuclidLikelihood_3x2pt.

    Parameters
    ----------
    data : dict
        Input data, as for EuclidLikelihood_3x2pt.
    settings : dict
        Configuration settings (see the module docstring).
    Background : object
        `JAXBackground`.
    LinPerturbations : object
        `JAXLinearPerturbations`.
    NonLinPerturbations : object
        `GODMAXNonLinearPerturbations`.
    mode : str, optional
        Mode of operation, default is "coupled".
    """

    pass


class EuclidLikelihood_3x2pt_tSZx(
    GCphMixin,
    GGLMixin,
    WLMixin,
    tSZxtSZMixin,
    tSZxGCMixin,
    tSZxWLMixin,
    GODMAXMixin,
    PhotoLikelihoodBase,
):
    """
    EuclidLikelihood_3x2pt_tSZx combines the 3x2pt likelihood (WL, GGL, GCph) with
    the tSZ x WL, tSZ x GCph and tSZ auto-spectrum likelihoods on the GODMAX halo
    model.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GODMAXMixin: Mixin running the likelihood on the GODMAX halo model.
        tSZxWLMixin, tSZxGCMixin, tSZxtSZMixin: the tSZ blocks.
        WLMixin, GGLMixin, GCphMixin: the 3x2pt blocks.

    Note: The order of inheritance matters due to the method resolution order (MRO) in Python
    and how mixins extend the base class functionality. Each mixin appends its block after
    the blocks of the mixins that follow it in the MRO, so the data vector, and therefore
    the covariance matrix blocks, are ordered tSZxWL, tSZxGC, tSZxtSZ, WL, GGL and GCph.
    The whole likelihood runs in decoupled mode.

    Parameters
    ----------
    data : dict
        Input data, as for EuclidLikelihood_3x2pt plus the tSZ cells.
    settings : dict
        Configuration settings (see the module docstring).
    Background : object
        `JAXBackground`.
    LinPerturbations : object
        `JAXLinearPerturbations`.
    NonLinPerturbations : object
        `GODMAXNonLinearPerturbations`.
    mode : str
        Must be "decoupled".
    """

    pass
