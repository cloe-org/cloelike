import numpy as np
from functools import lru_cache
from typing import Protocol, runtime_checkable
from copy import deepcopy
from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.observables.photo import PositionsTracer, ShearTracer


@runtime_checkable
class PhotoLikelihoodProtocol(Protocol):
    """
    Protocol for photo-z likelihood classes.

    This protocol defines the required interface for photometric likelihood implementations,
    specifying initialization, required attributes, and methods for computing data vectors,
    theory vectors, covariance matrices, and log-likelihoods.

    Attributes:
        data (dict): Observational data dictionary.
        settings (dict): Configuration settings dictionary.
        Background (Background): Cosmological background instance.
        LinPerturbations (Perturbations): Linear perturbations instance.
        NonLinPerturbations (Perturbations): Non-linear perturbations instance.
        derived (dict): Dictionary for derived quantities.
        mode (str): Mode of operation (e.g., "coupled").

    Methods:
        __init__(data, settings, Background, LinPerturbations, NonLinPerturbations, mode):
            Initializes the likelihood protocol.
        get_data_vector_full() -> np.ndarray:
            Returns the full data vector.
        get_data_vector_masked() -> np.ndarray:
            Returns the masked data vector.
        get_theory_vector_full(parameters: dict) -> np.ndarray:
            Returns the full theory vector for given parameters.
        get_theory_vector_masked(parameters: dict) -> np.ndarray:
            Returns the masked theory vector for given parameters.
        get_covariance_matrix_full() -> np.ndarray:
            Returns the full covariance matrix.
        get_covariance_matrix_masked_inv() -> np.ndarray:
            Returns the inverse of the masked covariance matrix.
        loglike(parameters: dict) -> float:
            Computes the log-likelihood for the given parameters.
    """

    def __init__(
        self,
        data: dict,
        settings: dict,
        Background: Background,
        LinPerturbations: Perturbations,
        NonLinPerturbations: Perturbations,
    ) -> None: ...

    data: dict
    settings: dict
    Background: Background
    LinPerturbations: Perturbations
    NonLinPerturbations: Perturbations
    derived: dict

    def get_data_vector_full(self) -> np.ndarray: ...
    def get_data_vector_masked(self) -> np.ndarray: ...
    def get_theory_vector_full(self, parameters: dict) -> np.ndarray: ...
    def get_theory_vector_masked(self, parameters: dict) -> np.ndarray: ...
    def get_covariance_matrix_full(self) -> np.ndarray: ...
    def get_covariance_matrix_masked_inv(self) -> np.ndarray: ...
    def loglike(self, parameters: dict) -> float: ...


class PhotoLikelihoodBase:
    """
    Base class for photometric likelihood calculations using angular correlation functions.
    This class provides methods for preparing, binning, and masking data, as well as computing
    likelihoods based on theoretical predictions and observed data vectors.
    Args:
        data (dict): Dictionary containing observational data, including '2pcf', 'theta', 'z_arr', and 'cov'.
        settings (dict): Configuration settings, including 'scale_cuts'.
        Background: Object representing background cosmology.
        LinPerturbations: Object representing linear perturbations.
        NonLinPerturbations: Object representing non-linear perturbations.
        mode (str, optional): Mode of operation, default is "coupled".
    Attributes:
        data (dict): Observational data.
        settings (dict): Configuration settings.
        derived (dict): Dictionary for storing derived quantities.
        Background: Background cosmology object.
        LinPerturbations: Linear perturbations object.
        NonLinPerturbations: Non-linear perturbations object.
        scale_cuts: Scale cuts from settings.
        selected_modes: COSEBIs, selected modes from settings, defaults to the first seven
        w_ells: COSEBI, kernel functions for the COSEBIs, contains the scale cut
        ells_integration_COSEBI: COSEBI, ells for the integration, need to match the w_ells
        zs: Redshift array from data.
        mixmat: Mixing matrix, possibly rebinned.
        weight_mat: Weight matrix used for binning.
        masking_vector: Boolean mask for selecting data vector elements.
    Methods:
        _masking(arr, interval):
            Returns a boolean mask for elements within the specified interval.
        get_covariance_matrix_full():
            Returns the full covariance matrix from the data.
        get_data_vector_masked():
            Returns the masked data vector.
        get_covariance_matrix_masked_inv():
            Returns the inverse of the masked covariance matrix.
        get_theory_vector_full(parameters):
            Returns the full theoretical prediction vector for given parameters.
        get_theory_vector_masked(parameters):
            Returns the masked theoretical prediction vector for given parameters.
        loglike(parameters):
            Computes the log-likelihood for the given parameters using the masked data and theory vectors.
    """

    def __init__(
        self,
        data,
        settings,
        Background,
        LinPerturbations,
        NonLinPerturbations,
        ells_integration=None,
        mode="coupled",
    ):
        self.data = data
        self.settings = settings
        self.derived = {}
        self.Background = Background
        self.LinPerturbations = LinPerturbations
        self.NonLinPerturbations = NonLinPerturbations
        self.theory_prediction = {}
        self.mode = mode
        if "EE" in data:
            self.w_ells = settings["w_ells"]
            if settings.get("scale_cuts", None) is not None:
                print(
                    "For COSEBIs the scale cuts need to be applied via the W_ells, the passed scale cuts are ignored now"
                )
            self.scale_cuts = None

        else:
            self.scale_cuts = settings["scale_cuts"]
        self.selected_modes = settings.get("selected_modes", np.arange(1, 8))
        self.ells_integration_COSEBI = settings.get("ells_integration_COSEBI", None)

        if (self.ells_integration_COSEBI is None) and ("EE" in data):
            raise ValueError(
                "an ells array corresponding to the W_ells is needed for the COSEBIs"
            )

        self.rebin = False
        self.zs = data["z_arr"]
        if self.mode == "coupled":
            self.mixmat = deepcopy(data.get("mixmat", {}))
        else:
            self.mixmat = None

        if (ells_integration is None) and ("2pcf" in data):
            self.ells_integration = np.arange(2, 40000)
        else:
            self.ells_integration = ells_integration

        if ("cells" in data) and ("n_ell_bins" in settings):
            self._prepare()

    def _prepare(self):
        """Perform rebinning if required by settings."""
        if self.settings["n_ell_bins"] < len(self.data["ells"]):
            self.rebin = True
            self.data["cells_unbin"] = self.data["cells"]
            self.data["ells_unbin"] = self.data["ells"]
            self._bin_data(
                self.data["cells"], self.data["ells"], self.settings["n_ell_bins"]
            )
            self._bin_mixmat()

    def _bin_data(self, cells_data, ells, n_bins):
        """Geometric rebinning of data vectors."""
        bin_edges = np.geomspace(10, ells[-1], n_bins + 1)
        mask_bins = [
            (ells >= bin_edges[i]) & (ells < bin_edges[i + 1]) for i in range(n_bins)
        ]
        self.weight_mat = np.asarray(mask_bins, dtype=float)
        self.weight_mat /= np.sum(self.weight_mat, axis=1)[:, None]
        self.data["ells"] = np.array([np.mean(ells[mb]) for mb in mask_bins])
        for k in cells_data.keys():
            self.data["cells"][k] = cells_data[k] @ self.weight_mat.T

    def _bin_mixmat(self):
        """Apply the same binning to the mixing matrix."""
        for k in self.mixmat.keys():
            new_array = np.tensordot(self.weight_mat, self.mixmat[k], axes=([1], [-2]))
            if k[:2] == ("SHE", "SHE"):
                new_array = np.transpose(new_array, axes=(1, 0, 2))
            self.mixmat[k] = new_array

    def _masking(self, arr, interval):
        return (arr >= interval[0]) & (arr <= interval[1])

    # Parameters passed to Background(...) -- log10TAGN is not one of them (only
    # NonLinPerturbations consumes it), so it can't be folded into this list.
    _BACKGROUND_PARAM_KEYS = (
        "H0",
        "Omega_cdm0",
        "Omega_b0",
        "Omega_k0",
        "w0",
        "wa",
        "ns",
        "As",
        "mnu",
        "gamma_MG",
        "N_mnu",
        "alpha_s",
    )

    # Full set of parameters that determine Background/LinPerturbations/NonLinPerturbations
    # -- i.e. everything the perturbations/tracer caches below key on.
    _COSMO_PARAM_KEYS = _BACKGROUND_PARAM_KEYS + ("log10TAGN",)

    def _get_perturbations(self, parameters):
        """Build (and cache) Background/LinPerturbations/NonLinPerturbations for
        the given parameters.

        Combined likelihoods (e.g. 3x2pt) mix several probe-specific mixins in
        one MRO chain, each of which needs the perturbations. Caching here
        (instead of rebuilding in every mixin's get_theory_vector_full) avoids
        recomputing the same cosmology multiple times per call. The cache is
        keyed on the cosmology-relevant subset of `parameters` so a change in
        any of those values rebuilds it, while repeated calls with an unchanged
        cosmology (e.g. only nuisance parameters varying) reuse it.
        """
        key = self._cosmo_key(parameters)
        cached = getattr(self, "_perturbations_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        background = self.Background(
            **{k: parameters[k] for k in self._BACKGROUND_PARAM_KEYS}
        )
        lp = self.LinPerturbations(background, self.zs)
        nlp = self.NonLinPerturbations(
            background, lp, self.zs, log10TAGN=parameters["log10TAGN"]
        )
        # sigma8_0() self-memoizes by overwriting `nlp.sigma8_0` with its
        # result on first call, so it must be called exactly once per nlp
        # instance -- here, rather than in each mixin (which would fail on
        # the second mixin to see this shared, cached nlp).
        self.derived["sigma8_0"] = nlp.sigma8_0()
        result = (background, lp, nlp)
        self._perturbations_cache = (key, result)
        return result

    def _cosmo_key(self, parameters):
        """The same cache key used by `_get_perturbations`, exposed so tracer
        caching below can be keyed on values rather than on `nlp`'s object
        identity (which could in principle be reused after garbage collection)."""
        return tuple(parameters[k] for k in self._COSMO_PARAM_KEYS)

    def _get_pos_tracer(self, parameters, nlp):
        """Build (and cache) the PositionsTracer for the given parameters/nlp,
        so GCph and GGL do not each rebuild it within the same call."""
        key = self._cosmo_key(parameters) + tuple(
            parameters[k] for k in self.full_pos_keys
        )
        cached = getattr(self, "_pos_tracer_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        tracer = PositionsTracer(
            nlp,
            self.data["dndz_pos"],
            self.zs,
            nuisance_params={k: parameters[k] for k in self.full_pos_keys},
            galaxy_bias_model="poly",
        )
        self._pos_tracer_cache = (key, tracer)
        return tracer

    def _get_she_tracer(self, parameters, nlp):
        """Build (and cache) the ShearTracer for the given parameters/nlp, so
        WL and GGL do not each rebuild it within the same call."""
        key = self._cosmo_key(parameters) + tuple(
            parameters[k] for k in self.full_she_keys
        )
        cached = getattr(self, "_she_tracer_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        tracer = ShearTracer(
            nlp,
            self.data["dndz_she"],
            self.zs,
            nuisance_params={k: parameters[k] for k in self.full_she_keys},
        )
        self._she_tracer_cache = (key, tracer)
        return tracer

    def get_masking_vector(self):
        return np.array([], dtype=bool)

    def get_covariance_matrix_full(self):
        return self.data["cov"]

    def get_data_vector_full(self):
        return np.array([])

    @lru_cache(maxsize=None)
    def get_masking_vector_cached(self):
        """Compute (once) and cache the combined boolean mask."""
        return self.get_masking_vector()

    @lru_cache(maxsize=None)
    def get_data_vector_masked(self):
        mask = self.get_masking_vector_cached()
        return self.get_data_vector_full()[mask]

    @lru_cache(maxsize=None)
    def get_covariance_matrix_masked_inv(self):
        """Compute (once) and cache inverse masked covariance matrix."""
        cov = self.get_covariance_matrix_full()
        mask = self.get_masking_vector_cached()
        cov_masked = cov[mask][:, mask]
        return np.linalg.inv(cov_masked)

    def get_theory_vector_full(self, parameters):
        return np.array([])

    def get_theory_vector_masked(self, parameters):
        mask = self.get_masking_vector_cached()
        return self.get_theory_vector_full(parameters)[mask]

    def loglike(self, parameters):
        diff = self.get_theory_vector_masked(parameters) - self.get_data_vector_masked()
        inv_cov = self.get_covariance_matrix_masked_inv()
        return -0.5 * diff @ inv_cov @ diff
