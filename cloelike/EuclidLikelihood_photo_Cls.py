# file: euclid_likelihoods.py
import numpy as np
from copy import deepcopy
from typing import Protocol, runtime_checkable

from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint


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
        mode: str = "coupled",
    ) -> None: ...

    data: dict
    settings: dict
    Background: Background
    LinPerturbations: Perturbations
    NonLinPerturbations: Perturbations
    derived: dict
    mode: str

    def get_data_vector_full(self) -> np.ndarray: ...
    def get_data_vector_masked(self) -> np.ndarray: ...
    def get_theory_vector_full(self, parameters: dict) -> np.ndarray: ...
    def get_theory_vector_masked(self, parameters: dict) -> np.ndarray: ...
    def get_covariance_matrix_full(self) -> np.ndarray: ...
    def get_covariance_matrix_masked_inv(self) -> np.ndarray: ...
    def loglike(self, parameters: dict) -> float: ...


class PhotoLikelihoodBase:
    """
    Base class for photometric likelihood calculations using angular power spectra (Cls).
    This class provides methods for preparing, binning, and masking data, as well as computing
    likelihoods based on theoretical predictions and observed data vectors.
    Args:
        data (dict): Dictionary containing observational data, including 'cells', 'ells', 'z_arr', 'mixmat', and 'cov'.
        settings (dict): Configuration settings, including 'scale_cuts' and 'n_ell_bins'.
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
        mode (str): Mode of operation.
        scale_cuts: Scale cuts from settings.
        rebin (bool): Flag indicating if data has been rebinned.
        zs: Redshift array from data.
        mixmat: Mixing matrix, possibly rebinned.
        weight_mat: Weight matrix used for binning.
        masking_vector: Boolean mask for selecting data vector elements.
    Methods:
        _prepare():
            Prepares the data, performing binning if necessary.
        _bin_data(cells_data, ells, n_bins):
            Bins the data vectors and updates the weight matrix.
        _bin_mixmat():
            Bins the mixing matrix according to the weight matrix.
        _masking(arr, interval):
            Returns a boolean mask for elements within the specified interval.
        get_covariance_matrix_full():
            Returns the full covariance matrix from the data.
        get_data_vector_masked():
            Returns the masked data vector.
        get_covariance_matrix_masked_inv():
            Returns the inverse of the masked covariance matrix.
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
        mode="coupled",
    ):
        self.data = data
        self.settings = settings
        self.derived = {}
        self.Background = Background
        self.LinPerturbations = LinPerturbations
        self.NonLinPerturbations = NonLinPerturbations
        self.mode = mode
        self.scale_cuts = settings["scale_cuts"]
        self.rebin = False
        self.zs = data["z_arr"]
        self.mixmat = deepcopy(data["mixmat"])
        self._prepare()

    def _prepare(self):
        if self.settings["n_ell_bins"] < len(self.data["ells"]):
            self.rebin = True
            self.data["cells_unbin"] = self.data["cells"]
            self.data["ells_unbin"] = self.data["ells"]
            self._bin_data(
                self.data["cells"], self.data["ells"], self.settings["n_ell_bins"]
            )
            self._bin_mixmat()

    def _bin_data(self, cells_data, ells, n_bins):
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
        for k in self.mixmat.keys():
            new_array = np.tensordot(self.weight_mat, self.mixmat[k], axes=([1], [-2]))
            if k[:2] == ("SHE", "SHE"):
                new_array = np.transpose(new_array, axes=(1, 0, 2))
            self.mixmat[k] = new_array

    def _masking(self, arr, interval):
        return (arr >= interval[0]) & (arr <= interval[1])

    def get_covariance_matrix_full(self):
        return self.data["cov"]

    def get_data_vector_masked(self):
        return self.get_data_vector_full()[self.masking_vector]

    def get_covariance_matrix_masked_inv(self):
        cov = self.get_covariance_matrix_full()
        return np.linalg.inv(cov[self.masking_vector][:, self.masking_vector])

    def get_theory_vector_masked(self, parameters):
        return self.get_theory_vector_full(parameters)[self.masking_vector]

    def loglike(self, parameters):
        t_vec = self.get_theory_vector_masked(parameters)
        d_vec = self.get_data_vector_masked()
        inv_cov = self.get_covariance_matrix_masked_inv()
        diff = t_vec - d_vec
        return -0.5 * diff @ inv_cov @ diff


class WLMixin:
    def _init_wl(self):
        self.n_she_bins = self.data["dndz_she"].shape[0]
        IA_keys = ["AIA", "EtaIA", "CIA"]
        mul_bias_keys = [
            f"multiplicative_bias_{i}" for i in range(1, self.n_she_bins + 1)
        ]
        dz_she_keys = [f"dz_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        self.full_she_keys = IA_keys + mul_bias_keys + dz_she_keys
        self.WL_keys = [
            ("SHE", "SHE", i, j)
            for i in range(1, self.n_she_bins + 1)
            for j in range(i, self.n_she_bins + 1)
        ]

    def get_data_vector_full(self):
        return np.array(
            [self.data["cells"][key][0, 0] for key in self.WL_keys]
        ).flatten()

    def get_theory_vector_full(self, parameters):
        background = self.Background(
            **{
                k: parameters[k]
                for k in [
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
                ]
            }
        )
        lp = self.LinPerturbations(background, self.zs)
        nlp = self.NonLinPerturbations(
            background, lp, self.zs, log10TAGN=parameters["log10TAGN"]
        )
        she = ShearTracer(
            nlp,
            self.data["dndz_she"],
            self.zs,
            nuisance_params={key: parameters[key] for key in self.full_she_keys},
        )
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(she, she).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key][0, 0] for key in self.WL_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(she, she).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key][0, 0] for key in self.WL_keys]).flatten()
        self.derived["sigma8_0"] = nlp.sigma8_0()
        self.theory_prediction = cell_all_th
        return vec


class GCphMixin:
    def _init_gcph(self):
        self.n_pos_bins = self.data["dndz_pos"].shape[0]
        bias_keys = [f"b1_photo_poly{i}" for i in range(4)]
        mag_bias_keys = [
            f"magnification_bias_{i}" for i in range(1, self.n_pos_bins + 1)
        ]
        dz_pos_keys = [f"dz_pos_{i}" for i in range(1, self.n_pos_bins + 1)]
        self.full_pos_keys = bias_keys + mag_bias_keys + dz_pos_keys
        self.GG_keys = [
            ("POS", "POS", i, j)
            for i in range(1, self.n_pos_bins + 1)
            for j in range(i, self.n_pos_bins + 1)
        ]

    def get_data_vector_full(self):
        return np.array([self.data["cells"][key] for key in self.GG_keys]).flatten()

    def get_theory_vector_full(self, parameters):
        background = self.Background(
            **{
                k: parameters[k]
                for k in [
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
                ]
            }
        )
        lp = self.LinPerturbations(background, self.zs)
        nlp = self.NonLinPerturbations(
            background, lp, self.zs, log10TAGN=parameters["log10TAGN"]
        )
        pos = PositionsTracer(
            nlp,
            self.data["dndz_pos"],
            self.zs,
            nuisance_params={key: parameters[key] for key in self.full_pos_keys},
            galaxy_bias_model="poly",
        )
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(pos, pos).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key] for key in self.GG_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(pos, pos).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key] for key in self.GG_keys]).flatten()
        self.derived["sigma8_0"] = nlp.sigma8_0()
        self.theory_prediction = cell_all_th
        return vec


class GGLMixin:
    def _init_ggl(self):
        self.n_pos_bins = self.data["dndz_pos"].shape[0]
        self.n_she_bins = self.data["dndz_she"].shape[0]
        bias_keys = [f"b1_photo_poly{i}" for i in range(4)]
        mag_bias_keys = [
            f"magnification_bias_{i}" for i in range(1, self.n_pos_bins + 1)
        ]
        dz_pos_keys = [f"dz_pos_{i}" for i in range(1, self.n_pos_bins + 1)]
        IA_keys = ["AIA", "EtaIA", "CIA"]
        mul_bias_keys = [
            f"multiplicative_bias_{i}" for i in range(1, self.n_she_bins + 1)
        ]
        dz_she_keys = [f"dz_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        self.full_pos_keys = bias_keys + mag_bias_keys + dz_pos_keys
        self.full_she_keys = IA_keys + mul_bias_keys + dz_she_keys
        self.GGL_keys = [
            ("POS", "SHE", i, j)
            for i in range(1, self.n_pos_bins + 1)
            for j in range(1, self.n_she_bins + 1)
        ]

    def get_data_vector_full(self):
        return np.array([self.data["cells"][key][0] for key in self.GGL_keys]).flatten()

    def get_theory_vector_full(self, parameters):
        background = self.Background(
            **{
                k: parameters[k]
                for k in [
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
                ]
            }
        )
        lp = self.LinPerturbations(background, self.zs)
        nlp = self.NonLinPerturbations(
            background, lp, self.zs, log10TAGN=parameters["log10TAGN"]
        )
        pos = PositionsTracer(
            nlp,
            self.data["dndz_pos"],
            self.zs,
            nuisance_params={key: parameters[key] for key in self.full_pos_keys},
            galaxy_bias_model="poly",
        )
        she = ShearTracer(
            nlp,
            self.data["dndz_she"],
            self.zs,
            nuisance_params={key: parameters[key] for key in self.full_she_keys},
        )
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(pos, she).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key][0] for key in self.GGL_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(pos, she).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key][0] for key in self.GGL_keys]).flatten()
        self.derived["sigma8_0"] = nlp.sigma8_0()
        self.theory_prediction = cell_all_th
        return vec


class EuclidLikelihood_WL(PhotoLikelihoodBase, WLMixin):
    """
    EuclidLikelihood_WL computes the weak lensing (WL) likelihood for photometric surveys using Euclid data.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        WLMixin: Mixin providing weak lensing specific functionality.

    Args:
        data (dict): Input data containing observed quantities, including 'ells'.
        settings (dict): Configuration settings for the likelihood calculation.
        Background: Cosmological background model instance.
        LinPerturbations: Linear perturbations model instance.
        NonLinPerturbations: Non-linear perturbations model instance.
        mode (str, optional): Mode for likelihood calculation. Default is "coupled".

    Attributes:
        masking_vector (np.ndarray): Vector indicating which multipoles are masked based on scale cuts.
        WL_keys (list): Keys corresponding to weak lensing bins or probes.
        scale_cuts (dict): Dictionary specifying scale cuts for each WL key.

    Methods:
        _init_wl(): Initializes weak lensing specific parameters and settings.

    Example:
        likelihood = EuclidLikelihood_WL(
            data=my_data,
            settings=my_settings,
            Background=background_model,
            LinPerturbations=lin_pert_model,
            NonLinPerturbations=nonlin_pert_model,
            mode="coupled"
    """

    def __init__(
        self,
        *,
        data,
        settings,
        Background,
        LinPerturbations,
        NonLinPerturbations,
        mode="coupled",
    ):
        super().__init__(
            data, settings, Background, LinPerturbations, NonLinPerturbations, mode
        )
        self._init_wl()
        self.masking_vector = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.WL_keys
            ]
        )


class EuclidLikelihood_GCph(PhotoLikelihoodBase, GCphMixin):
    """
    EuclidLikelihood_GCph computes the likelihood for galaxy clustering photometric (GCph) data
    using the Euclid survey specifications.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GCphMixin: Mixin providing GCph-specific functionality.

    Args:
        data (dict): Input observational data, including 'ells' and other relevant fields.
        settings (dict): Configuration settings for the likelihood calculation.
        Background: Instance providing background cosmology calculations.
        LinPerturbations: Instance for linear perturbation theory calculations.
        NonLinPerturbations: Instance for non-linear perturbation theory calculations.
        mode (str, optional): Calculation mode, defaults to "coupled".

    Attributes:
        masking_vector (np.ndarray): Vector used to mask multipoles according to scale cuts.
        GG_keys (list): Keys identifying galaxy-galaxy correlations.
        scale_cuts (dict): Dictionary specifying scale cuts for each GG_key.

    Methods:
        _init_gcph(): Initializes GCph-specific parameters and settings.

    Notes:
        - The masking_vector is constructed by concatenating masks for each GG_key using the
          corresponding scale cuts.
        - This class is tailored for Euclid GCph likelihood analysis and expects data in euclidlib
          specific format.
    """

    def __init__(
        self,
        *,
        data,
        settings,
        Background,
        LinPerturbations,
        NonLinPerturbations,
        mode="coupled",
    ):
        super().__init__(
            data, settings, Background, LinPerturbations, NonLinPerturbations, mode
        )
        self._init_gcph()
        self.masking_vector = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.GG_keys
            ]
        )


class EuclidLikelihood_GGL(PhotoLikelihoodBase, GGLMixin):
    """
    EuclidLikelihood_GGL class for galaxy-galaxy lensing likelihood computation.

    This class combines the functionalities of PhotoLikelihoodBase and GGLMixin to compute
    the likelihood for galaxy-galaxy lensing (GGL) using photometric data. It initializes
    the necessary components and constructs a masking vector based on scale cuts for each
    GGL key.

    Parameters
    ----------
    data : dict
        Input data required for likelihood computation, including observed ells and other relevant quantities.
    settings : dict
        Configuration settings for the likelihood calculation.
    Background : object
        Instance representing the cosmological background model.
    LinPerturbations : object
        Instance representing linear perturbations.
    NonLinPerturbations : object
        Instance representing non-linear perturbations.
    mode : str, optional
        Mode of operation, default is "coupled".

    Attributes
    ----------
    masking_vector : np.ndarray
        Concatenated masking vector for all GGL keys, constructed using scale cuts.
    GGL_keys : list
        List of keys corresponding to different GGL components (inherited from GGLMixin).
    scale_cuts : dict
        Dictionary specifying scale cuts for each GGL key (inherited from GGLMixin).

    Methods
    -------
    _init_ggl()
        Initializes GGL-specific components.
    _masking(ells, scale_cut)
        Applies masking to the input ells based on the provided scale cut.
    """

    def __init__(
        self,
        *,
        data,
        settings,
        Background,
        LinPerturbations,
        NonLinPerturbations,
        mode="coupled",
    ):
        super().__init__(
            data, settings, Background, LinPerturbations, NonLinPerturbations, mode
        )
        self._init_ggl()
        self.masking_vector = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.GGL_keys
            ]
        )


class EuclidLikelihood_3x2pt(PhotoLikelihoodBase, WLMixin, GCphMixin, GGLMixin):
    """
    EuclidLikelihood_3x2pt combines weak lensing (WL), galaxy clustering (GCph), and galaxy-galaxy lensing (GGL)
    likelihoods for photometric cosmological analyses, supporting scale cuts and masking.
    Parameters
    ----------
    data : dict
        Dictionary containing observational data vectors and related metadata.
    settings : dict
        Configuration settings for the likelihood calculation.
    Background : object
        Instance providing background cosmology calculations.
    LinPerturbations : object
        Instance for linear perturbation theory calculations.
    NonLinPerturbations : object
        Instance for non-linear perturbation theory calculations.
    mode : str, optional
        Mode for likelihood calculation, default is "coupled".
    Attributes
    ----------
    masking_vector : np.ndarray
        Boolean or integer array used to mask data and theory vectors according to scale cuts.
    Methods
    -------
    get_data_vector_full()
        Returns the full concatenated data vector from WL, GCph, and GGL components.
    get_theory_vector_full(parameters)
        Returns the full concatenated theory vector for given parameters.
    get_data_vector_masked()
        Returns the masked data vector according to the masking_vector.
    get_theory_vector_masked(parameters)
        Returns the masked theory vector for given parameters.
    get_covariance_matrix_masked_inv()
        Returns the inverse of the masked covariance matrix.
    loglike(parameters)
        Computes the log-likelihood for the given parameters using masked data, theory, and covariance.
    """

    def __init__(
        self,
        *,
        data,
        settings,
        Background,
        LinPerturbations,
        NonLinPerturbations,
        mode="coupled",
    ):
        super().__init__(
            data, settings, Background, LinPerturbations, NonLinPerturbations, mode
        )
        self._init_wl()
        self._init_gcph()
        self._init_ggl()
        self.masking_vector = np.concatenate(
            [
                np.concatenate(
                    [
                        self._masking(self.data["ells"], self.scale_cuts[key])
                        for key in keys
                    ]
                )
                for keys in (self.WL_keys, self.GG_keys, self.GGL_keys)
            ]
        )

    def get_data_vector_full(self):
        return np.concatenate(
            [
                WLMixin.get_data_vector_full(self),
                GCphMixin.get_data_vector_full(self),
                GGLMixin.get_data_vector_full(self),
            ]
        )

    def get_theory_vector_full(self, parameters):
        return np.concatenate(
            [
                WLMixin.get_theory_vector_full(self, parameters),
                GCphMixin.get_theory_vector_full(self, parameters),
                GGLMixin.get_theory_vector_full(self, parameters),
            ]
        )


class EuclidLikelihood_2x2pt(PhotoLikelihoodBase, GCphMixin, GGLMixin):
    """
    Likelihood class for Euclid 2x2pt photometric clustering and galaxy-galaxy lensing analysis.

    This class combines galaxy clustering (GCph) and galaxy-galaxy lensing (GGL) likelihoods,
    providing methods to compute the full data and theory vectors for both probes.

    Parameters
    ----------
    data : dict
        Input data dictionary containing observed power spectra and related quantities.
    settings : dict
        Configuration settings for the likelihood analysis.
    Background : object
        Cosmological background model instance.
    LinPerturbations : object
        Linear perturbations model instance.
    NonLinPerturbations : object
        Non-linear perturbations model instance.
    mode : str, optional
        Mode for likelihood computation, default is "coupled".

    Attributes
    ----------
    masking_vector : np.ndarray
        Combined masking vector for all GCph and GGL keys, used to apply scale cuts.

    Methods
    -------
    get_data_vector_full()
        Returns the concatenated data vector for both GCph and GGL probes.
    get_theory_vector_full(parameters)
        Returns the concatenated theory vector for both GCph and GGL probes, given model parameters.
    """

    def __init__(
        self,
        *,
        data,
        settings,
        Background,
        LinPerturbations,
        NonLinPerturbations,
        mode="coupled",
    ):
        super().__init__(
            data, settings, Background, LinPerturbations, NonLinPerturbations, mode
        )
        self._init_gcph()
        self._init_ggl()
        self.masking_vector = np.concatenate(
            [
                np.concatenate(
                    [
                        self._masking(self.data["ells"], self.scale_cuts[key])
                        for key in keys
                    ]
                )
                for keys in (self.GG_keys, self.GGL_keys)
            ]
        )

    def get_data_vector_full(self):
        return np.concatenate(
            [
                GCphMixin.get_data_vector_full(self),
                GGLMixin.get_data_vector_full(self),
            ]
        )

    def get_theory_vector_full(self, parameters):
        return np.concatenate(
            [
                GCphMixin.get_theory_vector_full(self, parameters),
                GGLMixin.get_theory_vector_full(self, parameters),
            ]
        )
