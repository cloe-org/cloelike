# file: euclid_likelihoods.py
import numpy as np
from copy import deepcopy
from typing import Protocol, runtime_checkable
from functools import lru_cache

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

    Provides methods for preparing, binning, and masking data, and computing
    likelihoods based on theoretical predictions and observed data vectors.

    Parameters
    ----------
    data : dict
        Input data required for likelihood computation.
    settings : dict
        Configuration settings for the likelihood calculation.
    Background : object
        Cosmological background instance.
    LinPerturbations : object
        Linear perturbations instance.
    NonLinPerturbations : object
        Non-linear perturbations instance.
    mode : str, optional
        Mode of operation, default is "coupled".
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

    # -------------------------------
    #  Core preparation utilities
    # -------------------------------
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
        """Helper: boolean mask for elements within a given interval."""
        return (arr >= interval[0]) & (arr <= interval[1])

    # -------------------------------
    #  Default methods
    # -------------------------------
    def get_masking_vector(self):
        """Return the combined masking vector (to be extended by mixins)."""
        return np.array([], dtype=bool)

    def get_covariance_matrix_full(self):
        """Return the full covariance matrix."""
        return self.data["cov"]

    def get_data_vector_full(self):
        """Return the full observed data vector (to be extended by mixins)."""
        return np.array([])

    def get_theory_vector_full(self, parameters):
        """Return the full theoretical data vector (to be extended by mixins)."""
        return np.array([])

    @lru_cache(maxsize=None)
    def get_masking_vector_cached(self):
        """Compute (once) and cache the combined boolean mask."""
        return self.get_masking_vector()

    @lru_cache(maxsize=None)
    def get_data_vector_masked(self):
        """Compute (once) and cache masked data vector."""
        mask = self.get_masking_vector_cached()
        return self.get_data_vector_full()[mask]

    @lru_cache(maxsize=None)
    def get_covariance_matrix_masked_inv(self):
        """Compute (once) and cache inverse masked covariance matrix."""
        cov = self.get_covariance_matrix_full()
        mask = self.get_masking_vector_cached()
        cov_masked = cov[mask][:, mask]
        return np.linalg.inv(cov_masked)

    def get_theory_vector_masked(self, parameters):
        """Return theoretical vector masked identically to the data vector."""
        mask = self.get_masking_vector_cached()
        return self.get_theory_vector_full(parameters)[mask]

    def loglike(self, parameters):
        """Compute Gaussian log-likelihood."""
        diff = self.get_theory_vector_masked(parameters) - self.get_data_vector_masked()
        inv_cov = self.get_covariance_matrix_masked_inv()
        return -0.5 * diff @ inv_cov @ diff


class WLMixin:
    """
    Mixin class providing weak lensing (WL) specific functionality for photometric likelihoods.
    This mixin extends the base photometric likelihood class to include methods and attributes
    necessary for handling weak lensing data, such as initializing shear bins, constructing
    masking vectors, and computing data and theory vectors specific to weak lensing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the next class in the MRO
        self._init_wl()

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

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.WL_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array(
            [self.data["cells"][key][0, 0] for key in self.WL_keys]
        ).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
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
                    "N_mnu",
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
        return np.concatenate([v, vec])


class GCphMixin:
    """
    Mixin class providing photometric angular galaxy clustering specific functionality for photometric likelihoods.
    This mixin extends the base photometric likelihood class to include methods and attributes
    necessary for handling weak lensing data, such as initializing shear bins, constructing
    masking vectors, and computing data and theory vectors specific to weak lensing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_gcph()

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

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.GG_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array([self.data["cells"][key] for key in self.GG_keys]).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
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
                    "N_mnu",
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
        return np.concatenate([v, vec])


class GGLMixin:
    """
    Mixin class providing galaxy-galaxy lensing specific functionality for photometric likelihoods.
    This mixin extends the base photometric likelihood class to include methods and attributes
    necessary for handling weak lensing data, such as initializing shear bins, constructing
    masking vectors, and computing data and theory vectors specific to weak lensing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_ggl()

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

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.GGL_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array([self.data["cells"][key][0] for key in self.GGL_keys]).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
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
                    "N_mnu",
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
        return np.concatenate([v, vec])


class BNTMixin:
    """
    Mixin that applies a BNT transform to shear-related blocks (WL and GGL)
    in the photometric likelihood.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # test to make sure n(z) is in data
        if "dndz_she" not in self.data:
            raise ValueError(
                "BNTMixin requires 'dndz_she' in self.data to determine "
                "the number of shear bins. This is needed to validate the "
                "shape of the provided BNT matrix."
            )

        n_she_bins = self.data["dndz_she"].shape[0]

        # check to make sure BNT matrix is present.
        if "BNT_matrix" not in self.data:
            raise ValueError(
                "BNTMixin requires a precomputed BNT matrix to be provided in "
                "data['BNT_matrix']. Please compute the BNT matrix "
                "(shape [n_she_bins, n_she_bins]) elsewhere and store it "
                "in the data dict before constructing the likelihood."
            )

        # check to make sure BNT matrix is valid.
        T = np.asarray(self.data["BNT_matrix"])
        if T.ndim != 2 or T.shape[0] != T.shape[1]:
            raise ValueError(
                f"BNT_matrix must be a square 2D array, got shape {T.shape}."
            )
        if T.shape[0] != n_she_bins:
            raise ValueError(
                f"BNT_matrix shape {T.shape} is inconsistent with the number of "
                f"shear bins inferred from data['dndz_she'] "
                f"({n_she_bins} bins)."
            )

        self.BNT_matrix = T

        #  Build projection matrix P and precompute transformed data/cov
        self._P = self._build_full_projection_matrix(T)

        base_data = super().get_data_vector_full()
        base_cov = super().get_covariance_matrix_full()

        self._data_vec_full_BNT = self._P @ base_data
        self._cov_full_BNT = self._P @ base_cov @ self._P.T

    def _build_full_projection_matrix(self, T: np.ndarray) -> np.ndarray:
        """
        Build the full projection matrix P given a BNT matrix T acting on WL
        and on the shear leg of GGL.

        P acts on the concatenated data/theory vector:

            [ GCph ; GGL ; WL ]

        where some blocks may be absent depending on the specific likelihood
        class (WL only, GCph+GGL, or full 3x2pt).
        """
        base_vec = super().get_data_vector_full()
        dim_total = base_vec.size
        n_ell = len(self.data["ells"])
        n_she = self.data["dndz_she"].shape[0]

        # How many pairs in each block?
        n_GC_pairs = len(self.GG_keys) if hasattr(self, "GG_keys") else 0
        n_GGL_pairs = len(self.GGL_keys) if hasattr(self, "GGL_keys") else 0
        n_WL_pairs = len(self.WL_keys) if hasattr(self, "WL_keys") else 0

        dim_GC = n_GC_pairs * n_ell
        dim_GGL = n_GGL_pairs * n_ell
        dim_WL = n_WL_pairs * n_ell

        if dim_GC + dim_GGL + dim_WL != dim_total:
            raise RuntimeError(
                "Inconsistent data-vector dimensions when building BNT "
                f"projection: dim_total={dim_total}, "
                f"GC={dim_GC}, GGL={dim_GGL}, WL={dim_WL}."
            )

        # Start with identity.
        P = np.eye(dim_total)

        # Offsets in the full vector.
        start_GC = 0
        end_GC = start_GC + dim_GC
        start_GGL = end_GC
        end_GGL = start_GGL + dim_GGL
        start_WL = end_GGL
        end_WL = start_WL + dim_WL

        # GGL block: transform shear index only.
        if dim_GGL > 0:
            P_GGL = self._build_GGL_projection(T, n_ell)
            P[start_GGL:end_GGL, start_GGL:end_GGL] = P_GGL

        # WL block: transform both shear indices.
        if dim_WL > 0:
            P_WL = self._build_WL_projection(T, n_ell, n_she)
            P[start_WL:end_WL, start_WL:end_WL] = P_WL

        # GCph block is left as identity (no transform).
        return P

    def _build_WL_projection(self, T: np.ndarray, n_ell: int, n_she: int) -> np.ndarray:
        """
        Build the WL subspace projection matrix P_WL.

        WL block layout (from WLMixin):
        - WL_keys:
            [("SHE","SHE", i, j) for i in 1..n_she for j in i..n_she]
        - for each key, we take cells[key][0,0], a 1D array over ell
        - final WL vector is flattened in (pair, ell) order.

        For each ell, we implement:
            C'_{ab}(ell) = sum_{i,j} T_{ai} T_{bj} C_{ij}(ell)
        using symmetry C_{ij} = C_{ji}.
        """
        if not hasattr(self, "WL_keys"):
            raise RuntimeError(
                "BNTMixin: WL projection requested but 'WL_keys' is missing. "
                "Make sure WLMixin is in the MRO."
            )

        # Use the WL pair ordering used to build the data vector
        pairs = [(key[2], key[3]) for key in self.WL_keys]
        n_pairs = len(pairs)

        # Sanity check: consistent with number of shear bins
        expected_pairs = n_she * (n_she + 1) // 2
        if n_pairs != expected_pairs:
            raise RuntimeError(
                f"BNTMixin: WL_keys length ({n_pairs}) is inconsistent with "
                f"n_she={n_she} (expected {expected_pairs} pairs)."
            )

        pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}

        # Build pair-level transformation L_pairs (n_pairs x n_pairs).
        L_pairs = np.zeros((n_pairs, n_pairs))

        for a in range(1, n_she + 1):
            for b in range(a, n_she + 1):
                out_pair = (a, b)
                out_idx = pair_to_idx[out_pair]
                row = L_pairs[out_idx]

                for i in range(1, n_she + 1):
                    for j in range(1, n_she + 1):
                        coeff = T[a - 1, i - 1] * T[b - 1, j - 1]
                        if coeff == 0.0:
                            continue
                        # Symmetry C_{ij} = C_{ji}
                        in_pair = (i, j) if i <= j else (j, i)
                        idx_in = pair_to_idx[in_pair]
                        row[idx_in] += coeff

        # Lift L_pairs to include ell: WL vector has shape (n_pairs * n_ell,).
        dim = n_pairs * n_ell
        P_WL = np.zeros((dim, dim))

        for p_out in range(n_pairs):
            for p_in in range(n_pairs):
                c = L_pairs[p_out, p_in]
                if c == 0.0:
                    continue
                for ell in range(n_ell):
                    out_idx = p_out * n_ell + ell
                    in_idx = p_in * n_ell + ell
                    P_WL[out_idx, in_idx] = c

        return P_WL

    def _build_GGL_projection(self, T: np.ndarray, n_ell: int) -> np.ndarray:
        """
        Build the GGL subspace projection matrix P_GGL.

        GGL block layout (from GGLMixin):
        - GGL_keys:
            [("POS","SHE", i, j)
             for i in 1..n_pos_bins
             for j in 1..n_she_bins]
        - for each key, we take cells[key][0], a 1D array over ell
        - final GGL vector is flattened in (pair, ell) order, where
          'pair' is (pos_bin, she_bin).

        For each ell and each position bin p, we implement:
            C'_{p,a}(ell) = sum_{j} T_{a j} C_{p,j}(ell)
        i.e. a BNT transform on the shear index only.
        """
        if "dndz_pos" not in self.data:
            raise RuntimeError(
                "GGL projection requested but 'dndz_pos' is missing in data."
            )
        if not hasattr(self, "GGL_keys"):
            raise RuntimeError(
                "BNTMixin: GGL projection requested but 'GGL_keys' is missing. "
                "Make sure GGLMixin is in the MRO."
            )

        n_pos = self.data["dndz_pos"].shape[0]
        n_she = self.data["dndz_she"].shape[0]

        # Use the GGL pair ordering used to build the data vector.
        pairs = [(key[2], key[3]) for key in self.GGL_keys]
        n_pairs = len(pairs)

        expected_pairs = n_pos * n_she
        if n_pairs != expected_pairs:
            raise RuntimeError(
                f"BNTMixin: GGL_keys length ({n_pairs}) is inconsistent with "
                f"n_pos={n_pos}, n_she={n_she} (expected {expected_pairs} pairs)."
            )

        pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}

        # Build pair-level transformation L_GGL (n_pairs x n_pairs).
        L_GGL = np.zeros((n_pairs, n_pairs))

        for p in range(1, n_pos + 1):
            for a in range(1, n_she + 1):
                out_pair = (p, a)
                out_idx = pair_to_idx[out_pair]
                row = L_GGL[out_idx]

                for j in range(1, n_she + 1):
                    in_pair = (p, j)
                    in_idx = pair_to_idx[in_pair]
                    row[in_idx] += T[a - 1, j - 1]

        # Lift L_GGL to include ell: GGL vector has shape (n_pairs * n_ell,).
        dim = n_pairs * n_ell
        P_GGL = np.zeros((dim, dim))

        for p_out in range(n_pairs):
            for p_in in range(n_pairs):
                c = L_GGL[p_out, p_in]
                if c == 0.0:
                    continue
                for ell in range(n_ell):
                    out_idx = p_out * n_ell + ell
                    in_idx = p_in * n_ell + ell
                    P_GGL[out_idx, in_idx] = c

        return P_GGL

    #  Overrides of base interface.
    def get_data_vector_full(self) -> np.ndarray:
        """Return BNT-transformed full data vector."""
        return self._data_vec_full_BNT

    def get_covariance_matrix_full(self) -> np.ndarray:
        """Return BNT-transformed full covariance matrix."""
        return self._cov_full_BNT

    def get_theory_vector_full(self, parameters: dict) -> np.ndarray:
        """
        Return BNT-transformed theory vector.

        Theory is recomputed for each parameter set using the base class,
        then transformed by the precomputed projection matrix `_P`.
        """
        base_theory = super().get_theory_vector_full(parameters)
        return self._P @ base_theory


class EuclidLikelihood_WL(WLMixin, PhotoLikelihoodBase):
    """
    EuclidLikelihood_WL computes the weak lensing (WL) likelihood for photometric surveys using Euclid data.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        WLMixin: Mixin providing weak lensing specific functionality.

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
    """

    pass


class EuclidLikelihood_GCph(GCphMixin, PhotoLikelihoodBase):
    """
    EuclidLikelihood_GCph computes the likelihood for galaxy clustering photometric (GCph) data
    using the Euclid survey specifications.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GCphMixin: Mixin providing GCph-specific functionality.

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
    """

    pass


class EuclidLikelihood_GGL(GGLMixin, PhotoLikelihoodBase):
    """
    EuclidLikelihood_GGL class for galaxy-galaxy lensing likelihood computation.

    This class combines the functionalities of PhotoLikelihoodBase and GGLMixin to compute
    the likelihood for galaxy-galaxy lensing (GGL) using photometric data. It initializes
    the necessary components and constructs a masking vector based on scale cuts for each
    GGL key.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GGLMixin: Mixin providing GGL-specific functionality.

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
    """

    pass


class EuclidLikelihood_3x2pt(GCphMixin, GGLMixin, WLMixin, PhotoLikelihoodBase):
    """
    EuclidLikelihood_3x2pt combines weak lensing (WL), galaxy clustering (GCph), and galaxy-galaxy lensing (GGL)
    likelihoods for photometric cosmological analyses, supporting scale cuts and masking.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GCphMixin: Mixin providing galaxy clustering specific functionality.
        GGLMixin: Mixin providing galaxy-galaxy lensing specific functionality.
        WLMixin: Mixin providing weak lensing specific functionality.

    Note: The order of inheritance matters due to the method resolution order (MRO) in Python
    and how mixins extend the base class functionality. Also, the order of the mixins assumes
    the ordering of the covariance matrix blocks is GCph, GGL and WL.

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
    """

    pass


class EuclidLikelihood_2x2pt(GCphMixin, GGLMixin, PhotoLikelihoodBase):
    """
    Likelihood class for Euclid 2x2pt photometric clustering and galaxy-galaxy lensing analysis.

    This class combines galaxy clustering (GCph) and galaxy-galaxy lensing (GGL) likelihoods,
    providing methods to compute the full data and theory vectors for both probes.

    Note: The order of inheritance matters due to the method resolution order (MRO) in Python
    and how mixins extend the base class functionality. Also, the order of the mixins assumes
    the ordering of the covariance matrix blocks is GCph, GGL.

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
    """

    pass


class EuclidLikelihood_WL_BNT(BNTMixin, EuclidLikelihood_WL):
    """
    EuclidLikelihood_WL_BNT computes the weak lensing (WL) likelihood in BNT basis for photometric surveys using Euclid data.

    Inherits from:
        EuclidLikelihood_WL: Base class for weak lensing likelihoods.
        BNTMixin: Mixin providing BNT specific functionality.

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
    """

    pass


class EuclidLikelihood_GGL_BNT(BNTMixin, EuclidLikelihood_GGL):
    """
    EuclidLikelihood_GGL_BNT class for galaxy-galaxy lensing likelihood computation in the BNT basis.


    Inherits from:
        EuclidLikelihood_GGL: Base class for galaxy-galaxy lensing likelihoods.
        BNTMixin: Mixin providing BNT specific functionality.

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
    """

    pass


class EuclidLikelihood_3x2pt_BNT(BNTMixin, EuclidLikelihood_3x2pt):
    """
    EuclidLikelihood__3x2pt_BNT weak lensing (WL), galaxy clustering (GCph), and galaxy-galaxy lensing (GGL)
    likelihoods for photometric cosmological analyses, supporting scale cuts and masking.
    with WL in the BNT basis.

    Inherits from:
        EuclidLikelihood_3x2pt: Base class for 3x2pt likelihoods.
        BNTMixin: Mixin providing BNT specific functionality.

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
    """

    pass


class EuclidLikelihood_2x2pt_BNT(BNTMixin, EuclidLikelihood_2x2pt):
    """
    EuclidLikelihood__2x2pt_BNT weak lensing (WL) and galaxy-galaxy lensing (GGL)
    likelihoods for photometric cosmological analyses, supporting scale cuts and masking.
    with WL in the BNT basis.

    Inherits from:
        EuclidLikelihood_2x2pt: Base class for 2x2pt likelihoods.
        BNTMixin: Mixin providing BNT specific functionality.

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
    """

    pass
