"""Real-space 3x2pt likelihood (GC, GGL, WL) using Dark Emulator.

Single class for real data: optional data_gc (w_p), data_ggl (ΔΣ), data_wl (ξ±).
Use one or any combination; loglike is Gaussian (χ²) over provided probes.
"""

import numpy as np
from typing import Dict, Optional, List

from cloelib.cosmology.darkemu_cosmology import DarkEmuHODPerturbations
from cloelib.observables.darkemu_hod import DarkEmuHODParameters


def _extract_cosmo_params(parameters: dict) -> dict:
    cosmo_keys = [
        "H0",
        "Omega_cdm0",
        "Omega_b0",
        "Omega_k0",
        "w0",
        "wa",
        "ns",
        "As",
        "gamma_MG",
        "mnu",
        "N_mnu",
    ]
    out = {k: parameters[k] for k in cosmo_keys if k in parameters}
    out.setdefault("Omega_k0", 0.0)
    out.setdefault("wa", 0.0)
    out.setdefault("gamma_MG", 0.55)
    return out


def _extract_hod_params(parameters: dict, fixed_hod: dict) -> DarkEmuHODParameters:
    if "hod_params" in parameters:
        return parameters["hod_params"]
    hod_keys = [
        "logMmin",
        "sigma_sq",
        "logM1",
        "alpha",
        "kappa",
        "poff",
        "Roff",
        "sat_dist_type",
        "alpha_inc",
        "logM_inc",
    ]
    hod_dict = {}
    for k in hod_keys:
        if k in parameters:
            hod_dict[k] = parameters[k]
        elif k in fixed_hod:
            hod_dict[k] = fixed_hod[k]
    hod_dict.setdefault("kappa", 0.0)
    hod_dict.setdefault("poff", 0.0)
    hod_dict.setdefault("Roff", 0.0)
    hod_dict.setdefault("sat_dist_type", "emulator")
    return DarkEmuHODParameters(**hod_dict)


class EuclidLikelihood_DarkEmu_RealSpace:
    """Single real-space likelihood: GC (w_p), GGL (ΔΣ), WL (ξ±) in any combination.

    Pass only the data you have: data_gc, data_ggl, and/or data_wl.
    loglike(parameters) returns the combined Gaussian log-likelihood.
    """

    def __init__(
        self,
        data_gc: Optional[Dict] = None,
        data_ggl: Optional[Dict] = None,
        data_wl: Optional[Dict] = None,
        settings: Optional[Dict] = None,
        Background: type = None,
        LinPerturbations: type = None,
        NonLinPerturbations: type = None,
    ):
        if settings is None:
            settings = {}
        self.settings = settings
        self.Background = Background
        self.data_gc = data_gc
        self.data_ggl = data_ggl
        self.data_wl = data_wl
        self.fixed_hod = settings.get("hod_params_fixed", {})

        self.has_gc = data_gc is not None
        self.has_ggl = data_ggl is not None
        self.has_wl = data_wl is not None
        self.LinPerturbations = LinPerturbations
        self.NonLinPerturbations = NonLinPerturbations

        self._setup_data()

    def _setup_data(self):
        parts: List[np.ndarray] = []
        inv_blocks: List[np.ndarray] = []

        if self.has_gc:
            R_min = self.settings.get("R_min_gc", 0.0)
            R_max = self.settings.get("R_max_gc", np.inf)
            R = np.asarray(self.data_gc["R_bins"])
            mask = (R >= R_min) & (R <= R_max)
            self._R_gc = R[mask]
            self._wp_data = np.asarray(self.data_gc["wp"])[mask]
            cov = np.asarray(self.data_gc["covariance"])
            n = int(mask.sum())
            inv_cov = np.linalg.inv(cov[np.outer(mask, mask)].reshape(n, n))
            parts.append(self._wp_data)
            inv_blocks.append(inv_cov)
            self._z_sample = self.data_gc["z_sample"]
            self._pimax = self.data_gc.get("pimax", None)

        if self.has_ggl:
            R_min = self.settings.get("R_min_ggl", 0.0)
            R_max = self.settings.get("R_max_ggl", np.inf)
            R = np.asarray(self.data_ggl["R_bins"])
            mask = (R >= R_min) & (R <= R_max)
            self._R_ggl = R[mask]
            self._ds_data = np.asarray(self.data_ggl["delta_sigma"])[mask]
            cov = np.asarray(self.data_ggl["covariance"])
            n = int(mask.sum())
            inv_cov = np.linalg.inv(cov[np.outer(mask, mask)].reshape(n, n))
            parts.append(self._ds_data)
            inv_blocks.append(inv_cov)
            self._z_lens = self.data_ggl["z_lens"]
            if not self.has_gc:
                self._z_sample = (
                    self._z_lens if np.isscalar(self._z_lens) else self._z_lens[0]
                )

        if self.has_wl:
            self._setup_wl_data()
            parts.append(self._xi_data)
            inv_blocks.append(self._inv_cov_wl)

        self._data_vector = np.concatenate(parts) if parts else np.array([])
        if inv_blocks:
            from scipy.linalg import block_diag

            self._inv_cov = block_diag(*inv_blocks)
        else:
            self._inv_cov = np.array([]).reshape(0, 0)

        # Expose data point counts and slices for reporting/plotting
        self.n_gc = len(self._wp_data) if self.has_gc else 0
        self.n_ggl = len(self._ds_data) if self.has_ggl else 0
        self.n_wl = len(self._xi_data) if self.has_wl else 0
        self.n_xi_plus = getattr(self, "_n_xi_plus", 0)
        self.n_xi_minus = getattr(self, "_n_xi_minus", 0)
        self._start_gc = 0
        self._start_ggl = self.n_gc
        self._start_wl = self.n_gc + self.n_ggl

    @property
    def data_vector(self) -> np.ndarray:
        """Combined data vector (GC, GGL, WL after scale cuts)."""
        return self._data_vector

    @property
    def start_ggl(self) -> int:
        """Start index of GGL block in data_vector / theory vector."""
        return self._start_ggl

    @property
    def R_bins_ggl(self) -> np.ndarray:
        """GGL R bins after scale cuts (None if no GGL)."""
        return self._R_ggl if self.has_ggl else None

    @property
    def R_bins_gc(self) -> np.ndarray:
        """GC R bins after scale cuts (None if no GC)."""
        return self._R_gc if self.has_gc else None

    def _setup_wl_data(self):
        import jax.numpy as jnp

        wl_settings = self.settings.get("wl_settings", {})
        theta_min_p = wl_settings.get("theta_min_plus", 0.0)
        theta_max_p = wl_settings.get("theta_max_plus", np.inf)
        theta_min_m = wl_settings.get("theta_min_minus", 0.0)
        theta_max_m = wl_settings.get("theta_max_minus", np.inf)

        theta = np.asarray(self.data_wl["theta"])
        self._theta_rad = theta * np.pi / 180.0 / 60.0
        self._mask_xi_p = (theta >= theta_min_p) & (theta <= theta_max_p)
        self._mask_xi_m = (theta >= theta_min_m) & (theta <= theta_max_m)

        self._dndz = jnp.asarray(self.data_wl["dndz"])
        self._z_arr_wl = np.asarray(self.data_wl["z_arr"])
        n_tomo = self._dndz.shape[0]
        self._bin_pairs = [
            (i, j) for i in range(1, n_tomo + 1) for j in range(i, n_tomo + 1)
        ]

        xi_p_list = []
        xi_m_list = []
        for i, j in self._bin_pairs:
            xi_p_list.append(
                np.asarray(self.data_wl["xi_plus"][(i, j)])[self._mask_xi_p]
            )
            xi_m_list.append(
                np.asarray(self.data_wl["xi_minus"][(i, j)])[self._mask_xi_m]
            )
        self._xi_data = np.concatenate(xi_p_list + xi_m_list)
        self._n_xi_plus = sum(len(x) for x in xi_p_list)
        self._n_xi_minus = sum(len(x) for x in xi_m_list)

        n_theta = len(theta)
        n_pairs = len(self._bin_pairs)
        idx_p = []
        idx_m = []
        for p in range(n_pairs):
            idx_p.extend(p * n_theta + np.where(self._mask_xi_p)[0])
            idx_m.extend(n_pairs * n_theta + p * n_theta + np.where(self._mask_xi_m)[0])
        idx_all = np.array(idx_p + idx_m)
        cov_full = np.asarray(self.data_wl["covariance"])
        cov_cut = cov_full[np.ix_(idx_all, idx_all)]
        self._inv_cov_wl = np.linalg.inv(cov_cut)

        self._wl_nuisance = {"AIA": 1.0, "EtaIA": 0.0, "CIA": 0.0134}
        for i in range(1, n_tomo + 1):
            self._wl_nuisance[f"multiplicative_bias_{i}"] = 0.0
            self._wl_nuisance[f"dz_shear_{i}"] = 0.0
        self._ells = jnp.unique(jnp.geomspace(2, 15000, 500).astype(int))
        self._ks = jnp.geomspace(1e-4, 50.0, 200)

    def get_theory_vector(self, parameters: dict) -> np.ndarray:
        parts: List[np.ndarray] = []
        cosmo = _extract_cosmo_params(parameters)
        hod = _extract_hod_params(parameters, self.fixed_hod)
        background = self.Background(**cosmo)

        if self.has_gc or self.has_ggl:
            z = self._z_sample if np.isscalar(self._z_sample) else self._z_sample[0]
            pert = DarkEmuHODPerturbations(
                background=background,
                redshifts=np.array([z]),
                hod_params=hod,
                validate_params=False,
            )
        if self.has_gc:
            wp = pert.projected_correlation(self._R_gc, z, pimax=self._pimax)
            parts.append(wp)
        if self.has_ggl:
            z_ggl = self._z_lens if np.isscalar(self._z_lens) else self._z_lens[0]
            if not self.has_gc:
                pert = DarkEmuHODPerturbations(
                    background=background,
                    redshifts=np.array([z_ggl]),
                    hod_params=hod,
                    validate_params=False,
                )
            ds = pert.delta_sigma(self._R_ggl, z_ggl)
            parts.append(ds)

        if self.has_wl:
            xi_wl = self._get_wl_theory(parameters, cosmo)
            parts.append(xi_wl)

        return np.concatenate(parts) if parts else np.array([])

    def _get_wl_theory(self, parameters: dict, cosmo: dict) -> np.ndarray:
        import jax.numpy as jnp
        from cloelib.cosmology.HMcode2020Emu_cosmology import (
            HMemuLinearPerturbations,
            HMemuNonLinearPerturbations,
        )
        from cloelib.observables.photo import ShearTracer
        from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
        from cloelib.summary_statistics.angular_correlation_function_wigner import (
            AngularCorrelationFunctionWigner,
        )

        background = self.Background(**cosmo)
        lp = HMemuLinearPerturbations(background, self._z_arr_wl)
        nlp = HMemuNonLinearPerturbations(
            background,
            lp,
            self._z_arr_wl,
            log10TAGN=parameters.get("log10TAGN", 7.8),
        )
        nuisance = {k: parameters.get(k, v) for k, v in self._wl_nuisance.items()}
        shear = ShearTracer(
            perturbations=nlp,
            dndz=self._dndz,
            z=jnp.asarray(self._z_arr_wl),
            nuisance_params=nuisance,
        )
        atp = AngularTwoPoint(shear, shear)
        acf = AngularCorrelationFunctionWigner(atp, self._ells, self._ks)
        xi_dict = acf.get_xi(jnp.asarray(self._theta_rad))

        xi_p_list = []
        xi_m_list = []
        for i, j in self._bin_pairs:
            key = ("SHE", "SHE", i, j)
            tpcf = xi_dict[key]
            xi_p_list.append(np.asarray(tpcf.array[0, 0, :])[self._mask_xi_p])
            xi_m_list.append(np.asarray(tpcf.array[1, 1, :])[self._mask_xi_m])
        return np.concatenate(xi_p_list + xi_m_list)

    def loglike(self, parameters: dict) -> float:
        try:
            theory = self.get_theory_vector(parameters)
            diff = theory - self._data_vector
            return float(-0.5 * (diff @ self._inv_cov @ diff))
        except (ValueError, RuntimeError):
            return -np.inf
