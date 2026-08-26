from typing import Optional

import numpy as np
from cloelib.summary_statistics.bao_alphas import BaryonAcousticOscillations


class EuclidLikelihood_GCspectro_BAO:
    # Single source of truth for BAO alpha ordering: this order drives both
    # the data vector and the covariance block assembly in
    # _build_observable, so the two are always kept consistent, per redshift
    # bin.
    _BAO_ORDER = ["alpha_perp", "alpha_par", "alpha_iso", "alpha_ap"]
    _BAO_FITS_KEY = {key: key.upper() for key in _BAO_ORDER}
    _BACKGROUND_GAMMA_MG = 0.545
    # As is not directly constrained by BAO data (which store sigma_8
    # instead); this placeholder is rescaled in
    # _build_fiducial_cosmology_dictionary to match the fiducial sigma_8.
    _BACKGROUND_As = 2.1e-9

    def __init__(
        self,
        data: dict,
        Background: type,
        LinearPerturbations: type,
        N_s: Optional[int] = None,
    ):
        r"""Class constructor

        Parameters
        ----------
        data: dict
            Data dictionary, whose structure is as follows. The first layer
            contains a single key, ``"GCspectro"``, for future homogenisation
            with photometric probes. In the second layer
            ``data["GCspectro"][z]`` is expected to hold the raw euclidlib
            objects for that redshift bin, under the keys ``"datavec"``
            (``BAO_alphas``) and ``"covariance"`` (``BAO_alphas_covariance``).
        Background: type
            Protocol-consistent background class
        LinearPerturbations: type
            Protocol-consistent linear perturbation class
        N_s: int
            Number of mock realisations used to estimate the covariance
            matrix, if it is a numerical (sample) covariance -- triggers the
            Hartlap (2007) debiasing correction on the inverse covariance
            matrix. ``None`` (default) means the covariance is analytic/
            theoretical, so no correction is applied.
        """
        data = data["GCspectro"]
        self.redshifts = list(data.keys())
        self.Background = Background
        self.N_s = N_s

        self._prepare(data, LinearPerturbations)

    def _prepare(self, data: dict, LinearPerturbations: type):
        r"""Builds the fiducial cosmology, ingests the raw euclidlib data for
        each redshift bin, and arranges data vectors and covariance matrices
        in the format required for :math:`\chi^2` calculation

        Parameters
        ----------
        data: dict
            ``data[z]``, for each redshift bin ``z``, holds the raw
            euclidlib objects as documented in the class constructor.
        LinearPerturbations: type
            Protocol-consistent linear perturbation class, needed to
            translate the fiducial sigma_8 into a fiducial As.
        """
        params_fid = self._build_fiducial_cosmology_dictionary(
            data[self.redshifts[0]]["datavec"].fiducial_cosmology, LinearPerturbations
        )
        self.background_fiducial = self.Background(**params_fid)

        self.data = {z: self._build_observable(data[z]) for z in self.redshifts}

        self._build_data_vector()
        self._build_covariance_matrix()
        self._invert_covariance_matrix()

    def _build_fiducial_cosmology_dictionary(
        self, fiducial_cosmology: dict, LinearPerturbations: type
    ) -> dict:
        r"""Builds the fiducial cosmology dictionary expected by ``Background``
        from the euclidlib fiducial cosmology of one redshift bin.

        Parameters
        ----------
        fiducial_cosmology: dict
            Fiducial cosmology as read by euclidlib from a FITS header.
            Stores ``Omega_m0`` and ``sigma_8``, which need to be converted
            to ``Omega_cdm0`` and ``As`` respectively.
        LinearPerturbations: type
            Protocol-consistent linear perturbation class, needed to
            translate the fiducial sigma_8 into a fiducial As.

        Returns
        -------
        params_fid: dict
            Fiducial cosmology dictionary, ready to be passed to ``Background``
        """
        Omega_cdm0 = (
            fiducial_cosmology["Omega_m0"] - fiducial_cosmology["Omega_b0"]
            if "Omega_m0" in fiducial_cosmology
            else fiducial_cosmology["Omega_cdm0"]
        )
        params_fid = {
            "H0": fiducial_cosmology["H0"],
            "Omega_cdm0": Omega_cdm0,
            "Omega_b0": fiducial_cosmology["Omega_b0"],
            "Omega_k0": fiducial_cosmology.get("Omega_k0", 0.0),
            "mnu": fiducial_cosmology.get("mnu", 0.0),
            "N_mnu": fiducial_cosmology.get("N_mnu", 0),
            "w0": fiducial_cosmology.get("w0", -1.0),
            "wa": fiducial_cosmology.get("wa", 0.0),
            "ns": fiducial_cosmology["ns"],
            "gamma_MG": self._BACKGROUND_GAMMA_MG,
            "As": self._BACKGROUND_As,
        }

        # BAO fiducial cosmology stores sigma_8, we need As
        tmp_background = self.Background(**params_fid)
        tmp_lin_pert = LinearPerturbations(tmp_background, np.array([0.0]))
        params_fid["As"] *= (
            fiducial_cosmology["sigma_8"] / tmp_lin_pert.sigma8_0()
        ) ** 2

        return params_fid

    def _build_observable(self, data_z: dict) -> dict:
        r"""Assembles the BAO alphas and covariance for one redshift bin,
        from raw euclidlib objects.

        Parameters
        ----------
        data_z: dict
            Raw euclidlib objects for one redshift bin: ``"datavec"`` and
            ``"covariance"`` are required.

        Returns
        -------
        entry: dict
            Data dictionary for this redshift bin, in the internal format
            used by the rest of the class.
        """
        dv, cv = data_z["datavec"], data_z["covariance"]
        params = [
            key for key in self._BAO_ORDER if not np.isnan(getattr(dv, key)).all()
        ]
        observables = [self._BAO_FITS_KEY[key] for key in params]
        return {
            "params": params,
            "data": {param: getattr(dv, param) for param in params},
            "covariance": np.block(
                [
                    [cv.covariance[f"{oi}-{oj}"] for oj in observables]
                    for oi in observables
                ]
            ),
        }

    def _build_data_vector(self):
        r"""Arranges the BAO data into a flattened data vector"""
        self.data_vector = np.squeeze(
            np.concatenate(
                [
                    [self.data[z]["data"][param] for param in self.data[z]["params"]]
                    for z in self.redshifts
                ]
            )
        )

    def _build_covariance_matrix(self):
        r"""Arranges the BAO covariance into a matrix form"""
        cov_blocks = [self.data[z]["covariance"] for z in self.redshifts]
        self.flattened_covariance_matrix = np.block(
            [
                [
                    block if i == j else np.zeros_like(block)
                    for j, block in enumerate(cov_blocks)
                ]
                for i, _ in enumerate(cov_blocks)
            ]
        )

    def hartlap_factor(self) -> float:
        r"""Hartlap factor to correct the biased estimator for the inverse covariance
        matrix as obtained from a finite number of mock realisations.

        Returns
        -------
        factor: float
            Hartlap factor.

        Raises
        ------
        ValueError
            If ``N_s <= N_d + 2``, since the debiasing factor would then be
            non-positive.
        """
        N_d = self.flattened_covariance_matrix.shape[0]
        if self.N_s <= N_d + 2:
            raise ValueError(
                f"Hartlap correction requires N_s > N_d + 2 (got N_s={self.N_s}, "
                f"N_d={N_d}): the debiasing factor would be non-positive."
            )
        return (self.N_s - N_d - 2) / (self.N_s - 1)

    def _invert_covariance_matrix(self):
        r"""Invert GCspectro covariance matrix, and apply Hartlap factor if needed."""
        self.inverse_covariance_matrix = np.linalg.inv(self.flattened_covariance_matrix)
        if self.N_s is not None:
            self.inverse_covariance_matrix *= self.hartlap_factor()

    def get_theory_vector(self, parameters: dict) -> np.ndarray:
        r"""Generate theory vectors based on specified parameters
        Parameters
        ----------
        parameters: dict
            Input parameters
        Return
        ------
        theory_vec: np.ndarray
            Stacked theory vector
        """
        background = self.Background(
            H0=parameters["H0"],
            Omega_cdm0=parameters["Omega_cdm0"],
            Omega_b0=parameters["Omega_b0"],
            Omega_k0=parameters["Omega_k0"],
            w0=parameters["w0"],
            wa=parameters["wa"],
            ns=parameters["ns"],
            As=parameters["As"],
            gamma_MG=parameters["gamma_MG"],
            mnu=parameters["mnu"],
            N_mnu=parameters["N_mnu"],
        )

        zs = np.float64(self.redshifts)
        alphas_dict = BaryonAcousticOscillations(
            background=background,
            background_fiducial=self.background_fiducial,
            redshifts=zs,
        ).alphas_dict

        theory_vec = np.concatenate(
            [
                [alphas_dict[zf][params] for params in self.data[z]["params"]]
                for z, zf in zip(self.redshifts, zs)
            ]
        )
        return theory_vec

    def loglike(self, parameters: dict):
        r"""Log-likelihood of BAO probe
        Parameters
        ----------
        parameters: dict
            Ensemble of cosmological parameters
        """
        self.theory_vector = self.get_theory_vector(parameters)
        diff = self.theory_vector - self.data_vector
        chi2 = np.dot(np.dot(diff, self.inverse_covariance_matrix), diff)
        return -0.5 * chi2
