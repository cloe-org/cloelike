import numpy as np

from copy import deepcopy
from dataclasses import replace
from scipy.linalg import block_diag
from typing import Optional

from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles
from cloelib.summary_statistics.bao_alphas import BaryonAcousticOscillations


class EuclidLikelihood_GCspectro_Pls_BAO:
    # Single source of truth for BAO alpha ordering: this order drives both
    # the alphas data vector and the covariance block assembly in
    # _build_observable, so the two are always kept consistent, per redshift bin.
    _BAO_ORDER = ["alpha_perp", "alpha_par", "alpha_iso", "alpha_ap"]
    _BAO_FITS_KEY = {
        "alpha_perp": "ALPHA_PERP",
        "alpha_par": "ALPHA_PAR",
        "alpha_iso": "ALPHA_ISO",
        "alpha_ap": "ALPHA_AP",
    }
    # Background requires As and gamma_MG to be instantiated, but
    # background_fiducial is only ever used for rdrag/H(z)/D_A(z) (see
    # BaryonAcousticOscillations, LegendreMultipoles), which do not depend
    # on either -- these are placeholders, not fiducial choices.
    _BACKGROUND_As = 2.1e-9
    _BACKGROUND_GAMMA_MG = 0.545

    def __init__(
        self,
        data: dict,
        settings: dict,
        Background: type,
        SpectroPower: type,
        Perturbations: Optional[type] = None,
        AM_priors: Optional[dict] = None,
    ):
        r"""Class constructor

        Parameters
        ----------
        data: dict
            Data dictionary, whose structure is as follows. The first layer
            contains a single key, ``"GCspectro"``, for future homogenisation
            with photometric probes. In the second layer
            ``data["GCspectro"][z]`` is expected to hold the raw euclidlib
            objects for that redshift bin, under the keys ``"datavec_FS"``
            (``PowerSpectrumMultipoles``), ``"datavec_BAO"``
            (``BaryonAcousticOscillations``, from
            ``euclidlib.le3.bao_gc.BAO_alphas``), and ``"covariance"`` (the
            joint FS+BAO ``PowerSpectrumMultipolesCovariance``) -- all three
            required; optionally ``"mixing"``
            (``PowerSpectrumMultipolesMixingMatrix``).
        settings: dict
            Settings dictionary, whose structure is as follows. The first layer
            contains a single key, "GCspectro", for future homogenisation with
            photometric probes. The second layer also contains a single keyword,
            ``"scale_cuts"``. in the third layer ``settings["GCspectro"][z]``
            is expected to be a dictionary containing pairs ``"ell<i>: [a,b]"``,
            where ``"i"`` runs over 0,2,4 and ``"a,b"`` represent the range of
            modes included in the fit.
        Background: type
            Protocol-consistent Background class
        SpectroPower: type
            Protocol-consistent SpectroPower class
        Perturbations: type
            Protocol-consistent Perturbations class (optional, only for
            cases that require linear P(k) input, such as PBJ)
        AM_priors: dict
            Mean and standard deviation of gaussian priors for analytical
            marginalisation
        """
        data = data["GCspectro"]
        self.settings = settings["GCspectro"]
        self.scale_cuts = self.settings["scale_cuts"]
        self.redshifts = list(data.keys())

        self.NLcode = SpectroPower.NLcode
        if self.NLcode == "COMET":
            self.RSDmodel = SpectroPower.RSDmodel

        self.ells = [0, 2, 4]
        self.Background = Background
        self.Perturbations = Perturbations
        self.SpectroPower = SpectroPower

        self._prepare(data)

        if self.NLcode == "COMET":
            self.RSD_parameter_names = [
                "b1",
                "b2",
                "bG2",
                "bGam3",
                "c0",
                "c2",
                "c4",
            ]
            if self.RSDmodel == "EFTofLSS":
                self.RSD_parameter_names.append("cnlo")
            elif self.RSDmodel == "VDG_infty":
                self.RSD_parameter_names.append("avir")
        elif self.NLcode == "PBJ":
            self.RSD_parameter_names = [
                "b1",
                "b2",
                "bG2",
                "bG3",
                "c0",
                "c2",
                "c4",
                "ck4",
            ]
        else:
            raise ValueError(f"Unsupported NL code: {self.NLcode}")

        self.noise_syst_parameter_names = ["NP0", "NP20", "NP22", "fout", "sigmaz"]

        self.AM_priors = AM_priors
        if self.AM_priors:
            unmatched_z = [z for z in AM_priors.keys() if z not in self.redshifts]
            if unmatched_z:
                raise ValueError(f"Redshifts {unmatched_z} not found in data.")

            if self.NLcode == "COMET":
                self.AM_par_to_diag = {
                    "bGam3": ["b1-bGam3", "bGam3"],
                    "c0": ["c0"],
                    "c2": ["c2"],
                    "c4": ["c4"],
                    "NP0": ["noise_k0"],
                    "NP20": ["noise_k2"],
                    "NP22": ["noise_k2mu2"],
                }
                if self.RSDmodel == "EFTofLSS":
                    self.AM_par_to_diag["cnlo"] = ["b1-b1-cnlo", "b1-cnlo", "cnlo"]
            elif self.NLcode == "PBJ":
                self.AM_par_to_diag = {
                    "bG3": ["bG3"],
                    "c0": ["c0"],
                    "c2": ["c2"],
                    "c4": ["c4"],
                    "ck4": ["ck4"],
                    "NP0": ["noise_k0"],
                    "NP20": ["noise_k2"],
                    "NP22": ["noise_k2mu2"],
                }

            self.AM_diagrams = [
                term for values in self.AM_par_to_diag.values() for term in values
            ]

            self.AM_params = {}
            AM_means = []
            AM_sigmas = []
            for z in self.AM_priors.keys():
                self.AM_params[z] = list(self.AM_priors[z].keys())
                AM_means.extend(
                    [self.AM_priors[z][par][0] for par in self.AM_priors[z].keys()]
                )
                AM_sigmas.extend(
                    [self.AM_priors[z][par][1] for par in self.AM_priors[z].keys()]
                )
            self.AM_means = np.array(AM_means)
            self.AM_sigmas = np.array(AM_sigmas)

    def _prepare(self, data: dict):
        r"""Builds the fiducial cosmology, ingests the raw euclidlib data for
        each redshift bin, and arranges data vectors and covariance matrices
        in the format required for :math:`\chi^2` calculation

        Parameters
        ----------
        data: dict
            ``data[z]``, for each redshift bin ``z``, holds the raw
            euclidlib objects as documented in the class constructor.
        """
        params_fid = self._build_fiducial_cosmology_dictionary(
            data[self.redshifts[0]]["datavec_FS"].fiducial_cosmology
        )
        self.background_fiducial = self.Background(**params_fid)
        fid_h = params_fid["H0"] / 100.0

        self.data = {z: self._build_observable(data[z], fid_h) for z in self.redshifts}
        self.BAO_params = {z: self.data[z]["alphas"].keys() for z in self.redshifts}
        self.nbar = [self.data[z]["nbar"] for z in self.redshifts]

        self.mixmat = (
            {z: self.data[z]["mixing_matrix"] for z in self.redshifts}
            if all("mixing_matrix" in self.data[z] for z in self.redshifts)
            else None
        )

        self._build_data_vector()
        self._build_covariance_matrix()
        self._build_masking_vector()
        self._mask_data_vector()
        self._mask_covariance_matrix()
        self._invert_covariance_matrix()

    def _build_fiducial_cosmology_dictionary(self, fiducial_cosmology: dict) -> dict:
        r"""Builds the fiducial cosmology dictionary expected by ``Background``
        from the euclidlib fiducial cosmology of one redshift bin.

        Parameters
        ----------
        fiducial_cosmology: dict
            Fiducial cosmology as read by euclidlib from a FITS header

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
        return {
            "H0": fiducial_cosmology["H0"],
            "Omega_cdm0": Omega_cdm0,
            "Omega_b0": fiducial_cosmology["Omega_b0"],
            "Omega_k0": fiducial_cosmology.get("Omega_k0", 0.0),
            "mnu": fiducial_cosmology.get("mnu", 0.0),
            "N_mnu": fiducial_cosmology.get("N_mnu", 0),
            "w0": fiducial_cosmology.get("w0", -1.0),
            "wa": fiducial_cosmology.get("wa", 0.0),
            "ns": fiducial_cosmology["ns"],
            "As": self._BACKGROUND_As,
            "gamma_MG": self._BACKGROUND_GAMMA_MG,
        }

    def _build_observable(self, data_z: dict, fid_h: float) -> dict:
        r"""Rescales units (Mpc/h to Mpc) and assembles the BAO alphas and
        covariance for one redshift bin, from raw euclidlib objects.

        The order in which BAO alphas appear in the data vector for this
        redshift bin is determined here, once, from ``_BAO_ORDER`` and the
        alphas actually available (i.e. not all-NaN) in
        ``data_z["datavec_BAO"]``. The same order is reused immediately below
        to assemble the matching covariance block, so the two can never
        drift out of sync -- and different redshift bins are free to carry
        different sets of alphas.

        Parameters
        ----------
        data_z: dict
            Raw euclidlib objects for one redshift bin: ``"datavec_FS"``,
            ``"datavec_BAO"``, and ``"covariance"`` (the joint FS+BAO
            covariance) are required; ``"mixing"`` is optional.
        fid_h: float
            Fiducial value of :math:`H_0/100`, used for unit rescaling.

        Returns
        -------
        entry: dict
            Data dictionary for this redshift bin, in the internal format
            used by the rest of the class.
        """
        dv, cv = data_z["datavec_FS"], data_z["covariance"]
        k_fac, pk_fac, cov_fac = fid_h, 1.0 / fid_h**3, 1.0 / fid_h**6

        entry = {"nbar": dv.nbar * fid_h**3, "k": dv.keff * k_fac}
        entry.update({f"pk{ell}": dv.multipoles[ell] * pk_fac for ell in self.ells})

        bao = data_z["datavec_BAO"]
        bao_keys = [
            key for key in self._BAO_ORDER if not np.isnan(getattr(bao, key)).all()
        ]
        entry["alphas"] = {
            key: np.asarray(getattr(bao, key)).item() for key in bao_keys
        }

        ells = [str(ell) for ell in self.ells]
        observables = ells + [self._BAO_FITS_KEY[key] for key in bao_keys]
        entry["covariance"] = np.block(
            [
                [
                    cv.covariance[f"{oi}-{oj}"]
                    * (
                        cov_fac
                        if oi in ells and oj in ells
                        else np.sqrt(cov_fac)
                        if oi in ells or oj in ells
                        else 1.0
                    )
                    for oj in observables
                ]
                for oi in observables
            ]
        )

        if "mixing" in data_z:
            mm = data_z["mixing"]
            entry["mixing_matrix"] = replace(
                mm,
                kout=mm.kout * k_fac,
                kin={ell: val * k_fac for ell, val in mm.kin.items()},
                mixing={key: val.squeeze() for key, val in mm.mixing.items()},
            )

        return entry

    def _build_data_vector(self):
        r"""Arranges full shape and BAO data into a flattened data vector"""
        self.data_vector = np.concatenate(
            [
                np.concatenate(
                    [self.data[z][f"pk{ell}"] for ell in self.ells]
                    + [list(self.data[z]["alphas"].values())]
                )
                for z in self.redshifts
            ]
        )

    def _build_covariance_matrix(self):
        r"""Arranges the full shape and BAO covariances into a matrix form"""
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

    def _masking(self, arr: np.ndarray, interval: list) -> np.ndarray:
        r"""Get a 1/0 mask for the elements of arr contained in interval
        Parameters
        ----------
        arr: numpy.ndarray
            Input array
        interval: list
            Edges defining the masking region
        Returns
        -------
        masked_arr: numpy.ndarray
            Masked array
        """
        return (arr >= interval[0]) & (arr <= interval[1])

    def _build_masking_vector(self):
        r"""Computes the masking vector for the combination of full shape and BAO"""
        self.masking_vector = np.concatenate(
            [
                np.concatenate(
                    [
                        self._masking(
                            self.data[z]["k"], np.array(self.scale_cuts[z][f"ell{ell}"])
                        )
                        for ell in self.ells
                    ]
                    + [np.array([True]) for _ in self.BAO_params[z]]
                )
                for z in self.redshifts
            ]
        )

    def _mask_data_vector(self):
        r"""Mask GCspectro data vector"""
        self.masked_data_vector = self.data_vector[self.masking_vector]

    def _mask_covariance_matrix(self):
        r"""Mask GCspectro covariance matrix"""
        self.masked_covariance_matrix = self.flattened_covariance_matrix[
            self.masking_vector
        ][:, self.masking_vector]

    def _invert_covariance_matrix(self):
        r"""Invert GCspectro covariance matrix"""
        self.inverse_masked_covariance_matrix = np.linalg.inv(
            self.masked_covariance_matrix
        )

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

        if self.Perturbations is not None and self.NLcode in ["PBJ"]:
            zs = np.float64(self.redshifts)
            cosmo_input = self.Perturbations(background, zs)
        elif self.NLcode in ["COMET"]:
            cosmo_input = background
        else:
            raise ValueError("Perturbations are required for PBJ, but not for COMET.")

        alphas_dict = BaryonAcousticOscillations(
            background=background,
            background_fiducial=self.background_fiducial,
            redshifts=self.redshifts,
        ).alphas_dict

        theory_vec = []

        for i, z in enumerate(self.redshifts):
            RSD_params = {key: parameters[key][i] for key in self.RSD_parameter_names}
            syst_params = {
                key: parameters[key][i] for key in self.noise_syst_parameter_names
            }

            power = self.SpectroPower(
                cosmo_input, RSD_parameters=RSD_params, redshift=float(z)
            )
            obs = LegendreMultipoles(
                spectro_power=power,
                background_fiducial=self.background_fiducial,
                parameters=syst_params,
                nbar=self.nbar[i],
            )

            if self.mixmat:
                mps = obs.convolved_power_multipoles(
                    self.mixmat[z], ells=self.ells, use_AP=True
                )
            else:
                k = self.data[z]["k"]
                mps = obs.power_multipoles(k=k, ells=self.ells, use_AP=True)

            vector = np.concatenate(
                [mps[f"ell{ell}"] for ell in self.ells]
                + [np.atleast_1d(alphas_dict[z][param]) for param in self.BAO_params[z]]
            )
            theory_vec.extend(vector)

        return np.array(theory_vec)

    def _mask_theory_vector(self):
        r"""Mask theory vector"""
        self.masked_theory_vector = self.theory_vector[self.masking_vector]

    def loglike(self, parameters: dict):
        r"""Log-likelihood of GCspectro probe

        Parameters
        ----------
        parameters: dict
            Ensemble of cosmological and nuisance parameters

        Returns
        -------
            loglike: float
                Value of the log-likelihood
        """
        self.theory_vector = self.get_theory_vector(parameters)
        self._mask_theory_vector()
        diff = self.masked_theory_vector - self.masked_data_vector
        chi2 = np.dot(np.dot(diff, self.inverse_masked_covariance_matrix), diff)
        return -0.5 * chi2

    def get_theory_vector_AM(self, parameters: dict, term_list: dict):
        r"""Generate theory vectors based on specified parameters for
        analytical marginalization

        Parameters
        ----------
        parameters: dict
            Input parameters

        Return
        ------
        theory_vec: np.ndarray
            Stacked theory vector
        theory_vec_AM: np.ndarray
            Stacked terms for analytical marginalization
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

        if self.Perturbations is not None and self.NLcode in ["PBJ"]:
            zs = np.float64(self.redshifts)
            cosmo_input = self.Perturbations(background, zs)
        elif self.NLcode in ["COMET"]:
            cosmo_input = background
        else:
            raise ValueError("Perturbations are required for PBJ, but not for COMET.")

        alphas_dict = BaryonAcousticOscillations(
            background=background,
            background_fiducial=self.background_fiducial,
            redshifts=self.redshifts,
        ).alphas_dict

        theory_vec = []
        theory_vec_AM = {}
        coeff = self._coeff_AM(parameters)
        for i, z in enumerate(self.redshifts):
            RSD_parameters = {
                key: parameters[key][i] for key in self.RSD_parameter_names
            }
            power = self.SpectroPower(
                cosmo_input, RSD_parameters=RSD_parameters, redshift=float(z)
            )
            nois_syst_parameters = {
                key: parameters[key][i] for key in self.noise_syst_parameter_names
            }
            obs = LegendreMultipoles(
                spectro_power=power,
                background_fiducial=self.background_fiducial,
                parameters=nois_syst_parameters,
                nbar=self.nbar[i],
            )
            k = self.data[z]["k"]
            if self.mixmat:
                mps_dict = obs.convolved_power_multipoles(
                    self.mixmat[z], ells=self.ells, use_AP=True
                )
            else:
                mps_dict = obs.power_multipoles(k=k, ells=self.ells, use_AP=True)

            vector = np.concatenate(
                [mps_dict[f"ell{ell}"] for ell in self.ells]
                + [np.atleast_1d(alphas_dict[z][param]) for param in self.BAO_params[z]]
            )
            theory_vec = np.concatenate((theory_vec, vector))

            if z in self.AM_params.keys():
                if self.mixmat:
                    mps_AM_dict = obs.convolved_power_term_multipoles(
                        self.mixmat[z],
                        term_list=term_list[z],
                        ells=self.ells,
                        use_AP=True,
                    )
                else:
                    mps_AM_dict = obs.power_term_multipoles(
                        k=k, term_list=term_list[z], ells=self.ells, use_AP=True
                    )
                terms_to_scale = [term for term in term_list[z] if term in coeff]
                indices_to_scale = [term_list[z].index(term) for term in terms_to_scale]
                for ell in self.ells:
                    for idx, term in zip(indices_to_scale, terms_to_scale):
                        mps_AM_dict[f"ell{ell}"][idx, :] *= coeff[term][i]
                mps_AM_list = [mps_AM_dict[f"ell{ell}"] for ell in self.ells]
                theory_vec_AM[z] = np.hstack(mps_AM_list)
                theory_vec_AM[z] = np.hstack(
                    (
                        theory_vec_AM[z],
                        np.zeros((theory_vec_AM[z].shape[0], len(self.BAO_params[z]))),
                    )
                )
            else:
                Nk = sum(len(self.data[z][f"pk{ell}"]) for ell in self.ells)
                theory_vec_AM[z] = np.zeros((0, Nk))
                theory_vec_AM[z] = np.hstack(
                    (
                        theory_vec_AM[z],
                        np.zeros((theory_vec_AM[z].shape[0], len(self.BAO_params[z]))),
                    )
                )

        return theory_vec, theory_vec_AM

    def _mask_theory_vector_AM(self):
        r"""Mask theory vector"""
        self.masked_theory_vector = self.theory_vector[self.masking_vector]
        self.masked_theory_vector_AM_reduced = self.theory_vector_AM_reduced[
            :, self.masking_vector
        ]

    # This should be different for non-linear codes different from COMET
    def _coeff_AM(self, parameters: dict):
        r"""Coefficients of individual terms for analytical marginalisation

        Parameters
        ----------
        parameters: dict
            Input parameters

        Returns
        -------
        coeff: dict
            Coefficients of individual terms to be analytically marginalised
        """
        coeff = {
            "COMET": {
                "b1-bGam3": -4.0 / 7.0 * parameters["b1"],
                "bGam3": -4.0 / 7.0 * np.ones_like(parameters["b1"]),
                "b1-b1-cnlo": parameters["b1"] ** 2,
                "b1-cnlo": parameters["b1"],
            },
            "PBJ": {},
        }
        return coeff[self.NLcode]

    def loglike_AM(self, parameters: dict, use_Jeffreys: Optional[bool] = False):
        r"""Log-likelihood of GCspectro probe with analytical marginalisation

        Parameters
        ----------
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        use_Jeffreys: bool
            Flag to decide use of Jeffreys priors on linear parameters during AM

        Returns
        -------
        loglike_AM: float
            Value of analytically-marginalised log-likelihood
        """
        # Create copy of dictionary, to avoid modifying the external one
        parameters = deepcopy(parameters)
        # Setting to 0 the parameters to be analytically marginalised
        for z in self.AM_params.keys():
            for p in self.AM_params[z]:
                parameters[p][self.redshifts.index(z)] = 0.0
        # Obtaining list of terms to marginalise for each redshift bin
        term_list = {
            z: [
                value
                for key in self.AM_params[z]
                if key in self.AM_par_to_diag
                for value in self.AM_par_to_diag[key]
            ]
            for z in self.AM_params.keys()
        }
        # Creating theory vectors (both the part non-marginalisable
        # and the individual marginalisable terms)
        self.theory_vector, self.theory_vector_AM = self.get_theory_vector_AM(
            parameters, term_list
        )
        # Recombining the marginalisable terms corresponding to the same
        # linear parameter
        theory_vector_AM_reduced = []
        for z in self.redshifts:
            # To preserve the correct matrix dimensions, we need to add a
            # block of zeros for those bins in which we don't do
            # analytical marginalisation
            if z not in self.AM_params or len(self.AM_params[z]) == 0:
                matrix_reduced_z = np.zeros((0, self.theory_vector_AM[z].shape[1]))
            # Otherwise, we correctly recombine the terms that requires it
            else:
                indices_groups = [
                    [
                        term_list[z].index(term)
                        for term in self.AM_par_to_diag.get(key, [])
                    ]
                    for key in self.AM_params[z]
                ]
                matrix_reduced_z = np.zeros(
                    (len(indices_groups), self.theory_vector_AM[z].shape[1])
                )
                for i, indices in enumerate(indices_groups):
                    matrix_reduced_z[i, :] = np.sum(
                        self.theory_vector_AM[z][indices, :], axis=0
                    )
            theory_vector_AM_reduced.append(matrix_reduced_z)
        # Constructing the block diagonal matrix for analytical marginalisation
        self.theory_vector_AM_reduced = block_diag(*theory_vector_AM_reduced)
        # Masking it (This works since the size of the matrix is consistent)
        self._mask_theory_vector_AM()
        # Calculating chi2 including analytical marginalisation
        diff = self.masked_theory_vector - self.masked_data_vector
        F0 = np.einsum(
            "i,ij,j->", diff, self.inverse_masked_covariance_matrix, diff
        ) + np.sum((self.AM_means / self.AM_sigmas) ** 2)
        F1i = (
            -np.einsum(
                "ij,jk,k->i",
                self.masked_theory_vector_AM_reduced,
                self.inverse_masked_covariance_matrix,
                diff,
            )
            + self.AM_means / self.AM_sigmas**2
        )
        F2ij = np.einsum(
            "ik,kp,jp->ij",
            self.masked_theory_vector_AM_reduced,
            self.inverse_masked_covariance_matrix,
            self.masked_theory_vector_AM_reduced,
        ) + np.diag(1.0 / self.AM_sigmas**2)
        chi2 = F0 - np.einsum("i,ij,j->", F1i, np.linalg.inv(F2ij), F1i)
        if not use_Jeffreys:
            chi2 += np.log(np.linalg.det(F2ij))

        # Access to means of marginalised params
        self.marg_pars_means_raw = np.dot(F1i, np.linalg.inv(F2ij))

        # Also as a dictionary for easy interpretation
        vals = iter(self.marg_pars_means_raw)

        self.marg_pars_means_dict = {
            z: {par: next(vals) for par in self.AM_params[z]} for z in self.redshifts
        }

        return -0.5 * chi2
