import numpy as np

from copy import deepcopy
from scipy.linalg import block_diag
from typing import Optional

from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles


class EuclidLikelihood_GCspectro_Pls:
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
            Data dictionary
        settings: dict
            Settings dictionary
        Background: type
            Protocol-consistent Background class
        SpectroPower: type
            Protocol-consistent SpectroPower class
        AM_priors: dict
            Mean and standard deviation of gaussian priors for analytical
            marginalisation
        """
        self.data = data
        self.settings = settings
        self.NLcode = SpectroPower.NLcode

        # Assuming that GCspectro data will be arranged with hierarchy
        # redshift -> multipole -> wavemodes
        self.ells = [0, 2, 4]
        self.redshifts = list(data["GCspectro"].keys())
        self.nbar = [data["GCspectro"][z]["nbar"] for z in self.redshifts]

        self.scale_cuts = settings["scale_cuts"]

        self.mixmat = (
            {z: data["GCspectro"][z]["mixing_matrix"] for z in self.redshifts}
            if all("mixing_matrix" in data["GCspectro"][z] for z in self.redshifts)
            else None
        )

        self.Background = Background
        self.Perturbations = Perturbations
        self.SpectroPower = SpectroPower

        params_fid = data["fiducial_cosmology"]
        self.background_fiducial = Background(**params_fid)

        self._prepare()

        # We need to change this for e.g. VDG, since cnlo is not a parameter
        # of that model
        self.RSD_parameter_names = {
            "COMET": ["b1", "b2", "bG2", "bGam3", "c0", "c2", "c4", "cnlo"],
            "PBJ": ["b1", "b2", "bG2", "bG3", "c0", "c2", "c4", "ck4"],
        }

        self.noise_syst_parameter_names = ["NP0", "NP20", "NP22", "fout", "sigmaz"]

        self.AM_priors = AM_priors
        if self.AM_priors:
            unmatched_z = [z for z in AM_priors.keys() if z not in self.redshifts]
            if unmatched_z:
                raise ValueError(f"Redshifts {unmatched_z} not found in data.")

            self.AM_par_to_diag = {
                "COMET": {
                    "bGam3": ["b1-bGam3", "bGam3"],
                    "c0": ["c0"],
                    "c2": ["c2"],
                    "c4": ["c4"],
                    "cnlo": ["b1-b1-cnlo", "b1-cnlo", "cnlo"],
                    "NP0": ["noise_k0"],
                    "NP20": ["noise_k2"],
                    "NP22": ["noise_k2mu2"],
                },
                "PBJ": {
                    "bG3": ["bG3"],
                    "c0": ["c0"],
                    "c2": ["c2"],
                    "c4": ["c4"],
                    "ck4": ["ck4"],
                    "NP0": ["noise_k0"],
                    "NP20": ["noise_k2"],
                    "NP22": ["noise_k2mu2"],
                },
            }
            self.AM_diagrams = [
                term
                for values in self.AM_par_to_diag[self.NLcode].values()
                for term in values
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

    def _prepare(self):
        r"""Arrange data vectors and covariance matrices in format required
        for :math:`\chi^2` calculation
        """
        self._flatten_data_vector()
        self._flatten_covariance_matrix()
        self._create_masking_vector()
        self._mask_data_vector()
        self._mask_covariance_matrix()
        self._invert_covariance_matrix()

    def _flatten_data_vector(self):
        r"""Arranges the GCspectro data into a flattened data vector"""
        self.data_vector = np.concatenate(
            [
                self.data["GCspectro"][z][f"pk{ell}"]
                for z in self.redshifts
                for ell in self.ells
            ]
        )

    def _flatten_covariance_matrix(self):
        r"""Arranges the GCspectro covariance into a matrix form"""
        cov_blocks = [self.data["GCspectro"][z]["cov"] for z in self.redshifts]
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

    def _create_masking_vector(self):
        r"""Computes the masking vector for GCspectro"""
        self.masking_vector = np.concatenate(
            [
                self._masking(
                    self.data["GCspectro"][z]["k"],
                    np.array(self.scale_cuts["GCspectro"][f"bin{i + 1}"][f"ell{ell}"]),
                )
                for i, z in enumerate(self.redshifts)
                for ell in self.ells
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

        if self.Perturbations is not None:
            zs = np.float64(self.redshifts)
            lp = self.Perturbations(background, zs)
            cosmo_input = lp
        else:
            zs = None
            lp = None
            cosmo_input = background

        theory_vec = []

        for i, z in enumerate(self.redshifts):
            RSD_params = {
                key: parameters[key][i] for key in self.RSD_parameter_names[self.NLcode]
            }
            syst_params = {
                key: parameters[key][i] for key in self.noise_syst_parameter_names
            }

            power = self.SpectroPower(cosmo_input, RSD_params, redshift=float(z))
            obs = LegendreMultipoles(
                spectro_power=power,
                background_fiducial=self.background_fiducial,
                parameters=syst_params,
                nbar=self.nbar[i],
            )

            k = self.data["GCspectro"][z]["k"]
            if self.mixmat:
                mps = obs.convolved_power_multipoles(
                    self.mixmat[z], ells=self.ells, use_AP=True
                )
            else:
                mps = obs.power_multipoles(k=k, ells=self.ells, use_AP=True)
            theory_vec.extend(np.concatenate([mps[f"ell{ell}"] for ell in self.ells]))

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

        if self.Perturbations is not None:
            zs = np.float64(self.redshifts)
            lp = self.Perturbations(background, zs)
            cosmo_input = lp
        else:
            zs = None
            lp = None
            cosmo_input = background

        theory_vec = []
        theory_vec_AM = {}
        coeff = self._coeff_AM(parameters)
        for i, z in enumerate(self.redshifts):
            RSD_parameters = {
                key: parameters[key][i] for key in self.RSD_parameter_names[self.NLcode]
            }
            power = self.SpectroPower(cosmo_input, RSD_parameters, redshift=float(z))
            nois_syst_parameters = {
                key: parameters[key][i] for key in self.noise_syst_parameter_names
            }
            obs = LegendreMultipoles(
                spectro_power=power,
                background_fiducial=self.background_fiducial,
                parameters=nois_syst_parameters,
                nbar=self.nbar[i],
            )
            k = self.data["GCspectro"][z]["k"]
            if self.mixmat:
                mps_dict = obs.convolved_power_multipoles(
                    self.mixmat[z], ells=self.ells, use_AP=True
                )
            else:
                mps_dict = obs.power_multipoles(k=k, ells=self.ells, use_AP=True)
            mps_list = [mps_dict[f"ell{ell}"] for ell in self.ells]
            theory_vec = np.concatenate((theory_vec, np.concatenate(mps_list)))
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
            else:
                Nk = sum(
                    len(self.data["GCspectro"][z][f"pk{ell}"]) for ell in self.ells
                )
                theory_vec_AM[z] = np.zeros((0, Nk))
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
                if key in self.AM_par_to_diag[self.NLcode]
                for value in self.AM_par_to_diag[self.NLcode][key]
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
                        for term in self.AM_par_to_diag[self.NLcode].get(key, [])
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

        self.marg_pars_means = np.dot(F1i, np.linalg.inv(F2ij))
        
        return -0.5 * chi2
