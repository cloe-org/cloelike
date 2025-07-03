import numpy as np

from copy import deepcopy
from dataclasses import replace

from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuNonLinearPerturbations
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint


class EuclidLikelihood_3x2pt_Cls:
    def __init__(
        self,
        data: dict,
        settings: dict,
        Background: type,
        LinPerturbations: type,
        NonLinPerturbations: type,
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
        LinPerturbations: type
            Protocol-consistent Perturbations class
        NonLinPerturbations: type
            Protocol-consistent Perturbation class
        """
        self.data = data
        self.settings = settings

        self.Background = Background
        self.LinPerturbations = LinPerturbations
        if NonLinPerturbations == HMemuNonLinearPerturbations:
            self.NonLinPerturbations = NonLinPerturbations
        else:
            raise TypeError(
                "Currenty, this only works for the HMcode "
                "emulator, so NonLinPerturbations must be of "
                "type 'HMemuNonLinearPerturbations'"
            )

        self.scale_cuts = settings["scale_cuts"]

        self.rebin = False
        self.zs = data["z_arr"]
        self.mixmat = deepcopy(data["mixmat"])

        self.n_pos_bins = self.data["dndz_pos"].shape[0]
        bias_keys = [f"b1_photo_poly{i}" for i in range(4)]
        mag_bias_keys = [
            f"magnification_bias_{i}" for i in range(1, self.n_pos_bins + 1)
        ]
        dz_pos_keys = [f"dz_pos_{i}" for i in range(1, self.n_pos_bins + 1)]
        self.full_pos_keys = bias_keys + mag_bias_keys + dz_pos_keys

        self.n_she_bins = self.data["dndz_she"].shape[0]
        IA_keys = ["AIA", "EtaIA"]
        mul_bias_keys = [
            "multiplicative_bias_%d" % i for i in range(1, self.n_she_bins + 1)
        ]
        dz_she_keys = [f"dz_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        self.full_she_keys = IA_keys + mul_bias_keys + dz_she_keys

        self.WL_keys, self.GG_keys, self.GGL_keys = [], [], []
        for i in range(1, self.n_pos_bins + 1):
            for j in range(i, self.n_pos_bins + 1):
                self.GG_keys.append(("POS", "POS", i, j))
        for i in range(1, self.n_pos_bins + 1):
            for j in range(1, self.n_she_bins + 1):
                self.GGL_keys.append(("POS", "SHE", i, j))
        for i in range(1, self.n_she_bins + 1):
            for j in range(i, self.n_she_bins + 1):
                self.WL_keys.append(("SHE", "SHE", i, j))

        self._prepare()

    def _prepare(self):
        r"""Arrange data vectors and covariance matrices in format required
        for :math:`\chi^2` calculation
        """
        if self.settings["n_ell_bins"] < len(self.data["ells"]):
            self.rebin = True
            self.data["cells_unbin"] = self.data["cells"]
            self.data["ells_unbin"] = self.data["ells"]
            self._bin_data(
                self.data["cells"], self.data["ells"], self.settings["n_ell_bins"]
            )
            self._bin_mixmat()

        self._flatten_data_vector_and_mask()
        self._mask_covariance_and_invert()

    def _bin_data(self, cells_data, ells, n_bins):
        r"""Rebin 3x2pt photometric data
        Parameters
        ----------
        Returns
        -------
        """
        bin_edges = np.geomspace(10, ells[-1], n_bins + 1)
        mask_bins = [
            (ells > bin_edges[i]) & (ells < bin_edges[i + 1]) for i in range(n_bins)
        ]
        self.weight_mat = np.asarray(mask_bins, dtype=float)
        self.weight_mat /= np.sum(self.weight_mat, axis=1)[:, None]

        cells_ave = {k: (cells_data[k] @ self.weight_mat.T) for k in cells_data.keys()}
        self.data["ells"] = np.array([np.mean(ells[mb]) for mb in mask_bins])
        self.data["cells"] = cells_ave

    def _bin_mixmat(self):
        r"""Rebin 3x2pt mixing matrix"""
        # self.mixmat = {k: np.tensordot(self.weight_mat, self.mixmat[k], axes=([1], [-2]))
        #                for k in self.mixmat}
        for k in self.mixmat.keys():
            new_array = np.tensordot(self.weight_mat, self.mixmat[k], axes=([1], [-2]))
            if k[:2] == ("SHE", "SHE"):
                new_array = np.transpose(new_array, axes=(1, 0, 2))

            self.mixmat[k] = replace(self.mixmat[k], array=new_array)

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

    def _flatten_data_vector_and_mask(self):
        r"""Arranges data vectors into flattened vectors and mask them"""
        self.data_vector = np.concatenate(
            [
                np.transpose(
                    [self.data["cells"][key][:2] for key in self.WL_keys],
                    axes=(1, 0, 2),
                ).flatten(),
                np.array(
                    [self.data["cells"][key][:1] for key in self.GGL_keys]
                ).flatten(),
                np.array([self.data["cells"][key] for key in self.GG_keys]).flatten(),
            ]
        )

        self.masking_vector = np.concatenate(
            [
                np.transpose(
                    [
                        [
                            self._masking(self.data["ells"], self.scale_cuts[key][i])
                            for i in [0, 1]
                        ]
                        for key in self.WL_keys
                    ],
                    axes=(1, 0, 2),
                ).flatten(),
                np.transpose(
                    [
                        self._masking(self.data["ells"], self.scale_cuts[key])
                        for key in self.GGL_keys
                    ]
                ).flatten(),
                np.transpose(
                    [
                        self._masking(self.data["ells"], self.scale_cuts[key])
                        for key in self.GG_keys
                    ]
                ).flatten(),
            ]
        )

        self.masked_data_vector = self.data_vector[self.masking_vector]

    def _mask_covariance_and_invert(self):
        r"""Arrange, mask, and invert covariance matrices"""
        self.covariance_matrix = self.data["cov"]

        self.inverse_masked_covariance_matrix = np.linalg.inv(
            self.covariance_matrix[self.masking_vector][:, self.masking_vector]
        )

    def get_theory_vector(self, parameters):
        r"""Generate theory vectors based on specified parameters
        Parameters
        ----------
        parameters: dict
            Input parameters
        Return
        ------
        theory_vector: dict
            Stacked theory vectors
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
            mnu=parameters["mnu"],
            gamma_MG=parameters["gamma_MG"],
        )

        lp = self.LinPerturbations(background, self.zs)
        nlp = self.NonLinPerturbations(
            background, lp, self.zs, log10TAGN=parameters["log10TAGN"]
        )

        pos = PositionsTracer(
            nlp,
            self.data["dndz_pos"],
            self.zs,
            galaxy_bias_model="poly",
            nuisance_params={key: parameters[key] for key in self.full_pos_keys},
        )
        she = ShearTracer(
            nlp,
            self.data["dndz_she"],
            self.zs,
            nuisance_params={key: parameters[key] for key in self.full_she_keys}
            | {"CIA": 0.0134},
        )

        cell_all_th = {
            **AngularTwoPoint(she, she).get_pseudo_Cl(0, nlp.k, self.mixmat),
            **AngularTwoPoint(pos, she).get_pseudo_Cl(0, nlp.k, self.mixmat),
            **AngularTwoPoint(pos, pos).get_pseudo_Cl(0, nlp.k, self.mixmat),
        }

        theory_vector = np.concatenate(
            [
                np.transpose(
                    [cell_all_th[key][:2] for key in self.WL_keys], axes=(1, 0, 2)
                ).flatten(),
                np.array([cell_all_th[key] for key in self.GGL_keys]).flatten(),
                np.array([cell_all_th[key] for key in self.GG_keys]).flatten(),
            ]
        )

        return theory_vector

    def _mask_theory_vector(self):
        r"""Mask theory vector"""
        self.masked_theory_vector = self.theory_vector[self.masking_vector]

    def loglike(self, parameters: dict):
        r"""Log-likelihood of 3x2pt probe
        Parameters
        ----------
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        Returns
        -------
        loglike: float
            Log-likelihood
        """
        self.theory_vector = self.get_theory_vector(parameters)
        self._mask_theory_vector()

        diff = self.masked_theory_vector - self.masked_data_vector
        chi2 = np.dot(np.dot(diff, self.inverse_masked_covariance_matrix), diff)

        return -0.5 * chi2
