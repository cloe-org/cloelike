import numpy as np

from cloelib.cosmology.cosmology import Background
from cloelib.summary_statistics.bao_alphas import BaryonAcousticOscillations

class EuclidLikelihood_BAO:

    def __init__(self, data: dict,
                 Background: type):
        r"""Class constructor

        Parameters
        ----------
        data: dict
            Data dictionary
        Background: type
            Protocol-consistent background class
        """
        self.data = data

        self.redshifts = np.array(list(data['BAO'].keys()))

        self.Background = Background

        # All data files have a fiducial_cosmology entry, we take the
        # first one as they are supposed to be the same
        params_fid = data['BAO'][self.redshifts[0]]['fiducial_cosmology']
        self.background_fiducial = Background(**params_fid)

        self._prepare()

    def _prepare(self):
        r"""Arrange data vectors and covariance matrices in format required
        for :math:`\chi^2` calculation
        """
        self._flatten_data_vector()
        self._flatten_covariance_matrix()
        self._invert_covariance_matrix()

    def _flatten_data_vector(self):
        r""" Arranges the BAO data into a flattened data vector
        """
        self.data_vector = np.concatenate([
            [self.data['BAO'][z]['data'][param]
             for param in self.data['BAO'][z]['params']]
            for z in self.redshifts
        ])

    def _flatten_covariance_matrix(self):
        r""" Arranges the BAO covariance into a matrix form
        """
        cov_blocks = [self.data['BAO'][z]['covariance'] for z in self.redshifts]
        self.flattened_covariance_matrix = np.block([
            [block if i == j else np.zeros_like(block)
             for j, block in enumerate(cov_blocks)]
            for i,_ in enumerate(cov_blocks)
            ])

    def _invert_covariance_matrix(self):
        r""" Invert BAO covariance matrix
        """
        self.inverse_covariance_matrix = \
            np.linalg.inv(self.flattened_covariance_matrix)

    def get_theory_vector(self, parameters: dict) -> np.ndarray:
        r""" Generate theory vectors based on specified parameters
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
            H0=parameters['H0'], Omega_cdm0=parameters['Omega_cdm0'],
            Omega_b0=parameters['Omega_b0'], Omega_k0=parameters['Omega_k0'],
            w0=parameters['w0'], wa=parameters['wa'], ns=parameters['ns'],
            As=parameters['As'], gamma_MG=parameters['gamma_MG'],
            mnu=parameters['mnu'])

        alphas_dict = BaryonAcousticOscillations(
            background=background,
            background_fiducial=self.background_fiducial,
            redshifts=self.redshifts).alphas_dict

        theory_vec = np.concatenate([[alphas_dict[z][params]
                                      for params in self.data['BAO'][z]['params']]
                                     for z in self.redshifts])
        return theory_vec

    def loglike(self, parameters: dict):
        r""" Log-likelihood of BAO probe
        Parameters
        ----------
        parameters: dict
            Ensemble of cosmological parameters
        """
        self.theory_vector = self.get_theory_vector(parameters)
        diff = self.theory_vector - self.data_vector
        chi2 = np.dot(np.dot(diff, self.inverse_covariance_matrix), diff)
        return -0.5 * chi2

