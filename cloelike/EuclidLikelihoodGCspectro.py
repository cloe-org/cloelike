import numpy as np

from cloelib.cosmology.cosmology import Background
from cloelib.observables.spectro import SpectroPower
from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles


class EuclidLikelihoodGCspectro:

    def __init__(self, data: dict, settings: dict,
                 Background: type, SpectroPower: type):
        r""" Class constructor
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
        """
        self.data = data
        self.settings = settings

        # Assuming that GCspectro data will be arranged with hierarchy
        # redshift -> multipole -> wavemodes
        self.redshifts = data['GCspectro'].keys()
        self.ells = [0, 2, 4]
        self.nbar = [data['GCspectro'][z]['nbar'] for z in self.redshifts]

        self.scale_cuts = settings['scale_cuts']

        self.Background = Background
        self.SpectroPower = SpectroPower

        params_fid = self.data['fiducial_cosmology']
        self.background_fiducial = self.Background(
            H0=params_fid['H0'], Omega_cdm0=params_fid['Omega_cdm0'],
            Omega_b0=params_fid['Omega_b0'], Omega_k0=params_fid['Omega_k0'],
            w0=params_fid['w0'], wa=params_fid['wa'], ns=params_fid['ns'],
            As=params_fid['As'], gamma_MG=params_fid['gamma_MG']
        )

        self._flatten_data_vector()
        self._flatten_covariance_matrix()
        self._create_masking_vector()
        self._mask_data_vector()
        self._mask_covariance_matrix()
        self._invert_covariance_matrix()

    def _flatten_data_vector(self):
        r""" Arranges the GCspectro data into a flattened data vector
        """
        data_vec = np.concatenate([self.data['GCspectro'][z][f'pk{ell}']
                                    for z in self.redshifts
                                    for ell in self.ells])
        self.data_vector = data_vec

    def _flatten_covariance_matrix(self):
        r""" Arranges the GCspectro covariance into a matrix form
        """
        cov_blocks = [self.data['GCspectro'][z]['cov'] for z in self.redshifts]
        cov_mat = np.block([[block if i == j else np.zeros_like(block)
                             for j, block in enumerate(cov_blocks)]
                            for i, _ in enumerate(cov_blocks)])
        self.flattened_covariance_matrix = cov_mat

    def _masking(self, arr: np.ndarray, interval: list) -> np.ndarray:
        r""" Get a 1/0 mask for the elements of arr contained in interval
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
        return ((arr >= interval[0]) & (arr <= interval[1]))

    def _create_masking_vector(self):
        r""" Computes the masking vector for GCspectro
        """
        masking_vec = np.concatenate([
            self._masking(self.data['GCspectro'][z]['k'],
            np.array(self.scale_cuts['GCspectro'][f'bin{i+1}'][f'ell{ell}']))
            for i, z in enumerate(self.redshifts) for ell in self.ells],
            axis=None)
        self.masking_vector = masking_vec

    def _mask_data_vector(self):
        r""" Mask GCspectro data vector
        """
        self.masked_data_vector = self.data_vector[self.masking_vector]

    def _mask_covariance_matrix(self):
        r""" Mask GCspectro covariance matrix
        """
        self.masked_covariance_matrix = \
            self.flattened_covariance_matrix[self.masking_vector][:, self.masking_vector]

    def _invert_covariance_matrix(self):
        r""" Invert GCspectro covariance matrix
        """
        self.inverse_masked_covariance_matrix = \
            np.linalg.inv(self.masked_covariance_matrix)

    def _create_theory_vector(self, parameters: dict):
        r""" Generate theory vectors based on specified parameters
        """
        background = self.Background(
            H0=parameters['H0'], Omega_cdm0=parameters['Omega_cdm0'],
            Omega_b0=parameters['Omega_b0'], Omega_k0=parameters['Omega_k0'],
            w0=parameters['w0'], wa=parameters['wa'], ns=parameters['ns'],
            As=parameters['As'], gamma_MG=parameters['gamma_MG']
        )

        theory_vec = []
        for i,z in enumerate(self.redshifts):
            RSD_parameters = {key: parameters[key][i] for key in
                              ['b1', 'b2', 'bG2','bGam3', 'c0', 'c2', 'c4', 'cnlo']}
            spectro_power = self.SpectroPower(
                background=background,
                RSD_parameters=RSD_parameters,
                redshift=float(z))
            nois_syst_parameters = {key: parameters[key][i] for key in
                                    ['NP0', 'NP20', 'NP22', 'fout', 'sigmaz']}
            spectro_obs = LegendreMultipoles(
                spectro_power=spectro_power,
                background_fiducial=self.background_fiducial,
                parameters=nois_syst_parameters,
                nbar=self.nbar[i])
            mps_dict = spectro_obs.power_multipoles(
                k=self.data['GCspectro'][z]['k'],
                ells=self.ells, use_AP=True
            )
            mps_list = [mps_dict[f'ell{ell}'] for ell in self.ells]
            theory_vec = np.concatenate((theory_vec, np.concatenate(mps_list)))
        self.theory_vector = theory_vec

    def _mask_theory_vector(self):
        r""" Mask theory vector
        """
        self.masked_theory_vector = self.theory_vector[self.masking_vector]

    def loglike(self, parameters: dict):
        r""" Log-likelihood of GCspectro probe
        """
        self._create_theory_vector(parameters)
        self._mask_theory_vector()
        diff = self.masked_theory_vector - self.masked_data_vector
        chi2 = np.dot(np.dot(diff, self.inverse_masked_covariance_matrix), diff)
        return -0.5 * chi2
