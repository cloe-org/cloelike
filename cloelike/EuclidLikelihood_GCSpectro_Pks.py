import numpy as np
from cloelib.cosmology.cosmology import Background
from cloelib.observables.spectro import SpectroPower
from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles


class EuclidLikelihoodGCspectroPks:

    def __init__(self, data: dict, settings: dict, Background: type, SpectroPower: type):
        self.data = data
        self.settings = settings
        self.ells = [0, 2, 4]
        self.redshifts = data['GCspectro'].keys()
        self.nbar = [data['GCspectro'][z]['nbar'] for z in self.redshifts]
        self.scale_cuts = settings['scale_cuts']
        self.Background = Background
        self.SpectroPower = SpectroPower

        params_fid = data['fiducial_cosmology']
        self.background_fiducial = Background(**params_fid)

        self._prepare()

    def _prepare(self):
        self._flatten_data_vector()
        self._flatten_covariance_matrix()
        self._create_masking_vector()
        self._mask_data_vector()
        self._mask_covariance_matrix()
        self._invert_covariance_matrix()

    def _flatten_data_vector(self):
        self.data_vector = np.concatenate([
            self.data['GCspectro'][z][f'pk{ell}']
            for z in self.redshifts for ell in self.ells
        ])

    def _flatten_covariance_matrix(self):
        cov_blocks = [self.data['GCspectro'][z]['cov'] for z in self.redshifts]
        self.flattened_covariance_matrix = np.block([
            [block if i == j else np.zeros_like(block)
             for j, block in enumerate(cov_blocks)]
            for i, _ in enumerate(cov_blocks)
        ])

    def _masking(self, arr: np.ndarray, interval: list) -> np.ndarray:
        return ((arr >= interval[0]) & (arr <= interval[1]))

    def _create_masking_vector(self):
        self.masking_vector = np.concatenate([
            self._masking(
                self.data['GCspectro'][z]['k'],
                np.array(self.scale_cuts['GCspectro'][f'bin{i+1}'][f'ell{ell}'])
            ) for i, z in enumerate(self.redshifts) for ell in self.ells
        ])

    def _mask_data_vector(self):
        self.masked_data_vector = self.data_vector[self.masking_vector]

    def _mask_covariance_matrix(self):
        self.masked_covariance_matrix = self.flattened_covariance_matrix[
            self.masking_vector][:, self.masking_vector]

    def _invert_covariance_matrix(self):
        self.inverse_masked_covariance_matrix = np.linalg.inv(self.masked_covariance_matrix)

    def get_theory_vector(self, parameters: dict) -> np.ndarray:
        background = self.Background(
            H0=parameters['H0'], Omega_cdm0=parameters['Omega_cdm0'],
            Omega_b0=parameters['Omega_b0'], Omega_k0=parameters['Omega_k0'],
            w0=parameters['w0'], wa=parameters['wa'], ns=parameters['ns'],
            As=parameters['As'], gamma_MG=parameters['gamma_MG'],
            mnu=parameters['mnu'])
        theory_vec = []

        for i, z in enumerate(self.redshifts):
            RSD_params = {key: parameters[key][i] for key in ['b1', 'b2', 'bG2', 'bGam3', 'c0', 'c2', 'c4', 'cnlo']}
            syst_params = {key: parameters[key][i] for key in ['NP0', 'NP20', 'NP22', 'fout', 'sigmaz']}

            power = self.SpectroPower(background=background, RSD_parameters=RSD_params, redshift=float(z))
            obs = LegendreMultipoles(spectro_power=power, background_fiducial=self.background_fiducial,
                                     parameters=syst_params, nbar=self.nbar[i])

            k = self.data['GCspectro'][z]['k']
            mps = obs.power_multipoles(k=k, ells=self.ells, use_AP=True)
            theory_vec.extend(np.concatenate([mps[f'ell{ell}'] for ell in self.ells]))

        return np.array(theory_vec)

    def _mask_theory_vector(self):
        self.masked_theory_vector = self.theory_vector[self.masking_vector]

    def loglike(self, parameters: dict):
        self.theory_vector = self.get_theory_vector(parameters)
        self._mask_theory_vector()
        diff = self.masked_theory_vector - self.masked_data_vector
        chi2 = np.dot(np.dot(diff, self.inverse_masked_covariance_matrix), diff)
        return -0.5 * chi2
