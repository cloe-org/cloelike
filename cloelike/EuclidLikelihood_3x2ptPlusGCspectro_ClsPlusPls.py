import numpy as np

from copy import deepcopy
from scipy.linalg import block_diag
from dataclasses import replace

from cloelib.cosmology.cosmology import Background, Perturbations
from cloelib.cosmology.HMcode2020Emu_cosmology \
    import HMemuNonLinearPerturbations
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.observables.spectro import SpectroPower
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles


class EuclidLikelihood_3x2ptPlusGCspectro_ClsPlusPls:

    def __init__(self, data: dict, settings: dict,
                 Background: type, LinPerturbations: type,
                 NonLinPerturbations: type, SpectroPower: type):
        r""" Class constructor
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
        SpectroPower: type
            Protocol-consistent SpectroPower class
        """
        self.data = data
        self.settings = settings

        self.Background = Background
        self.LinPerturbations = LinPerturbations
        if NonLinPerturbations == HMemuNonLinearPerturbations:
            self.NonLinPerturbations = NonLinPerturbations
        else:
            raise TypeError("Currenty, this only works for the HMcode "
                            "emulator, so NonLinPerturbations must be of "
                            "type 'HMemuNonLinearPerturbations'")
        self.SpectroPower = SpectroPower

        self.scale_cuts = {probe: settings[probe]['scale_cuts']
                           for probe in ['3x2pt', 'GCspectro']}

        # 3x2pt
        self.rebin = False
        # Need to think how to homogenise redshifts
        self.zs = data['3x2pt']['z_arr']
        self.mixmat = {}
        self.mixmat['3x2pt'] = deepcopy(data['3x2pt']['mixmat'])
        self.mixmat['GCspectro'] = (
            {z: data['GCspectro'][z]['mixing_matrix']
             for z in self.redshifts}
            if all('mixing_matrix' in data['GCspectro'][z]
                   for z in self.redshifts)
            else None)

        self.n_pos_bins = self.data['3x2pt']['dndz_pos'].shape[0]
        bias_keys = [f'b1_photo_poly{i}' for i in range(4)]
        mag_bias_keys = [f'magnification_bias_{i}'
                         for i in range(1, self.n_pos_bins + 1)]
        dz_pos_keys = [f'dz_pos_{i}' for i in range(1, self.n_pos_bins + 1)]
        self.full_pos_keys = bias_keys + mag_bias_keys + dz_pos_keys

        self.n_she_bins = self.data['3x2pt']['dndz_she'].shape[0]
        IA_keys = ['AIA', 'EtaIA']
        mul_bias_keys = ['multiplicative_bias_%d'%i
                         for i in range(1,self.n_she_bins+1)]
        dz_she_keys = [f'dz_shear_{i}' for i in range(1, self.n_she_bins + 1)]
        self.full_she_keys = IA_keys + mul_bias_keys + dz_she_keys

        self.WL_keys, self.GG_keys, self.GGL_keys = [], [], []
        for i in range(1,self.n_pos_bins+1):
            for j in range(i,self.n_pos_bins+1):
                self.GG_keys.append(('POS', 'POS', i, j))
        for i in range(1,self.n_pos_bins+1):
            for j in range(1,self.n_she_bins+1):
                self.GGL_keys.append(('POS', 'SHE', i, j))
        for i in range(1,self.n_she_bins+1):
            for j in range(i,self.n_she_bins+1):
                self.WL_keys.append(('SHE', 'SHE', i, j))

        # GCspectro
        # Assuming that GCspectro data will be arranged with hierarchy
        # redshift -> multipole -> wavemodes
        self.ells = [0, 2, 4]
        self.redshifts = list(data['GCspectro'].keys())
        self.nbar = [data['GCspectro'][z]['nbar'] for z in self.redshifts]

        params_fid = data['fiducial_cosmology']
        self.background_fiducial = Background(**params_fid)

        # We need to change this for e.g. VDG, since cnlo is not a parameter
        # of that model
        self.RSD_parameter_names = \
            ['b1', 'b2', 'bG2', 'bGam3', 'c0', 'c2', 'c4', 'cnlo']
        self.noise_syst_parameter_names = \
            ['NP0', 'NP20', 'NP22', 'fout', 'sigmaz']

        self._prepare()

    def _prepare(self):
        r"""Arrange data vectors and covariance matrices in format required
        for :math:`\chi^2` calculation
        """
        if (self.settings['3x2pt']['n_ell_bins']
                < len(self.data['3x2pt']['ells'])):
            self.rebin = True
            self.data['3x2pt']['cells_unbin'] = self.data['3x2pt']['cells']
            self.data['3x2pt']['ells_unbin'] = self.data['3x2pt']['ells']
            self._bin_data(
                self.data['3x2pt']['cells'], self.data['3x2pt']['ells'],
                self.settings['3x2pt']['n_ell_bins'])
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
        mask_bins = [(ells > bin_edges[i]) & (ells < bin_edges[i + 1]) for i in range(n_bins)]
        self.weight_mat = np.asarray(mask_bins, dtype=float)
        self.weight_mat /= np.sum(self.weight_mat, axis=1)[:, None]

        cells_ave = {k: (cells_data[k] @ self.weight_mat.T) for k in cells_data.keys()}
        self.data['3x2pt']['ells'] = np.array([np.mean(ells[mb]) for mb in mask_bins])
        self.data['3x2pt']['cells'] = cells_ave

    def _bin_mixmat(self):
        r"""Rebin 3x2pt mixing matrix
        """
        for k in self.mixmat['3x2pt'].keys():
            new_array = np.tensordot(self.weight_mat, self.mixmat['3x2pt'][k], axes=([1], [-2]))
            if k[:2]==('SHE','SHE'):
                new_array = np.transpose(new_array,axes=(1,0,2))
            self.mixmat['3x2pt'][k] = replace(self.mixmat['3x2pt'][k], array=new_array)

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

    def _flatten_data_vector_and_mask(self):
        r""" Arranges data vectors into flattened vectors and mask them
        """
        self.data_vector = {}
        self.masking_vector = {}
        self.masked_data_vector = {}

        self.data_vector['3x2pt'] = np.concatenate([
            np.transpose([self.data['3x2pt']['cells'][key][:2]
                          for key in self.WL_keys], axes=(1, 0, 2)).flatten(),
            np.array([self.data['3x2pt']['cells'][key][:1]
                      for key in self.GGL_keys]).flatten(),
            np.array([self.data['3x2pt']['cells'][key]
                      for key in self.GG_keys]).flatten()])

        self.data_vector['GCspectro'] = np.concatenate([
            self.data['GCspectro'][z][f'pk{ell}']
            for z in self.redshifts for ell in self.ells])

        self.masking_vector['3x2pt'] = np.concatenate([
            np.transpose([[
                self._masking(self.data['3x2pt']['ells'],
                              self.scale_cuts['3x2pt'][key][i])
                for i in [0, 1]] for key in self.WL_keys],
            axes=(1, 0, 2)).flatten(),
            np.transpose([
                self._masking(self.data['3x2pt']['ells'],
                              self.scale_cuts['3x2pt'][key])
                for key in self.GGL_keys]).flatten(),
            np.transpose([
                self._masking(self.data['3x2pt']['ells'],
                              self.scale_cuts['3x2pt'][key])
                for key in self.GG_keys]).flatten()])

        self.masking_vector['GCspectro'] = np.concatenate([
            self._masking(
                self.data['GCspectro'][z]['k'],
                np.array(self.scale_cuts['GCspectro'][f'bin{i+1}'][f'ell{ell}'])
            ) for i, z in enumerate(self.redshifts) for ell in self.ells])

        self.masked_data_vector['3x2pt'] = \
            self.data_vector['3x2pt'][self.masking_vector['3x2pt']]

        self.masked_data_vector['GCspectro'] = \
            self.data_vector['GCspectro'][self.masking_vector['GCspectro']]

    def _mask_covariance_and_invert(self):
        r"""Arrange, mask, and invert covariance matrices
        """
        self.covariance_matrix = {}

        self.covariance_matrix['3x2pt'] = self.data['3x2pt']['cov']

        cov_blocks = [self.data['GCspectro'][z]['cov'] for z in self.redshifts]
        self.covariance_matrix['GCspectro'] = np.block([
            [block if i == j else np.zeros_like(block)
             for j, block in enumerate(cov_blocks)]
            for i, _ in enumerate(cov_blocks)])

        self.inverse_masked_covariance_matrix = {}

        self.inverse_masked_covariance_matrix['3x2pt'] = \
            np.linalg.inv(self.covariance_matrix['3x2pt'][
                self.masking_vector['3x2pt']][
                    :, self.masking_vector['3x2pt']])

        self.inverse_masked_covariance_matrix['GCspectro'] = \
            np.linalg.inv(self.covariance_matrix['GCspectro'][
                self.masking_vector['GCspectro']][
                    :, self.masking_vector['GCspectro']])

    def get_theory_vector(self, parameters: dict):
        r""" Generate theory vectors based on specified parameters
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
            H0=parameters['H0'], Omega_cdm0=parameters['Omega_cdm0'],
            Omega_b0=parameters['Omega_b0'], Omega_k0=parameters['Omega_k0'],
            w0=parameters['w0'], wa=parameters['wa'],  ns=parameters['ns'],
            As=parameters['As'], mnu=parameters['mnu'],
            gamma_MG=parameters['gamma_MG'])

        lp = self.LinPerturbations(background, self.zs)
        nlp = self.NonLinPerturbations(
            background, lp, self.zs, log10TAGN=parameters['log10TAGN'])

        pos = PositionsTracer(
            nlp, self.data['3x2pt']['dndz_pos'], self.zs,
            galaxy_bias_model='poly',
            nuisance_params={key: parameters[key]
                             for key in self.full_pos_keys})
        she = ShearTracer(
            nlp, self.data['3x2pt']['dndz_she'], self.zs,
            nuisance_params={key: parameters[key]
                             for key in self.full_she_keys} | {'CIA': 0.0134})

        cell_all_th = {
            **AngularTwoPoint(she, she).get_pseudo_Cl(0, nlp.k, self.mixmat['3x2pt']),
            **AngularTwoPoint(pos, she).get_pseudo_Cl(0, nlp.k, self.mixmat['3x2pt']),
            **AngularTwoPoint(pos, pos).get_pseudo_Cl(0, nlp.k, self.mixmat['3x2pt'])}

        theory_vector_3x2pt = np.concatenate([
            np.transpose([cell_all_th[key][:2] for key in self.WL_keys],
                         axes=(1, 0, 2)).flatten(),
            np.array([cell_all_th[key] for key in self.GGL_keys]).flatten(),
            np.array([cell_all_th[key] for key in self.GG_keys]).flatten()])

        theory_vector_GCspectro = []
        for i, z in enumerate(self.redshifts):
            RSD_params = {key: parameters[key][i]
                          for key in self.RSD_parameter_names}
            syst_params = {key: parameters[key][i]
                           for key in self.noise_syst_parameter_names}
            power = self.SpectroPower(
                background=background, RSD_parameters=RSD_params,
                redshift=float(z))
            obs = LegendreMultipoles(
                spectro_power=power,
                background_fiducial=self.background_fiducial,
                parameters=syst_params, nbar=self.nbar[i])

            k = self.data['GCspectro'][z]['k']
            if self.mixmat['GCspectro']:
                mps = obs.convolved_power_multipoles(
                    self.mixmat['GCspectro'][z], ells=self.ells, use_AP=True)
            else:
                mps = obs.power_multipoles(k=k, ells=self.ells, use_AP=True)
            theory_vector_GCspectro.extend(
                np.concatenate([mps[f'ell{ell}'] for ell in self.ells]))

        theory_vector = {'3x2pt': theory_vector_3x2pt,
                         'GCspectro': np.array(theory_vector_GCspectro)}

        return theory_vector

    def _mask_theory_vector(self):
        r""" Mask theory vector
        """
        self.masked_theory_vector = {}

        self.masked_theory_vector['3x2pt'] = \
            self.theory_vector['3x2pt'][self.masking_vector['3x2pt']]
        self.masked_theory_vector['GCspectro'] = \
            self.theory_vector['GCspectro'][self.masking_vector['GCspectro']]

    def loglike(self, parameters: dict):
        r""" Combined log-likelihood of 3x2pt and GCspectro probes
        Parameters
        ----------
        parameters: dict
            Ensemble of cosmological and nuisance parameters
        """
        self.theory_vector = self.get_theory_vector(parameters)
        self._mask_theory_vector()

        diff_3x2pt = \
            self.masked_theory_vector['3x2pt'] - self.masked_data_vector['3x2pt']
        chi2_3x2pt = np.dot(
            np.dot(diff_3x2pt, self.inverse_masked_covariance_matrix['3x2pt']),
            diff_3x2pt)

        diff_GCspectro = \
            self.masked_theory_vector['GCspectro'] - self.masked_data_vector['GCspectro']
        chi2_GCspectro = np.dot(
            np.dot(diff_GCspectro,
                   self.inverse_masked_covariance_matrix['GCspectro']),
            diff_GCspectro)

        chi2 = chi2_3x2pt + chi2_GCspectro

        return -0.5 * chi2
