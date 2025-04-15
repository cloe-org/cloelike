import numpy as np
from cloelib.cosmology.cosmology import Background, Peturbations
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint


class EuclidLikelihood3x2ptCls:

    def __init__(self, data: dict, settings: dict, Background: type, Perturbations: type):
        self.data = data
        self.settings = settings
        self.Background = Background
        self.Perturbations = Perturbations
        self.scale_cuts = settings['scale_cuts']
        self.rebin = False
        self.cov = data['cov']
        self.mixmat = data['mixmat']

        self._prepare()

    def _prepare(self):
        if self.settings['n_ell_bins'] < len(self.data['ells']):
            self.rebin = True
            self.data['cells_unbin'] = self.data['cells']
            self.data['ells_unbin'] = self.data['ells']
            self._bin_data(self.data['cells'], self.data['ells'], self.settings['n_ell_bins'])
            self._bin_mixmat()

        self._flatten_data_vector_and_mask()
        self._mask_covariance_and_invert()

    def _bin_data(self, cells_data, ells, n_bins):
        bin_edges = np.geomspace(10, ells[-1], n_bins + 1)
        mask_bins = [(ells > bin_edges[i]) & (ells < bin_edges[i + 1]) for i in range(n_bins)]
        self.weight_mat = np.asarray(mask_bins, dtype=float)
        self.weight_mat /= np.sum(self.weight_mat, axis=1)[:, None]

        cells_ave = {k: (self.weight_mat @ cells_data[k]).T for k in cells_data}
        self.data['ells'] = np.array([np.mean(ells[mb]) for mb in mask_bins])
        self.data['cells'] = cells_ave

    def _bin_mixmat(self):
        self.mixmat = {k: np.tensordot(self.weight_mat, self.mixmat[k], axes=([1], [0]))
                       for k in self.mixmat}

    def _flatten_data_vector_and_mask(self):
        self.WL_keys, self.GG_keys, self.GGL_keys = [], [], []
        for i in range(6):
            for j in range(6):
                if j >= i:
                    self.WL_keys.append(('SHE', 'SHE', i, j))
                    self.GG_keys.append(('POS', 'POS', i, j))
                self.GGL_keys.append(('POS', 'SHE', i, j))

        self.dv_3x2 = np.concatenate([
            np.transpose([self.data['cells'][k][:2] for k in self.WL_keys], axes=(1, 0, 2)).flatten(),
            np.array([self.data['cells'][k][:1] for k in self.GGL_keys]).flatten(),
            np.array([self.data['cells'][k] for k in self.GG_keys]).flatten()
        ])

        self.mask_all = np.concatenate([
            np.transpose([[(self.data['ells'] >= self.scale_cuts[k][i][0]) &
                           (self.data['ells'] <= self.scale_cuts[k][i][1])
                           for i in [0, 1]] for k in self.WL_keys], axes=(1, 0, 2)).flatten(),
            np.transpose([(self.data['ells'] >= self.scale_cuts[k][0]) &
                          (self.data['ells'] <= self.scale_cuts[k][1]) for k in self.GGL_keys]).flatten(),
            np.transpose([(self.data['ells'] >= self.scale_cuts[k][0]) &
                          (self.data['ells'] <= self.scale_cuts[k][1]) for k in self.GG_keys]).flatten()
        ])

        self.dv_3x2_masked = self.dv_3x2[self.mask_all]

    def _mask_covariance_and_invert(self):
        self.inv_cov = np.linalg.inv(self.cov[self.mask_all][:, self.mask_all])

    def get_theory_vector(self, par_dict):
        zs = self.data['z_arr']
        background = self.Background(**{
            'H0': par_dict['H0'], 'Omega_cdm0': par_dict['Omega_cdm0'], 'Omega_b0': par_dict['Omega_b0'],
            'w0': -1, 'wa': 0, 'Omega_k0': 0.0, 'ns': par_dict['ns'], 'As': par_dict['As'], 'gamma_MG': 0.545
        })
        lp = self.Perturbations(background, zs)
        nlp = self.Perturbations(background, lp, zs, log10TAGN=7.8)

        pos = PositionsTracer(nlp, self.data['dndz_pos'], zs)
        she = ShearTracer(nlp, self.data['dndz_she'], zs,
                          nuisance_params={'AIA': 1.72, 'CIA': 0.0134, 'EtaIA': -0.41})

        self.cell_all_th = {
            **AngularTwoPoint(she, she).get_pseudo_Cl(0, nlp.k, self.mixmat),
            **AngularTwoPoint(she, pos).get_pseudo_Cl(0, nlp.k, self.mixmat),
            **AngularTwoPoint(pos, pos).get_pseudo_Cl(0, nlp.k, self.mixmat)
        }

    def flatten_theory_vector_and_mask(self):
        self.thv_3x2 = np.concatenate([
            np.transpose([self.cell_all_th[k][:2] for k in self.WL_keys], axes=(1, 0, 2)).flatten(),
            np.array([self.cell_all_th[k] for k in self.GGL_keys]).flatten(),
            np.array([self.cell_all_th[k] for k in self.GG_keys]).flatten()
        ])
        self.thv_3x2_masked = self.thv_3x2[self.mask_all]

    def loglike(self, parameters: dict):
        self.get_theory_vector(parameters)
        self.flatten_theory_vector_and_mask()
        chi2 = np.dot(np.dot(self.thv_3x2_masked - self.dv_3x2_masked, self.inv_cov),
                      self.thv_3x2_masked - self.dv_3x2_masked)
        return -0.5 * chi2

