import numpy as np

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations, CAMBNonLinearPerturbations
from cloelib.cosmology.HMcode2020Emu_cosmology import HMemuLinearPerturbations, HMemuNonLinearPerturbations
from cloelib.observables.photo import ShearTracer, PositionsTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint


class EuclidLikelihood3x2:

    def __init__(self,data,cov,mixmat,settings):

        self.data = data
        self.cov = cov
        self.mixmat = mixmat
        self.settings = settings
        self.scale_cuts = settings['scale_cuts']
        self.rebin = False

        # Here I was using a data file with a lot of ells, so I would re-bin
        # here. This is unlikely to be necessary, since the data should already
        # have the right binning, but just leaving it here in case we want it
        if settings['n_ell_bins']<len(data['ells']):

            self.rebin = True

            self.data['cells_unbin'] = self.data['cells']
            self.data['ells_unbin'] = self.data['ells']

            self.bin_data(data['cells'],data['ells'],settings['n_ell_bins'])

        # We then flatten the data vector and mask it according to scale cuts
        self.flatten_dv_and_mask()
        # Mask also the covariance, which is here assumed to be an array in
        # order probe_zpair_ell.
        self.mask_cov_and_invert()


    def bin_data(self,cells_data,ell_cell,n_ell_bins):

        # Here I assume lmin=10, but should change
        bin_edges=np.geomspace(10,ell_cell[-1],n_ell_bins+1)
        mask_bins = []
        cells_data_ave={}
        ells_ave = []
        weight_mat = []
        for i in range(n_ell_bins):
            mask_bins.append((ell_cell>bin_edges[i]) & (ell_cell<bin_edges[i+1]))
            ells_ave.append(np.mean(ell_cell[mask_bins[i]]))

        mask_bins_arr = np.asarray(mask_bins, dtype=float)
        # Creates a weight matrix to also re-bin theory
        self.weight_mat = mask_bins_arr/np.sum(mask_bins_arr,axis=1)[:,None]

        for key in cells_data.keys():
            cells_data_ave[key] = (self.weight_mat@cells_data[key]).T

        # Re-write data with re-binned case (un-binned are saved elsewhere)
        self.data['ells']=np.asarray(ells_ave)
        self.data['cells']=cells_data_ave

    def flatten_dv_and_mask(self):

        # Organize dictionary keys so that ordering is fixed
        WL_keys = []
        GG_keys = []
        GGL_keys = []
        for i in range(6):
            for j in range(6):
                if j>=i:
                    WL_keys.append(('SHE','SHE',i,j))
                    GG_keys.append(('POS','POS',i,j))
                GGL_keys.append(('POS','SHE',i,j))

        self.WL_keys=WL_keys
        self.GG_keys=GG_keys
        self.GGL_keys=GGL_keys

        # Create flattened data and masks for later
        cell_data_WL_flatten=np.transpose(np.asarray([self.data['cells'][key][:2] for key in WL_keys]),axes=(1,0,2)).flatten()
        self.mask_WL = np.transpose(np.asarray([[(self.data['ells']>=self.scale_cuts[key][0][0]) & (self.data['ells']<=self.scale_cuts[key][0][1]),
                                                 (self.data['ells']>=self.scale_cuts[key][1][0]) & (self.data['ells']<=self.scale_cuts[key][1][1])] for key in WL_keys]),axes=(1,0,2)).flatten()
        cell_data_GGL_flatten=np.asarray([self.data['cells'][key][:1] for key in GGL_keys]).flatten()
        self.mask_GGL = np.transpose(np.asarray([(self.data['ells']>=self.scale_cuts[key][0]) & (self.data['ells']<=self.scale_cuts[key][1]) for key in GGL_keys])).flatten()
        cell_data_GG_flatten=np.asarray([self.data['cells'][key] for key in GG_keys]).flatten()
        self.mask_GG = np.transpose(np.asarray([(self.data['ells']>=self.scale_cuts[key][0]) & (self.data['ells']<=self.scale_cuts[key][1]) for key in GG_keys])).flatten()

        self.mask_all = np.concatenate((self.mask_WL,self.mask_GGL,self.mask_GG))

        # Output full and masked data vector with appropriate scale cuts
        self.dv_3x2 = np.concatenate((cell_data_WL_flatten,cell_data_GGL_flatten,cell_data_GG_flatten))
        self.dv_3x2_masked = self.dv_3x2[self.mask_all]

    def mask_cov_and_invert(self):

        # Mask input covariance and invert it

        masked_cov=self.cov[self.mask_all,:][:,self.mask_all]
        self.inv_cov=np.linalg.inv(masked_cov)

    def loglike(self,par_dict):

        # Simple log like as a function of parameters

        # Build theory vector
        self.get_theory(par_dict)

        # Re-bin it if necessary
        if self.rebin:
            self.bin_theory()

        # Flatten it and mask it
        self.flatten_thv_and_mask()

        # Finally compute the loglike
        return self.calc_like()

    def get_theory(self,par_dict):

        zs = self.data['z_arr']

        # Assuming dictionary only has LCDM params, but of course this will have to change later
        background = CAMBBackground(H0=par_dict['H0'],
                                    Omega_cdm0=par_dict['Omega_cdm0'],
                                    Omega_b0=par_dict['Omega_b0'],
                                    w0=-1,
                                    wa=0,
                                    Omega_k0 = 0.0,
                                    ns = par_dict['ns'],
                                    As = par_dict['As'],
                                    gamma_MG = 0.545)
        # Using emulator here, but should probably create different options
        linear_perturbations_emu = HMemuLinearPerturbations(background, zs)
        nonlinear_perturbations_emu = HMemuNonLinearPerturbations(background, linear_perturbations_emu,zs,log10TAGN=7.8)

        # Define the tracers
        tracer_pos = PositionsTracer(perturbations=nonlinear_perturbations_emu,
                                     dndz=self.data['dndz_pos'],
                                     z = zs)

        tracer_she = ShearTracer(perturbations=nonlinear_perturbations_emu,
                                     dndz=self.data['dndz_she'],
                                     z = zs,
                                     nuisance_params={'AIA': 1.72, 'CIA': 0.0134, 'EtaIA':-0.41})


        twopoint_pospos = AngularTwoPoint(tracer_pos, tracer_pos)
        twopoint_shepos = AngularTwoPoint(tracer_she, tracer_pos)
        twopoint_sheshe = AngularTwoPoint(tracer_she, tracer_she)

        self.cell_GG_th = twopoint_pospos.get_pseudo_Cl(0, nonlinear_perturbations_emu.k,self.mixmat)
        self.cell_GGL_th = twopoint_shepos.get_pseudo_Cl(0, nonlinear_perturbations_emu.k,self.mixmat)
        self.cell_WL_th = twopoint_sheshe.get_pseudo_Cl(0, nonlinear_perturbations_emu.k,self.mixmat)

        self.cell_all_th = self.cell_WL_th | self.cell_GGL_th | self.cell_GG_th

    def bin_theory(self):

        for key in self.cell_all_th.keys():
            self.cell_all_th[key] = (self.weight_mat@(self.cell_all_th[key].T)).T

    def flatten_thv_and_mask(self):

        cell_WL_flatten=np.transpose(np.asarray([self.cell_all_th[key][:2] for key in self.WL_keys]),axes=(1,0,2)).flatten()
        cell_GGL_flatten=np.asarray([self.cell_all_th[key] for key in self.GGL_keys]).flatten()
        cell_GG_flatten=np.asarray([self.cell_all_th[key] for key in self.GG_keys]).flatten()

        self.thv_3x2 = np.concatenate((cell_WL_flatten,cell_GGL_flatten,cell_GG_flatten))
        self.thv_3x2_masked = self.thv_3x2[self.mask_all]

    def calc_like(self):

        th_m_dat = self.thv_3x2_masked - self.dv_3x2_masked

        chi2 = np.dot(np.dot(th_m_dat, self.inv_cov),th_m_dat)

        return -0.5*chi2
