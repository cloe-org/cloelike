import numpy as np
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelike.EuclidLikelihood_photo_base import PhotoLikelihoodBase
from cloelike.EuclidLikelihood_photo_Cls import GCphMixin, GGLMixin, WLMixin


class CMBLxWLMixin:
    """
    Mixin class providing CMB lensing (CMBL) x WL specific functionality for CMB lensing x WL likelihoods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the next class in the MRO
        self._init_cmblxwl()

    def _init_cmblxwl(self):
        self.n_she_bins = self.data["dndz_she"].shape[0]
        self._init_she_keys()
        self.CMBLxWL_keys = [
            ("CMBL", "SHE", 1, i)
            for i in range(1, self.n_she_bins + 1)
        ]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.CMBLxWL_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array(
            [self.data["cells"][key][0] for key in self.CMBLxWL_keys]
        ).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        _, _, nlp = self._get_perturbations(parameters)
        cmbl = self._get_cmbl_tracer(parameters, nlp)
        she = self._get_she_tracer(parameters, nlp)
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(cmbl, she).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key][0] for key in self.CMBLxWL_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(cmbl, she).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key][0] for key in self.CMBLxWL_keys]).flatten()
        self.theory_prediction.update(cell_all_th)
        return np.concatenate([v, vec])

class CMBLxGCMixin:
    """
    Mixin class providing CMB lensing (CMBL) x GC specific functionality for CMB lensing x GC likelihoods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the next class in the MRO
        self._init_cmblxgc()

    def _init_cmblxgc(self):
        self.n_pos_bins = self.data["dndz_pos"].shape[0]
        self._init_pos_keys()
        self.CMBLxGC_keys = [
            ("CMBL", "POS", 1, i)
            for i in range(1, self.n_pos_bins + 1)
        ]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.CMBLxGC_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array(
            [self.data["cells"][key] for key in self.CMBLxGC_keys]
        ).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        _, _, nlp = self._get_perturbations(parameters)
        cmbl = self._get_cmbl_tracer(parameters, nlp)
        pos = self._get_pos_tracer(parameters, nlp)
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(cmbl, pos).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key] for key in self.CMBLxGC_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(cmbl, pos).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key] for key in self.CMBLxGC_keys]).flatten()
        self.theory_prediction.update(cell_all_th)
        return np.concatenate([v, vec])


class EuclidLikelihood_CMBLxWL(CMBLxWLMixin, PhotoLikelihoodBase):
    """
    EuclidLikelihood_CMBLxWL computes the CMB lensing x weak lensing (CMBxWL) likelihood for photometric surveys using Euclid data.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        CMBLxWLMixin: Mixin providing CMB lensing x weak lensing specific functionality.

    Parameters
    ----------
    data : dict
        Input data required for likelihood computation, including observed ells and other relevant quantities.
    settings : dict
        Configuration settings for the likelihood calculation.
    Background : object
        Instance representing the cosmological background model.
    LinPerturbations : object
        Instance representing linear perturbations.
    NonLinPerturbations : object
        Instance representing non-linear perturbations.
    mode : str, optional
        Mode of operation, default is "coupled".
    """

    pass


class EuclidLikelihood_CMBLxGC(CMBLxGCMixin, PhotoLikelihoodBase):
    """
    EuclidLikelihood_CMBLxGC computes the CMB lensing x galaxy clustering likelihood for photometric data
    using the Euclid survey specifications.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        CMBLxGCMixin: Mixin providing CMB lensing x GCph-specific functionality.

    Parameters
    ----------
    data : dict
        Input data required for likelihood computation, including observed ells and other relevant quantities.
    settings : dict
        Configuration settings for the likelihood calculation.
    Background : object
        Instance representing the cosmological background model.
    LinPerturbations : object
        Instance representing linear perturbations.
    NonLinPerturbations : object
        Instance representing non-linear perturbations.
    mode : str, optional
        Mode of operation, default is "coupled".
    """

    pass


class EuclidLikelihood_5x2pt(GCphMixin, GGLMixin, WLMixin, CMBLxGCMixin, CMBLxWLMixin, PhotoLikelihoodBase):
    """
    EuclidLikelihood_5x2pt combines weak lensing (WL), galaxy clustering (GCph), galaxy-galaxy lensing (GGL), CMB lensing x weak lensing and CMB lensing x GCph likelihoods for photometric cosmological analyses, supporting scale cuts and masking.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GCphMixin: Mixin providing galaxy clustering specific functionality.
        GGLMixin: Mixin providing galaxy-galaxy lensing specific functionality.
        WLMixin: Mixin providing weak lensing specific functionality.
        CMBLxGCMixin: Mixin providing CMB lensing x GCph-specific functionality.
        CMBLxWLMixin: Mixin providing CMB lensing x WL-specific functionality.

    Note: The order of inheritance matters due to the method resolution order (MRO) in Python
    and how mixins extend the base class functionality. Each mixin appends its block after
    the blocks of the mixins that follow it in the MRO, so the data vector, and therefore
    the covariance matrix blocks, are ordered CMBLxWL, CMBLxGC, WL GGL and GCph.

    Parameters
    ----------
    data : dict
        Dictionary containing observational data vectors and related metadata.
    settings : dict
        Configuration settings for the likelihood calculation.
    Background : object
        Instance providing background cosmology calculations.
    LinPerturbations : object
        Instance for linear perturbation theory calculations.
    NonLinPerturbations : object
        Instance for non-linear perturbation theory calculations.
    mode : str, optional
        Mode for likelihood calculation, default is "coupled".
    """

    pass


class EuclidLikelihood_CMBLx(CMBLxGCMixin, CMBLxWLMixin, PhotoLikelihoodBase):
    """
    EuclidLikelihood_CMBLx combines CMB lensing x weak lensing and CMB lensing x GCph likelihoods for photometric cosmological analyses, supporting scale cuts and masking.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        CMBLxGCMixin: Mixin providing CMB lensing x GCph-specific functionality.
        CMBLxWLMixin: Mixin providing CMB lensing x WL-specific functionality.

    Note: The order of inheritance matters due to the method resolution order (MRO) in Python
    and how mixins extend the base class functionality. Each mixin appends its block after
    the blocks of the mixins that follow it in the MRO, so the data vector, and therefore
    the covariance matrix blocks, are ordered CMBLxWL and CMBLxGC.

    Parameters
    ----------
    data : dict
        Dictionary containing observational data vectors and related metadata.
    settings : dict
        Configuration settings for the likelihood calculation.
    Background : object
        Instance providing background cosmology calculations.
    LinPerturbations : object
        Instance for linear perturbation theory calculations.
    NonLinPerturbations : object
        Instance for non-linear perturbation theory calculations.
    mode : str, optional
        Mode for likelihood calculation, default is "coupled".
    """

    pass