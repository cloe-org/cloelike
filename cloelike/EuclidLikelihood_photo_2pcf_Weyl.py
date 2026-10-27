import numpy as np
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelib.summary_statistics.angular_correlation_function_wigner import (
    AngularCorrelationFunctionWigner,
)
from cloelike.EuclidLikelihood_photo_base_Weyl import PhotoLikelihoodBase_Weyl


class GCphMixin_Weyl:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_gcph()

    def _init_gcph(self):
        self.n_pos_bins = self.data["dndz_pos"].shape[
            0
        ]  # Note: _init_pos_keys has been modified to include bhat_bin parameters
        self._init_pos_keys()
        self.GG_keys = [
            ("POS", "POS", i, j)
            for i in range(1, self.n_pos_bins + 1)
            for j in range(i, self.n_pos_bins + 1)
        ]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["theta"], self.scale_cuts[key])
                for key in self.GG_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array([self.data["2pcf"][key] for key in self.GG_keys]).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        Weyl_p = self._get_Weyl_perturbations(parameters)
        pos = self._get_pos_tracer_Weyl(parameters, Weyl_p, mode="GC")
        cf_all_th = AngularCorrelationFunctionWigner(
            AngularTwoPoint(pos, pos), self.ells_integration, Weyl_p.k
        ).get_xi(np.radians(self.data["theta"] / 60))
        vec = np.array([cf_all_th[key] for key in self.GG_keys]).flatten()
        self.theory_prediction = cf_all_th
        return np.concatenate([v, vec])


class GGLMixin_Weyl:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_ggl()

    def _init_ggl(self):
        self.n_pos_bins = self.data["dndz_pos"].shape[0]
        self.n_she_bins = self.data["dndz_she"].shape[0]
        self._init_pos_keys()  # Note: _init_pos_keys has been modified to include bhat_bin parameters
        self.Jhat_keys = [
            f"Jhat_bin{i}" for i in range(self.n_pos_bins)
        ]  # Added Jhat keys for Weyl implementation
        self._init_she_keys()
        self.GGL_keys = [
            ("POS", "SHE", i, j)
            for i in range(1, self.n_pos_bins + 1)
            for j in range(1, self.n_she_bins + 1)
        ]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["theta"], self.scale_cuts[key])
                for key in self.GGL_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array([self.data["2pcf"][key][0] for key in self.GGL_keys]).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        _, _, nlp = self._get_perturbations(parameters)
        Weyl_p = self._get_Weyl_perturbations(parameters)

        pos_Weyl = self._get_pos_tracer_Weyl(parameters, Weyl_p, mode="GGL")
        she = self._get_she_tracer(parameters, Weyl_p)

        pos_standard = self._get_pos_tracer(parameters, nlp)
        she_IAOnly = self._get_she_tracer_IAOnly(parameters, nlp)

        cf_all_th = AngularCorrelationFunctionWigner(
            AngularTwoPoint(pos_Weyl, she), self.ells_integration, Weyl_p.k
        ).get_xi(np.radians(self.data["theta"] / 60))
        cf_all_th_IAOnly = AngularCorrelationFunctionWigner(
            AngularTwoPoint(pos_standard, she_IAOnly), self.ells_integration, Weyl_p.k
        ).get_xi(np.radians(self.data["theta"] / 60))
        vec = np.array(
            [cf_all_th[key][0] + cf_all_th_IAOnly[key][0] for key in self.GGL_keys]
        ).flatten()

        self.theory_prediction = cf_all_th
        return np.concatenate([v, vec])


class EuclidLikelihood_GCph_Weyl(GCphMixin_Weyl, PhotoLikelihoodBase_Weyl):
    """
    EuclidLikelihood_GCph computes the likelihood for galaxy clustering photometric (GCph) data
    using the Euclid survey specifications.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GCphMixin: Mixin providing GCph-specific functionality.

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


class EuclidLikelihood_GGL_Weyl(GGLMixin_Weyl, PhotoLikelihoodBase_Weyl):
    """
    EuclidLikelihood_GGL class for galaxy-galaxy lensing likelihood computation.

    This class combines the functionalities of PhotoLikelihoodBase and GGLMixin to compute
    the likelihood for galaxy-galaxy lensing (GGL) using photometric data. It initializes
    the necessary components and constructs a masking vector based on scale cuts for each
    GGL key.

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


class EuclidLikelihood_2x2pt_Weyl(
    GCphMixin_Weyl, GGLMixin_Weyl, PhotoLikelihoodBase_Weyl
):
    """
    Likelihood class for Euclid 2x2pt photometric clustering and galaxy-galaxy lensing analysis.

    This class combines galaxy clustering (GCph) and galaxy-galaxy lensing (GGL) likelihoods,
    providing methods to compute the full data and theory vectors for both probes.

    Parameters
    ----------
    data : dict
        Input data dictionary containing observed power spectra and related quantities.
    settings : dict
        Configuration settings for the likelihood analysis.
    Background : object
        Cosmological background model instance.
    LinPerturbations : object
        Linear perturbations model instance.
    NonLinPerturbations : object
        Non-linear perturbations model instance.
    mode : str, optional
        Mode for likelihood computation, default is "coupled".
    """

    pass
