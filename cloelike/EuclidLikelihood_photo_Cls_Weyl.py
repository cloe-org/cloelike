import numpy as np
from cloelib.observables.photo import ShearTracer
from cloelib.cosmology.Weyl_cosmology import Weyl_Perturbations
from cloelib.observables.photo_Weyl import PositionsTracer_Weyl_GC, PositionsTracer_Weyl_GGL
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelike.EuclidLikelihood_photo_base import PhotoLikelihoodBase


class GCphMixin_Weyl:
    """
    Mixin class providing photometric angular galaxy clustering specific functionality for photometric likelihoods.
    This mixin extends the base photometric likelihood class to include methods and attributes
    necessary for handling weak lensing data, such as initializing shear bins, constructing
    masking vectors, and computing data and theory vectors specific to weak lensing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_gcph()

    def _init_gcph(self):
        self.n_pos_bins = self.data["dndz_pos"].shape[0]
        bias_keys = [f"bhat_bin{i}" for i in range(self.n_pos_bins)] # Adjusted bias keys for Weyl case
        mag_bias_keys = [
            f"magnification_bias_{i}" for i in range(1, self.n_pos_bins + 1)
        ]
        dz_pos_keys = [f"dz_pos_{i}" for i in range(1, self.n_pos_bins + 1)]
        width_pos_keys = [f"width_pos_{i}" for i in range(1, self.n_pos_bins + 1)]
        self.full_pos_keys = bias_keys + mag_bias_keys + dz_pos_keys + width_pos_keys
        self.GG_keys = [
            ("POS", "POS", i, j)
            for i in range(1, self.n_pos_bins + 1)
            for j in range(i, self.n_pos_bins + 1)
        ]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.GG_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array([self.data["cells"][key] for key in self.GG_keys]).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        background = self.Background(
            **{
                k: parameters[k]
                for k in [
                    "H0",
                    "Omega_cdm0",
                    "Omega_b0",
                    "Omega_k0",
                    "w0",
                    "wa",
                    "ns",
                    "As",
                    "mnu",
                    "gamma_MG",
                    "N_mnu",
                ]
            }
        )

        # For Weyl imlementation: check if settings contains 'z_ini'
        if self.settings.get('z_ini') is None:
            raise ValueError("z_ini must be set in settings")
        z_ini = self.settings.get('z_ini')
        
        # For Weyl implementation: create lp and nlp objects, add z_ini to zs array
        lp = self.LinPerturbations(background, np.append(self.zs, z_ini))
        nlp = self.NonLinPerturbations(
            background, lp, 
            np.append(self.zs, z_ini),
            log10TAGN=parameters["log10TAGN"]
        ) 

        # Use Weyl perturbations class
        Weyl_p = Weyl_Perturbations(nlp, lp, self.zs, z_ini)
 
        # Weyl: Replaced PositionsTracer with PositionsTracer_Weyl_GC
        pos = PositionsTracer_Weyl_GC(
            Weyl_p,
            self.data["dndz_pos"],
            self.zs,
            nuisance_params={key: parameters[key] for key in self.full_pos_keys},
            include_rsd = self.settings.get('include_rsd', False),
        )
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(pos, pos).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key] for key in self.GG_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(pos, pos).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key] for key in self.GG_keys]).flatten()
        self.derived["sigma8_0"] = nlp.sigma8_0()
        self.theory_prediction.update(cell_all_th)
        return np.concatenate([v, vec])


class GGLMixin_Weyl:
    """
    Mixin class providing galaxy-galaxy lensing specific functionality for photometric likelihoods.
    This mixin extends the base photometric likelihood class to include methods and attributes
    necessary for handling weak lensing data, such as initializing shear bins, constructing
    masking vectors, and computing data and theory vectors specific to weak lensing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_ggl()

    def _init_ggl(self):
        self.n_pos_bins = self.data["dndz_pos"].shape[0]
        self.n_she_bins = self.data["dndz_she"].shape[0]
        bias_keys = [f"bhat_bin{i}" for i in range(self.n_pos_bins)] # Adjusted bias keys for Weyl implementation
        self.Jhat_keys = [f"Jhat_bin{i}" for i in range(self.n_pos_bins)] # Added Jhat keys for Weyl implementation
        mag_bias_keys = [
            f"magnification_bias_{i}" for i in range(1, self.n_pos_bins + 1)
        ]
        dz_pos_keys = [f"dz_pos_{i}" for i in range(1, self.n_pos_bins + 1)]
        width_pos_keys = [f"width_pos_{i}" for i in range(1, self.n_pos_bins + 1)]
        IA_keys = ["AIA", "EtaIA", "CIA"]
        mul_bias_keys = [
            f"multiplicative_bias_{i}" for i in range(1, self.n_she_bins + 1)
        ]
        dz_she_keys = [f"dz_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        width_she_keys = [f"width_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        self.full_pos_keys = bias_keys + mag_bias_keys + dz_pos_keys + width_pos_keys # Contains only nuisance params., not Jhat params. which are passed separately to the tracer
        self.full_she_keys = IA_keys + mul_bias_keys + dz_she_keys + width_she_keys
        self.GGL_keys = [
            ("POS", "SHE", i, j)
            for i in range(1, self.n_pos_bins + 1)
            for j in range(1, self.n_she_bins + 1)
        ]

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(self.data["ells"], self.scale_cuts[key])
                for key in self.GGL_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array([self.data["cells"][key][0] for key in self.GGL_keys]).flatten()
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        background = self.Background(
            **{
                k: parameters[k]
                for k in [
                    "H0",
                    "Omega_cdm0",
                    "Omega_b0",
                    "Omega_k0",
                    "w0",
                    "wa",
                    "ns",
                    "As",
                    "mnu",
                    "gamma_MG",
                    "N_mnu",
                ]
            }
        )

        # Use Weyl perturbations class
        if self.settings.get('z_ini') is None:
            raise ValueError("z_ini must be set in settings")
        z_ini = self.settings.get('z_ini')
        
        # For Weyl implementation: create lp and nlp objects, add z_ini to zs array
        lp = self.LinPerturbations(background, np.append(self.zs, z_ini))
        nlp = self.NonLinPerturbations(
            background, lp, 
            np.append(self.zs, z_ini), log10TAGN=parameters["log10TAGN"]
        ) 


        Weyl_p = Weyl_Perturbations(nlp, lp, self.zs, z_ini)

        # Weyl: Replaced PositionsTracer with PositionsTracer_Weyl_GGL
        pos = PositionsTracer_Weyl_GGL(
            Weyl_p,
            self.data["dndz_pos"],
            self.zs,
            nuisance_params={key: parameters[key] for key in self.full_pos_keys},
            Jhat_params={key: parameters[key] for key in self.Jhat_keys},
            include_rsd = self.settings.get('include_rsd', False),
        )
        she = ShearTracer(
            nlp,
            self.data["dndz_she"],
            self.zs,
            nuisance_params={key: parameters[key] for key in self.full_she_keys},
        )
        if self.mode == "coupled":
            cell_all_th = AngularTwoPoint(pos, she).get_pseudo_Cl(0, nlp.k, self.mixmat)
            vec = np.array([cell_all_th[key][0] for key in self.GGL_keys]).flatten()
        else:
            cell_all_th = AngularTwoPoint(pos, she).get_Cl(self.data["ells"], 0, nlp.k)
            vec = np.array([cell_all_th[key][0] for key in self.GGL_keys]).flatten()
        self.derived["sigma8_0"] = nlp.sigma8_0()
        self.theory_prediction.update(cell_all_th)
        return np.concatenate([v, vec])


class EuclidLikelihood_GCph_Weyl(GCphMixin_Weyl, PhotoLikelihoodBase):
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


class EuclidLikelihood_GGL_Weyl(GGLMixin_Weyl, PhotoLikelihoodBase):
    """
    EuclidLikelihood_GGL class for galaxy-galaxy lensing likelihood computation.

    This class combines the functionalities of PhotoLikelihoodBase and GGLMixin to compute
    the likelihood for galaxy-galaxy lensing (GGL) using photometric data. It initializes
    the necessary components and constructs a masking vector based on scale cuts for each
    GGL key.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        GGLMixin: Mixin providing GGL-specific functionality.

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


class EuclidLikelihood_2x2pt_Weyl(GCphMixin_Weyl, GGLMixin_Weyl, PhotoLikelihoodBase):
    """
    Likelihood class for Euclid 2x2pt photometric clustering and galaxy-galaxy lensing analysis.

    This class combines galaxy clustering (GCph) and galaxy-galaxy lensing (GGL) likelihoods,
    providing methods to compute the full data and theory vectors for both probes.

    Note: The order of inheritance matters due to the method resolution order (MRO) in Python
    and how mixins extend the base class functionality. The order of the mixins assumes
    the ordering of the covariance matrix blocks is GCph, GGL.

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

