import numpy as np
from cloelib.observables.photo import ShearTracer
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelike.EuclidLikelihood_photo_base import PhotoLikelihoodBase
from cloelib.auxiliary.cosebi_helpers import get_W_ell


class WLCosebi:
    """
    WLCosebi class providing COSEBI weak lensing (WL) specific functionality for photometric likelihoods.
    This mixin extends the base photometric likelihood class to include methods and attributes
    necessary for handling COSEBI weak lensing data, such as initializing shear bins, constructing
    masking vectors, and computing data and theory vectors specific to weak lensing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the next class in the MRO
        self._init_wl()
        self._w_ells = None

    def _init_wl(self):
        self.n_she_bins = self.data["dndz_she"].shape[0]
        IA_keys = ["AIA", "EtaIA", "CIA"]
        mul_bias_keys = [
            f"multiplicative_bias_{i}" for i in range(1, self.n_she_bins + 1)
        ]
        dz_she_keys = [f"dz_shear_{i}" for i in range(1, self.n_she_bins + 1)]
        self.full_she_keys = IA_keys + mul_bias_keys + dz_she_keys
        self.WL_keys = [
            (i, j)
            for i in range(1, self.n_she_bins + 1)
            for j in range(i, self.n_she_bins + 1)
        ]

    def _compute_w_ells(self):
        nmax = int(self.selected_modes[-1])
        print("Start computing the W_ells, this only has to be done once!")
        return get_W_ell(
            self.thetagrid, nmax, self.ells_integration_COSEBI, self.n_thread
        )

    @property
    def w_ells(self):
        if self._w_ells is None:
            self._w_ells = self._compute_w_ells()
        return self._w_ells

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                self._masking(
                    self.data["MODE"][key],
                    [self.selected_modes[0], self.selected_modes[-1]],
                )
                for key in self.WL_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.array([self.data["EE"][key] for key in self.WL_keys]).flatten()
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
        lp = self.LinPerturbations(background, self.zs)
        nlp = self.NonLinPerturbations(
            background, lp, self.zs, log10TAGN=parameters["log10TAGN"]
        )
        she = ShearTracer(
            nlp,
            self.data["dndz_she"],
            self.zs,
            nuisance_params={key: parameters[key] for key in self.full_she_keys},
        )
        cosebi_all_th = AngularTwoPoint(she, she).get_cosebis(
            self.ells_integration_COSEBI, 0, nlp.k, self.w_ells, self.selected_modes
        )

        withpadding = {}
        for key in self.WL_keys:
            withpadding[key] = np.pad(
                np.asarray(cosebi_all_th[key], dtype=float),
                (0, len(self.data["MODE"][key]) - len(cosebi_all_th[key][0][0])),
                constant_values=np.nan,
            )
        cosebi_all_th = withpadding
        vec = np.array([cosebi_all_th[key][0][0] for key in self.WL_keys]).flatten()
        self.derived["sigma8_0"] = nlp.sigma8_0()
        self.theory_prediction = cosebi_all_th
        return np.concatenate([v, vec])


class EuclidLikelihood_WLCosebi(WLCosebi, PhotoLikelihoodBase):
    """
    EuclidLikelihood_WL computes the weak lensing (WL) likelihood for photometric surveys using Euclid data.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        WLMixin: Mixin providing weak lensing specific functionality.

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
