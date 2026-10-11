import numpy as np
from cloelib.summary_statistics.angular_two_point import AngularTwoPoint
from cloelike.EuclidLikelihood_photo_base import PhotoLikelihoodBase


class WLCosebi:
    """
    WLCosebi class providing COSEBI weak lensing (WL) specific functionality for photometric likelihoods.
    This mixin extends the base photometric likelihood class to include methods and attributes
    necessary for handling COSEBI weak lensing data, such as initializing shear bins, constructing
    masking vectors, and computing data and theory vectors specific to weak lensing.

    Only the E-modes (``array[0, 0]``) enter the likelihood. Data and theory are
    ``COSEBI`` objects of shape ``(2, 2, n_modes)`` (``[[EE, EB], [EB, BB]]``)
    and are matched by mode number, so ``selected_modes`` can be any subset of
    the data modes.

    Required entries in ``data``:
        cosebis: dict of ``COSEBI`` objects keyed by ``('SHE', 'SHE', i, j)``,
            as returned by ``euclidlib.le3.twopcf_wl.cosebis``. ``thmin`` and
            ``thmax`` are in arcmin, as in the Euclid LE3 product.

    Required entries in ``settings``:
        w_ells: harmonic-space COSEBI kernels (global or per-bin layout, see
            ``cloelib.summary_statistics.angular_two_point.get_cosebis_from_cl``).
            They set the angular range (THMIN/THMAX in radians), which must
            match the data.
        ells_integration_COSEBI: multipoles at which ``w_ells`` are evaluated.
        selected_modes (optional): modes entering the likelihood, defaults to 1..7.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the next class in the MRO
        self._init_wl()
        self._init_cosebis()

    def _init_wl(self):
        self.n_she_bins = self.data["dndz_she"].shape[0]
        self._init_she_keys()
        self.WL_keys = [
            ("SHE", "SHE", i, j)
            for i in range(1, self.n_she_bins + 1)
            for j in range(i, self.n_she_bins + 1)
        ]

    def _init_cosebis(self):
        if "w_ells" not in self.settings:
            raise ValueError("settings['w_ells'] (COSEBI kernels) is required.")
        self.w_ells = self.settings["w_ells"]
        self.ells_integration_COSEBI = self.settings.get("ells_integration_COSEBI")
        if self.ells_integration_COSEBI is None:
            raise ValueError(
                "settings['ells_integration_COSEBI'] matching the w_ells is required."
            )
        self.selected_modes = [
            int(n) for n in np.unique(self.settings.get("selected_modes", range(1, 8)))
        ]

        # Position of each selected mode in the data vector of every bin pair
        self._mode_index = {}
        for key in self.WL_keys:
            data_modes = [int(n) for n in np.asarray(self.data["cosebis"][key].mode)]
            missing = [n for n in self.selected_modes if n not in data_modes]
            if missing:
                raise ValueError(f"selected_modes {missing} not in the data for {key}.")
            self._mode_index[key] = np.array(
                [data_modes.index(n) for n in self.selected_modes]
            )
        self._check_kernel_scales()

    def _kernel_dicts(self):
        """All kernel dicts in `w_ells`, for both the global and per-bin layouts."""
        first_val = next(v for k, v in self.w_ells.items() if k != "metadata")
        if isinstance(first_val, dict):
            return list(self.w_ells.values())
        return [self.w_ells]

    def _check_kernel_scales(self):
        """Check that the kernels cover the same angular range as the data."""
        for kernel in self._kernel_dicts():
            missing = [n for n in self.selected_modes if n not in kernel]
            if missing:
                raise ValueError(f"selected_modes {missing} not in the w_ells kernels.")
        for key in self.WL_keys:
            c = self.data["cosebis"][key]
            data_range = np.radians(np.array([c.thmin, c.thmax]) / 60)
            for kernel in self._kernel_dicts():
                meta = kernel["metadata"]
                kernel_range = np.array([meta["THMIN"], meta["THMAX"]])
                if not np.allclose(kernel_range, data_range, rtol=1e-3):
                    raise ValueError(
                        f"w_ells angular range {np.degrees(kernel_range) * 60} arcmin "
                        f"does not match the data range [{c.thmin}, {c.thmax}] arcmin "
                        f"for {key}."
                    )

    def get_masking_vector(self):
        v = super().get_masking_vector()
        vec = np.concatenate(
            [
                np.isin(self.data["cosebis"][key].mode, self.selected_modes)
                for key in self.WL_keys
            ]
        )
        return np.concatenate([v, vec])

    def get_data_vector_full(self):
        v = super().get_data_vector_full()
        vec = np.concatenate(
            [np.asarray(self.data["cosebis"][key].array[0, 0]) for key in self.WL_keys]
        )
        return np.concatenate([v, vec])

    def get_theory_vector_full(self, parameters):
        v = super().get_theory_vector_full(parameters)
        _, _, nlp = self._get_perturbations(parameters)
        she = self._get_she_tracer(parameters, nlp)
        cosebi_all_th = AngularTwoPoint(she, she).get_cosebis(
            self.ells_integration_COSEBI, 0, nlp.k, self.w_ells, self.selected_modes
        )

        # Place the theory at the data positions of each selected mode; the
        # remaining modes are NaN and removed by the masking vector.
        vecs = []
        for key in self.WL_keys:
            full = np.full(len(self.data["cosebis"][key].mode), np.nan)
            full[self._mode_index[key]] = np.asarray(cosebi_all_th[key].array[0, 0])
            vecs.append(full)
        self.theory_prediction = cosebi_all_th
        return np.concatenate([v, np.concatenate(vecs)])


class EuclidLikelihood_WLCosebi(WLCosebi, PhotoLikelihoodBase):
    """
    EuclidLikelihood_WLCosebi computes the weak lensing (WL) COSEBI likelihood for photometric surveys using Euclid data.

    Inherits from:
        PhotoLikelihoodBase: Base class for photometric likelihoods.
        WLCosebi: Mixin providing COSEBI weak lensing specific functionality.

    Parameters
    ----------
    data : dict
        Input data required for likelihood computation: ``cosebis`` (as read by
        ``euclidlib.le3.twopcf_wl.cosebis``), ``dndz_she``, ``z_arr`` and ``cov``.
    settings : dict
        Configuration settings: ``w_ells``, ``ells_integration_COSEBI`` and
        optionally ``selected_modes``. Scale cuts are set by the ``w_ells``.
    Background : object
        Instance representing the cosmological background model.
    LinPerturbations : object
        Instance representing linear perturbations.
    NonLinPerturbations : object
        Instance representing non-linear perturbations.
    """

    pass
