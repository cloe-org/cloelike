"""Spectroscopic power-spectrum likelihood with PBJ beyond-LCDM support."""

from typing import Optional

import numpy as np

from cloelib.summary_statistics.legendre_multipoles import LegendreMultipoles

from .EuclidLikelihood_GCspectro_Pls import EuclidLikelihood_GCspectro_Pls


class EuclidLikelihood_GCspectro_Pls_beyond_LCDM(EuclidLikelihood_GCspectro_Pls):
    """GCspectro likelihood that can pass separate LCDM and MG perturbations to PBJ."""

    MG_PARAMETER_ALIASES = {
        "fr0": ("fr0", "fR0"),
        "omega_rc": ("omega_rc", "omegarc", "Omrc"),
        "gamma0": ("gamma0", "gamma"),
        "gamma1": ("gamma1",),
        "xi": ("xi",),
        "mu0": ("mu0",),
        "sigma0": ("sigma0",),
        "w0": ("w0",),
        "wa": ("wa",),
    }
    MODEL_ALIASES = {
        "growth_model": ("growth_model", "gravity_model", "model"),
        "gravity_model": ("gravity_model", "growth_model", "model"),
    }
    GROWTH_PERTURBATION_MODEL_MAP = {
        "darkscattering": "ide",
        "growthindex": "gamma",
        "gamma": "gamma",
        "ide": "ide",
        "fr": "fr",
        "dgp": "dgp",
        "wcdm": "w0wacdm",
        "w0wacdm": "w0wacdm",
    }

    def __init__(
        self,
        data: dict,
        settings: dict,
        Background: type,
        SpectroPower: type,
        Perturbations: Optional[type] = None,
        GrowthPerturbations: Optional[type] = None,
        gravity_model: Optional[str] = None,
        AM_priors: Optional[dict] = None,
    ):
        r"""Class constructor.

        Parameters
        ----------
        data, settings, Background, SpectroPower, Perturbations, AM_priors
            Same meaning as in :class:`EuclidLikelihood_GCspectro_Pls`.
        GrowthPerturbations
            Optional CLOE MGrowth perturbation class. When supplied it is built
            from the LCDM `Perturbations` instance and passed to the PBJ
            beyond-LCDM spectro interface as `growth_perturbations`.
        gravity_model
            Optional MG model tag, e.g. `fr`, `dgp`, `ndgp`, `gamma`, or `ide`.
            If omitted, the value is read from `settings` when present.
        """
        self.GrowthPerturbations = GrowthPerturbations
        self.gravity_model = gravity_model or self._get_setting(
            settings, self.MODEL_ALIASES["gravity_model"]
        )
        self.fixed_mgpars = settings.get("mgpars", {})
        try:
            super().__init__(
                data=data,
                settings=settings,
                Background=Background,
                SpectroPower=SpectroPower,
                Perturbations=Perturbations,
                AM_priors=AM_priors,
            )
        except TypeError as exc:
            if "Perturbations" not in str(exc):
                raise
            super().__init__(
                data=data,
                settings=settings,
                Background=Background,
                SpectroPower=SpectroPower,
                AM_priors=AM_priors,
            )
            self.Perturbations = Perturbations
            self.NLcode = SpectroPower.NLcode
            self._ensure_pbj_base_attributes()

    def _ensure_pbj_base_attributes(self):
        """Provide PBJ bookkeeping when inheriting from older likelihood bases."""
        if (
            not isinstance(getattr(self, "RSD_parameter_names", None), dict)
            or self.NLcode not in self.RSD_parameter_names
        ):
            self.RSD_parameter_names = {
                "COMET": ["b1", "b2", "bG2", "bGam3", "c0", "c2", "c4", "cnlo"],
                "PBJ": ["b1", "b2", "bG2", "bG3", "c0", "c2", "c4", "ck4"],
            }
        if not hasattr(self, "noise_syst_parameter_names"):
            self.noise_syst_parameter_names = ["NP0", "NP20", "NP22", "fout", "sigmaz"]

        if (
            self.AM_priors
            and (
                not isinstance(getattr(self, "AM_par_to_diag", None), dict)
                or self.NLcode not in self.AM_par_to_diag
            )
        ):
            self.AM_par_to_diag = {
                "COMET": {
                    "bGam3": ["b1-bGam3", "bGam3"],
                    "c0": ["c0"],
                    "c2": ["c2"],
                    "c4": ["c4"],
                    "cnlo": ["b1-b1-cnlo", "b1-cnlo", "cnlo"],
                    "NP0": ["noise_k0"],
                    "NP20": ["noise_k2"],
                    "NP22": ["noise_k2mu2"],
                },
                "PBJ": {
                    "bG3": ["bG3"],
                    "c0": ["c0"],
                    "c2": ["c2"],
                    "c4": ["c4"],
                    "cnlo": ["ck4"],
                    "NP0": ["noise_k0"],
                    "NP20": ["noise_k2"],
                    "NP22": ["noise_k2mu2"],
                },
            }
            self.AM_diagrams = [
                term
                for values in self.AM_par_to_diag[self.NLcode].values()
                for term in values
            ]

    @staticmethod
    def _get_setting(settings: dict, names):
        """Read the first available setting across a list of aliases."""
        for name in names:
            if name in settings and settings[name] is not None:
                return settings[name]
        return None

    @staticmethod
    def _as_scalar_if_single(value):
        """Return a Python scalar for one-element arrays, otherwise the value."""
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        if arr.size == 1:
            return arr.ravel()[0].item()
        return value

    def _parameter_value(self, parameters: dict, names):
        """Read the first available parameter across aliases."""
        for name in names:
            if name in parameters and parameters[name] is not None:
                return self._as_scalar_if_single(parameters[name])
        return None

    def _mg_parameters(self, parameters: dict) -> dict:
        """Build a CLOE-style MG parameter dictionary from sampled parameters."""
        mgpars = dict(self.fixed_mgpars)
        for target_name, aliases in self.MG_PARAMETER_ALIASES.items():
            value = self._parameter_value(parameters, aliases)
            if value is not None:
                mgpars[target_name] = value
        return mgpars

    def _growth_perturbation_model(self) -> str:
        """Translate PBJ/CLOE model tags to the CLOE MGrowth perturbation tag."""
        model = str(self.gravity_model).lower()
        if model not in self.GROWTH_PERTURBATION_MODEL_MAP:
            raise ValueError(
                f"GrowthPerturbations does not support gravity_model={self.gravity_model!r}. "
                "For PBJ-internal growth models such as 'ndgp', omit GrowthPerturbations."
            )
        return self.GROWTH_PERTURBATION_MODEL_MAP[model]

    def _spectro_extra_parameters(self, parameters: dict) -> dict:
        """Parameters passed through the PBJ spectro interface but not varied as RSD terms."""
        extras = self._mg_parameters(parameters)
        if self.gravity_model is not None:
            extras["gravity_model"] = self.gravity_model
        return extras

    def _build_background(self, parameters: dict):
        """Build the background cosmology from likelihood parameters."""
        return self.Background(
            H0=parameters["H0"],
            Omega_cdm0=parameters["Omega_cdm0"],
            Omega_b0=parameters["Omega_b0"],
            Omega_k0=parameters["Omega_k0"],
            w0=parameters["w0"],
            wa=parameters["wa"],
            ns=parameters["ns"],
            As=parameters["As"],
            gamma_MG=parameters["gamma_MG"],
            mnu=parameters["mnu"],
            N_mnu=parameters["N_mnu"],
        )

    def _build_perturbations(self, background, parameters: dict):
        """Build LCDM baseline and optional MGrowth perturbations."""
        if self.Perturbations is None:
            raise ValueError(
                "EuclidLikelihood_GCspectro_Pls_beyond_LCDM requires a Perturbations "
                "class to provide the LCDM baseline linear spectrum."
            )

       # zs = np.float64(self.redshifts)
        
        zs_interp = np.asarray(self.settings.get(
                "hmcode_emu_redshifts",
                np.linspace(1e-4, 3.0, 100),
            ),
            dtype=float,
        )

        linear_perturbations = self.Perturbations(background, zs_interp)
        if self.GrowthPerturbations is None:
            return linear_perturbations, None

        if self.gravity_model is None:
            raise ValueError(
                "GrowthPerturbations was supplied, but no gravity_model was provided "
                "or found in settings."
            )
        growth_perturbations = self.GrowthPerturbations(
            background,
            linear_perturbations,
            self._growth_perturbation_model(),
            self._mg_parameters(parameters),
        )
        return linear_perturbations, growth_perturbations

    def _build_spectro_power(
        self,
        cosmo_input,
        growth_perturbations,
        RSD_parameters: dict,
        parameters: dict,
        redshift: float,
    ):
        """Instantiate the beyond-LCDM PBJ spectro power object."""
        spectro_parameters = {
            **RSD_parameters,
            **self._spectro_extra_parameters(parameters),
        }
        return self.SpectroPower(
            cosmo_input,
            spectro_parameters,
            growth_perturbations=growth_perturbations,
            redshift=redshift,
        )

    def get_theory_vector(self, parameters: dict) -> np.ndarray:
        r"""Generate the stacked GCspectro theory vector."""
        background = self._build_background(parameters)
        cosmo_input, growth_perturbations = self._build_perturbations(
            background, parameters
        )

        theory_vec = []
        for i, z in enumerate(self.redshifts):
            RSD_params = {
                key: parameters[key][i] for key in self.RSD_parameter_names[self.NLcode]
            }
            syst_params = {
                key: parameters[key][i] for key in self.noise_syst_parameter_names
            }

            power = self._build_spectro_power(
                cosmo_input,
                growth_perturbations,
                RSD_params,
                parameters,
                redshift=float(z),
            )
            obs = LegendreMultipoles(
                spectro_power=power,
                background_fiducial=self.background_fiducial,
                parameters=syst_params,
                nbar=self.nbar[i],
            )

            k = self.data["GCspectro"][z]["k"]
            if self.mixmat:
                mps = obs.convolved_power_multipoles(
                    self.mixmat[z], ells=self.ells, use_AP=True
                )
            else:
                mps = obs.power_multipoles(k=k, ells=self.ells, use_AP=True)
            theory_vec.extend(np.concatenate([mps[f"ell{ell}"] for ell in self.ells]))

        return np.array(theory_vec)

    def get_theory_vector_AM(self, parameters: dict, term_list: dict):
        r"""Generate the GCspectro theory vector and analytical-marginalisation terms."""
        background = self._build_background(parameters)
        cosmo_input, growth_perturbations = self._build_perturbations(
            background, parameters
        )

        theory_vec = []
        theory_vec_AM = {}
        coeff = self._coeff_AM(parameters)
        for i, z in enumerate(self.redshifts):
            RSD_parameters = {
                key: parameters[key][i] for key in self.RSD_parameter_names[self.NLcode]
            }
            power = self._build_spectro_power(
                cosmo_input,
                growth_perturbations,
                RSD_parameters,
                parameters,
                redshift=float(z),
            )
            noise_syst_parameters = {
                key: parameters[key][i] for key in self.noise_syst_parameter_names
            }
            obs = LegendreMultipoles(
                spectro_power=power,
                background_fiducial=self.background_fiducial,
                parameters=noise_syst_parameters,
                nbar=self.nbar[i],
            )

            k = self.data["GCspectro"][z]["k"]
            if self.mixmat:
                mps_dict = obs.convolved_power_multipoles(
                    self.mixmat[z], ells=self.ells, use_AP=True
                )
            else:
                mps_dict = obs.power_multipoles(k=k, ells=self.ells, use_AP=True)
            mps_list = [mps_dict[f"ell{ell}"] for ell in self.ells]
            theory_vec = np.concatenate((theory_vec, np.concatenate(mps_list)))
            if z in self.AM_params.keys():
                if self.mixmat:
                    mps_AM_dict = obs.convolved_power_term_multipoles(
                        self.mixmat[z],
                        term_list=term_list[z],
                        ells=self.ells,
                        use_AP=True,
                    )
                else:
                    mps_AM_dict = obs.power_term_multipoles(
                        k=k, term_list=term_list[z], ells=self.ells, use_AP=True
                    )
                terms_to_scale = [term for term in term_list[z] if term in coeff]
                indices_to_scale = [term_list[z].index(term) for term in terms_to_scale]
                for ell in self.ells:
                    for idx, term in zip(indices_to_scale, terms_to_scale):
                        mps_AM_dict[f"ell{ell}"][idx, :] *= coeff[term][i]
                mps_AM_list = [mps_AM_dict[f"ell{ell}"] for ell in self.ells]
                theory_vec_AM[z] = np.hstack(mps_AM_list)
            else:
                Nk = sum(
                    len(self.data["GCspectro"][z][f"pk{ell}"]) for ell in self.ells
                )
                theory_vec_AM[z] = np.zeros((0, Nk))
        return theory_vec, theory_vec_AM
