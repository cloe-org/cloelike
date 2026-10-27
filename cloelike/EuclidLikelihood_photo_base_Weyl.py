import numpy as np
from cloelib.cosmology.Weyl_cosmology import WeylPerturbations
from cloelib.observables.photo.positions_Weyl import PositionsTracer_Weyl_GC, PositionsTracer_Weyl_GGL, ShearTracer_IAOnly
from cloelib.observables.photo.positions import PositionsTracer
from cloelib.observables.photo.shear import PBJTATTLoopComputer
from cloelike.EuclidLikelihood_photo_base import PhotoLikelihoodBase

class PhotoLikelihoodBase_Weyl(PhotoLikelihoodBase):
    """Base likelihood class for the Weyl-potential photometric model.

    This class reuses the common data preparation, masking, covariance, caching,
    and likelihood machinery implemented by :class:`PhotoLikelihoodBase`.  It
    adds the Weyl-specific perturbation construction and tracer factory used by
    the Weyl galaxy-clustering (GC) and galaxy-galaxy lensing (GGL) probes.

    The inherited base-class API is intentionally unchanged, so existing
    photometric likelihood subclasses can use this class wherever the ordinary
    photometric base class was used.

    Args:
        data (dict): Observational data, including the redshift grid and, when
            applicable, angular power-spectrum measurements and covariance.
        settings (dict): Likelihood configuration. The Weyl implementation
            additionally expects ``z_ini`` for the Weyl perturbation wrappers.
        Background: Background cosmology constructor/class used by the parent
            likelihood implementation.
        LinPerturbations: Linear perturbation constructor/class used by the
            parent likelihood implementation.
        NonLinPerturbations: Non-linear perturbation constructor/class used by
            the parent likelihood implementation.
        ells_integration (array-like, optional): Multipoles used for theory
            calculations. If omitted, the parent class uses its default range.
        mode (str, optional): Likelihood assembly mode. Defaults to ``"coupled"``.

    Attributes inherited from :class:`PhotoLikelihoodBase` include ``data``,
    ``settings``, ``derived``, ``scale_cuts``, ``zs``, ``mixmat``, and the
    cached masking/covariance helpers.
    """

    def __init__(
        self,
        data,
        settings,
        Background,
        LinPerturbations,
        NonLinPerturbations,
        ells_integration=None,
        mode="coupled",
    ):
        """Initialize the shared photometric likelihood state via the parent."""
        super().__init__(
            data=data,
            settings=settings,
            Background=Background,
            LinPerturbations=LinPerturbations,
            NonLinPerturbations=NonLinPerturbations,
            ells_integration=ells_integration,
            mode=mode,
        )

    def _init_pos_keys(self):
        """Position nuisance parameters required by the selected bias model."""
        if self.galaxy_bias_model != "per_bin":
            raise ValueError(
                "Weyl tracer classes require "
                "galaxy_bias_model='per_bin'."
            )

        n = self.data["dndz_pos"].shape[0]
        bias_keys = [f"bhat_bin{i}" for i in range(n)]
        optional = []

        self.full_pos_keys = (
            bias_keys
            + [f"magnification_bias_{i}" for i in range(1, n + 1)]
            + [f"dz_pos_{i}" for i in range(1, n + 1)]
            + [f"width_pos_{i}" for i in range(1, n + 1)]
        )
        self.optional_pos_keys = optional

    def _get_perturbations(self, parameters):
        """Build (and cache) Background/LinPerturbations/NonLinPerturbations for
        the given parameters.

        Combined likelihoods (e.g. 3x2pt) mix several probe-specific mixins in
        one MRO chain, each of which needs the perturbations. Caching here
        (instead of rebuilding in every mixin's get_theory_vector_full) avoids
        recomputing the same cosmology multiple times per call. The cache is
        keyed on the cosmology-relevant subset of `parameters` so a change in
        any of those values rebuilds it, while repeated calls with an unchanged
        cosmology (e.g. only nuisance parameters varying) reuse it.
        """
        key = self._cosmo_key(parameters)
        cached = getattr(self, "_perturbations_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        background = self.Background(
            **{k: parameters[k] for k in self._BACKGROUND_PARAM_KEYS}
        )

        # Add z_ini to the redshift array for the perturbation objects, since the Weyl wrapper will require it.
        z_vals = np.append(self.zs, self.settings.get("z_ini"))

        lp = self.LinPerturbations(background=background, redshifts=z_vals)

        nl_kwargs = dict(self.settings.get("nonlinear_kwargs", {}))
        nl_kwargs.update({k: parameters[k] for k in self.nonlinear_param_keys})
        if self.baryon_param_keys:
            nl_kwargs["baryon_kwargs"] = {
                k: parameters[k] for k in self.baryon_param_keys
            }

        nlp = self.NonLinPerturbations(
            background=background,
            linearperturbations=lp,
            redshifts=z_vals,
            **nl_kwargs,
        )

        self.derived["sigma8_0"] = nlp.sigma8_0()
        result = (background, lp, nlp)
        self._perturbations_cache = (key, result)
        # Loop computers depend only on the cosmology; drop the old ones.
        self._loop_computers = {}
        return result

    def _get_Weyl_perturbations(self, parameters):
        """Return cosmological perturbations adapted to the Weyl potential.

        The ordinary perturbation objects are built by
        :meth:`PhotoLikelihoodBase._get_perturbations` and then wrapped by
        :class:`WeylPerturbations`.
        The resulting tuple has the same background object as the parent and
        Weyl-specific perturbation object for tracer use.
        """
        if self.settings.get("z_ini") is None:
            raise ValueError("z_ini must be set in settings")

        key = self._cosmo_key(parameters)

        cached = getattr(self, "_Weyl_perturbations_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        _, lp, nlp = self._get_perturbations(parameters)

        Weyl_p = WeylPerturbations(
            nlp, lp, self.zs, self.settings.get("z_ini")
        )

        result = (Weyl_p)
        self._Weyl_perturbations_cache = (key, result)
        return result
    
    def _get_pos_tracer_Weyl(self, parameters, nlp, mode):
        """Build the Weyl-specific position tracer.
        Note: Unlike the parent class, which has a single PositionsTracer, the Weyl likelihood has two different position tracers: one for GC and one for GGL. 
        This method returns the appropriate tracer based on the `mode` argument.
        We do not cache the positions tracer here, since GGL and GC need to use two different once. 

        Args:
            parameters (dict): Cosmological and nuisance parameters.
            Weyl_nlp: Weyl non-linear perturbations used by the tracer.
            mode (str): Probe type. Must be ``"GC"`` or ``"GGL"``.

        Returns:
            PositionsTracer_Weyl_GC or PositionsTracer_Weyl_GGL: Tracer
            instance corresponding to ``mode``.

        Raises:
            ValueError: If ``mode`` is not ``"GC"`` or ``"GGL"``.
        """

        # Removed checking for cashed object (GC and GGL need to use two different pos tracers)

        nuisance_params = {k: parameters[k] for k in self.full_pos_keys}
        include_rsd = self.settings.get("include_rsd", False)

        if mode == "GC":
            tracer = PositionsTracer_Weyl_GC(
                nlp,
                self.data["dndz_pos"],
                self.zs,
                nuisance_params=nuisance_params,
                include_rsd=include_rsd,
            )

        elif mode == "GGL":
            Jhat_params = {k: parameters[k] for k in self.Jhat_keys}
            tracer = PositionsTracer_Weyl_GGL(
                nlp,
                self.data["dndz_pos"],
                self.zs,
                nuisance_params=nuisance_params,
                Jhat_params=Jhat_params,
                include_rsd=include_rsd,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'GC', 'GGL' or 'standard'.")


        # Removed caching (GC and GGL need to use two different pos tracers)
        return tracer

    def _get_pos_tracer(self, parameters, nlp):
        """Build (and cache) the PositionsTracer for the given parameters/nlp,
        so GCph and GGL do not each rebuild it within the same call."""
        nuisance = self._select(parameters, self.full_pos_keys, self.optional_pos_keys)

        #key = self._cosmo_key(parameters) + tuple(sorted(nuisance.items()))
        #cached = getattr(self, "_pos_tracer_cache", None)
        #if cached is not None and cached[0] == key:
        #    return cached[1]

        # Calculate sigma8(z_effs)
        dndz_pos = self.data["dndz_pos"]
        zs = self.zs
        dz = np.gradient(zs)
        z_effs = []
        for i in range(dndz_pos.shape[0]):
            n = dndz_pos[i,:]
            z_eff = np.sum(zs * n * dz) / np.sum(n * dz)
            z_effs.append(z_eff)
        sigma8_zeffs = self.derived["sigma8_0"]*np.squeeze(nlp.growth_factor(z_effs, np.array([0.1]))) 

        # Add b1_photo_bin parameters by dividing bhat through respective sigma8(z_eff) value
        for i, sigma8 in enumerate(np.atleast_1d(sigma8_zeffs)):
            nuisance[f"b1_photo_bin{i+1}"] = nuisance[f"bhat_bin{i}"] / sigma8

        tracer = PositionsTracer(
            nlp,
            dndz_pos,
            zs,
            nuisance_params=nuisance,
            galaxy_bias_model='per_bin',
            include_rsd=self.settings.get("include_rsd", False),
        )

        #self._pos_tracer_cache = (key, tracer)
        return tracer

    def _get_she_tracer_IAOnly(self, parameters, nlp):
        """Build (and cache) the ShearTracer for the given parameters/nlp, so
        WL and GGL do not each rebuild it within the same call."""
        nuisance = self._select(parameters, self.full_she_keys, self.optional_she_keys)

        if self.ia_model == None:
            raise ValueError("ia_model is required to compute IA only contribution")

        if self.ia_model == "TATT":
            loop_computer = self._get_loop_computer(
                "tatt", PBJTATTLoopComputer, parameters
            )
        else:
            loop_computer = None

        tracer = ShearTracer_IAOnly(
            nlp,
            self.data["dndz_she"],
            self.zs,
            nuisance_params=nuisance,
            ia_model=self.ia_model,
            tatt_loop_computer=loop_computer,
        )

        return tracer
