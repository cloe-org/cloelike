from cloelib.cosmology.Weyl_cosmology import WeylNonLinearPerturbations, WeylLinearPerturbations
from cloelib.observables.photo_Weyl import PositionsTracer_Weyl_GC, PositionsTracer_Weyl_GGL
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

    def _get_Weyl_perturbations(self, parameters):
        """Return cosmological perturbations adapted to the Weyl potential.

        The ordinary perturbation objects are built by
        :meth:`PhotoLikelihoodBase._get_perturbations` and then wrapped by
        :class:`WeylLinearPerturbations` and :class:`WeylNonLinearPerturbations`.
        The resulting tuple has the same background object as the parent and
        Weyl-specific linear/non-linear perturbation objects for tracer use.
        """
        # Check if settings contains 'z_ini'
        if self.settings.get('z_ini') is None:
            raise ValueError("z_ini must be set in settings")

        background, lp, nlp = self._get_perturbations(parameters)
        Weyl_lp = WeylLinearPerturbations(lp, self.zs, self.settings.z_ini)
        Weyl_nlp = WeylNonLinearPerturbations(
            nlp, Weyl_lp, self.zs, self.settings.z_ini
        )
        return background, Weyl_lp, Weyl_nlp

    def _get_pos_tracer(self, parameters, Weyl_nlp, mode):
        """Build and cache the Weyl-specific position tracer.

        Args:
            parameters (dict): Cosmological and nuisance parameters.
            Weyl_nlp: Weyl non-linear perturbations used by the tracer.
            mode (str): Probe type. Must be ``"GC"`` or ``"GGL"``.

        Returns:
            PositionsTracer_Weyl_GC or PositionsTracer_Weyl_GGL: Cached tracer
            instance corresponding to ``mode``.

        Raises:
            ValueError: If ``mode`` is not ``"GC"`` or ``"GGL"``.
        """
        key = self._cosmo_key(parameters) + tuple(
            parameters[k] for k in self.full_pos_keys
        )
        cached = getattr(self, "_pos_tracer_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        nuisance_params = {k: parameters[k] for k in self.full_pos_keys}
        include_rsd = self.settings.get("include_rsd", False)

        if mode == "GC":
            tracer = PositionsTracer_Weyl_GC(
                Weyl_nlp,
                self.data["dndz_pos"],
                self.zs,
                nuisance_params=nuisance_params,
                include_rsd=include_rsd,
            )
        elif mode == "GGL":
            tracer = PositionsTracer_Weyl_GGL(
                Weyl_nlp,
                self.data["dndz_pos"],
                self.zs,
                nuisance_params=nuisance_params,
                include_rsd=include_rsd,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'GC' or 'GGL'.")

        self._pos_tracer_cache = (key, tracer)
        return tracer
