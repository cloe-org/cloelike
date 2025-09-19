import pytest

# --- cloe-org imports ---
from cloelike.EuclidLikelihood_photo_Cls import (
    EuclidLikelihood_WL,
    EuclidLikelihood_GCph,
    EuclidLikelihood_GGL,
    EuclidLikelihood_3x2pt,
    EuclidLikelihood_2x2pt,
)


def test_euclidlikelihood_wl_initialization():
    # Test initialization of EuclidLikelihood_WL
    likelihood = EuclidLikelihood_WL()
    assert likelihood is not None


def test_euclidlikelihood_gcph_initialization():
    # Test initialization of EuclidLikelihood_GCph
    likelihood = EuclidLikelihood_GCph()
    assert likelihood is not None


def test_euclidlikelihood_ggl_initialization():
    # Test initialization of EuclidLikelihood_GGL
    likelihood = EuclidLikelihood_GGL()
    assert likelihood is not None


def test_euclidlikelihood_3x2pt_initialization():
    # Test initialization of EuclidLikelihood_3x2pt
    likelihood = EuclidLikelihood_3x2pt()
    assert likelihood is not None


def test_euclidlikelihood_2x2pt_initialization():
    # Test initialization of EuclidLikelihood_2x2pt
    likelihood = EuclidLikelihood_2x2pt()
    assert likelihood is not None


def test_euclidlikelihood_wl_attributes():
    likelihood = EuclidLikelihood_WL()
    assert hasattr(likelihood, "compute_loglike")
    assert callable(likelihood.loglike)


def test_euclidlikelihood_gcph_attributes():
    likelihood = EuclidLikelihood_GCph()
    assert hasattr(likelihood, "compute_loglike")
    assert callable(likelihood.loglike)


def test_euclidlikelihood_ggl_attributes():
    likelihood = EuclidLikelihood_GGL()
    assert hasattr(likelihood, "compute_loglike")
    assert callable(likelihood.loglike)


def test_euclidlikelihood_3x2pt_attributes():
    likelihood = EuclidLikelihood_3x2pt()
    assert hasattr(likelihood, "compute_loglike")
    assert callable(likelihood.loglike)


def test_euclidlikelihood_2x2pt_attributes():
    likelihood = EuclidLikelihood_2x2pt()
    assert hasattr(likelihood, "compute_loglike")
    assert callable(likelihood.loglike)


@pytest.mark.parametrize(
    "LikelihoodClass",
    [
        EuclidLikelihood_WL,
        EuclidLikelihood_GCph,
        EuclidLikelihood_GGL,
        EuclidLikelihood_3x2pt,
        EuclidLikelihood_2x2pt,
    ],
)
def test_likelihood_compute_loglike_returns_float(LikelihoodClass):
    likelihood = LikelihoodClass()
    # Assuming cloglike can be called with no arguments for a default test
    result = likelihood.loglike()
    assert isinstance(result, float)
