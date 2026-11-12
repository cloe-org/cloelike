import numpy as np
import pytest

from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import (
    HMemuLinearPerturbations,
    HMemuNonLinearPerturbations,
)
from cloelike.EuclidLikelihood_CMBLensingCross import (
    EuclidLikelihood_CMBLxGC,
    EuclidLikelihood_CMBLxWL,
    EuclidLikelihood_CMBLx,
    EuclidLikelihood_5x2pt,
)

# We create mock data for testing
nb_bins = 3
z = np.linspace(1e-3, 3.0, 100)
ells = np.linspace(10, 3000, 20)


# We make gaussian n(z)
def gaussian_nz(mean, width=0.15):
    nz = np.exp(-0.5 * ((z - mean) / width) ** 2)
    return nz / np.trapezoid(nz, z)


# The same n(z) is used for positions and shear.
dndz = np.array([gaussian_nz(mean) for mean in np.linspace(0.5, 1.5, nb_bins)])

bins = range(1, nb_bins + 1)
CMBLxWL_keys = [("CMBL", "SHE", 1, i) for i in bins]
CMBLxGC_keys = [("CMBL", "POS", 1, i) for i in bins]
WL_keys = [("SHE", "SHE", i, j) for i in bins for j in bins if j >= i]
GGL_keys = [("POS", "SHE", i, j) for i in bins for j in bins]
GC_keys = [("POS", "POS", i, j) for i in bins for j in bins if j >= i]

# Keys of each likelihood, in the order of its data vector.
keys = {
    EuclidLikelihood_CMBLxWL: CMBLxWL_keys,
    EuclidLikelihood_CMBLxGC: CMBLxGC_keys,
    EuclidLikelihood_CMBLx: CMBLxWL_keys + CMBLxGC_keys,
    EuclidLikelihood_5x2pt: CMBLxWL_keys + CMBLxGC_keys + WL_keys + GGL_keys + GC_keys,
}


def fiducial_params():
    return {
        "H0": 70,
        "Omega_cdm0": 0.25,
        "Omega_b0": 0.05,
        "w0": -1,
        "wa": 0,
        "Omega_k0": 0,
        "ns": 0.96,
        "As": 2e-9,
        "alpha_s": 0.0,
        "mnu": 0.06,
        "gamma_MG": 0.545,
        "AIA": 0.5,
        "CIA": 0.0134,
        "EtaIA": 0,
        "log10TAGN": 7.75,
        "N_mnu": 1,
        "b1_photo_poly0": 1.0,
        "b1_photo_poly1": 0.5,
        "b1_photo_poly2": 0.0,
        "b1_photo_poly3": 0.0,
        **{f"magnification_bias_{i}": 0.0 for i in bins},
        **{f"dz_pos_{i}": 0.0 for i in bins},
        **{f"width_pos_{i}": 1.0 for i in bins},
        **{f"dz_shear_{i}": 0.0 for i in bins},
        **{f"width_shear_{i}": 1.0 for i in bins},
        **{f"multiplicative_bias_{i}": 0.0 for i in bins},
    }


def build_settings():
    all_keys = keys[EuclidLikelihood_5x2pt]
    scale_cuts = {key: [ells[0], ells[-1]] for key in all_keys}
    return {"scale_cuts": scale_cuts, "include_rsd": True}


def make_likelihood(cls, cells, settings=None):
    """Build a decoupled-mode likelihood with a diagonal covariance."""
    data = {
        "cells": cells,
        "ells": ells,
        "z_arr": z,
        "dndz_pos": dndz,
        "dndz_she": dndz,
        "cov": None,
    }
    like = cls(
        data=data,
        settings=settings if settings is not None else build_settings(),
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="decoupled",
    )
    # We build a fake, diagonal, covariance, equal to 10% of the datavector.
    sigma = 0.1 * np.abs(like.get_data_vector_full())
    like.data["cov"] = np.diag(sigma**2)
    return like


@pytest.fixture(scope="module")
def mock_cells():
    """Noiseless theory spectra at the fiducial cosmology, for every key."""
    data = {"ells": ells, "z_arr": z, "dndz_pos": dndz, "dndz_she": dndz}
    like = EuclidLikelihood_5x2pt(
        data=data,
        settings=build_settings(),
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="decoupled",
    )
    like.get_theory_vector_full(fiducial_params())
    return {key: np.asarray(cl.array) for key, cl in like.theory_prediction.items()}


def test_CMBLxWL_likelihood(mock_cells):
    like = make_likelihood(EuclidLikelihood_CMBLxWL, mock_cells)
    assert like.loglike(fiducial_params()) == pytest.approx(0, abs=1e-8)


def test_CMBLxGC_likelihood(mock_cells):
    like = make_likelihood(EuclidLikelihood_CMBLxGC, mock_cells)
    assert like.loglike(fiducial_params()) == pytest.approx(0, abs=1e-8)


def test_CMBLx_likelihood(mock_cells):
    like = make_likelihood(EuclidLikelihood_CMBLx, mock_cells)
    assert like.loglike(fiducial_params()) == pytest.approx(0, abs=1e-8)


def test_5x2pt_likelihood(mock_cells):
    like = make_likelihood(EuclidLikelihood_5x2pt, mock_cells)
    assert like.loglike(fiducial_params()) == pytest.approx(0, abs=1e-8)
