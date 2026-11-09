"""Tests of the tSZ x photo likelihoods on the GODMAX halo model.

The data are synthetic and decoupled: the theory of `EuclidLikelihood_3x2pt_tSZx`
at the fiducial parameters, with a diagonal covariance of 10% errors.
"""

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("godmax") is None:
    pytest.skip("godmax is not installed", allow_module_level=True)
# Skip until the installed cloelib has the GODMAX backend.
pytest.importorskip("cloelib.cosmology.godmax_cosmology")

from cloelib.cosmology.godmax_cosmology import (  # noqa: E402
    GODMAXNonLinearPerturbations,
)
from cloelib.cosmology.jax_cosmology import (  # noqa: E402
    JAXBackground,
    JAXLinearPerturbations,
    JAXNonLinearPerturbations,
)
from cloelib.observables.cmb import tSZTracer  # noqa: E402
from cloelib.observables.kernel_conventions import kernel_convention  # noqa: E402
from cloelib.observables.photo import PositionsTracer, ShearTracer  # noqa: E402
from cloelib.summary_statistics.angular_two_point import (  # noqa: E402
    AngularTwoPoint,
)

from cloelike.EuclidLikelihood_tsz import (  # noqa: E402
    EuclidLikelihood_3x2pt_tSZx,
    EuclidLikelihood_tSZx,
    EuclidLikelihood_tSZxGC,
    EuclidLikelihood_tSZxtSZ,
    EuclidLikelihood_tSZxWL,
)

N_BINS = 2
Z = np.linspace(0.05, 2.0, 40)
ELLS = np.geomspace(20.0, 2000.0, 6)


def gaussian_nz(mean, sigma):
    nz = np.exp(-0.5 * ((Z - mean) / sigma) ** 2)
    return nz / np.trapezoid(nz, Z)


DNDZ_POS = np.array([gaussian_nz(0.4, 0.1), gaussian_nz(0.8, 0.15)])
DNDZ_SHE = np.array([gaussian_nz(0.6, 0.2), gaussian_nz(1.1, 0.25)])

TSZ_KEYS = (
    [("Y", "Y", 1, 1)]
    + [("POS", "Y", i, 1) for i in range(1, N_BINS + 1)]
    + [("SHE", "Y", i, 1) for i in range(1, N_BINS + 1)]
)
PHOTO_KEYS = (
    [("POS", "POS", i, j) for i in range(1, N_BINS + 1) for j in range(i, N_BINS + 1)]
    + [("POS", "SHE", i, j) for i in range(1, N_BINS + 1) for j in range(1, N_BINS + 1)]
    + [("SHE", "SHE", i, j) for i in range(1, N_BINS + 1) for j in range(i, N_BINS + 1)]
)


@pytest.fixture(scope="module")
def fiducial():
    params = {
        "H0": 67.2,
        "Omega_cdm0": 0.261,
        "Omega_b0": 0.049,
        "Omega_k0": 0.0,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.96,
        "As": 2.1e-9,
        "alpha_s": 0.0,
        # GODMAX has no massive neutrinos.
        "mnu": 0.0,
        "N_mnu": 0,
        "gamma_MG": 0.55,
        # GODMAX defaults.
        "theta_ej_0": 2.0,
        "log10M1_fshmr": 12.35,
        "AIA": 0.5,
        "EtaIA": 1.0,
        "CIA": 0.0134,
    }
    for i in range(1, N_BINS + 1):
        params.update(
            {
                f"multiplicative_bias_{i}": 0.0,
                f"dz_shear_{i}": 0.0,
                f"width_shear_{i}": 1.0,
                f"dz_pos_{i}": 0.0,
                f"width_pos_{i}": 1.0,
                f"magnification_bias_{i}": 0.0,
            }
        )
    return params


def make_settings(**overrides):
    settings = {
        "scale_cuts": {key: [0.0, 1e5] for key in TSZ_KEYS + PHOTO_KEYS},
        "nonlinear_param_keys": ("theta_ej_0",),
        "nonlinear_kwargs": {
            "model_tSZ": True,
            "galaxies": {"nbar": 5e-4},
            "godmax_config": {"halo_params": {"nM": 12}},
        },
        "hod_param_keys": ("log10M1_fshmr",),
        "nonlinear_redshifts": np.linspace(Z[0], Z[-1], 20),
        "kernel_convention": "godmax",
    }
    settings.update(overrides)
    return settings


def make_data(cells=None, cov=None):
    return {
        "cells": {} if cells is None else cells,
        "ells": ELLS,
        "z_arr": Z,
        "dndz_pos": DNDZ_POS,
        "dndz_she": DNDZ_SHE,
        "cov": cov,
    }


def make_likelihood(cls, data=None, nonlinear=GODMAXNonLinearPerturbations, **kw):
    mode = kw.pop("mode", "decoupled")
    return cls(
        make_data() if data is None else data,
        make_settings(**kw),
        JAXBackground,
        JAXLinearPerturbations,
        nonlinear,
        mode=mode,
    )


@pytest.fixture(scope="module")
def synthetic_cells(fiducial):
    """Theory Cls of every tSZ and 3x2pt block at the fiducial parameters."""
    like = make_likelihood(EuclidLikelihood_3x2pt_tSZx)
    like.get_theory_vector_full(fiducial)
    return dict(like.theory_prediction)


def synthetic_likelihood(cls, cells, fiducial):
    """A likelihood of `cls` whose data are its own theory at `fiducial`."""
    probe = make_likelihood(cls, data=make_data(cells))
    theory = probe.get_data_vector_full()
    cov = np.diag((0.1 * np.abs(theory)) ** 2 + 1e-60)
    return make_likelihood(cls, data=make_data(cells, cov))


@pytest.mark.parametrize(
    "cls, n_blocks",
    [
        (EuclidLikelihood_tSZxtSZ, 1),
        (EuclidLikelihood_tSZxGC, N_BINS),
        (EuclidLikelihood_tSZxWL, N_BINS),
        (EuclidLikelihood_tSZx, 1 + 2 * N_BINS),
        (EuclidLikelihood_3x2pt_tSZx, len(TSZ_KEYS) + len(PHOTO_KEYS)),
    ],
)
def test_loglike_is_zero_at_fiducial(synthetic_cells, fiducial, cls, n_blocks):
    like = synthetic_likelihood(cls, synthetic_cells, fiducial)
    assert like.get_data_vector_full().shape == (n_blocks * len(ELLS),)
    assert like.loglike(fiducial) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("name, shift", [("theta_ej_0", 1.0), ("log10M1_fshmr", 0.3)])
def test_parameter_change_lowers_loglike(synthetic_cells, fiducial, name, shift):
    like = synthetic_likelihood(EuclidLikelihood_tSZx, synthetic_cells, fiducial)
    moved = dict(fiducial, **{name: fiducial[name] + shift})
    assert like.loglike(moved) < -1.0


def test_block_order(synthetic_cells, fiducial):
    """Blocks are ordered tSZxWL, tSZxGC, tSZxtSZ, WL, GGL, GCph."""
    like = make_likelihood(EuclidLikelihood_3x2pt_tSZx, data=make_data(synthetic_cells))
    expected = np.concatenate(
        [np.asarray(synthetic_cells[k][0]).ravel() for k in like.tSZxWL_keys]
        + [np.asarray(synthetic_cells[k]).ravel() for k in like.tSZxGC_keys]
        + [np.asarray(synthetic_cells[k]).ravel() for k in like.tSZxtSZ_keys]
        + [np.asarray(synthetic_cells[k][0, 0]).ravel() for k in like.WL_keys]
        + [np.asarray(synthetic_cells[k][0]).ravel() for k in like.GGL_keys]
        + [np.asarray(synthetic_cells[k]).ravel() for k in like.GG_keys]
    )
    np.testing.assert_allclose(like.get_data_vector_full(), expected)


def test_godmax_convention_matches_cloelib_tracers(fiducial):
    """The likelihood applies the kernel convention as cloelib documents it."""
    like = make_likelihood(EuclidLikelihood_tSZx)
    theory = like.get_theory_vector_full(fiducial)
    _, _, nlp = like._get_perturbations(fiducial)

    convention = kernel_convention("godmax")
    nuisance = convention.nuisance(fiducial)
    she = ShearTracer(
        nlp, DNDZ_SHE, Z, nuisance_params=nuisance, **convention.shear_options()
    )
    pos = PositionsTracer(
        nlp, DNDZ_POS, Z, nuisance_params=nuisance, **convention.positions_options()
    )
    tsz = tSZTracer(nlp, Z, **convention.tsz_options())
    expected = np.concatenate(
        [
            np.asarray(AngularTwoPoint(she, tsz).get_Cl(ELLS, 0, nlp.k)[k][0])
            for k in like.tSZxWL_keys
        ]
        + [
            np.asarray(AngularTwoPoint(pos, tsz).get_Cl(ELLS, 0, nlp.k)[k])
            for k in like.tSZxGC_keys
        ]
        + [
            np.asarray(AngularTwoPoint(tsz, tsz).get_Cl(ELLS, 0, nlp.k)[k])
            for k in like.tSZxtSZ_keys
        ]
    )
    np.testing.assert_allclose(theory, expected, rtol=1e-10)


def test_hod_needs_no_bias_parameters():
    like = make_likelihood(EuclidLikelihood_tSZxGC)
    assert like.galaxy_bias_model == "hod"
    assert not [k for k in like.full_pos_keys if k.startswith("b1_")]


def test_hod_parameter_change_rebuilds_halo_model(fiducial):
    like = make_likelihood(EuclidLikelihood_tSZxtSZ)
    _, _, nlp = like._get_perturbations(fiducial)
    assert like._get_perturbations(dict(fiducial))[2] is nlp
    moved = dict(fiducial, log10M1_fshmr=fiducial["log10M1_fshmr"] + 0.1)
    assert like._get_perturbations(moved)[2] is not nlp


@pytest.mark.parametrize(
    "cls, kwargs, error",
    [
        (EuclidLikelihood_tSZxtSZ, {"mode": "coupled"}, NotImplementedError),
        (
            EuclidLikelihood_tSZxWL,
            {"kernel_convention": None, "ia_model": "TATT"},
            NotImplementedError,
        ),
        (
            EuclidLikelihood_tSZxGC,
            {"kernel_convention": None, "galaxy_bias_model": "nonlinear"},
            NotImplementedError,
        ),
        (EuclidLikelihood_tSZx, {"galaxy_bias_model": "poly"}, ValueError),
        (EuclidLikelihood_tSZx, {"hod_param_keys": ("not_an_hod_param",)}, ValueError),
        (
            EuclidLikelihood_tSZx,
            {"nonlinear": JAXNonLinearPerturbations, "kernel_convention": None},
            ValueError,
        ),
    ],
)
def test_construction_errors(cls, kwargs, error):
    with pytest.raises(error):
        make_likelihood(cls, **kwargs)
