import numpy as np
import pytest
import requests
import euclidlib as el

from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.auxiliary.bnt import BNTMatrixCalculator
from cloelib.cosmology.cosmology import with_baryon_boost
from cloelib.cosmology.HMcode2020Emu_cosmology import (
    HMcode2020BaryonBoostMixin,
    HMemuLinearPerturbations,
    HMemuNonLinearPerturbations,
)
from cloelike.EuclidLikelihood_photo_Cls import (
    EuclidLikelihood_WL,
    EuclidLikelihood_GCph,
    EuclidLikelihood_GGL,
    EuclidLikelihood_3x2pt,
    EuclidLikelihood_WL_BNT,
    EuclidLikelihood_GGL_BNT,
    EuclidLikelihood_3x2pt_BNT,
)

urls = {
    "nz_example.fits": "https://zenodo.org/records/15092862/files/nz_example.fits",
    "example-spectra.npy": "https://zenodo.org/records/17191365/files/example-spectra.fits",
    "example-mixmats.fits": "https://zenodo.org/records/17191365/files/example-mixmats.fits",
    "cov_3x2pt.npz": "https://zenodo.org/records/17191365/files/cov_3x2pt.npz",
    "cov_WL.npz": "https://zenodo.org/records/17191365/files/cov_WL.npz",
    "cov_GC.npz": "https://zenodo.org/records/17191365/files/cov_GC.npz",
    "cov_GGL.npz": "https://zenodo.org/records/17191365/files/cov_GGL.npz",
}


@pytest.fixture
def fiducial_params():
    """Minimal mock parameters for testing likelihoods."""
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
        "AIA": 0,
        "CIA": 0,
        "EtaIA": 0,
        "log10TAGN": 7.75,
        "N_mnu": 1,
        **{f"b1_photo_poly{i}": 0.0 for i in range(4)},
        **{f"magnification_bias_{i}": 0.0 for i in range(1, 7)},
        **{f"dz_pos_{i}": 0.0 for i in range(1, 7)},
        **{f"dz_shear_{i}": 0.0 for i in range(1, 7)},
        **{f"width_pos_{i}": 1.0 for i in range(1, 7)},
        **{f"width_shear_{i}": 1.0 for i in range(1, 7)},
        **{f"multiplicative_bias_{i}": 0.0 for i in range(1, 7)},
    }


@pytest.fixture(scope="module")
def data_setup(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("data")

    for filename, url in urls.items():
        r = requests.get(url)
        r.raise_for_status()
        with open(tmpdir / filename, "wb") as f:
            f.write(r.content)

    z_nz, nz_heracles = el.phz.redshift_distributions(tmpdir / "nz_example.fits")
    myz = np.linspace(1e-4, 3.0, 100)

    def normalize_and_resample(nz_dict, z_grid, z_target):
        nz_array = np.vstack([nz / np.trapezoid(nz, z_grid) for nz in nz_dict.values()])
        return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])

    my_dndz_pos_norm = normalize_and_resample(nz_heracles, z_nz, myz)
    my_dndz_she_norm = normalize_and_resample(nz_heracles, z_nz, myz)
    cells_binned = el.le3.pk_wl.angular_power_spectra(tmpdir / "example-spectra.npy")
    mixmat_binned = el.le3.pk_wl.mixing_matrices(tmpdir / "example-mixmats.fits")
    covmat_wl = np.load(tmpdir / "cov_WL.npz")["arr_0"]
    covmat_gc = np.load(tmpdir / "cov_GC.npz")["arr_0"]
    covmat_ggl = np.load(tmpdir / "cov_GGL.npz")["arr_0"]
    covmat_3x2pt = np.load(tmpdir / "cov_3x2pt.npz")["arr_0"]

    return {
        "myz": myz,
        "my_dndz_pos_norm": my_dndz_pos_norm,
        "my_dndz_she_norm": my_dndz_she_norm,
        "cells_binned": cells_binned,
        "mixmat_binned": mixmat_binned,
        "covmat_wl": covmat_wl,
        "covmat_gc": covmat_gc,
        "covmat_ggl": covmat_ggl,
        "covmat_3x2pt": covmat_3x2pt,
    }


def build_data(ds, mapa_key, cov, nz_indices, include_pos=False, include_she=False):
    """Use data_setup output `ds` to build likelihood data."""
    data = {
        "cells": ds["cells_binned"],
        "ells": ds["cells_binned"][mapa_key].ell,
        "z_arr": ds["myz"],
        "cov": cov,
        "mixmat": ds["mixmat_binned"],
    }
    if include_pos:
        data["dndz_pos"] = ds[nz_indices[0]]
    if include_she:
        data["dndz_she"] = ds[nz_indices[1] if len(nz_indices) > 1 else nz_indices[0]]
    return data


def build_settings_DR1(ds):
    cls = ds["cells_binned"]
    scale_cuts = {key: [4, 3000] for key in cls}
    return {"n_ell_bins": 32, "scale_cuts": scale_cuts, "include_rsd": True}


def test_euclid_likelihood_wl(data_setup, fiducial_params):
    ds = data_setup
    data_ll_DR1 = build_data(
        ds,
        ("SHE", "SHE", 1, 1),
        ds["covmat_wl"],
        ["my_dndz_she_norm"],
        include_she=True,
    )
    settings_ll_DR1 = build_settings_DR1(ds)

    like = EuclidLikelihood_WL(
        data=data_ll_DR1,
        settings=settings_ll_DR1,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="coupled",
    )
    val = like.loglike(fiducial_params)
    assert isinstance(val, float)
    assert not np.isnan(val)
    assert not np.isinf(val)


def test_euclid_likelihood_gcph(data_setup, fiducial_params):
    ds = data_setup
    data_gc_DR1 = build_data(
        ds,
        ("POS", "POS", 1, 1),
        ds["covmat_gc"],
        ["my_dndz_pos_norm"],
        include_pos=True,
    )
    settings_gc_DR1 = build_settings_DR1(ds)

    like = EuclidLikelihood_GCph(
        data=data_gc_DR1,
        settings=settings_gc_DR1,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="coupled",
    )
    val = like.loglike(fiducial_params)
    assert isinstance(val, float)
    assert not np.isnan(val)
    assert not np.isinf(val)


def test_euclid_likelihood_ggl(data_setup, fiducial_params):
    ds = data_setup
    data_ggl_DR1 = build_data(
        ds,
        ("POS", "SHE", 1, 1),
        ds["covmat_ggl"],
        ["my_dndz_pos_norm", "my_dndz_she_norm"],
        include_pos=True,
        include_she=True,
    )
    settings_ggl_DR1 = build_settings_DR1(ds)

    like = EuclidLikelihood_GGL(
        data=data_ggl_DR1,
        settings=settings_ggl_DR1,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="coupled",
    )
    val = like.loglike(fiducial_params)
    assert isinstance(val, float)
    assert not np.isnan(val)
    assert not np.isinf(val)


def test_euclid_likelihood_3x2pt(data_setup, fiducial_params):
    ds = data_setup
    data_3x2pt_DR1 = build_data(
        ds,
        ("POS", "POS", 1, 1),
        ds["covmat_3x2pt"],
        ["my_dndz_pos_norm", "my_dndz_she_norm"],
        include_pos=True,
        include_she=True,
    )
    settings_3x2pt_DR1 = build_settings_DR1(ds)

    like = EuclidLikelihood_3x2pt(
        data=data_3x2pt_DR1,
        settings=settings_3x2pt_DR1,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="coupled",
    )
    val = like.loglike(fiducial_params)
    assert isinstance(val, float)
    assert not np.isnan(val)
    assert not np.isinf(val)


@pytest.fixture
def bnt_matrix(data_setup, fiducial_params):
    """Compute the BNT matrix once per module."""
    ds = data_setup

    dndz_she = ds["my_dndz_she_norm"]
    z_arr = ds["myz"]

    fid_bg = {
        k: fiducial_params[k]
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

    background = CAMBBackground(**fid_bg)

    bnt_calc = BNTMatrixCalculator(
        dndz_list=dndz_she,
        z=z_arr,
        background=background,
    )
    return bnt_calc.get_bnt_matrix()


def test_euclid_likelihood_wl_bnt(data_setup, fiducial_params, bnt_matrix):
    ds = data_setup

    data_wl = build_data(
        ds,
        ("SHE", "SHE", 1, 1),
        ds["covmat_wl"],
        ["my_dndz_she_norm"],
        include_she=True,
    )
    settings_wl = build_settings_DR1(ds)

    like = EuclidLikelihood_WL(
        data=data_wl,
        settings=settings_wl,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="coupled",
    )
    val_base = like.loglike(fiducial_params)

    data_wl_bnt = dict(data_wl)
    data_wl_bnt["BNT_matrix"] = bnt_matrix

    like_bnt = EuclidLikelihood_WL_BNT(
        data=data_wl_bnt,
        settings=settings_wl,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="coupled",
    )
    val_bnt = like_bnt.loglike(fiducial_params)

    assert isinstance(val_bnt, float)
    assert not np.isnan(val_bnt)
    assert not np.isinf(val_bnt)
    assert round(val_bnt, 3) == round(val_base, 3)


def test_euclid_likelihood_ggl_bnt(data_setup, fiducial_params, bnt_matrix):
    ds = data_setup

    # --- Base (non-BNT) GGL likelihood ---
    data_ggl = build_data(
        ds,
        ("POS", "SHE", 1, 1),
        ds["covmat_ggl"],
        ["my_dndz_pos_norm", "my_dndz_she_norm"],
        include_pos=True,
        include_she=True,
    )
    settings_ggl = build_settings_DR1(ds)

    like_ggl = EuclidLikelihood_GGL(
        data=data_ggl,
        settings=settings_ggl,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="coupled",
    )
    val_base = like_ggl.loglike(fiducial_params)

    # --- BNT GGL likelihood (same data + BNT matrix) ---
    data_ggl_bnt = dict(data_ggl)
    data_ggl_bnt["BNT_matrix"] = bnt_matrix

    like_ggl_bnt = EuclidLikelihood_GGL_BNT(
        data=data_ggl_bnt,
        settings=settings_ggl,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="coupled",
    )
    val_bnt = like_ggl_bnt.loglike(fiducial_params)

    assert isinstance(val_bnt, float)
    assert not np.isnan(val_bnt)
    assert not np.isinf(val_bnt)
    assert round(val_bnt, 3) == round(val_base, 3)


def test_euclid_likelihood_3x2pt_bnt(data_setup, fiducial_params, bnt_matrix):
    ds = data_setup

    data_3x2 = build_data(
        ds,
        ("POS", "POS", 1, 1),
        ds["covmat_3x2pt"],
        ["my_dndz_pos_norm", "my_dndz_she_norm"],
        include_pos=True,
        include_she=True,
    )
    settings_3x2 = build_settings_DR1(ds)

    like_3x2 = EuclidLikelihood_3x2pt(
        data=data_3x2,
        settings=settings_3x2,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="coupled",
    )
    val_base = like_3x2.loglike(fiducial_params)

    data_3x2_bnt = dict(data_3x2)
    data_3x2_bnt["BNT_matrix"] = bnt_matrix

    like_3x2_bnt = EuclidLikelihood_3x2pt_BNT(
        data=data_3x2_bnt,
        settings=settings_3x2,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="coupled",
    )
    val_bnt = like_3x2_bnt.loglike(fiducial_params)

    assert isinstance(val_bnt, float)
    assert not np.isnan(val_bnt)
    assert not np.isinf(val_bnt)
    assert round(val_bnt, 3) == round(val_base, 3)


def _wl_theory(ds, parameters, extra_settings, NonLinPerturbations=None):
    data = build_data(
        ds,
        ("SHE", "SHE", 1, 1),
        ds["covmat_wl"],
        ["my_dndz_she_norm"],
        include_she=True,
    )
    settings = {**build_settings_DR1(ds), **extra_settings}
    like = EuclidLikelihood_WL(
        data=data,
        settings=settings,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=NonLinPerturbations or HMemuNonLinearPerturbations,
        mode="coupled",
    )
    return like.get_theory_vector_full(parameters)


def _gc_theory(ds, parameters, extra_settings):
    data = build_data(
        ds,
        ("POS", "POS", 1, 1),
        ds["covmat_gc"],
        ["my_dndz_pos_norm"],
        include_pos=True,
    )
    # cloelib's generalized Cl engine (used by the nonlinear bias) has no RSD.
    settings = {**build_settings_DR1(ds), "include_rsd": False, **extra_settings}
    like = EuclidLikelihood_GCph(
        data=data,
        settings=settings,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
        mode="coupled",
    )
    return like.get_theory_vector_full(parameters)


def test_ia_model_selection(data_setup, fiducial_params):
    params = {**fiducial_params, "AIA": 1.0, "EtaIA": 1.0, "CIA": 0.0134}
    no_ia = _wl_theory(data_setup, params, {"ia_model": None})
    nla = _wl_theory(data_setup, params, {"ia_model": "NLA"})
    assert not np.allclose(nla, no_ia, rtol=1e-2, atol=0)

    # With A2 = b_TA = 0 and NLA's pivot z0 = 0, TATT reduces to NLA.
    tatt_params = {**params, "A2IA": 0.0, "bTA": 0.0, "z0IA": 0.0}
    tatt = _wl_theory(data_setup, tatt_params, {"ia_model": "TATT"})
    np.testing.assert_allclose(tatt, nla, rtol=2e-3)

    tatt_full = _wl_theory(
        data_setup, {**tatt_params, "A2IA": 1.0, "bTA": 1.0}, {"ia_model": "TATT"}
    )
    assert not np.allclose(tatt_full, tatt, rtol=1e-2, atol=0)


def test_nonlinear_galaxy_bias_selection(data_setup, fiducial_params):
    n_bins = data_setup["my_dndz_pos_norm"].shape[0]
    b1 = {i: 1.0 + 0.2 * i for i in range(n_bins)}
    linear = _gc_theory(
        data_setup,
        {**fiducial_params, **{f"b1_photo_bin{i}": b for i, b in b1.items()}},
        {"galaxy_bias_model": "per_bin"},
    )
    nl_params = {
        **fiducial_params,
        **{f"b1_photo_nl_bin{i}": b for i, b in b1.items()},
    }
    # Without higher-order terms the nonlinear bias reduces to linear bias.
    nonlinear = _gc_theory(data_setup, nl_params, {"galaxy_bias_model": "nonlinear"})
    np.testing.assert_allclose(nonlinear, linear, rtol=1e-5)

    with_b2 = _gc_theory(
        data_setup,
        {**nl_params, **{f"b2_photo_nl_bin{i}": 0.5 for i in range(n_bins)}},
        {"galaxy_bias_model": "nonlinear"},
    )
    assert not np.allclose(with_b2, nonlinear, rtol=1e-2, atol=0)


def test_baryon_boost_selection(data_setup, fiducial_params):
    builtin = _wl_theory(data_setup, fiducial_params, {})
    dmo = _wl_theory(data_setup, fiducial_params, {"nonlinear_param_keys": ()})
    assert not np.allclose(dmo, builtin, rtol=1e-2, atol=0)

    boosted = _wl_theory(
        data_setup,
        fiducial_params,
        {"nonlinear_param_keys": (), "baryon_param_keys": ("log10TAGN",)},
        NonLinPerturbations=with_baryon_boost(
            HMemuNonLinearPerturbations, HMcode2020BaryonBoostMixin
        ),
    )
    np.testing.assert_allclose(boosted, builtin, rtol=1e-10)


@pytest.mark.parametrize(
    "extra_settings",
    [
        {"ia_model": "TATT-M"},
        {"galaxy_bias_model": "cubic"},
        {"galaxy_bias_model": "nonlinear", "include_rsd": True},
    ],
)
def test_invalid_model_settings(data_setup, extra_settings):
    data = build_data(
        data_setup,
        ("SHE", "SHE", 1, 1),
        data_setup["covmat_wl"],
        ["my_dndz_she_norm"],
        include_she=True,
    )
    with pytest.raises(ValueError):
        EuclidLikelihood_WL(
            data=data,
            settings={**build_settings_DR1(data_setup), **extra_settings},
            Background=CAMBBackground,
            LinPerturbations=HMemuLinearPerturbations,
            NonLinPerturbations=HMemuNonLinearPerturbations,
            mode="coupled",
        )
