import numpy as np
import pytest
import requests
import euclidlib as el

from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import (
    HMemuLinearPerturbations,
    HMemuNonLinearPerturbations,
)
from cloelike.EuclidLikelihood_photo_Cls import (
    EuclidLikelihood_WL,
    EuclidLikelihood_GCph,
    EuclidLikelihood_GGL,
    EuclidLikelihood_3x2pt,
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

    z_nz, nz_heracles = el.photo.redshift_distributions(tmpdir / "nz_example.fits")
    myz = np.linspace(1e-4, 3.0, 100)

    def normalize_and_resample(nz_dict, z_grid, z_target):
        nz_array = np.vstack([nz / np.trapz(nz, z_grid) for nz in nz_dict.values()])
        return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])

    my_dndz_pos_norm = normalize_and_resample(nz_heracles, z_nz, myz)
    my_dndz_she_norm = normalize_and_resample(nz_heracles, z_nz, myz)
    cells_binned = el.photo.angular_power_spectra(tmpdir / "example-spectra.npy")
    mixmat_binned = el.photo.mixing_matrices(tmpdir / "example-mixmats.fits")
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
    return {"n_ell_bins": 32, "scale_cuts": scale_cuts}

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
