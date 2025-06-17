import numpy as np
import pytest
from scipy import integrate
import requests

# --- External Libraries ---
import euclidlib as el
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import (
    HMemuLinearPerturbations,
    HMemuNonLinearPerturbations,
)
from cloelike.EuclidLikelihood_3x2pt_Cls import EuclidLikelihood_3x2pt_Cls

# --- Default Parameters ---
default_pars = {
    "H0": 67,
    "Omega_cdm0": 0.27,
    "Omega_b0": 0.049,
    "ns": 0.96,
    "As": 2.1e-9,
    "w0": -1,
    "wa": 0,
    "Omega_k0": 0,
    "mnu": 0.0,
    "gamma_MG": 0.545,
    "log10TAGN": 7.75,
    "AIA": 0.16,
    "EtaIA": 1.66,
    "b1_photo_poly0": 1.33291,
    "b1_photo_poly1": -0.72414,
    "b1_photo_poly2": 1.0183,
    "b1_photo_poly3": -0.14913,
    "magnification_bias_1": 0.0,
    "magnification_bias_2": 0.0,
    "magnification_bias_3": 0.0,
    "magnification_bias_4": 0.0,
    "magnification_bias_5": 0.0,
    "magnification_bias_6": 0.0,
    "dz_pos_1": 0.0,
    "dz_pos_2": 0.0,
    "dz_pos_3": 0.0,
    "dz_pos_4": 0.0,
    "dz_pos_5": 0.0,
    "dz_pos_6": 0.0,
    "multiplicative_bias_1": 0.0,
    "multiplicative_bias_2": 0.0,
    "multiplicative_bias_3": 0.0,
    "multiplicative_bias_4": 0.0,
    "multiplicative_bias_5": 0.0,
    "multiplicative_bias_6": 0.0,
    "dz_shear_1": 0.0,
    "dz_shear_2": 0.0,
    "dz_shear_3": 0.0,
    "dz_shear_4": 0.0,
    "dz_shear_5": 0.0,
    "dz_shear_6": 0.0,
}

urls = {
    "nz_example.fits": "https://zenodo.org/records/15092862/files/nz_example.fits",
    "cov_Gauss_3x2pt_2D_probe_zpair_ell_2500deg2_ellmax5000_Bmode_copy.npy": "https://zenodo.org/records/15496892/files/cov_Gauss_3x2pt_2D_probe_zpair_ell_2500deg2_ellmax5000_Bmode_copy.npy",
    "mixmat_identity_5000_binned.fits": "https://zenodo.org/records/15496892/files/mixmat_identity_5000_binned.fits",
    "synth_cells_5000_binned.fits": "https://zenodo.org/records/15496892/files/synth_cells_5000_binned.fits",
}


@pytest.fixture(scope="module")
def data_setup(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("data")

    def download_file(url, dest_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)

    for filename, url in urls.items():
        download_file(url, tmpdir / filename)

    z_nz, nz_heracles = el.photo.redshift_distributions(tmpdir / "nz_example.fits")
    myz = np.linspace(1e-4, 3.0, 100)

    def normalize_and_resample(nz_dict, z_grid, z_target):
        nz_array = np.vstack(
            [nz / integrate.trapezoid(nz, z_grid) for nz in nz_dict.values()]
        )
        return np.array([np.interp(z_target, z_grid, nz) for nz in nz_array])

    my_dndz_pos_norm = normalize_and_resample(nz_heracles, z_nz, myz)
    my_dndz_she_norm = normalize_and_resample(nz_heracles, z_nz, myz)
    cells_data = el.photo.angular_power_spectra(tmpdir / "synth_cells_5000_binned.fits")
    mixmat = el.photo.mixing_matrices(tmpdir / "mixmat_identity_5000_binned.fits")
    full_cov = np.load(
        tmpdir / "cov_Gauss_3x2pt_2D_probe_zpair_ell_2500deg2_ellmax5000_Bmode_copy.npy"
    )

    return {
        "myz": myz,
        "my_dndz_pos_norm": my_dndz_pos_norm,
        "my_dndz_she_norm": my_dndz_she_norm,
        "cells_data": cells_data,
        "mixmat": mixmat,
        "full_cov": full_cov,
    }


def build_data(ell_key, cov, dset, include_pos=False, include_she=False):
    data = {
        "cells": dset["cells_data"],
        "ells": dset["cells_data"][ell_key].ell,
        "z_arr": dset["myz"],
        "cov": cov,
        "mixmat": dset["mixmat"],
    }
    if include_pos:
        data["dndz_pos"] = dset["my_dndz_pos_norm"]
    if include_she:
        data["dndz_she"] = dset["my_dndz_she_norm"]
    return data


def build_settings(dset):
    scale_cuts = {key: [10, 1500] for key in dset["cells_data"]}
    for key in dset["cells_data"]:
        if key[:2] == ("SHE", "SHE"):
            scale_cuts[key] = [scale_cuts[key], [0, 0]]
    return {
        "n_ell_bins": 32,
        "scale_cuts": scale_cuts,
    }


# --- Tests ---


def test_likelihood_negative_or_zero(data_setup):
    data = build_data(
        ("POS", "POS", 1, 1),
        data_setup["full_cov"],
        data_setup,
        include_pos=True,
        include_she=True,
    )
    settings = build_settings(data_setup)
    like = EuclidLikelihood_3x2pt_Cls(
        data=data,
        settings=settings,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
    )
    logl = like.loglike(default_pars)
    assert np.isfinite(logl), "Likelihood should be finite"
    assert logl <= 1e-8, "Likelihood should be negative or zero within small tolerance"


def test_likelihood_changes_with_parameters(data_setup):
    data = build_data(
        ("POS", "POS", 1, 1),
        data_setup["full_cov"],
        data_setup,
        include_pos=True,
        include_she=True,
    )
    settings = build_settings(data_setup)
    like = EuclidLikelihood_3x2pt_Cls(
        data=data,
        settings=settings,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
    )
    logl_default = like.loglike(default_pars)
    test_pars = default_pars.copy()
    test_pars["H0"] += 5
    logl_changed = like.loglike(test_pars)
    assert logl_default != logl_changed, "Likelihood should change with parameters"


def test_likelihood_handles_bad_parameters(data_setup):
    data = build_data(
        ("POS", "POS", 1, 1),
        data_setup["full_cov"],
        data_setup,
        include_pos=True,
        include_she=True,
    )
    settings = build_settings(data_setup)
    like = EuclidLikelihood_3x2pt_Cls(
        data=data,
        settings=settings,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
    )
    bad_pars = default_pars.copy()
    bad_pars["H0"] = -100  # Unphysical value
    with pytest.raises(Exception):
        like.loglike(bad_pars)
