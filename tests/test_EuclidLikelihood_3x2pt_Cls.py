import numpy as np
import pytest
from scipy import integrate

# --- External Libraries ---
import euclidlib as el
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import (
    HMemuLinearPerturbations,
    HMemuNonLinearPerturbations,
)
from cloelike.EuclidLikelihood_3x2pt_Cls import EuclidLikelihood_3x2pt_Cls

# --- Data Preparation ---

import os

# Load redshift distributions
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
print(os.path.join(DATA_DIR, 'nz_example.fits'))
z_nz, nz_heracles = el.photo.redshift_distributions(os.path.join(DATA_DIR, 'nz_example.fits'))

# Define a common redshift grid
myz = np.linspace(1e-4, 3.0, 100)

def normalize_and_resample(nz_dict, z_grid, z_target):
    """Normalize and resample n(z) distributions onto a target grid."""
    nz_array = np.vstack([
        nz / integrate.trapezoid(nz, z_grid) for nz in nz_dict.values()
    ])
    return np.array([
        np.interp(z_target, z_grid, nz) for nz in nz_array
    ])

my_dndz_pos_norm = normalize_and_resample(
    nz_heracles, z_nz, myz
)
my_dndz_she_norm = normalize_and_resample(
    nz_heracles, z_nz, myz
)

# Normalized and resampled n(z) for position and shear
# Load angular power spectra and mixing matrices
cells_data = el.photo.angular_power_spectra(os.path.join(DATA_DIR, 'synth_cells_5000_binned.fits'))
mixmat = el.photo.mixing_matrices(os.path.join(DATA_DIR, 'mixmat_identity_5000_binned.fits'))

# Load covariance matrix
full_cov = np.load(
    os.path.join(DATA_DIR, 'cov_Gauss_3x2pt_2D_probe_zpair_ell_2500deg2_ellmax5000_Bmode.npy')
)

def build_data(ell_key, cov, include_pos=False, include_she=False):
    """Build the data dictionary for the likelihood."""
    data = {
        'cells': cells_data,
        'ells': cells_data[ell_key].ell,
        'z_arr': myz,
        'cov': cov,
        'mixmat': mixmat,
    }
    if include_pos:
        data['dndz_pos'] = my_dndz_pos_norm
    if include_she:
        data['dndz_she'] = my_dndz_she_norm
    return data

def build_settings():
    """Build the settings dictionary for the likelihood."""
    scale_cuts = {key: [10, 1500] for key in cells_data}
    for key in cells_data:
        if key[:2] == ('SHE', 'SHE'):
            scale_cuts[key] = [scale_cuts[key], [0, 0]]
    return {
        'n_ell_bins': 32,
        'scale_cuts': scale_cuts,
    }

# --- Default Parameters ---
default_pars = {
    'H0': 67, 'Omega_cdm0': 0.27, 'Omega_b0': 0.049, 'ns': 0.96, 'As': 2.1e-9,
    'w0': -1, 'wa': 0, 'Omega_k0': 0, 'mnu': 0.0, 'gamma_MG': 0.545,
    'log10TAGN': 7.75,
    'AIA': 0.16, 'EtaIA': 1.66,
    'b1_photo_poly0': 1.33291, 'b1_photo_poly1': -0.72414,
    'b1_photo_poly2': 1.0183, 'b1_photo_poly3': -0.14913,
    'magnification_bias_1': 0.0, 'magnification_bias_2': 0.0,
    'magnification_bias_3': 0.0, 'magnification_bias_4': 0.0,
    'magnification_bias_5': 0.0, 'magnification_bias_6': 0.0,
    'dz_pos_1': 0.0, 'dz_pos_2': 0.0, 'dz_pos_3': 0.0,
    'dz_pos_4': 0.0, 'dz_pos_5': 0.0, 'dz_pos_6': 0.0,
    'multiplicative_bias_1': 0.0, 'multiplicative_bias_2': 0.0,
    'multiplicative_bias_3': 0.0, 'multiplicative_bias_4': 0.0,
    'multiplicative_bias_5': 0.0, 'multiplicative_bias_6': 0.0,
    'dz_shear_1': 0.0, 'dz_shear_2': 0.0, 'dz_shear_3': 0.0,
    'dz_shear_4': 0.0, 'dz_shear_5': 0.0, 'dz_shear_6': 0.0,
}

# --- Tests ---

def test_likelihood_negative_or_zero():
    """Test that the likelihood is finite and negative (or zero within tolerance) for default parameters."""
    data = build_data(('POS', 'POS', 1, 1), full_cov, include_pos=True, include_she=True)
    settings = build_settings()
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

def test_likelihood_changes_with_parameters():
    """Test that the likelihood changes when cosmological parameters are varied."""
    data = build_data(('POS', 'POS', 1, 1), full_cov, include_pos=True, include_she=True)
    settings = build_settings()
    like = EuclidLikelihood_3x2pt_Cls(
        data=data,
        settings=settings,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
    )
    logl_default = like.loglike(default_pars)
    test_pars = default_pars.copy()
    test_pars['H0'] += 5
    logl_changed = like.loglike(test_pars)
    assert logl_default != logl_changed, "Likelihood should change with parameters"

def test_likelihood_handles_bad_parameters():
    """Test that the likelihood raises an error for unphysical parameters."""
    data = build_data(('POS', 'POS', 1, 1), full_cov, include_pos=True, include_she=True)
    settings = build_settings()
    like = EuclidLikelihood_3x2pt_Cls(
        data=data,
        settings=settings,
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
    )
    bad_pars = default_pars.copy()
    bad_pars['H0'] = -100  # Unphysical value
    with pytest.raises(Exception):
        like.loglike(bad_pars)
