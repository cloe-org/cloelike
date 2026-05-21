import numpy as np
import pytest
import requests
from dataclasses import replace

# --- cloe-org imports ---
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.CometEFT_spectro import CometEFT_SpectroPower
from cloelike.EuclidLikelihood_GCspectro_Pls import EuclidLikelihood_GCspectro_Pls

# --- euclidlib imports ---
from euclidlib.le3.pk_gc import (
    power_spectrum_multipoles,
    power_spectrum_multipole_covariance,
    power_spectrum_multipole_mixing_matrix,
)

# --- Default redshifts ---
redshifts = np.array([1.0, 1.2, 1.4, 1.65])
labels = [str(z).strip("0") for z in redshifts]

# --- Default Legendre multipoles ---
multipoles = np.array([0, 2, 4])

# --- Default Parameters ---
default_pars = {
    "Omega_cdm0": 0.27,
    "Omega_b0": 0.049,
    "mnu": 0.0,
    "ns": 0.96,
    "As": 2.1e-9,
    "H0": 67.0,
    "w0": -1.0,
    "wa": 0.0,
    "Omega_k0": 0.0,
    "gamma_MG": 0.545,
    "N_mnu": 0,
    "b1": np.array([1.412, 1.769, 2.039, 2.496]),
    "b2": np.array([0.695, 0.870, 1.162, 2.010]),
    "bG2": np.array([-0.156, -0.299, -0.400, -0.555]),
    "bGam3": np.array([0.323, 0.621, 0.827, 1.137]),
    "c0": np.array([30.948, 37.116, 36.738, 53.627]),
    "c2": np.array([46.233, 53.071, 48.626, 60.962]),
    "c4": np.array([10.057, 10.385, 8.643, 8.711]),
    "cnlo": np.array([0.0, 0.0, 0.0, 0.0]),
    "NP0": np.array([1.056, 1.152, 1.144, 1.309]),
    "NP20": np.array([0.0, 0.0, 0.0, 0.0]),
    "NP22": np.array([0.0, 0.0, 0.0, 0.0]),
    "fout": np.array([0.0, 0.0, 0.0, 0.0]),
    "sigmaz": np.array([0.0, 0.0, 0.0, 0.0]),
}

# --- Default number densities ---
nbar = np.array([2.042611e-03, 1.02876011e-03, 0.58531983e-03, 0.313402e-03])

# --- Zenodo path and filenames ---
path = "https://zenodo.org/records/18711304/files/"
files = [
    "mps_pk_GCspectro_comet_EFT_z{}.fits",
    "cov_pk_Gauss_GCspectro_comet_EFT_z{}_2500deg2.fits",
    "mixmat_pk_GCspectro_identity_z{}.fits",
]


@pytest.fixture(scope="module")
def data_setup(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("data")

    def download_file(url, dest_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)

    data = {"GCspectro": {}, "fiducial_cosmology": {}}

    for ii, z in enumerate(labels):
        data["GCspectro"][z] = {}

        for file in files:
            download_file(
                path + file.format(str(z).strip("0")),
                tmpdir / file.format(str(z).strip("0")),
            )

        datavec = power_spectrum_multipoles(
            tmpdir / files[0].format(str(z).strip("0"))
        )[("SPE", "SPE", 0, 0)]
        covariance = power_spectrum_multipole_covariance(
            tmpdir / files[1].format(str(z).strip("0"))
        )[("SPE", "SPE", 0, 0)]
        mixing = power_spectrum_multipole_mixing_matrix(
            tmpdir / files[2].format(str(z).strip("0"))
        )[("SPE", "SPE", 0, 0)]

        if ii == 0:
            data["fiducial_cosmology"]["H0"] = datavec.fiducial_cosmology["H0"]
            data["fiducial_cosmology"]["Omega_cdm0"] = (
                datavec.fiducial_cosmology["Omega_m0"]
                - datavec.fiducial_cosmology["Omega_b0"]
            )
            data["fiducial_cosmology"]["Omega_b0"] = datavec.fiducial_cosmology[
                "Omega_b0"
            ]
            data["fiducial_cosmology"]["Omega_k0"] = datavec.fiducial_cosmology[
                "Omega_k0"
            ]
            data["fiducial_cosmology"]["mnu"] = 0.0
            data["fiducial_cosmology"]["N_mnu"] = 0
            data["fiducial_cosmology"]["w0"] = datavec.fiducial_cosmology["w0"]
            data["fiducial_cosmology"]["wa"] = 0.0
            data["fiducial_cosmology"]["ns"] = datavec.fiducial_cosmology["ns"]
            data["fiducial_cosmology"]["As"] = 2.1e-9
            data["fiducial_cosmology"]["gamma_MG"] = 0.545

            fid_h = data["fiducial_cosmology"]["H0"] / 100.0
            k_fac = fid_h
            pk_fac = 1.0 / fid_h**3
            cov_fac = 1.0 / fid_h**6

        data["GCspectro"][z]["nbar"] = datavec.nbar * fid_h**3

        data["GCspectro"][z]["k"] = datavec.keff * k_fac
        data["GCspectro"][z]["pk0"] = datavec.multipoles[0] * pk_fac
        data["GCspectro"][z]["pk2"] = datavec.multipoles[2] * pk_fac
        data["GCspectro"][z]["pk4"] = datavec.multipoles[4] * pk_fac

        full_matrix = np.block(
            [
                [covariance.covariance[f"{i}-{j}"] for j in [0, 2, 4]]
                for i in [0, 2, 4]
            ]
        )
        data["GCspectro"][z]["cov"] = full_matrix * cov_fac

        resc_kout = mixing.kout * k_fac
        resc_kin = {ell: val * k_fac for ell, val in mixing.kin.items()}
        squeezed_mixing = {key: val.squeeze() for key, val in mixing.mixing.items()}

        data["GCspectro"][z]["mixing_matrix"] = replace(
            mixing, kout=resc_kout, kin=resc_kin, mixing=squeezed_mixing
        )

    return data


@pytest.fixture(scope="module")
def settings_setup(data_setup):
    fid_h = data_setup["fiducial_cosmology"]["H0"] / 100.0

    settings = {
        "scale_cuts": {
            "GCspectro": {
                "bin1": {
                    "ell0": [0.0, 0.20 * fid_h],
                    "ell2": [0.0, 0.15 * fid_h],
                    "ell4": [0.0, 0.15 * fid_h],
                },
                "bin2": {
                    "ell0": [0.0, 0.25 * fid_h],
                    "ell2": [0.0, 0.20 * fid_h],
                    "ell4": [0.0, 0.20 * fid_h],
                },
                "bin3": {
                    "ell0": [0.0, 0.25 * fid_h],
                    "ell2": [0.0, 0.20 * fid_h],
                    "ell4": [0.0, 0.20 * fid_h],
                },
                "bin4": {
                    "ell0": [0.0, 0.30 * fid_h],
                    "ell2": [0.0, 0.25 * fid_h],
                    "ell4": [0.0, 0.25 * fid_h],
                },
            }
        }
    }

    return settings


@pytest.fixture(scope="module")
def AM_priors_setup():
    AM_priors = {
        "1.": {"bGam3": [0.0, 5.0]},
        "1.2": {"c0": [0.0, 200.0]},
        "1.4": {"bGam3": [0.0, 5.0], "c0": [0.0, 200.0]},
        "1.65": {
            "bGam3": [0.0, 5.0],
            "c0": [0.0, 200.0],
            "c2": [0.0, 200.0],
            "c4": [0.0, 200.0],
            "cnlo": [0.0, 200.0],
        },
    }

    return AM_priors


def test_likelihood_negative_or_zero(data_setup, settings_setup):
    like = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
    )
    logl = like.loglike(default_pars)
    assert np.isfinite(logl), "Likelihood should be finite"
    assert logl <= 1e-8, "Likelihood should be negative or zero within small tolerance"


def test_likelihood_changes_with_parameters(data_setup, settings_setup):
    like = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
    )
    logl_default = like.loglike(default_pars)
    test_pars = default_pars.copy()
    test_pars["H0"] += 5
    logl_changed = like.loglike(test_pars)
    assert logl_default != logl_changed, "Likelihood should change with parameters"


def test_likelihood_value(data_setup, settings_setup):
    likelihood = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
    )

    parameters = default_pars.copy()
    parameters["H0"] += 3

    computed_loglike = likelihood.loglike(parameters)

    expected_loglike = -8463.924
    assert computed_loglike == pytest.approx(expected_loglike, rel=1e-5), (
        f"Expected log-likelihood to be approximately {expected_loglike}, "
        f"but got {computed_loglike}"
    )


def test_likelihood_handles_bad_parameters(data_setup, settings_setup):
    like = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
    )
    bad_pars = default_pars.copy()
    bad_pars["H0"] = -100
    with pytest.raises(Exception):
        like.loglike(bad_pars)


def test_likelihood_value_with_AM(data_setup, settings_setup, AM_priors_setup):
    likelihood = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
        AM_priors=AM_priors_setup,
    )

    parameters = default_pars.copy()
    parameters["H0"] += 3

    computed_loglike = likelihood.loglike_AM(parameters)

    expected_loglike = -113.5673
    assert computed_loglike == pytest.approx(expected_loglike, rel=1e-5), (
        f"Expected log-likelihood to be approximately {expected_loglike}, "
        f"but got {computed_loglike}"
    )
