import numpy as np
import pytest
import requests
from astropy.io import fits

# --- cloe-org imports ---
from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.observables.CometEFT_spectro import CometEFT_SpectroPower
from cloelike.EuclidLikelihood_GCspectro_Pls import EuclidLikelihood_GCspectro_Pls

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
path = "https://zenodo.org/records/15543831/files/"
files = ["cov_Gauss_GCspectro_comet_EFT_z{}_2500deg2.fits", "mixmat_identity_z{}.fits"]


def get_index(multipole, scale, scale_dict):
    offset = sum(len(scale_dict[ell]) for ell in multipoles if ell < multipole)
    return offset + np.where(scale_dict[multipole] == scale)[0][0]


@pytest.fixture(scope="module")
def data_setup(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("data")

    def download_file(url, dest_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)

    data = {"GCspectro": {}}
    for ii, z in enumerate(labels):
        data["GCspectro"][z] = {}

        download_file(
            path + files[0].format(str(z).strip("0")),
            tmpdir / files[0].format(str(z).strip("0")),
        )

        with fits.open(tmpdir / files[0].format(str(z).strip("0"))) as hdul:
            hdr = hdul[1].header

            fiducial_cosmo = {
                key_cloe: hdr[key]
                for key_cloe, key in zip(
                    ["H0", "Omega_m0", "Omega_b0"], ["HUBBLE", "OMEGA_M", "OMEGA_B"]
                )
            }

            fiducial_cosmo["Omega_cdm0"] = (
                fiducial_cosmo.pop("Omega_m0") - fiducial_cosmo["Omega_b0"]
            )

            for key, val in zip(
                ["Omega_k0", "As", "ns", "mnu", "w0", "wa", "gamma_MG", "N_mnu"],
                [0.0, 2.1e-9, 0.96, 0.0, -1.0, 0.0, 0.545, 0],
            ):
                fiducial_cosmo.setdefault(key, val)

            fid_h = fiducial_cosmo["H0"] / 100.0
            k_fac = fid_h
            pk_fac = 1.0 / fid_h**3
            cov_fac = 1.0 / fid_h**6

            table = hdul["AVERAGE"].data
            data["GCspectro"][z]["k"] = table["SCALE_1DIM"] * k_fac
            for ell in multipoles:
                data["GCspectro"][z][f"pk{ell}"] = table[f"AVERAGE{ell}"] * pk_fac

            table = hdul["COVARIANCE"].data
            scale_i = table["SCALE_1DIM-I"]
            multipole_i = table["MULTIPOLE-I"]
            scale_j = table["SCALE_1DIM-J"]
            multipole_j = table["MULTIPOLE-J"]
            covariance = table["COVARIANCE"]

            scale_dict = {
                ell: np.unique(scale_i[multipole_i == ell]) for ell in multipoles
            }
            matrix_size = sum(len(scale_dict[ell]) for ell in multipoles)
            cov_matrix = np.zeros((matrix_size, matrix_size))
            for s_i, m_i, s_j, m_j, cov in zip(
                scale_i, multipole_i, scale_j, multipole_j, covariance
            ):
                i = get_index(m_i, s_i, scale_dict)
                j = get_index(m_j, s_j, scale_dict)
                cov_matrix[i, j] = cov
            data["GCspectro"][z]["cov"] = cov_matrix * cov_fac

        data["GCspectro"][z]["nbar"] = nbar[ii] * fid_h**3

        download_file(
            path + files[1].format(str(z).strip("0")),
            tmpdir / files[1].format(str(z).strip("0")),
        )

        with fits.open(tmpdir / files[1].format(str(z).strip("0"))) as hdul:
            kin0 = hdul["BINS_INPUT"].data["kp0"] * k_fac
            kin2 = hdul["BINS_INPUT"].data["kp2"] * k_fac
            kin4 = hdul["BINS_INPUT"].data["kp4"] * k_fac
            kout = hdul["BINS_OUTPUT"].data["k"] * k_fac
            mixing_matrix = hdul["MIXING_MATRIX"].data

            mixing_matrix_dict = {}

            mixing_matrix_dict["kout"] = kout
            mixing_matrix_dict["kin0"] = kin0
            mixing_matrix_dict["kin2"] = kin2
            mixing_matrix_dict["kin4"] = kin4
            for i in multipoles:
                for j in multipoles:
                    mm = f"W{i}{j}"
                    mixing_matrix_dict[mm] = mixing_matrix[mm]

            data["GCspectro"][z]["mixing_matrix"] = mixing_matrix_dict

    data["fiducial_cosmology"] = fiducial_cosmo

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

    expected_loglike = -8481.404
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

    expected_loglike = -113.8038
    assert computed_loglike == pytest.approx(expected_loglike, rel=1e-5), (
        f"Expected log-likelihood to be approximately {expected_loglike}, "
        f"but got {computed_loglike}"
    )
