import numpy as np
import pytest
import requests

from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelike.EuclidLikelihood_GCspectro_BAO import EuclidLikelihood_GCspectro_BAO

from euclidlib.le3.bao_gc import BAO_alphas, BAO_alphas_covariance

# Default redshifts
redshifts = np.array([1.00, 1.20, 1.40, 1.65])
labels = [f"{z:.2f}" for z in redshifts]

# Default Parameters
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
    "N_mnu": 0,
}

# Mock data from Zenodo
urls = {
    "bao_z1.00.fits": "https://zenodo.org/records/19729182/files/EUC_LE3_BAO_ALPHAS_Z1.0.fits",
    "bao_z1.20.fits": "https://zenodo.org/records/19729182/files/EUC_LE3_BAO_ALPHAS_Z1.2.fits",
    "bao_z1.40.fits": "https://zenodo.org/records/19729182/files/EUC_LE3_BAO_ALPHAS_Z1.4.fits",
    "bao_z1.65.fits": "https://zenodo.org/records/19729182/files/EUC_LE3_BAO_ALPHAS_Z1.65.fits",
    "cov_z1.00.fits": "https://zenodo.org/records/19729182/files/EUC_LE3_BAO_COV_Z1.0.fits",
    "cov_z1.20.fits": "https://zenodo.org/records/19729182/files/EUC_LE3_BAO_COV_Z1.2.fits",
    "cov_z1.40.fits": "https://zenodo.org/records/19729182/files/EUC_LE3_BAO_COV_Z1.4.fits",
    "cov_z1.65.fits": "https://zenodo.org/records/19729182/files/EUC_LE3_BAO_COV_Z1.65.fits",
}


@pytest.fixture(scope="module")
def data_setup(tmp_path_factory):
    # Downloads the mock data from Zenodo and constructs the `data` dictionary
    tmpdir = tmp_path_factory.mktemp("data")

    def download_file(url, dest_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)

    for filename, url in urls.items():
        download_file(url, tmpdir / filename)

    data = {"GCspectro": {}}

    # Download data
    filename = "bao_z{}.fits"
    datavec = BAO_alphas(tmpdir / filename, *labels)

    filename = "cov_z{}.fits"
    covariance = BAO_alphas_covariance(tmpdir / filename, *labels)

    # Store the raw euclidlib objects
    for i, z in enumerate(redshifts):
        data["GCspectro"][z] = {
            "datavec": datavec[("SPE", "SPE", i, i)],
            "covariance": covariance[("SPE", "SPE", i, i)],
        }

    return data


# Tests


def test_likelihood_negative_or_zero(data_setup):
    like = EuclidLikelihood_GCspectro_BAO(
        data=data_setup,
        Background=CAMBBackground,
        LinearPerturbations=CAMBLinearPerturbations,
    )
    logl = like.loglike(default_pars)
    assert np.isfinite(logl), "Likelihood should be finite"
    assert logl <= 1e-8, "Likelihood should be negative or zero within small tolerance"


def test_likelihood_changes_with_parameters(data_setup):
    like = EuclidLikelihood_GCspectro_BAO(
        data=data_setup,
        Background=CAMBBackground,
        LinearPerturbations=CAMBLinearPerturbations,
    )
    logl_default = like.loglike(default_pars)
    test_pars = default_pars.copy()
    test_pars["H0"] += 5
    logl_changed = like.loglike(test_pars)
    assert logl_default != logl_changed, "Likelihood should change with parameters"


def test_likelihood_value(data_setup):
    likelihood = EuclidLikelihood_GCspectro_BAO(
        data=data_setup,
        Background=CAMBBackground,
        LinearPerturbations=CAMBLinearPerturbations,
    )

    parameters = default_pars.copy()
    parameters["H0"] += 3

    computed_loglike = likelihood.loglike(parameters)

    expected_loglike = -1.93841
    assert computed_loglike == pytest.approx(expected_loglike, abs=5e-4), (
        f"Expected log-likelihood to be approximately {expected_loglike}, "
        f"but got {computed_loglike}"
    )


def test_likelihood_handles_bad_parameters(data_setup):
    like = EuclidLikelihood_GCspectro_BAO(
        data=data_setup,
        Background=CAMBBackground,
        LinearPerturbations=CAMBLinearPerturbations,
    )
    bad_pars = default_pars.copy()
    bad_pars["H0"] = -100
    with pytest.raises(Exception):
        like.loglike(bad_pars)
