import numpy as np
import pytest
import requests
from astropy.io import fits

from cloelib.cosmology.camb_cosmology i
mport CAMBBackground
from cloelike.EuclidLikelihood_BAO import EuclidLikelihood_BAO

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
}

# Mock data from Zenodo
urls = {
    "bao_z1.00.fits": "https://zenodo.org/records/15698427/files/bao_z1.00.fits",
    "bao_z1.20.fits": "https://zenodo.org/records/15698427/files/bao_z1.20.fits",
    "bao_z1.40.fits": "https://zenodo.org/records/15698427/files/bao_z1.40.fits",
    "bao_z1.65.fits": "https://zenodo.org/records/15698427/files/bao_z1.65.fits",
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

    data = {"BAO": {}}

    for filename, url in urls.items():
        download_file(url, tmpdir / filename)

        with fits.open(tmpdir / filename) as hdul:
            hdr = hdul[0].header
            zmean = hdr["ZMEAN"]
            param_names = hdr["PARAMS"].split(",")

            fiducial_cosmo = {
                key_cloe: hdr[key]
                for key_cloe, key in zip(
                    ["H0", "Omega_m0", "Omega_b0"], ["H0", "OMEGA_M", "OMEGA_B"]
                )
            }

            fiducial_cosmo["Omega_cdm0"] = (
                fiducial_cosmo.pop("Omega_m0") - fiducial_cosmo["Omega_b0"]
            )

            for key, val in zip(
                ["Omega_k0", "As", "ns", "mnu", "w0", "wa", "gamma_MG"],
                [0.0, 2.1e-9, 0.96, 0.0, -1.0, 0.0, 0.545],
            ):
                fiducial_cosmo.setdefault(key, val)

            table = hdul["MEASUREMENTS"].data
            values = {row["PARAM"]: row["VALUE"] for row in table}

            cov = hdul["COVMAT"].data
            data["BAO"][zmean] = {
                "params": param_names,
                "fiducial_cosmology": fiducial_cosmo,
                "data": values,
                "covariance": cov,
            }

    return data

def test_likelihood_negative_or_zero(data_setup):
    like = EuclidLikelihood_BAO(
        data=data_setup,
        Background=CAMBBackground,
    )
    logl = like.loglike(default_pars)
    assert np.isfinite(logl), "Likelihood should be finite"
    assert logl <= 1e-8, "Likelihood should be negative or zero within small tolerance"

def test_likelihood_changes_with_parameters(data_setup):
    like = EuclidLikelihood_BAO(
        data=data_setup,
        Background=CAMBBackground,
    )
    logl_default = like.loglike(default_pars)
    test_pars = default_pars.copy()
    test_pars["H0"] += 5
    logl_changed = like.loglike(test_pars)
    assert logl_default != logl_changed, "Likelihood should change with parameters"

def test_likelihood_value(data_setup):
    like = EuclidLikelihood_BAO(
        data=data_setup,
        Background=CAMBBackground,
    )
    test_pars = default_pars.copy()
    test_pars["H0"] += 3
    logl = like.loglike(test_pars)
    assert logl == pytest.approx(
        -3.2305
    ), "Likelihood for test parameters should be -3.23"

def test_likelihood_handles_bad_parameters(data_setup):
    like = EuclidLikelihood_BAO(
        data=data_setup,
        Background=CAMBBackground,
    )
    bad_pars = default_pars.copy()
    bad_pars["H0"] = -100
    with pytest.raises(Exception):
        like.loglike(bad_pars)
