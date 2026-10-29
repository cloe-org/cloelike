import numpy as np
import pytest
import requests

# --- cloe-org imports ---
from cloelib.cosmology.camb_cosmology import CAMBBackground, CAMBLinearPerturbations
from cloelib.observables.spectro.CometEFT_spectro import CometEFT_SpectroPower
from cloelib.summary_statistics.APDistortion import APDistortion
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
    "alpha_s": 0.0,
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

    data = {"GCspectro": {}}

    for z in labels:
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

        data["GCspectro"][z] = {
            "datavec": datavec,
            "covariance": covariance,
            "mixing": mixing,
        }

    return data


@pytest.fixture(scope="module")
def settings_setup(data_setup):
    fid_h = (
        data_setup["GCspectro"][labels[0]]["datavec"].fiducial_cosmology["H0"] / 100.0
    )

    settings = {
        "GCspectro": {
            "scale_cuts": {
                labels[0]: {
                    "ell0": [0.0, 0.20 * fid_h],
                    "ell2": [0.0, 0.15 * fid_h],
                    "ell4": [0.0, 0.15 * fid_h],
                },
                labels[1]: {
                    "ell0": [0.0, 0.25 * fid_h],
                    "ell2": [0.0, 0.20 * fid_h],
                    "ell4": [0.0, 0.20 * fid_h],
                },
                labels[2]: {
                    "ell0": [0.0, 0.25 * fid_h],
                    "ell2": [0.0, 0.20 * fid_h],
                    "ell4": [0.0, 0.20 * fid_h],
                },
                labels[3]: {
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


def test_hartlap_factor(data_setup, settings_setup):
    like_no_hartlap = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
    )
    num_data_points = like_no_hartlap.masked_covariance_matrix.shape[0]
    num_mocks = num_data_points + 200

    like_hartlap = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
        num_mocks=num_mocks,
    )

    expected_factor = (num_mocks - num_data_points - 2) / (num_mocks - 1)
    np.testing.assert_allclose(
        like_hartlap.inverse_masked_covariance_matrix,
        expected_factor * like_no_hartlap.inverse_masked_covariance_matrix,
    )


def test_hartlap_factor_with_few_realisations(data_setup, settings_setup):
    num_data_points = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
    ).masked_covariance_matrix.shape[0]

    with pytest.raises(ValueError):
        EuclidLikelihood_GCspectro_Pls(
            data=data_setup,
            settings=settings_setup,
            Background=CAMBBackground,
            SpectroPower=CometEFT_SpectroPower,
            num_mocks=num_data_points,
        )


def test_TCM_params_recovers_standard_value_loglike(data_setup, settings_setup):
    """Sampling b1 in TCM form (b1_TCM = b1 * Aap**n * sigma12**m)
    should give the exact same loglike as sampling the equivalent standard
    b1 directly, once the likelihood converts it back internally."""
    likelihood_standard = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
    )
    loglike_standard = likelihood_standard.loglike(default_pars)

    n, m = 1, 2
    background = CAMBBackground(
        H0=default_pars["H0"],
        Omega_cdm0=default_pars["Omega_cdm0"],
        Omega_b0=default_pars["Omega_b0"],
        Omega_k0=default_pars["Omega_k0"],
        w0=default_pars["w0"],
        wa=default_pars["wa"],
        ns=default_pars["ns"],
        As=default_pars["As"],
        gamma_MG=default_pars["gamma_MG"],
        mnu=default_pars["mnu"],
        N_mnu=default_pars["N_mnu"],
    )
    ap_distortion = APDistortion(background, likelihood_standard.background_fiducial)
    perturbations = CAMBLinearPerturbations(
        background, np.float64(likelihood_standard.redshifts)
    )
    sigma12_by_z = dict(zip(perturbations.z, perturbations.sigma12()))

    TCM_pars = dict(default_pars)
    b1_TCM = np.array(default_pars["b1"], dtype=float).copy()
    for i, z in enumerate(likelihood_standard.redshifts):
        Aap = 1.0 / (
            ap_distortion.q_AP_tr(float(z)) ** 2 * ap_distortion.q_AP_lo(float(z))
        )
        sigma12 = sigma12_by_z[float(z)]
        b1_TCM[i] = default_pars["b1"][i] * Aap**n * sigma12**m
    TCM_pars["b1"] = b1_TCM

    likelihood_TCM = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
        Perturbations=CAMBLinearPerturbations,
        TCM_params={"b1": (n, m)},
    )
    loglike_TCM = likelihood_TCM.loglike(TCM_pars)

    assert loglike_TCM == pytest.approx(loglike_standard, rel=1e-6)


def test_TCM_params_requires_perturbations_for_comet(data_setup, settings_setup):
    with pytest.raises(ValueError):
        likelihood = EuclidLikelihood_GCspectro_Pls(
            data=data_setup,
            settings=settings_setup,
            Background=CAMBBackground,
            SpectroPower=CometEFT_SpectroPower,
            TCM_params={"b1": (1, 2)},
        )
        likelihood.loglike(default_pars)


def test_TCM_params_recovers_standard_value_loglike_AM(data_setup, settings_setup):
    """Marginalising bGam3 (dist: AM) while it's also TCM-reparametrised
    should give the same loglike_AM as marginalising the equivalent standard
    bGam3 prior directly, up to a known additive offset -- exercises the
    extra coefficient rescaling in `_coeff_AM`/`AM_par_to_diag`, not just the
    plain parameter conversion already covered by the non-AM TCM test
    above.

    The offset: `loglike_AM`'s analytic marginalisation omits the
    `log det(Sigma_prior)` term (`sum(log(AM_sigmas**2))`) from its Gaussian
    integral normalisation -- harmless for ordinary use, where `AM_sigmas`
    is fixed for the whole run and the term is just a constant that cancels
    in any relative comparison. But `AM_sigmas` here is expressed in
    different units between the two likelihoods (TCM vs standard), so
    the omitted constant differs by `sum(log(alpha_i))`, where
    `alpha_i = sigma_TCM_i / sigma_std_i` per bin -- and that's the exact,
    analytically-known gap this test expects between the two loglikes.
    """
    AM_priors_standard = {
        "1.": {"bGam3": [0.3, 5.0]},
        "1.2": {"bGam3": [0.6, 5.0]},
        "1.4": {"bGam3": [0.8, 5.0]},
        "1.65": {"bGam3": [1.1, 5.0]},
    }
    likelihood_standard = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
        AM_priors=AM_priors_standard,
    )
    parameters = default_pars.copy()
    parameters["H0"] += 3
    loglike_standard = likelihood_standard.loglike_AM(parameters)

    n, m = 1, 2
    background = CAMBBackground(
        H0=parameters["H0"],
        Omega_cdm0=parameters["Omega_cdm0"],
        Omega_b0=parameters["Omega_b0"],
        Omega_k0=parameters["Omega_k0"],
        w0=parameters["w0"],
        wa=parameters["wa"],
        ns=parameters["ns"],
        As=parameters["As"],
        gamma_MG=parameters["gamma_MG"],
        mnu=parameters["mnu"],
        N_mnu=parameters["N_mnu"],
    )
    ap_distortion = APDistortion(background, likelihood_standard.background_fiducial)
    perturbations = CAMBLinearPerturbations(
        background, np.float64(likelihood_standard.redshifts)
    )
    sigma12_by_z = dict(zip(perturbations.z, perturbations.sigma12()))

    def to_TCM(loc, scale, z):
        Aap = 1.0 / (
            ap_distortion.q_AP_tr(float(z)) ** 2 * ap_distortion.q_AP_lo(float(z))
        )
        sigma12 = sigma12_by_z[float(z)]
        factor = Aap**n * sigma12**m
        return loc * factor, scale * factor

    AM_priors_TCM = {}
    alphas = []
    for z_label, entry in AM_priors_standard.items():
        loc_std, scale_std = entry["bGam3"]
        loc_TCM, scale_TCM = to_TCM(loc_std, scale_std, z_label)
        AM_priors_TCM[z_label] = {"bGam3": [loc_TCM, scale_TCM]}
        alphas.append(scale_TCM / scale_std)

    likelihood_TCM = EuclidLikelihood_GCspectro_Pls(
        data=data_setup,
        settings=settings_setup,
        Background=CAMBBackground,
        SpectroPower=CometEFT_SpectroPower,
        Perturbations=CAMBLinearPerturbations,
        AM_priors=AM_priors_TCM,
        TCM_params={"bGam3": (n, m)},
    )
    loglike_TCM = likelihood_TCM.loglike_AM(parameters)

    log_det_sigma_prior_offset = np.sum(np.log(alphas))
    assert loglike_TCM == pytest.approx(
        loglike_standard + log_det_sigma_prior_offset, rel=1e-6
    )
