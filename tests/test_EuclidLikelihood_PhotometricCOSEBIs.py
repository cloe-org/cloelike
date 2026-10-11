import numpy as np
import pytest
import euclidlib as el
from cosmolib.data import COSEBI

from cloelib.cosmology.camb_cosmology import CAMBBackground
from cloelib.cosmology.HMcode2020Emu_cosmology import (
    HMemuLinearPerturbations,
    HMemuNonLinearPerturbations,
)
from cloelike.EuclidLikelihood_photo_COSEBIs import EuclidLikelihood_WLCosebi

pytest.importorskip("pylevin")
pytest.importorskip("mpmath")
from cloelib.auxiliary.cosebi_helpers import get_W_ell  # noqa: E402

N_BINS = 2
N_MODES = 5
THMIN, THMAX = 1.0, 400.0  # arcmin, as in the Euclid LE3 product
WL_KEYS = [
    ("SHE", "SHE", i, j) for i in range(1, N_BINS + 1) for j in range(i, N_BINS + 1)
]
FIDUCIAL_PARAMS = {
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
    **{f"dz_shear_{i}": 0.0 for i in range(1, N_BINS + 1)},
    **{f"width_shear_{i}": 1.0 for i in range(1, N_BINS + 1)},
    **{f"multiplicative_bias_{i}": 0.0 for i in range(1, N_BINS + 1)},
}


@pytest.fixture
def fiducial_params():
    return dict(FIDUCIAL_PARAMS)


@pytest.fixture(scope="module")
def kernels():
    theta = np.radians(np.geomspace(THMIN, THMAX, 500) / 60)
    ells = np.geomspace(1, 1e5, 1000)
    return get_W_ell(theta, N_MODES, ells, 4), ells


def make_cosebis(values=None):
    """COSEBI data in the euclidlib format, shape (2, 2, n_modes)."""
    mode = np.arange(1, N_MODES + 1)
    cb = {}
    for n, key in enumerate(WL_KEYS):
        arr = np.zeros((2, 2, N_MODES))
        if values is not None:
            arr[0, 0] = values[n * N_MODES : (n + 1) * N_MODES]
        cb[key] = COSEBI(
            array=arr, axis=(2,), mode=mode, nmodes=N_MODES, thmin=THMIN, thmax=THMAX
        )
    return cb


def make_data(cosebis):
    z = np.linspace(1e-4, 3.0, 100)
    dndz = np.array([np.exp(-0.5 * ((z - zc) / 0.2) ** 2) for zc in (0.6, 1.2)])
    dndz /= np.trapezoid(dndz, z, axis=1)[:, None]
    n = len(WL_KEYS) * N_MODES
    return {"cosebis": cosebis, "dndz_she": dndz, "z_arr": z, "cov": np.eye(n) * 1e-22}


def make_like(cosebis, kernels, **settings):
    w_ells, ells = kernels
    return EuclidLikelihood_WLCosebi(
        data=make_data(cosebis),
        settings={"w_ells": w_ells, "ells_integration_COSEBI": ells, **settings},
        Background=CAMBBackground,
        LinPerturbations=HMemuLinearPerturbations,
        NonLinPerturbations=HMemuNonLinearPerturbations,
    )


@pytest.fixture(scope="module")
def fiducial_theory(kernels):
    """Fiducial E-mode COSEBIs for all modes, used as mock data."""
    like = make_like(make_cosebis(), kernels, selected_modes=np.arange(1, N_MODES + 1))
    return like.get_theory_vector_full(FIDUCIAL_PARAMS)


def test_theory_vector_all_modes(fiducial_theory):
    assert fiducial_theory.shape == (len(WL_KEYS) * N_MODES,)
    assert np.all(np.isfinite(fiducial_theory))


@pytest.mark.parametrize("selected_modes", [[1, 2, 3], [2, 3, 4], [1, 3, 5]])
def test_loglike_zero_at_fiducial(
    kernels, fiducial_theory, fiducial_params, selected_modes
):
    # Theory must line up with the data by mode number for any selection
    like = make_like(
        make_cosebis(fiducial_theory), kernels, selected_modes=selected_modes
    )
    mask = like.get_masking_vector_cached()
    assert mask.sum() == len(WL_KEYS) * len(selected_modes)
    np.testing.assert_allclose(
        like.get_theory_vector_masked(fiducial_params),
        like.get_data_vector_masked(),
        rtol=1e-6,
    )
    assert like.loglike(fiducial_params) == pytest.approx(0.0, abs=1e-6)

    # Theory carries the same angular range (arcmin) as the euclidlib data
    for key, th in like.theory_prediction.items():
        c = like.data["cosebis"][key]
        assert (th.thmin, th.thmax) == pytest.approx((c.thmin, c.thmax), rel=1e-3)


def test_euclidlib_roundtrip(tmp_path, kernels, fiducial_theory, fiducial_params):
    path = tmp_path / "cosebis.fits"
    el.le3.twopcf_wl.cosebis.write(path, make_cosebis(fiducial_theory))
    # Requires euclidlib with the (2, 2, n_modes) COSEBI format
    cosebis = el.le3.twopcf_wl.cosebis(path)
    assert cosebis[WL_KEYS[0]].array.shape == (2, 2, N_MODES)
    like = make_like(cosebis, kernels, selected_modes=[2, 3, 4])
    assert like.loglike(fiducial_params) == pytest.approx(0.0, abs=1e-6)


def test_kernel_range_mismatch_raises(kernels):
    w_ells, ells = kernels
    wrong = {**w_ells, "metadata": {"THMIN": np.radians(0.5 / 60), "THMAX": 0.1}}
    with pytest.raises(ValueError, match="angular range"):
        make_like(make_cosebis(), (wrong, ells), selected_modes=[1, 2])


def test_selected_mode_not_in_data_raises(kernels):
    with pytest.raises(ValueError, match="not in the data"):
        make_like(make_cosebis(), kernels, selected_modes=[1, N_MODES + 1])


def test_scale_cuts_ignored_warns(kernels):
    with pytest.warns(UserWarning, match="scale_cuts"):
        like = make_like(
            make_cosebis(), kernels, selected_modes=[1, 2], scale_cuts={"a": [1, 2]}
        )
    assert like.scale_cuts is None


def test_ia_model_selection(kernels):
    params = {**FIDUCIAL_PARAMS, "AIA": 1.0, "EtaIA": 1.0, "CIA": 0.0134}
    modes = np.arange(1, N_MODES + 1)

    def theory(ia_model, **extra):
        like = make_like(
            make_cosebis(), kernels, selected_modes=modes, ia_model=ia_model
        )
        return like.get_theory_vector_full({**params, **extra})

    nla = theory("NLA")
    assert not np.allclose(nla, theory(None), rtol=1e-2, atol=0)
    # With A2 = b_TA = 0 and NLA's pivot z0 = 0, TATT reduces to NLA.
    tatt = theory("TATT", A2IA=0.0, bTA=0.0, z0IA=0.0)
    np.testing.assert_allclose(tatt, nla, rtol=2e-3)
