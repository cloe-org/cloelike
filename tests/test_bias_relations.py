import numpy as np
import pytest

from cloelike.bias_relations import apply_bias_relations, check_bias_relations

RSD_parameter_names = ["b1", "b2", "bG2", "bGam3", "c0", "c2", "c4"]


def test_check_bias_relations_order():
    """Testing that relations are returned in the correct order."""
    relations = check_bias_relations(
        ["bGam3_coevolution", "bG2_coevolution"], RSD_parameter_names, set(), None
    )
    assert relations == [("bG2_coevolution", "bG2"), ("bGam3_coevolution", "bGam3")]


@pytest.mark.parametrize(
    "bias_relations, AM_parameter_names, TCM_params",
    [
        (["bG2_wrong"], set(), None),
        (["bG2_coevolution", "bG2_excursion_set"], set(), None),
        (["bGam3_coevolution"], {"bGam3"}, None),
        (["bG2_coevolution"], set(), {"bG2": (0.5, 2, 1)}),
    ],
)
def test_check_bias_relations_raises(bias_relations, AM_parameter_names, TCM_params):
    """Test that ValueError is raised for invalid inputs."""
    with pytest.raises(ValueError):
        check_bias_relations(
            bias_relations, RSD_parameter_names, AM_parameter_names, TCM_params
        )


def test_apply_bias_relations():
    """Test that bias relations are applied correctly."""
    b1 = np.array([1.0, 1.5, 2.0])
    parameters = {"b1": b1, "bG2": np.zeros(3), "bGam3": np.zeros(3)}
    relations = check_bias_relations(
        ["bG2_coevolution", "bGam3_coevolution"], RSD_parameter_names, set(), None
    )
    result = apply_bias_relations(parameters, relations)

    bG2 = -2.0 / 7.0 * (b1 - 1.0)
    np.testing.assert_allclose(result["bG2"], bG2)
    np.testing.assert_allclose(result["bGam3"], -1.0 / 6.0 * (b1 - 1.0) - 2.5 * bG2)
