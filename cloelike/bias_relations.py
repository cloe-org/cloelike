"""Relations fixing a bias parameter as a function of the others, applied by
the spectroscopic likelihoods to the standard values of the parameters."""

from typing import Optional

import numpy as np


def bG2_coevolution(parameters: dict) -> np.ndarray:
    r"""Coevolution bias relation for the tidal bias,
    :math:`b_{G_2} = -\frac{2}{7}(b_1 - 1)`.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary containing the key ``b1``.

    Returns
    -------
    bG2: np.ndarray
        Derived value of the tidal bias.
    """
    return -2.0 / 7.0 * (np.asarray(parameters["b1"]) - 1.0)


def bG2_excursion_set(parameters: dict) -> np.ndarray:
    r"""Excursion-set-derived relation for the tidal bias,
    :math:`b_{G_2} = 0.524 - 0.547\, b_1 + 0.046\, b_1^2`.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary containing the key ``b1``.

    Returns
    -------
    bG2: np.ndarray
        Derived value of the tidal bias.
    """
    b1 = np.asarray(parameters["b1"])
    return 0.524 - 0.547 * b1 + 0.046 * b1**2


def bGam3_coevolution(parameters: dict) -> np.ndarray:
    r"""Coevolution bias relation for the third-order tidal bias,
    :math:`b_{\Gamma_3} = -\frac{1}{6}(b_1 - 1) - \frac{5}{2} b_{G_2}`.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary containing the keys ``b1`` and ``bG2``.

    Returns
    -------
    bGam3: np.ndarray
        Derived value of the third-order tidal bias.
    """
    return -1.0 / 6.0 * (np.asarray(parameters["b1"]) - 1.0) - 5.0 / 2.0 * np.asarray(
        parameters["bG2"]
    )


# Order matters, since bGam3 relation depends on bG2
BIAS_RELATIONS = {
    "bG2_coevolution": (bG2_coevolution, ["bG2"]),
    "bG2_excursion_set": (bG2_excursion_set, ["bG2"]),
    "bGam3_coevolution": (bGam3_coevolution, ["bGam3", "bG3"]),
}


def check_bias_relations(
    bias_relations: Optional[list],
    RSD_parameter_names: list,
    AM_parameter_names: set,
    TCM_params: Optional[dict],
) -> list:
    r"""Checks the applicability of the specifiedbias relations given to a spectroscopic likelihood.

    Parameters
    ----------
    bias_relations: list
        Names of the relations to apply, keys of ``BIAS_RELATIONS``.
    RSD_parameter_names: list
        Parameters of the theory code.
    AM_parameter_names: set
        Analytically marginalised parameters.
    TCM_params: dict
        Parameters specified in the TCM reparametrisation.

    Returns
    -------
    bias_relations: list
        The relations, in the order they are applied, as pairs of their name
        and of the parameter they set.
    """
    bias_relations = list(bias_relations or [])
    unknown = [name for name in bias_relations if name not in BIAS_RELATIONS]
    if unknown:
        raise ValueError(
            f"Unknown bias relations {unknown}; available: {list(BIAS_RELATIONS)}"
        )
    set_by = {}
    for name in bias_relations:
        outputs = BIAS_RELATIONS[name][1]
        output = next((p for p in outputs if p in RSD_parameter_names), None)
        if output is None:
            raise ValueError(
                f"Bias relation {name} sets {outputs}, not a parameter of this "
                "theory code"
            )
        if output in set_by:
            raise ValueError(
                f"Bias relations {set_by[output]} and {name} both set '{output}'"
            )
        set_by[output] = name
        if output in AM_parameter_names:
            raise ValueError(
                f"Bias relation {name} sets '{output}', which cannot be "
                "analytically marginalised"
            )
        if output in (TCM_params or {}):
            raise ValueError(
                f"Bias relation {name} sets '{output}', which cannot be a TCM parameter"
            )
    output_of = {name: output for output, name in set_by.items()}
    return [(name, output_of[name]) for name in BIAS_RELATIONS if name in output_of]


def apply_bias_relations(parameters: dict, bias_relations: list) -> dict:
    r"""Applies the bias relations to a copy of the parameter dictionary.

    Parameters
    ----------
    parameters: dict
        Parameter dictionary.
    bias_relations: list
        Relations to apply, in order, as pairs of their name and of the
        parameter they set (as returned by ``check_bias_relations``).

    Returns
    -------
    parameters: dict
        Copy of the dictionary with the derived parameters set.
    """
    parameters = dict(parameters)
    for name, output in bias_relations:
        parameters[output] = BIAS_RELATIONS[name][0](parameters)
    return parameters
