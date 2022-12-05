from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, OrderedDict, Any

import numpy as np
from sympy import Symbol

from slimfit import NumExprBase
from slimfit.models import Model
from slimfit.operations import Mul

# from slimfit.operations import Mul
from slimfit.parameter import Parameter


def overlapping_model_parameters(
    model_callables: list[tuple[Symbol, NumExprBase]], symbols: set[Symbol],
) -> list[Model]:

    seen_models = []
    seen_sets = []
    for lhs, num_expr in model_callables:
        param_set = num_expr.symbols & symbols
        # param_set = set(num_expr.free_parameters.keys())

        found = False
        # look for sets of parameters we've seen so far, if found, append to the list of sets
        for i, test_set in enumerate(seen_sets):
            if param_set & test_set:
                # add additional items to this set of parameters
                test_set |= param_set
                seen_models[i].append((lhs, num_expr))
                found = True
        if not found:
            seen_sets.append(param_set)
            seen_models.append([(lhs, num_expr)])

    # Next, piece together the dependent model parts as Model objects, restoring original multiplications
    sub_models = []
    for components in seen_models:
        model_dict = defaultdict(list)
        for lhs, rhs in components:
            model_dict[lhs].append(rhs)

        model_dict = {
            lhs: rhs[0] if len(rhs) == 1 else Mul(*rhs) for lhs, rhs in model_dict.items()
        }
        sub_models.append(Model(model_dict))

    return sub_models


def get_bounds(
    parameters: Iterable[Parameter],
) -> Optional[list[tuple[Optional[float], Optional[float]]]]:
    """
    Get bounds for minimization.
    Args:
        parameters: Iterable of Parameter objects.

    Returns:
        Either a list of tuples to pass to `scipy.minimize` or None, if there are no bounds.
    """
    bounds = [(p.vmin, p.vmax) for p in parameters]

    if all([(None, None) == b for b in bounds]):
        return None
    else:
        return bounds


def clean_types(d: Any) -> Any:
    """cleans up nested dict/list/tuple/other `d` for exporting as yaml

    Converts library specific types to python native types, including numpy dtypes,
    OrderedDict, numpy arrays

    # https://stackoverflow.com/questions/59605943/python-convert-types-in-deeply-nested-dictionary-or-array

    """
    if isinstance(d, np.floating):
        return float(d)

    if isinstance(d, np.integer):
        return int(d)

    if isinstance(d, np.ndarray):
        return d.tolist()

    if isinstance(d, list):
        return [clean_types(item) for item in d]

    if isinstance(d, tuple):
        return tuple(clean_types(item) for item in d)

    if isinstance(d, OrderedDict):
        return clean_types(dict(d))

    if isinstance(d, dict):
        return {k: clean_types(v) for k, v in d.items()}

    else:
        return d
