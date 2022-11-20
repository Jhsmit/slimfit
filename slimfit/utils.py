from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, OrderedDict, Any

import numpy as np

from slimfit import NumExprBase, Model
from slimfit.operations import Mul
from slimfit.symbols import Parameter, FitSymbol


def overlapping_model_parameters(
    model_callables: list[tuple[FitSymbol, NumExprBase]]
) -> list[Model]:

    seen_models = []
    seen_sets = []
    for lhs, mc in model_callables:
        items = set(mc.free_parameters.keys())

        found = False
        # look for sets of parameters we've seen so far, if found, append to the list of sets
        for i, s in enumerate(seen_sets):
            if items & s:
                s |= items  # add in additional items
                seen_models[i].append((lhs, mc))
                found = True
        if not found:
            seen_sets.append(items)
            seen_models.append([(lhs, mc)])

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
