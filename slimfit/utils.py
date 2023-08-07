from __future__ import annotations

from collections import defaultdict
from functools import reduce
from typing import Iterable, Optional, OrderedDict, Any, Union

import numpy as np
from sympy import Symbol

from slimfit import NumExprBase
from slimfit.models import Model
from slimfit.operations import Mul

# from slimfit.operations import Mul
from slimfit.parameter import Parameter


def intersecting_component_symbols(
    model_components: list[tuple[Symbol, NumExprBase]],
    symbols: set[Symbol],
) -> list[Model]:
    """
    Finds and groups model components which have intersecting symbols.

    Args:
        model_components: Model components.
        symbols: Set of symbols to consider for intersections.

    Returns:
        Reconstructed models (assuming orignal model was a product of models).
    """

    seen_models = []
    seen_sets = []
    for lhs, num_expr in model_components:
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


def format_indexer(indexer: tuple[slice, int, None, Ellipsis]) -> str:
    """Format a tuple of slice objects into a string that can be used to index a numpy array.

    More or less the inverse of `numpy.index_exp`.


    Args:
        indexer: Tuple of indexing objects.

    """

    return f"[{', '.join(_format_indexer(sl) for sl in indexer)}]"


def _format_indexer(indexer: Union[slice, int, None, Ellipsis]) -> str:
    if isinstance(indexer, int):
        return str(indexer)
    elif isinstance(indexer, slice):
        # adapted from
        # https://stackoverflow.com/questions/24662999/how-do-i-convert-a-slice-object-to-a-string-that-can-go-in-brackets
        sl_start = "" if indexer.start is None else str(indexer.start)
        sl_stop = "" if indexer.stop is None else str(indexer.stop)
        if indexer.step is None:
            sl_str = "%s:%s" % (sl_start, sl_stop)
        else:
            sl_str = "%s:%s:%s" % (sl_start, sl_stop, indexer.step)
        return sl_str
    elif isinstance(indexer, type(None)):
        return "None"
    elif isinstance(indexer, type(Ellipsis)):
        return "..."
    else:
        raise TypeError(f"Unexpected type: {type(indexer)}")


# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
def rsetattr(obj: Any, attr: str, val: Any) -> Any:
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties/31174427#31174427
def rgetattr(obj: Any, attr: str, *default):
    try:
        return reduce(getattr, attr.split("."), obj)
    except AttributeError as e:
        if default:
            return default[0]
        else:
            raise e
