from __future__ import annotations

from numbers import Number
from typing import Union, Optional, Any, Type

import numpy as np
import numpy.typing as npt
from sympy import Symbol, zeros
from sympy.core.cache import clear_cache


def get_symbols(symbol_type: Optional[Type[FitSymbol]] = None) -> dict[str, FitSymbol]:
    """Returns a dictionary of all symbols created in the session"""
    if symbol_type is None:
        return FitSymbol._instances
    else:
        return {k: v for k, v in FitSymbol._instances.items() if isinstance(v, symbol_type)}


def symbol_dict(expr: Expr) -> dict[str, Symbol]:
    return {symbol.name: symbol for symbol in sorted(expr.free_symbols, key=str)}


def clear_symbols():
    clear_cache()
    FitSymbol._instances = {}


class FitSymbol(Symbol):
    _instances: dict[str, FitSymbol] = {}

    def __new__(cls, name: str):
        # Bypass the sympy cache
        if name.startswith("__"):
            raise ValueError("Double underscore leading names are limited to internal use.")
        if name in cls._instances:
            obj = cls._instances[name]
        else:
            obj = Symbol.__new__(cls, name)
            cls._instances[name] = obj
        return obj

    def _sympystr(self, printer, *args, **kwargs):
        return printer.doprint(self.name)

    _lambdacode = _sympystr
    _numpycode = _sympystr
    _pythoncode = _sympystr


def symbol_matrix(
    name: Optional[str] = None,
    shape: Optional[tuple[int, ...]] = None,
    names: Optional[npt.ArrayLike] = None,
    suffix: Optional[npt.ArrayLike] = None,
) -> Matrix:

    if shape is None:
        if names is not None:
            shape = (len(names), 1)
        elif suffix is not None:
            shape = (len(suffix), 1)
        else:
            raise ValueError("If 'shape' is not given, must specify 'names' or 'suffix'")

    # Generate names for parameters. Uses 'names' first, then <name>_<suffix> otherwise generates suffices
    # from indices
    if names is None and name is None:
        raise ValueError("Must specify either 'name' or 'names'")
    elif names is None:
        names = np.full(shape, fill_value="", dtype=object)
        if suffix is None:
            for i, j in np.ndindex(shape):
                names[i, j] = f"{name}_{i}_{j}"
        else:
            suffix = np.array(suffix).reshape(shape)
            for i, j in np.ndindex(shape):
                names[i, j] = f"{name}_{suffix[i, j]}"
    else:
        names = np.array(names)

    matrix = zeros(*shape)
    for i, j in np.ndindex(shape):
        matrix[i, j] = Symbol(
            name=names[i, j],
        )

    return matrix


# SORT_PRIORITY = [Variable, Probability, Parameter]
# SORT_KEY = lambda x: (SORT_PRIORITY.index(type(x)), x.name)
