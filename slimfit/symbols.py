from __future__ import annotations

from numbers import Number
from typing import Union, Optional, Any, Type

import numpy as np
import numpy.typing as npt
from sympy import Matrix, Symbol, zeros
from sympy.core.cache import clear_cache


def get_symbols(symbol_type: Optional[Type[FitSymbol]] = None) -> dict[str, FitSymbol]:
    """Returns a dictionary of all symbols created in the session"""
    if symbol_type is None:
        return FitSymbol._instances
    else:
        return {k: v for k, v in FitSymbol._instances.items() if isinstance(v, symbol_type)}


def clear_symbols():
    clear_cache()
    FitSymbol._instances = {}


def set_parameter_values(values: dict[str, float]):
    """Batch set parameter values through the magic of singleton Parameters"""

    for name, value in values.items():
        Parameter(name, value=value)


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


class Parameter(FitSymbol):
    def __new__(
        cls,
        name: str,
        value: Optional[float] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        fixed: Optional[bool] = None,
    ) -> Parameter:
        obj = super().__new__(cls, name)
        if not isinstance(obj, Parameter):
            raise TypeError(f"Variable name {name!r} already exists as '{obj.__class__.__name__}'")

        obj.set_attr("value", value, 1.0)
        obj.set_attr("vmin", vmin, None)
        obj.set_attr("vmax", vmax, None)
        obj.set_attr("fixed", fixed, False)

        if obj.vmin is not None and obj.vmin > obj.value:
            raise ValueError("Lower bound must be smaller than or equal to parameter value")
        if obj.vmax is not None and obj.vmax < obj.value:
            raise ValueError("Upper bound must be larger than or equal to parameter value")

        return obj

    def set_attr(self, attr: str, value: Any, default: Any) -> None:
        """
        Set attribute 'attr'; when it does not exist sets to either default or value,
        otherwise updates to 'value' if not None.

        Args:
            attr: Name of attribute to set.
            value: Value for the attribute.
            default: Value to use when 'value' is None.

        """
        if not hasattr(self, attr) or value is not None:
            setattr(self, attr, value if value is not None else default)

    def __repr__(self) -> str:
        attrs = ["name", "value", "vmin", "vmax", "fixed"]
        elements = [f"{attr}={repr(getattr(self, attr))}" for attr in attrs]
        return f"Parameter({', '.join(elements)})"


class Variable(FitSymbol):
    def __new__(cls, name):
        obj = super().__new__(cls, name)
        if isinstance(obj, Variable):
            return obj
        else:
            raise TypeError(f"Variable name {name!r} already exists as '{obj.__class__.__name__}'")


class Probability(FitSymbol):
    def __new__(cls, name):
        obj = super().__new__(cls, name)
        if isinstance(obj, Probability):
            return obj
        else:
            raise TypeError(
                f"Probability name {name!r} already exists as '{obj.__class__.__name__}'"
            )


def parameter_matrix(
    name: Optional[str] = None,
    values: Union[list, np.ndarray, None] = None,
    shape: Optional[tuple[int, ...]] = None,
    rand_init: Optional[bool] = False,
    norm: Optional[bool] = False,
    as_parameter: Union[npt.ArrayLike, bool] = True,
    vmin: Union[npt.ArrayLike, Number, None] = None,
    vmax: Union[npt.ArrayLike, Number, None] = None,
    fixed: Union[npt.ArrayLike, bool, None] = None,
    names: Optional[npt.ArrayLike] = None,
    suffix: Optional[npt.ArrayLike] = None,
):

    if values is None and shape is None:
        raise ValueError("Must specify either 'value' or 'shape'")
    elif values is not None and shape is None:
        value = np.array(values, dtype=object)
        shape = value.shape
    elif values is None and shape is not None:
        if rand_init:
            values = np.random.rand(*shape)
        elif norm:
            values = np.ones(shape)
        else:
            values = np.full(shape, fill_value=None)
    else:  # both value and shape are given
        values = np.array(values, dtype=object).reshape(shape)

    if norm:
        values = values / values.sum()

    if isinstance(fixed, bool) and fixed:
        fixed = np.ones(shape, dtype=bool)
    elif isinstance(fixed, bool) and not fixed:
        fixed = np.zeros(shape, dtype=bool)
    elif fixed is None:
        fixed = np.full(shape, fill_value=None)
    else:
        fixed = np.array(fixed).reshape(shape)

    if isinstance(as_parameter, bool) and as_parameter:
        as_parameter = np.ones(shape, dtype=bool)
    elif isinstance(as_parameter, bool) and not as_parameter:
        as_parameter = np.zeros(shape, dtype=bool)
    else:
        as_parameter = np.array(as_parameter).reshape(shape)

    if isinstance(vmin, (int, float)):
        vmin = np.ones(shape, dtype=float) * vmin
    elif isinstance(vmin, (np.ndarray, list)):
        vmin = np.array(vmin).reshape(shape)
        # if vmin.shape != shape:
        #     raise ValueError("Invalid shape for 'vmin'")
    else:
        vmin = np.full(shape, fill_value=None)

    if isinstance(vmax, (int, float)):
        vmax = np.ones(shape, dtype=float) * vmax
    elif isinstance(vmax, (np.ndarray, list)):
        vmax = np.array(vmax).reshape(shape)
        # if vmax.shape != shape:
        #     raise ValueError("Invalid shape for 'vmax'")
    else:
        vmax = np.full(shape, fill_value=None)

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
        if as_parameter[i, j]:
            matrix[i, j] = Parameter(
                name=names[i, j],
                value=values[i, j],
                fixed=fixed[i, j],
                vmin=vmin[i, j],
                vmax=vmax[i, j],
            )
        else:
            matrix[i, j] = values[i, j]

    return matrix


SORT_PRIORITY = [Variable, Probability, Parameter]
SORT_KEY = lambda x: (SORT_PRIORITY.index(type(x)), x.name)
