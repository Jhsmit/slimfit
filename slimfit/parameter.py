from __future__ import annotations

import re
from collections import UserList, UserDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

import numpy as np
import numpy.typing as npt
from sympy import Expr, Symbol


class ParamType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BOOLEAN = "boolean"


@dataclass
class Parameter:
    symbol: Expr
    guess: float | int | np.ndarray = field(default=1.0)
    lower_bound: float | int | np.ndarray = field(default=None)
    upper_bound: float | int | np.ndarray = field(default=None)
    # TODO partially fixing an array parameter is not supported
    # perhaps users should use Matrix instead if they want this type of functionality
    fixed: bool = field(default=False)
    param_type: ParamType = field(init=False)

    def __post_init__(self):
        if "boolean" in self.symbol.assumptions0:
            self.param_type = ParamType.BOOLEAN
        elif "integer" in self.symbol.assumptions0:
            self.param_type = ParamType.DISCRETE
        else:
            self.param_type = ParamType.CONTINUOUS

        # If the `guess` has a shape, it must be the same as the symbol shape,
        # if it has any.
        guess_shape = getattr(self.guess, "shape", None)
        symbol_shape = getattr(self.symbol, "shape", guess_shape)
        if guess_shape != symbol_shape:
            raise ValueError(f"Guess shape for symbol {self.symbol} does not match symbol shape")

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the Parameter. First tries to infer the shape from `Parameter.symbol`, otherwise
        from `Parameter.guess`, and returns an empty tuple if neither is found.

        """
        if shape := getattr(self.symbol, "shape", None):
            return shape
        elif shape := getattr(self.guess, "shape", None):
            return shape

        # when the parameter is a scalar, return an empty tuple, which is the same shape as
        # returned by np.asarray(3.).shape
        return tuple()

    @property
    def name(self) -> str:
        # Do symbols always have names?
        return self.symbol.name


class Parameters(UserDict):
    """Parameter dict object

    For now convenience
    Could potentially help the `Objective` to/from flat array of guesses for argument of scipy.minimize
    """

    @property
    def guess(self) -> dict[str, np.ndarray]:
        return {p.name: np.asarray(p.guess) for p in self.values()}

    def get_bounds(self) -> list[tuple[float | None, float | None]] | None:
        bounds = [(p.lower_bound, p.upper_bound) for p in self.values()]

        if all((None, None) == b for b in bounds):
            return None
        else:
            return bounds

    @classmethod
    def from_symbols(
        cls,
        symbols: dict[str, Symbol],
        parameters: dict[str, npt.ArrayLike] | Iterable[str] | str = None,
    ) -> Parameters:
        if isinstance(parameters, str):
            p_dict = {k: Parameter(symbols[k]) for k in re.split("; |, |\*|\s+", parameters)}
        elif isinstance(parameters, list):
            p_dict = {k: Parameter(symbols[k]) for k in parameters}
        elif isinstance(parameters, dict):
            p_dict = {k: Parameter(symbols[k], guess=v) for k, v in parameters.items()}
        elif parameters is None:
            p_dict = {symbol.name: Parameter(symbol) for symbol in symbols.values()}
        else:
            raise ValueError("Invalid values for 'parameters' or 'guess'")
        return cls(p_dict)

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Unpack a ndim 1 array of concatenated parameter values into a dictionary of
            parameter name: parameter_value where parameter values are cast back to their
            specified shapes.
        """
        sizes = [int(np.product(p.shape)) for p in self.values()]

        x_split = np.split(x, np.cumsum(sizes))
        p_values = {p.name: arr.reshape(p.shape) for arr, p in zip(x_split, self.values())}

        return p_values

    def pack(self, guess: Optional[dict] = None) -> np.ndarray:
        """Pack initial guesses together as array"""
        guess = guess or self.guess

        return np.concatenate(list(v.ravel() for v in guess.values()))
