from __future__ import annotations

from collections import UserList
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional
import re

import numpy as np
from sympy import Expr

from slimfit import Model


class ParamType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BOOLEAN = "boolean"


@dataclass
class Parameter:
    symbol: Expr
    guess: float | int | np.ndarray = field(default=1.)
    lower_bound: float | int | np.ndarray = field(default=None)
    upper_bound: float | int | np.ndarray = field(default=None)
    fixed: bool | np.ndarray = field(default=False)
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
        guess_shape = getattr(self.guess, 'shape', None)
        symbol_shape = getattr(self.symbol, 'shape', guess_shape)
        if guess_shape != symbol_shape:
            raise ValueError(f"Guess shape for symbol {self.symbol} does not match symbol shape")


    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the Parameter. First tries to infer the shape from `Parameter.symbol`, otherwise
        from `Parameter.guess`, and returns an empty tuple if neither is found.

        """
        if shape := getattr(self.symbol, 'shape', None):
            return shape
        elif shape := getattr(self.guess, 'shape', None):
            return shape
        else:
            # when the parameter is a scalar, return an empty tuple, which is the same shape as
            # returned by np.asarray(3.).shape
            return tuple()

    @property
    def name(self) -> str:
        # Do symbols always have names?
        return self.symbol.name


class Parameters(UserList):
    """Parameter list object

    For now convenience
    Could potentially help the `Objective` to/from flat array of guesses for argument of scipy.minimize
    """

    @property
    def guess(self) -> dict[str, np.ndarray]:
        return {p.name: np.asarray(p.guess) for p in self}

    def get_bounds(self) -> list[tuple[float | None, float | None]] | None:
        bounds = [(p.lower_bound, p.upper_bound) for p in self]

        if all((None, None) == b for b in bounds):
            return None
        else:
            return bounds

    @classmethod
    def from_model(cls,
           model: Model,
           parameters: Iterable[str] | str = None,
           guess: dict[str, np.ndarray] = None) -> Parameters:
        if isinstance(parameters, str):
            param_list = [Parameter(model.symbols[k]) for k in re.split('; |, |\*|\s+', parameters)]
        elif isinstance(parameters, list):
            param_list = [Parameter(model.symbols[k]) for k in parameters]
        elif isinstance(guess, dict):
            param_list = [Parameter(model.symbols[k], guess=v) for k, v in guess.items()]
        elif parameters is None and guess is None:
            param_list = [Parameter(symbol) for symbol in model.symbols.values()]
        else:
            raise ValueError("Invalid values for 'parameters' or 'guess'")
        return cls(param_list)

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Unpack a ndim 1 array of concatenated parameter values into a dictionary of
            parameter name: parameter_value where parameter values are cast back to their
            specified shapes.
        """
        sizes = [int(np.product(p.shape)) for p in self]

        x_split = np.split(x, np.cumsum(sizes))
        p_values = {p.name: arr.reshape(p.shape) for arr, p in zip(x_split, self)}

        return p_values

    def pack(self, guess: Optional[dict] = None) -> np.ndarray:
        """Pack initial guesses together as array"""
        guess = guess or self.guess

        return np.concatenate(list(v.ravel() for v in guess.values()))