from __future__ import annotations

import re
from collections import UserDict, UserList
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import cached_property
from typing import Iterable, Optional

import numpy as np
import numpy.typing as npt
from sympy import Expr, Symbol


class ParamType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class Parameter:
    symbol: Expr
    guess: float | int | np.ndarray = field(default=1.0)
    lower_bound: float | int | np.ndarray = field(default=None)
    upper_bound: float | int | np.ndarray = field(default=None)
    # TODO partially fixing an array parameter is not supported
    # perhaps users should use Matrix instead if they want this type of functionality
    fixed: bool = field(default=False)

    def __post_init__(self):

        # If the `guess` has a shape, it must be the same as the symbol shape,
        # if it has any.
        guess_shape = getattr(self.guess, "shape", None)
        symbol_shape = getattr(self.symbol, "shape", guess_shape)
        if guess_shape != symbol_shape:
            raise ValueError(f"Guess shape for symbol {self.symbol} does not match symbol shape")

    @property
    def param_type(self) -> ParamType:
        if "boolean" in self.symbol.assumptions0:
            return ParamType.BOOLEAN
        elif "integer" in self.symbol.assumptions0:
            return ParamType.DISCRETE
        else:
            return ParamType.CONTINUOUS

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


# frozen?
# frozen might not be nessecary but fit should make a copy to prevent modification
class Parameters(UserList):
    """Parameter list object

    or maybe it should be a dict?
    Could potentially help the `Objective` to/from flat array of guesses for argument of scipy.minimize
    """

    @classmethod
    def from_symbols(
        cls,
        symbols: Iterable[Symbol],
        parameters: dict[str, npt.ArrayLike] | Iterable[str] | str = None,
    ) -> Parameters:

        symbol_dict = {symbol.name: symbol for symbol in sorted(symbols, key=str)}

        if isinstance(parameters, str):
            p_list = [Parameter(symbol_dict[k]) for k in re.split("; |, |\*|\s+", parameters)]
        elif isinstance(parameters, list):
            p_list = [Parameter(symbol_dict[k]) for k in parameters]
        elif isinstance(parameters, dict):
            p_list = [Parameter(symbol_dict[k], guess=v) for k, v in parameters.items()]
        elif parameters is None:
            p_list = [Parameter(symbol) for symbol in symbol_dict.values()]
        else:
            raise ValueError("Invalid values for 'parameters' or 'guess'")
        return cls(p_list)

    @property
    def _symbols(self) -> list[Symbol]:
        return [p.symbol for p in self]

    @property
    def _names(self) -> list[str]:
        return [p.name for p in self]

    def index(self, item: Parameter | Symbol | str, *args) -> int:
        # are you really sure it shoulnt be a dict?
        # perhaps not since they 'keys' could be str or symbols ?
        if isinstance(item, Parameter):
            return super().index(item, *args)
        elif isinstance(item, Symbol):
            return self._symbols.index(item, *args)
        else:
            return self._names.index(item, *args)

    def set(self, symbol_or_name: Symbol | str, **kwargs):
        idx = self.index(symbol_or_name)

        # todo sanitize kwargs
        self[idx] = Parameter(**(asdict(self[idx]) | kwargs))

    def update_guess(self, guess: dict[str | Symbol, np.ndarray | float]) -> Parameters:
        """returns a new parameters object where """

        p_out = Parameters(self)
        for identifier, value in guess.items():
            idx = p_out.index(identifier)
            p_out[idx] = Parameter(**(asdict(self[idx]) | dict(guess=value)))

        return p_out

    @property
    def guess(self) -> dict[str, np.ndarray]:
        return {p.name: np.asarray(p.guess) for p in self}

    @property
    def symbols(self) -> set[Symbol]:
        return set(p.symbol for p in self)
