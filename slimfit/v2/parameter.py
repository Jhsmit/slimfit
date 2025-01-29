from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import sympy as sp

from slimfit.v2.typing import Numerical


@dataclass
class Parameter:
    """A mutable parameter class that supports method chaining"""

    symbol: sp.Symbol
    guess: Numerical = 1.0
    bounds: tuple[Optional[Numerical], Optional[Numerical]] = (None, None)
    fixed: bool = False

    @property
    def name(self) -> str:
        return self.symbol.name

    @property
    def shape(self) -> tuple[int, ...]:
        shape = getattr(self.guess, "shape", tuple())
        return shape

    def fix(self) -> Parameter:
        """Fix the parameter at its current guess value"""
        self.fixed = True
        return self

    def unfix(self) -> Parameter:
        """Make the parameter free to vary"""
        self.fixed = False
        return self

    def set_bounds(
        self, lower: Optional[Numerical] = None, upper: Optional[Numerical] = None
    ) -> Parameter:
        """Set parameter bounds"""
        self.bounds = (lower, upper)
        return self

    def set_guess(self, value: Numerical) -> Parameter:
        """Set initial guess value"""
        self.guess = value
        return self


class Parameters:
    """Container for managing multiple parameters"""

    def __init__(self, parameters: Iterable[Parameter]):
        self._parameters = {p.name: p for p in parameters}

    def __getitem__(self, key: Union[str, Sequence[str]]) -> Union[Parameter, Parameters]:
        """Get a parameter or create a new ParameterSet with selected parameters"""
        if isinstance(key, str):
            return self._parameters[key]
        return Parameters([self._parameters[k] for k in key])

    def __iter__(self):
        return iter(self._parameters.values())

    def __len__(self):
        return len(self._parameters)

    def __add__(self, other: Parameters) -> Parameters:
        return Parameters(self.to_list() + other.to_list())

    @classmethod
    def from_symbols(
        cls,
        symbols: Iterable[sp.Symbol],
    ) -> Parameters:
        symbol_dict = {symbol.name: symbol for symbol in sorted(symbols, key=str)}

        p_list = [Parameter(symbol) for symbol in symbol_dict.values()]
        return cls(p_list)

    @property
    def guess(self) -> dict[str, Numerical]:  # other types?
        return {p.name: np.asarray(p.guess) for p in self}

    def fix(self, *names: str) -> Parameters:
        """Fix specified parameters"""
        for name in names:
            self._parameters[name].fix()
        return self

    def unfix(self, *names: str) -> Parameters:
        """Unfix specified parameters"""
        for name in names:
            self._parameters[name].unfix()
        return self

    def set_bounds(
        self, bounds_dict: dict[str, tuple[Optional[Numerical], Optional[Numerical]]]
    ) -> Parameters:
        """Set bounds for multiple parameters at once"""
        for name, (lower, upper) in bounds_dict.items():
            self._parameters[name].set_bounds(lower, upper)
        return self

    def set_guesses(self, guess_dict: dict[str, Numerical]) -> Parameters:
        """Set initial guesses for multiple parameters at once"""
        for name, guess in guess_dict.items():
            self._parameters[name].set_guess(guess)
        return self

    @property
    def fixed(self) -> Parameters:
        """Get list of fixed parameters"""
        return Parameters([p for p in self._parameters.values() if p.fixed])

    @property
    def free(self) -> Parameters:
        """Get list of free parameters"""
        return Parameters([p for p in self._parameters.values() if not p.fixed])

    def to_list(self) -> list[Parameter]:
        """Convert to parameter list"""
        return list(self._parameters.values())

    def __repr__(self) -> str:
        return f"Parameters({list(self._parameters.values())})"
