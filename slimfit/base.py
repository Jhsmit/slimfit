from __future__ import annotations

import abc
from typing import Optional

from slimfit.symbols import FitSymbol, Variable, Parameter


class SymbolicBase(object, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def symbols(self) -> dict[str, FitSymbol]:
        """Returns a dict with all FitSymbols in the Callable"""

    @property
    def variables(self) -> dict[str, Variable]:
        return {name: p for name, p in self.symbols.items() if isinstance(p, Variable)}

    @property
    def variable(self) -> Optional[Variable]:
        if len(self.variables) == 1:
            return next(iter(self.variables.values()))
        else:
            return None

    @property
    def parameters(self) -> dict[str, Parameter]:
        return {name: p for name, p in self.symbols.items() if isinstance(p, Parameter)}

    @property
    def free_parameters(self) -> dict[str, Parameter]:
        return {name: p for name, p in self.parameters.items() if not p.fixed}

    @property
    def fixed_parameters(self) -> dict[str, Parameter]:
        return {name: p for name, p in self.parameters.items() if p.fixed}

    @property
    def free_symbols(self) -> dict[str, FitSymbol]:
        """variables + free parameters"""
        return {**self.variables, **self.free_parameters}

    @property
    def guess(self) -> dict[str, float]:
        return {name: p.value for name, p in self.free_parameters.items()}
