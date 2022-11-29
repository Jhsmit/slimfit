from __future__ import annotations

import abc
from typing import Optional

from sympy import Symbol

from slimfit.symbols import FitSymbol


class SymbolicBase(object, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def symbols(self) -> dict[str, Symbol]:
        """Returns a dict with all FitSymbols in the Callable"""
