from __future__ import annotations

import abc
from functools import cached_property

from sympy import Symbol
import numpy as np

from slimfit.parameter import Parameter, Parameters
from slimfit.typing import Shape


class SymbolicBase(metaclass=abc.ABCMeta):
    @cached_property
    @abc.abstractmethod
    def symbols(self) -> set[Symbol]:
        ...

    @cached_property
    def symbol_names(self) -> set[str]:
        return set(s.name for s in self.symbols)

    @property
    def shapes(self) -> dict[str, Shape]:
        """
        dict of symbol shapes
        """

    def filter_parameters(self, parameters: Parameters) -> Parameters:
        """Filters a list of parameters, returning only the ones whose symbols are
        in this model
        """
        return Parameters([p for p in parameters if p.symbol in self.symbols])

    # @property
    # @abc.abstractmethod
    # def shape(self) -> tuple:
    #     ...
