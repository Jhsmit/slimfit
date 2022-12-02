from __future__ import annotations

import abc

from sympy import Symbol
import numpy as np

from slimfit.parameter import Parameter


#
# class SymbolicBase(object, metaclass=abc.ABCMeta):
#     @property
#     @abc.abstractmethod
#     def symbols(self) -> dict[str, Symbol]:
#         """Returns a dict with all FitSymbols in the Callable"""


class SymbolicBase(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def symbols(self) -> dict[str, Symbol]:
        ...

    @property
    def free_symbols(self) -> dict[str, Symbol]:
        # all symbols which are not fixed parameters
        # these are the symbols which are arguments for lambdified,
        # fixed symbols (parameters) are substituted out before lambidification
        return {
            name: symbol
            for name, symbol in self.symbols.items()
            if name not in self.fixed_parameters
        }

    @property
    @abc.abstractmethod
    def parameters(self) -> dict[str, Parameter]:
        ...

    @property
    def fixed_parameters(self) -> dict[str, np.ndarray | float]:
        return {name: p.guess for name, p in self.parameters.items() if p.fixed}

    @property
    def free_parameters(self) -> dict[str, Parameter]:
        return {name: p for name, p in self.parameters.items() if not p.fixed}

    # @property
    # @abc.abstractmethod
    # def shape(self) -> tuple:
    #     ...
