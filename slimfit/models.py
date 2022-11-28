from __future__ import annotations
import itertools
from typing import Union, ItemsView, ValuesView, KeysView

import numpy.typing as npt
from sympy import Expr, MatrixBase, Symbol

from slimfit.base import SymbolicBase
from slimfit.numerical import to_numexpr, NumExprBase
from slimfit.symbols import (
    FitSymbol,
)


class Model(SymbolicBase):
    def __init__(
        self, model_dict: dict[Symbol, Expr | NumExprBase | MatrixBase],
    ):

        # Dict or original Sympy expressions
        self.symbolic = model_dict

        # Dict of converted numerical expresisons
        self.numerical: dict[Symbol, NumExprBase] = {
            lhs: to_numexpr(rhs) for lhs, rhs in model_dict.items()
        }

    def __call__(self, **kwargs) -> dict[str, npt.ArrayLike]:
        return {lhs.name: rhs(**kwargs) for lhs, rhs in self.numerical.items()}

    def __repr__(self):
        return f"Model({self.symbolic.__repr__()})"

    def __getitem__(self, item: Union[str, Symbol]) -> NumExprBase:
        if isinstance(item, str):
            item = self.symbols[item]

        return self.numerical[item]

    def items(self) -> ItemsView:
        return self.numerical.items()

    def values(self) -> ValuesView:
        return self.numerical.values()

    def keys(self) -> KeysView:
        return self.numerical.keys()

    def renew(self) -> None:
        """clears cached lamdified properties"""
        for rhs in self.values():
            rhs.renew()

    @property
    def symbols(self) -> dict[str, FitSymbol]:
        """Return symbols in the model.
        sorting is by dependent_variables, variables, parameters, then by alphabet
        """

        ch = itertools.chain(*(elem.symbols.values() for elem in self.numerical.values()))
        rhs_symbols = {s.name: s for s in sorted(ch, key=str)}

        return self.dependent_symbols | rhs_symbols

    #TODO
    # refactor: 'keys', 'ysymbols', 'lhs_symbols' ?
    @property
    def dependent_symbols(self) -> dict[str, Symbol]:
        """Variables corresponding to dependent (measured) data, given as keys in the model dict"""

        return {symbol.name: symbol for symbol in self.symbolic.keys()}

    # @property
    # def independent_variables(self) -> dict[str, Variable]:
    #     ch = itertools.chain(*(elem.variables.values() for elem in self.model_dict.values()))
    #
    #     return {s.name: s for s in sorted(ch, key=SORT_KEY)}


def independent_parts(model_dict):
    # determine which parts in the model dict are independent given their overlap in parameters
    # currently assumed to be probabilistic, top-level multiplication elements are considered to be indepentent as their terms are
    # seperable in log probability
    ...
