from __future__ import annotations
import itertools
from functools import reduce
from operator import or_
from typing import Union, ItemsView, ValuesView, KeysView, Optional

import numpy.typing as npt
from sympy import Expr, MatrixBase, Symbol

from slimfit.base import SymbolicBase
from slimfit.parameter import Parameters
from slimfit.symbols import (
    FitSymbol,
)

import slimfit.numerical as numerical


class Model(SymbolicBase):
    def __init__(
        self, model_dict: dict[Symbol, Expr | numerical.NumExprBase | MatrixBase],
    ):

        #!! symbolic or numerical models distinction
        # Dict of original Sympy expressions
        self.model_dict = model_dict


        # Dict of converted numerical expressions
        # TODO: cached ?
        # self.numerical: dict[Symbol, NumExprBase] = {
        #     lhs: to_numexpr(rhs) for lhs, rhs in model_dict.items()
        # }

    def __repr__(self):
        return f"Model({self.model_dict.__repr__()})"

    def __getitem__(self, item: Union[str, Symbol]) -> numerical.NumExprBase:
        if isinstance(item, str):
            item = self.symbols[item]

        return self.model_dict[item]

    def items(self) -> ItemsView:
        return self.model_dict.items()

    def values(self) -> ValuesView:
        return self.model_dict.values()

    def keys(self) -> KeysView:
        return self.model_dict.keys()

    @property
    def symbols(self) -> dict[str, Symbol]:
        """Return symbols in the model.
        sorting is by dependent_variables, variables, parameters, then by alphabet
        """

        # this fails because `free_symbols` is a dict on NumExpr but `set` on Expr

        symbols = set()
        for rhs in self.model_dict.values():
            try:
                # rhs is a sympy `Expr` and has `free_symbols` as a set
                symbols |= rhs.free_symbols
            except TypeError:
                # rhs is a slimfit `NumExpr` and has a `free_symbols` dictionary
                symbols |= set(rhs.free_symbols.values())

        rhs_symbols = {s.name: s for s in sorted(symbols, key=str)}

        return self.dependent_symbols | rhs_symbols

    #TODO
    # refactor: 'keys', 'ysymbols', 'lhs_symbols' ?
    # add 'rhs_symbols ?
    # used to split data in xdata / ydata
    @property
    def dependent_symbols(self) -> dict[str, Symbol]:
        """Variables corresponding to dependent (measured) data, given as keys in the model dict"""

        return {symbol.name: symbol for symbol in self.model_dict.keys()}

    def to_numerical(self, parameters: Optional[Parameters] = None) -> NumericalModel:
        return NumericalModel(model_dict=self.model_dict, parameters=parameters)


class NumericalModel(numerical.NumExprBase):

    # TODO or should the init convert to numerical model? probably yes
    def __init__(self,
                 model_dict: dict[Symbol, Expr | numerical.NumExprBase | MatrixBase],
                 parameters: Optional[Parameters] = None):

        self.model_dict: dict[Symbol, numerical.NumExprBase] = {lhs: numerical.to_numerical(rhs, parameters) for lhs, rhs in model_dict.items()}
        super().__init__(parameters)

    def __call__(self, **kwargs) -> dict[str, npt.ArrayLike]:
        return {lhs.name: rhs(**kwargs) for lhs, rhs in self.model_dict.items()}

    @property
    def symbols(self) -> dict[str, Symbol]:
        """Return symbols in the model.
        sorting is by dependent_variables, variables, parameters, then by alphabet
        """

        ch = itertools.chain(*(rhs.symbols.values() for rhs in self.model_dict.values()))
        rhs_symbols = {s.name: s for s in sorted(ch, key=str)}

        return self.dependent_symbols | rhs_symbols

    @property
    def dependent_symbols(self):
        return {symbol.name: symbol for symbol in self.model_dict.keys()}

    # @property
    # def free_parameters(self) -> dict[str, Symbol]:
    #     return reduce(or_, (rhs.free_parameters for rhs in self.values()))
    #
    # @property
    # def fixed_parameters(self) -> dict[str, Symbol]:
    #     return reduce(or_, (rhs.fixed_parameters for rhs in self.values()))

    # Mapping mixin ?
    def items(self) -> ItemsView:
        return self.model_dict.items()

    def values(self) -> ValuesView:
        return self.model_dict.values()

    def keys(self) -> KeysView:
        return self.model_dict.keys()
