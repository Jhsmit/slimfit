from __future__ import annotations

import itertools
from typing import Union, ItemsView, ValuesView, KeysView, Optional

import numpy.typing as npt
from sympy import Expr, MatrixBase, Symbol

import slimfit.numerical as numerical
from slimfit.parameter import Parameters


class Model(numerical.CompositeExpr):
    def __init__(
        self, model_dict: dict[Symbol | str, Expr | numerical.NumExprBase | MatrixBase],
    ):

        # currently typing has a small problem where keys are expected to be `str`, not symbol
        super().__init__(model_dict)

    def __repr__(self):
        return f"Model({self.expr.__repr__()})"

    # def __getitem__(self, item: Union[str, Symbol]) -> numerical.NumExprBase:
    #     if isinstance(item, str):
    #         item = self.symbols[item]
    #
    #     return self.expr[item]

    @property
    def dependent_symbols(self) -> dict[str, Symbol]:
        # todo needs to be updated
        """Variables corresponding to dependent (measured) data, given as keys in the model dict"""

        return {symbol.name: symbol for symbol in self.expr.keys()}
