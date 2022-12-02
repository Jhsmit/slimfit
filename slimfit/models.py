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

    def __getitem__(self, item: Union[str, Symbol]) -> numerical.NumExprBase:
        if isinstance(item, str):
            item = self.symbols[item]

        return self.expr[item]

    @property
    def dependent_symbols(self) -> dict[str, Symbol]:
        """Variables corresponding to dependent (measured) data, given as keys in the model dict"""

        return {symbol.name: symbol for symbol in self.expr.keys()}

    # def to_numerical(self, parameters: Optional[Parameters] = None) -> NumericalModel:
    #     return NumericalModel(model_dict=self.model_dict, parameters=parameters)

#
# class NumericalModel(numerical.NumExprBase):
#
#     # TODO or should the init convert to numerical model? probably yes
#     # actually no! it should be a composite where its elements ahve parameters
#     def __init__(
#         self,
#         model_dict: dict[Symbol, Expr | numerical.NumExprBase | MatrixBase],
#         parameters: Optional[Parameters] = None,
#     ):
#
#         self.model_dict: dict[Symbol, numerical.NumExprBase] = {
#             lhs: numerical.to_numerical(rhs, parameters) for lhs, rhs in model_dict.items()
#         }
#         super().__init__(parameters)
#
#     def __call__(self, **kwargs) -> dict[str, npt.ArrayLike]:
#         return {lhs.name: rhs(**kwargs) for lhs, rhs in self.model_dict.items()}
#
#     @property
#     def symbols(self) -> dict[str, Symbol]:
#         """Return symbols in the model.
#         sorting is by dependent_variables, variables, parameters, then by alphabet
#         """
#
#         ch = itertools.chain(*(rhs.symbols.values() for rhs in self.model_dict.values()))
#         rhs_symbols = {s.name: s for s in sorted(ch, key=str)}
#
#         return self.dependent_symbols | rhs_symbols
#
#     @property
#     def dependent_symbols(self):
#         return {symbol.name: symbol for symbol in self.model_dict.keys()}
#
#     # @property
#     # def free_parameters(self) -> dict[str, Symbol]:
#     #     return reduce(or_, (rhs.free_parameters for rhs in self.values()))
#     #
#     # @property
#     # def fixed_parameters(self) -> dict[str, Symbol]:
#     #     return reduce(or_, (rhs.fixed_parameters for rhs in self.values()))
#
#     # Mapping mixin ?
#     def items(self) -> ItemsView:
#         return self.model_dict.items()
#
#     def values(self) -> ValuesView:
#         return self.model_dict.values()
#
#     def keys(self) -> KeysView:
#         return self.model_dict.keys()
