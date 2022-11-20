import itertools
from typing import Union, ItemsView, ValuesView, KeysView

import numpy.typing as npt
from sympy import Expr, MatrixBase

from slimfit.base import SymbolicBase
from slimfit.callable import convert_callable, NumExprBase
from slimfit.symbols import (
    Variable,
    Probability,
    FitSymbol,
    SORT_KEY,
)


class Model(SymbolicBase):
    def __init__(
        self, model_dict: [Union[Variable, Probability], Union[Expr, NumExprBase, MatrixBase], ],
    ):

        if all(isinstance(lhs, Probability) for lhs in model_dict.keys()):
            self.probabilistic = True
        elif all(isinstance(lhs, Variable) for lhs in model_dict.keys()):
            self.probabilistic = False
        else:
            raise ValueError(
                "Model dictionary keys need to be either all `Variable` or `Probability`"
            )

        self.model_dict: [FitSymbol, NumExprBase] = {
            lhs: convert_callable(rhs) for lhs, rhs in model_dict.items()
        }

        self._model_dict: [str, NumExprBase] = {
            lhs.name: convert_callable(rhs) for lhs, rhs in model_dict.items()
        }

    def __call__(self, **kwargs) -> dict[str, npt.ArrayLike]:
        return {lhs.name: rhs(**kwargs) for lhs, rhs in self.model_dict.items()}

    def __repr__(self):
        return f"Model({self.model_dict.__repr__()})"

    def __getitem__(self, item) -> NumExprBase:
        if isinstance(item, str):
            return self._model_dict[item]
        else:
            return self.model_dict[item]

    def items(self) -> ItemsView:
        return self.model_dict.items()

    def values(self) -> ValuesView:
        return self.model_dict.values()

    def keys(self) -> KeysView:
        return self.model_dict.keys()

    def renew(self) -> None:
        """clears cached lamdified properties"""
        for rhs in self.values():
            rhs.renew()

    @property
    def symbols(self) -> dict[str, FitSymbol]:
        """Return symbols in the model.
        sorting is by dependent_variables, variables, parameters, then by alphabet
        """

        ch = itertools.chain(*(elem.symbols.values() for elem in self.model_dict.values()))
        rhs_symbols = {s.name: s for s in sorted(ch, key=SORT_KEY)}

        return self.dependent_variables | rhs_symbols

    @property
    def dependent_variables(self) -> dict[str, Variable]:
        """Variables corresponding to dependent (measured) data, given as keys in the model dict"""
        return {
            v.name: v
            for v in sorted(self.model_dict.keys(), key=SORT_KEY)
            if isinstance(v, Variable)
        }

    @property
    def independent_variables(self) -> dict[str, Variable]:
        ch = itertools.chain(*(elem.variables.values() for elem in self.model_dict.values()))

        return {s.name: s for s in sorted(ch, key=SORT_KEY)}


def independent_parts(model_dict):
    # determine which parts in the model dict are independent given their overlap in parameters
    # currently assumed to be probabilistic, top-level multiplication elements are considered to be indepentent as their terms are
    # seperable in log probability
    ...
