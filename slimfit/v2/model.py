from __future__ import annotations

import re
from typing import Iterable

import sympy as sp
from toposort import toposort

from slimfit.v2.expr import as_expr
from slimfit.v2.parameter import Parameter, Parameters
from slimfit.v2.typing import Numerical


class Model:
    def __init__(self, model: dict) -> None:
        self.model = model
        self.expr: dict = {k: as_expr(v) for k, v in self.model.items()}
        topology = {k: v.symbols for k, v in self.expr.items()}
        self.call_stack = [
            elem for subset in toposort(topology) for elem in subset if elem in model.keys()
        ]

    @property
    def x_symbols(self) -> set[sp.Symbol]:
        return set.union(*(v.symbols for v in self.expr.values())) - self.y_symbols

    @property
    def y_symbols(self) -> set[sp.Symbol]:
        return set(self.model.keys())

    def __call__(self, **kwargs):
        resolved = {}
        for key in self.call_stack:
            resolved[key.name] = self.expr[key](**kwargs, **resolved)
        return resolved

    def define_parameters(
        self, parameters: dict[str, Numerical] | Iterable[str] | str = "*"
    ) -> Parameters:
        symbols = {s.name: s for s in self.x_symbols}
        if parameters == "*":
            params = [Parameter(symbol) for symbol in self.x_symbols]
        elif isinstance(parameters, str):
            params = [Parameter(symbols[k]) for k in re.split(r"; |, |\*|\s+", parameters)]
        elif isinstance(parameters, dict):
            params = [Parameter(symbols[k], guess=v) for k, v in parameters.items()]
        else:
            raise TypeError("Invalid type")

        return Parameters(params)
