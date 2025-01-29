from __future__ import annotations

from functools import cached_property
from typing import Any, Optional

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from slimfit.v2.expr import Expr, as_expr


class CompositeExpr(Expr):
    @cached_property
    def symbols(self) -> set[sp.Symbol]:
        return set().union(*(expr.symbols for expr in self.expr.values()))

    def __call__(self, **kwargs):
        return {k: v(**kwargs) for k, v in self.expr.items()}


class MarkovIVP(CompositeExpr):
    """Uses scipy.integrate.solve_ivp to numerically find time evolution of a markov process
        given a transition rate matrix.

    Returned shape is <states>, <datapoints>

    """

    def __init__(
        self,
        t: sp.Symbol | sp.Expr | Expr,
        trs_matrix: sp.Matrix | Expr,
        y0: sp.Matrix | Expr,
        domain: Optional[tuple[float, float]] = None,
        **ivp_kwargs,
    ):
        expr = as_expr({"t": t, "trs_matrix": trs_matrix, "y0": y0})
        super().__init__(expr)

        ivp_defaults = {"method": "Radau"}
        self.ivp_defaults = ivp_defaults | ivp_kwargs
        self.domain = domain

    def __call__(self, **kwargs):
        components = super().__call__(**kwargs)

        # if `self['t']` does not depend on any parameters; domain can be precomputed and
        # does not have to be determined for every call
        # although its every fast to do so
        domain = self.domain or self.get_domain(components["t"])
        sol = solve_ivp(
            self.grad_func,
            domain,
            y0=components["y0"].squeeze(),
            t_eval=components["t"],
            args=(components["trs_matrix"],),
            **self.ivp_defaults,
        )

        return sol.y

    def get_domain(self, arr: np.ndarray) -> tuple[float, float]:
        # padding?
        return arr[0], arr[-1]

    @staticmethod
    def grad_func(t, y, trs_matrix):
        return trs_matrix @ y
