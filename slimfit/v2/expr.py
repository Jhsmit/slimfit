# %%
from __future__ import annotations

from functools import cached_property
from typing import Callable, Union

import numpy as np
import sympy as sp

from slimfit.v2.typing import Numerical


# %%
class Expr:
    def __init__(self, expr) -> None:
        self._expr = expr

    @property
    def expr(self):
        return self._expr

    @cached_property
    def symbols(self) -> set[sp.Symbol]:
        return set()

    def filter_kwargs(self, **kwargs) -> dict[str, Numerical]:
        """Parse kwargs and take only the ones in `free_parameters`"""
        try:
            kwargs = {k: kwargs[k] for k in {s.name for s in self.symbols}}
        except KeyError as e:
            raise KeyError(f"Missing value for {e}") from e

        return kwargs

    def __getitem__(self, item):
        return GetItem(self, item)

    def __call__(self, **kwargs):
        return self._expr


class GetItem(Expr):
    def __init__(self, expr: Expr, item: Union[tuple, slice, int]):
        # todo super
        self._expr = expr
        self.item = item

    def __call__(self, **kwargs):
        ans = self._expr(**kwargs)
        return ans[self.item]

    @cached_property
    def symbols(self) -> set[sp.Symbol]:
        return self._expr.symbols

    def __repr__(self) -> str:
        return f"{self.expr.__repr__()}[{self.item!r}]"


class SympyExpr(Expr):
    @cached_property
    def symbols(self) -> set[sp.Symbol]:
        return self._expr.free_symbols

    @cached_property
    def lambdified(self) -> Callable:
        ld = sp.lambdify(sorted(self.symbols, key=str), self._expr)

        return ld

    def __repr__(self) -> str:
        return f"SympyExpr({self._expr})"

    def __call__(self, **kwargs):
        return self.lambdified(**self.filter_kwargs(**kwargs))


class SympyMatrixExpr(Expr):
    def __init__(self, expr: sp.Matrix) -> None:
        super().__init__(expr)

    @cached_property
    def symbols(self) -> set[sp.Symbol]:
        return self._expr.free_symbols

    @cached_property
    def lambdified(self) -> dict[tuple[int, int], Callable]:
        lambdas = {}
        for i, j in np.ndindex(self.expr.shape):
            lambdas[(i, j)] = sp.lambdify(sorted(self.symbols, key=str), self.expr[i, j])

        return lambdas

    def __call__(self, **kwargs):
        # when marix elements != scalars, shape is expanded by the first dimensions to accomodate.
        ld_kwargs = self.filter_kwargs(**kwargs)

        base_shape = np.broadcast_shapes(
            *(getattr(value, "shape", tuple()) for value in ld_kwargs.values())
        )

        # squeeze last dim if shape is (1,)
        base_shape = () if base_shape == (1,) else base_shape
        shape = base_shape + self.expr.shape

        out = np.empty(shape)
        for i, j in np.ndindex(self.expr.shape):
            out[..., i, j] = self.lambdified[i, j](**ld_kwargs)

        return out


def as_expr(expr) -> Expr | dict[str, Expr]:
    if isinstance(expr, Expr):
        return expr
    elif isinstance(expr, (float, np.ndarray)):  # torch tensor, ...
        return Expr(expr)
    if isinstance(expr, sp.Expr):
        return SympyExpr(expr)
    elif isinstance(expr, sp.Matrix):
        return SympyMatrixExpr(expr)
    elif isinstance(expr, dict):
        return {k: as_expr(v) for k, v in expr.items()}  # type: ignore
    else:
        raise TypeError(f"Invalid type: {type(expr)}")
