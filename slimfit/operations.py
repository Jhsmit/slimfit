"""
Model operations
Currently only multiplications for probablities
"""
from __future__ import annotations

from functools import reduce
from operator import mul, add

import numpy as np
import numpy.typing as npt
from typing import Union

from slimfit.base import CompositeArgsExpr
from slimfit.numerical import to_numerical
from slimfit.typing import Shape


class Mul(CompositeArgsExpr):
    # might be subject to renaming
    """Mul elementwise lazily"""

    def __init__(self, *args):
        super().__init__(*args)

    def __call__(self, **kwargs) -> npt.ArrayLike:
        result = self._call(**kwargs)

        return reduce(mul, result.values())

    def __repr__(self) -> str:
        args = ", ".join([arg.__repr__() for arg in self.values()])
        return f"Mul({args})"


class MatMul(CompositeArgsExpr):

    """
    matmul composite callable

    arguments must have .shape attribute and shape must be compatible with matrix multiplication

    """

    def __init__(self, *args):
        if len(args) != 2:
            raise ValueError("MatMul takes exactly two arguments")
        super().__init__(*args)

    def __call__(self, **kwargs):
        result = self._call(**kwargs)
        return result[0] @ result[1]

    @property
    def shape(self) -> Shape:
        raise NotImplementedError()

    def __repr__(self) -> str:
        args = ", ".join([arg.__repr__() for arg in self.values()])
        return f"MatMul({args})"


class Add(CompositeArgsExpr):
    # TODO add flattening
    # identify removing / term sorting?

    def __init__(self, *args):
        super().__init__(*args)

    def __call__(self, **kwargs) -> npt.ArrayLike:
        result = self._call(**kwargs)

        return reduce(add, result.values())

    def __repr__(self) -> str:
        args = ", ".join([arg.__repr__() for arg in self.values()])
        return f"Add({args})"


class Sum(CompositeArgsExpr):
    def __init__(self, expr, axis: int = 0):
        super().__init__(expr)
        self.axis = axis

    def __call__(self, **kwargs) -> npt.ArrayLike:
        return np.sum(self.expr[0](**kwargs), axis=self.axis)

    def __repr__(self) -> str:
        args = ", ".join([arg.__repr__() for arg in self.values()])
        return f"Sum({args})"

    def to_numerical(self):
        args = (to_numerical(expr) for expr in self.values())
        instance = self.__class__(*args, axis=self.axis)

        return instance

    @property
    def shape(self) -> Shape:
        return tuple(elem for i, elem in enumerate(self.expr[0].shape) if i != self.axis)


class Indexer(CompositeArgsExpr):
    def __init__(self, expr, indexer: Union[tuple, slice, int]):
        super().__init__(expr)
        self.indexer = indexer

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        return result[0][self.indexer]

    def to_numerical(self) -> Indexer:
        # generalize this (see also Sum)
        args = (to_numerical(expr) for expr in self.values())
        return Indexer(*args, self.indexer)

    def __repr__(self) -> str:
        from slimfit.utils import format_indexer

        idx_fmt = format_indexer(
            self.indexer if isinstance(self.indexer, tuple) else (self.indexer,)
        )
        return self.expr[0].__repr__() + idx_fmt


# class Transpose(CompositeArgsExpr):
#     def __init__(self, expr):
#         super().__init__(expr)
#
#     def __call__(self, **kwargs):
#         result = super().__call__(**kwargs)
#         return result[0].T
#
#     def __repr__(self) -> str:
#         return f"{self.expr[0].__repr__()}.T"
