"""
Model operations
Currently only multiplications for probablities
"""
from functools import reduce
from operator import mul

import numpy as np
import numpy.typing as npt

from slimfit.numerical import to_numerical, CompositeExpr
from slimfit.parameter import Parameter
from slimfit.typing import Shape


class CompositeArgsExpr(CompositeExpr):
    """Composite expr which takes *args to init rather than dictionary of expressions"""

    def __init__(self, *args):
        expr = {i: arg for i, arg in enumerate(args)}
        super().__init__(expr)

    def to_numerical(self):
        args = (to_numerical(expr) for expr in self.values())
        instance = self.__class__(*args)

        return instance


class Mul(CompositeArgsExpr):
    # might be subject to renaming
    """Mul elementwise lazily
    """

    def __init__(self, *args):
        super().__init__(*args)

    def __call__(self, **kwargs) -> npt.ArrayLike:
        result = super().__call__(**kwargs)

        return reduce(mul, result.values())

    def __repr__(self):
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
        result = super().__call__(**kwargs)
        return result[0] @ result[1]

    @property
    def shape(self) -> Shape:
        raise NotImplementedError()

    def __repr__(self):
        args = ", ".join([arg.__repr__() for arg in self.values()])
        return f"MatMul({args})"


class Sum(object):
    def __init__(self, matrix, axis=0):
        self.matrix = matrix

    def __call__(self, **kwargs):

        ...
