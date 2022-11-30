"""
Model operations
Currently only multiplications for probablities
"""
from functools import reduce
from operator import or_, mul, add
from typing import Optional

import numpy.typing as npt

from slimfit.numerical import NumExprBase, to_numerical
from slimfit.parameter import Parameters

# a composite expression has multiple expr elements; connected by some operation but calculation is
# deferred so that fitting can inspect the composition and decide the best optimization strategy
class CompositeNumExpr(NumExprBase):
    """Operations base class"""

    kind = "composite"

    def __init__(self, *args, parameters: Optional[Parameters] = None):
        super().__init__(parameters)
        if len(args) == 1:
            raise ValueError("At least two arguments are required.")
        self.elements = [to_numerical(arg, parameters) for arg in args]

    def renew(self) -> None:
        """Renews component parts"""

        for elem in self.elements:
            elem.renew()

    @property
    def symbols(self) -> dict:
        """Return symbols in order or constituent elements and then by alphabet"""
        return reduce(or_, (elem.symbols for elem in self.elements))

    # @property
    # def parameters(self) -> Parameters:
    #     """Return parameters"""
    #     return reduce(or_, (elem.parameters for elem in self.elements))

    def __getitem__(self, item) -> NumExprBase:
        return self.elements.__getitem__(item)


class Sum(CompositeNumExpr):
    def __call__(self, **kwargs) -> npt.ArrayLike:
        eval_elems = (elem(**kwargs) for elem in self.elements)

        return reduce(add, eval_elems)


# elementwise !
class Mul(CompositeNumExpr):
    # might be subject to renaming
    """Mul elementwise lazily
    """

    def __call__(self, **kwargs) -> npt.ArrayLike:
        eval_elems = (elem(**kwargs) for elem in self.elements)

        return reduce(mul, eval_elems)

    def __repr__(self):
        args = ", ".join([arg.__repr__() for arg in self.elements])
        return f"Mul({args})"


class MatMul(CompositeNumExpr):

    """
    matmul composite callable

    arguments must have .shape attribute and shape must be compatible with matrix multiplication

    """

    def __init__(self, *args, parameters: Optional[Parameters] = None):
        if len(args) != 2:
            raise ValueError("MatMul takes exactly two arguments")
        super().__init__(*args, parameters=parameters)

    def __call__(self, **kwargs):
        return self.elements[0](**kwargs) @ self.elements[1](**kwargs)


class Sum(object):
    def __init__(self, matrix, axis=0):
        self.matrix = matrix

    def __call__(self, **kwargs):

        ...
