"""
Model operations
Currently only multiplications for probablities
"""
from functools import reduce
from operator import or_, mul, add

import numpy.typing as npt

from slimfit.callable import NumExprBase, convert_callable


# a composite expression has multiple expr elements; connected by some operation but calculation is
# deferred so that fitting can inspect the composition and decide the best optimization strategy
class CompositeNumExpr(NumExprBase):
    """Operations base class"""

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            raise ValueError("At least two arguments are required.")
        self.elements = [convert_callable(arg) for arg in args]

    def renew(self) -> None:
        """Renews component parts"""

        for elem in self.elements:
            elem.renew()

    @property
    def symbols(self) -> dict:
        """Return symbols in order or constituent elements and then by alphabet"""
        return reduce(or_, (elem.symbols for elem in self.elements))

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

    def __init__(self, *args):
        if len(args) != 2:
            raise ValueError("MatMul takes exactly two arguments")
        super().__init__(*args)

    def __call__(self, **kwargs):
        return self.elements[0](**kwargs) @ self.elements[1](**kwargs)


#
# class CompositeModel(StateProbabilityModel):
#     def __init__(
#             self, lhs: StateProbabilityModel, rhs: StateProbabilityModel, operator: Callable
#     ):
#         expressions = {**lhs.expressions, **rhs.expressions}
#         independent_vars = {**lhs.independent_vars, **rhs.independent_vars}.values()
#         super().__init__(independent_vars, expressions)
#         # TODO in sympy these are called args
#         """
#         >>> x*y + 3
#
#         >>> type(expr)
#         <class 'sympy.core.add.Add'>
#
#         >>> expr.args
#         (3, x * y)
#         """
#         self.components = [lhs, rhs]
#         for c in self.components:
#             if not isinstance(c, StateProbabilityModel):
#                 raise TypeError(
#                     "Only 'StateProbabilityModel' objects can be multiplied " "together"
#                 )
#         self.operator = operator
#         self.operator = operator
#         if not operator == mul:
#             raise TypeError("Composite Models support multiplication only")
#
#         if lhs.states != rhs.states:
#             raise ValueError("Mismatch between states of model components.")
#
#         self.states = lhs.states
#
#     def __call__(self, **kwargs):
#         return self.operator(self.components[0](**kwargs), self.components[1](**kwargs))
#
#     def __iter__(self):
#         return self.components.__iter__()
#
#     def f_ij(self, **kwargs):
#         lhs = self.components[0].f_ij(**kwargs)
#         rhs = self.components[1].f_ij(**kwargs)
#         return self.operator(lhs, rhs)
#
#     def get_component(self, component_type: Type) -> dict[int, Any]:
#         """
#         Returns model components of specifed type
#
#         Args:
#             component_type: The type to return
#
#         Returns:
#             Dictionary of components of the requested type, keys are their position.
#         """
#
#         components = {
#             i: c for i, c in enumerate(self.components) if isinstance(c, component_type)
#         }
#
#         return components
#
#


class Sum(object):
    def __init__(self, matrix, axis=0):
        self.matrix = matrix

    def __call__(self, **kwargs):

        ...
