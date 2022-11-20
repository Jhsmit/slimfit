from __future__ import annotations

import abc
from functools import cached_property
from numbers import Number
from typing import Union, Callable, Optional, Any

import numpy as np
from sympy import Expr, MatrixBase, lambdify, HadamardProduct, Matrix

from slimfit.base import SymbolicBase
from slimfit.symbols import (
    FitSymbol,
    Parameter,
    SORT_KEY,
    Variable,
)

#todo refactor NumExpr
class NumExprBase(SymbolicBase):
    """Symbolic expression which allows calling cached lambified expressions
    """

    @abc.abstractmethod
    def __call__(self, **kwargs):
        pass

    @property
    def symbols(self):
        return {}

    @abc.abstractmethod
    def renew(self) -> None:
        pass

    @property
    def value(self) -> float | np.ndarray:
        return self(**self.guess)


class DummyNumExpr(NumExprBase):
    """Dummy callable object which returns supplied 'obj' when called
    Has no parameters or variables
    """

    def __init__(self, obj: Any, **kwargs):
        self.obj = obj

    def __call__(self, **kwargs):
        return self.obj

    def renew(self) -> None:
        pass


class ArrayCallable(DummyNumExpr):
    @property
    def shape(self) -> tuple[int, int]:
        return self.obj.shape


class ScalarNumExpr(NumExprBase):
    def __init__(self, expr: Expr, **kwargs):
        if not isinstance(expr, Expr):
            raise TypeError(f"Expression must be an instance of MatrixParameter or sympy.Matrix")
        self.expr = expr

    # is same as callablematrix
    @property
    def symbols(self) -> dict[str, FitSymbol]:
        """all symbols, sorted first by sort priority (variable, probability, parameter), then alphabetized"""
        return {s.name: s for s in sorted(self.expr.free_symbols, key=SORT_KEY)}

    @cached_property
    def lambdified(self) -> Callable:

        # subtitute out fixed parameters
        subs = [(p, p.value) for p in self.fixed_parameters.values()]
        sub_expr = self.expr.subs(subs)

        ld = lambdify(self.free_symbols.values(), sub_expr)

        return ld

    def renew(self) -> None:
        try:
            del self.lambdified
        except AttributeError:
            pass

    def __call__(self, **kwargs: float) -> Union[np.ndarray, float]:
        try:
            val = self.lambdified(**{k: kwargs[k] for k in self.free_symbols.keys()})
            return val
        except KeyError as e:
            raise KeyError(f"Missing value for parameter {e}") from e


# todo name via kwargs to super
class MatrixNumExpr(NumExprBase):
    def __init__(self, expr: MatrixBase, name: Optional[str] = None, kind: Optional[str] = None):
        if not isinstance(expr, MatrixBase):
            raise TypeError(f"Expression must be an instance of MatrixParameter or sympy.Matrix")
        self.expr = expr
        self._name = name

        # Callable type is used by minimizers to determine solving strategy
        if kind is None:
            self.kind = identify_expression_kind(expr)
        elif isinstance(kind, str):
            self.kind: str = kind.lower()
        else:
            raise TypeError("Invalid type for 'kind', must be 'str'")

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        if self.kind == "constant":
            p_names = list(self.parameters.keys())
            prefix = [name.split("_")[0] for name in p_names]
            if len(set(prefix)) == 1:  # All prefixes are identical, prefix is the name
                return prefix[0]

        else:
            return "M"

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def symbols(self) -> dict[str, FitSymbol]:
        """all symbols, sorted first by sort priority (variable, probability, parameter), then alphabetized"""
        return {s.name: s for s in sorted(self.expr.free_symbols, key=SORT_KEY)}

    @property
    def shape(self) -> tuple[int, int]:
        return self.expr.shape

    @cached_property
    def elements(self) -> dict[str, tuple[int, int]]:
        """
        Dictionary mapping parameter name to matrix indices
        """

        element_mapping = {}
        for i, j in np.ndindex(self.shape):
            elem = self.expr[i, j]
            if isinstance(elem, Parameter):
                element_mapping[elem.name] = (i, j)
        return element_mapping

    @cached_property
    def lambdified(self) -> np.ndarray:
        """Array of lambdified function per matrix element"""
        #TODO scalercallable per element

        # subtitute out fixed parameters
        subs = [(p, p.value) for p in self.fixed_parameters.values()]
        sub_expr = self.expr.subs(subs)

        lambdas = np.empty(self.shape, dtype=object)
        for i, j in np.ndindex(self.shape):
            lambdas[i, j] = lambdify(self.free_symbols.values(), sub_expr[i, j])

        return lambdas

    def renew(self) -> None:
        try:
            del self.lambdified
        except AttributeError:
            pass

    # TODO allow positional args for variables
    def __call__(self, **kwargs: float) -> np.ndarray:
        # https://github.com/sympy/sympy/issues/5642
        # Prepare kwargs for lambdified
        try:
            ld_kwargs = {k: kwargs[k] for k in self.free_symbols.keys()}
        except KeyError as e:
            raise KeyError(f"Missing value for symbol {e}") from e

        # Find the shape of the output
        shapes = (getattr(arg, "shape", (1,)) for arg in ld_kwargs.values())
        shape = np.broadcast_shapes(*shapes)

        # squeeze last dim if shape is (1,)
        shape = () if shape == (1,) else shape

        out = np.empty(shape + self.shape)
        for i, j in np.ndindex(self.shape):
            out[..., i, j] = self.lambdified[i, j](**ld_kwargs)

        return out

    @property
    def values(self) -> np.ndarray:
        """

        Returns: Array with elements set to parameter values

        """
        raise DeprecationWarning("Deprecate in favoor of `value`")
        arr = np.empty(self.shape)
        for i, j in np.ndindex(self.shape):
            arr[i, j] = self.expr[i, j].value

        return arr

    def __getitem__(self, key):
        return self.expr[key]

    def __contains__(self, item) -> bool:
        return self.expr.__contains__(item)

    # this is more of a str than a repr
    def __repr__(self):
        var_names = ", ".join(self.variables.keys())
        par_names = ", ".join(self.parameters.keys())
        if var_names and par_names:
            return f"{self.name}({var_names}; {par_names})"
        elif var_names:
            return f"{self.name}({var_names})"
        elif par_names:
            return f"{self.name}({par_names})"
        else:
            return f"{self.name}()"

    def index(self, name: str) -> tuple[int, int]:
        """
        Returns indices of parameter for Matrix Expressions

        Args:
            name: Parameter name to find matrix elements of

        Returns: Tuple of matrix elements ij

        """

        return self.elements[name]


class Constant(MatrixNumExpr):
    # WIP of a class which has an additional variable which determines output shape but
    # is no symbol in underlying expression

    def __init__(self, x: Variable, m: Matrix, name: Optional[str] = None):
        self.x = x
        super().__init__(m, name=name)

    def __call__(self, **kwargs):
        m_vals = super().__call__(**kwargs)

        np.broadcast_to(...)


class DummyVariableMatrix(MatrixNumExpr):
    """
    Matrix callable which takes an additional variable such that called returned shape is expanded to accomodate its
    shapes
    """

    def __init__(
        self, x: Variable, m: Matrix, kind: Optional[str] = None, name: Optional[str] = None,
    ):
        self.x = x
        super().__init__(m, kind=kind, name=name)

    @property
    def symbols(self) -> dict[str, FitSymbol]:
        symbols = super().symbols | {self.x.name: self.x}
        return {s.name: s for s in sorted(symbols.values(), key=SORT_KEY)}


class GMM(MatrixNumExpr):
    def __init__(self, x: Variable, mu: Matrix, sigma: Matrix, name: Optional[str] = None):
        if mu.shape[1] != 1:
            raise ValueError(
                "GMM parameter matrices must be of shape (N, 1) where N is the number of states."
            )
        self.x = x
        self.mu = convert_callable(mu)
        self.sigma = convert_callable(sigma)

        from slimfit.functions import gaussian

        expr = gaussian(x, mu, sigma)
        name = name or "GMM"  # counter for number of instances?
        super().__init__(expr, kind="GMM", name=name)


def identify_expression_kind(sympy_expression: Union[Expr, MatrixBase]) -> str:
    """Find the type of expression

    Not implemented, currently always returns generic

    """

    if isinstance(sympy_expression, MatrixBase):
        # check for gaussian mixture model ...
        ...

        if all(isinstance(elem, (Number, Parameter)) for elem in sympy_expression):
            return "constant"

    return "generic"


def convert_callable(expression: Union[NumExprBase, Expr, MatrixBase], **kwargs) -> NumExprBase:
    """Converts sympy expression to slimfit Callable"""

    if isinstance(expression, HadamardProduct):
        from slimfit.operations import Mul

        return Mul(*(convert_callable(arg) for arg in expression.args), **kwargs)
    elif isinstance(expression, MatrixBase):
        return MatrixNumExpr(expression, **kwargs)
    elif isinstance(expression, Expr):
        return ScalarNumExpr(expression, **kwargs)
    elif isinstance(expression, np.ndarray):
        return ArrayCallable(expression, **kwargs)
    elif isinstance(expression, NumExprBase):
        return expression
    else:
        raise TypeError(f"Invalid type {type(expression)!r}")
