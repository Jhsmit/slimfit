from __future__ import annotations

import abc
import typing
from functools import cached_property
from typing import Union, Callable, Optional, Any, Mapping

import numpy as np
from scipy.integrate import solve_ivp
from sympy import Expr, MatrixBase, lambdify, HadamardProduct, Matrix, Symbol

from slimfit.base import SymbolicBase
from slimfit.parameter import Parameters, Parameter
from slimfit.symbols import (
    FitSymbol,
)

if typing.TYPE_CHECKING:
    from slimfit import Model


# todo refactor NumExpr
class NumExprBase(SymbolicBase):
    """Symbolic expression which allows calling cached lambified expressions
    """

    def __init__(self, parameters: Optional[Parameters] = None):
        parameters = parameters or {}

        #Accepted parameters are a subset of `symbols`
        #todo property with getter / setter where setter filters parameters?
        self.parameters = Parameters({name: p for name, p in parameters.items() if name in self.symbols})

    @abc.abstractmethod
    def __call__(self, **kwargs):
        pass

    @property
    def symbols(self) -> dict[str, Symbol]:
        return {}

    @property
    def free_symbols(self) -> dict[str, Symbol]:
        # all symbols which are not fixed parameters
        # these are the symbols which are arguments for lambdified,
        # fixed symbols (parameters) are substituted out before lambidification
        return {name: symbol for name, symbol in self.symbols.items() if name not in self.fixed_parameters}

    def set_parameters(self, parameters: Mapping | None):
        if parameters is not None:
            try:
                p = Parameters({name: p for name, p in parameters.items() if name in self.symbols})
                self.parameters = p
            except AttributeError:
                print(p)
                print('berak')

    @property
    def fixed_parameters(self) -> dict[str, Parameter]:
        return {name: p for name, p in self.parameters.items() if p.fixed}

    @property
    def free_parameters(self) -> dict[str, Parameter]:
        return {name: p for name, p in self.parameters.items() if not p.fixed}


class DummyNumExpr(NumExprBase):
    """Dummy callable object which returns supplied 'obj' when called
    Has no parameters or variables
    """

    def __init__(self, obj: Any, parameters: Optional[Parameters] = None):
        super().__init__(parameters)
        self.obj = obj

    def __call__(self, **kwargs):
        return self.obj


class ArrayCallable(DummyNumExpr):

    @property
    def shape(self) -> tuple[int, int]:
        return self.obj.shape


class ScalarNumExpr(NumExprBase):
    def __init__(self, expr: Expr, parameters: Optional[Parameters] = None):
        super().__init__(parameters)
        if not isinstance(expr, Expr):
            raise TypeError(f"Expression must be an instance of MatrixParameter or sympy.Matrix")
        self.expr = expr

    # is same as callablematrix
    @property
    def symbols(self) -> dict[str, FitSymbol]:
        """all symbols, sorted first by sort priority (variable, probability, parameter), then alphabetized"""
        return {s.name: s for s in sorted(self.expr.free_symbols, key=str)}

    @cached_property
    def lambdified(self) -> Callable:
        # substitute out fixed parameters
        subs = [(p, p.guess) for p in self.fixed_parameters.values()]
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
# = composite num expr"?
class MatrixNumExpr(NumExprBase):
    def __init__(self, expr: MatrixBase, parameters: Optional[Parameters] = None, name: Optional[str] = None, kind: Optional[str] = None):
        self.expr = expr

        if not isinstance(expr, MatrixBase):
            raise TypeError("Expression must be an instance of MatrixParameter or sympy.Matrix")
        self._name = name

        # Callable type is used by minimizers to determine solving strategy
        if kind is None:
            self.kind = identify_expression_kind(expr)
        elif isinstance(kind, str):
            self.kind: str = kind.lower()
        else:
            raise TypeError("Invalid type for 'kind', must be 'str'")

        super().__init__(parameters)


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
        return {s.name: s for s in sorted(self.expr.free_symbols, key=str)}

    @property
    def shape(self) -> tuple[int, int]:
        return self.expr.shape

    @cached_property
    def elements(self) -> dict[str, tuple[int, int]]:
        """
        Dictionary mapping parameter name to matrix indices
        """

        raise NotImplementedError("Elements is not implemented")

        # element_mapping = {}
        # for i, j in np.ndindex(self.shape):
        #     elem = self.expr[i, j]
        #     if isinstance(elem, Parameter):
        #         element_mapping[elem.name] = (i, j)
        # return element_mapping

    @cached_property
    def lambdified(self) -> np.ndarray:
        """Array of lambdified function per matrix element"""
        # TODO scalercallable per element

        # subtitute out fixed parameters
        # subs = [(p, p.value) for p in self.fixed_parameters.values()]
        # sub_expr = self.expr.subs(subs)

        lambdas = np.empty(self.shape, dtype=object)
        for i, j in np.ndindex(self.shape):
            lambdas[i, j] = lambdify(self.free_symbols.values(), self.expr[i, j])

        return lambdas

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
        return f"{self.name}({', '.join(self.symbols)})"

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

    def __init__(self, x: Symbol, m: Matrix, name: Optional[str] = None):
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
        self, x: Symbol, m: Matrix, parameters: Optional[Parameters] = None, kind: Optional[str] = None, name: Optional[str] = None,
    ):
        self.x = x
        super().__init__(m, parameters=parameters, kind=kind, name=name)

    @property
    def symbols(self) -> dict[str, FitSymbol]:
        symbols = super().symbols | {self.x.name: self.x}
        return {s.name: s for s in sorted(symbols.values(), key=str)}


class GMM(MatrixNumExpr):
    def __init__(self, x: Symbol, mu: Matrix, sigma: Matrix, parameters: Optional[Parameters] = None, name: Optional[str] = None):
        if mu.shape[1] != 1:
            raise ValueError(
                "GMM parameter matrices must be of shape (N, 1) where N is the number of states."
            )
        self.x = x
        self.mu = to_numerical(mu)
        self.sigma = to_numerical(sigma)

        from slimfit.functions import gaussian

        expr = gaussian(x, mu, sigma)
        name = name or "GMM"  # counter for number of instances?
        super().__init__(expr, parameters=parameters, kind="GMM", name=name)


class MarkovIVPNumExpr(NumExprBase):
    """Uses scipy.integrate.solve_ivp to numerically find time evolution of a markov process
        given a transition rate matrix.

    Returned shape is (len(t_var), len(y0), 1), or (<datapoints>, <states>, 1)

    """
    def __init__(
        self,
        t_var: Symbol,
        trs_matrix: Matrix,
        y0: Matrix,
        parameters: Optional[Parameters] = None,
        domain: Optional[tuple[float, float]] = None,
        **ivp_kwargs
    ):
        super().__init__(parameters)
        self.t_var = t_var
        self.trs_matrix = to_numerical(trs_matrix, parameters)
        self.y0 = to_numerical(y0, parameters)
        self.domain = domain

        ivp_defaults = {'method': 'Radau'}
        self.ivp_defaults = ivp_defaults | ivp_kwargs

    @property
    def symbols(self) -> dict[str, FitSymbol]:
        symbols = self.trs_matrix.symbols | self.y0.symbols | {self.t_var.name: self.t_var}
        return {s.name: s for s in sorted(symbols.values(), key=str)}

    def __call__(self, **kwargs):
        trs_matrix = self.trs_matrix(**kwargs)
        y0 = self.y0(**kwargs).squeeze()

        domain = self.domain or self.get_domain(**kwargs)
        sol = solve_ivp(
            self.grad_func,
            domain,
            y0=y0,
            t_eval=kwargs[self.t_var.name],
            args=(trs_matrix, ),
            **self.ivp_defaults

        )

        return np.expand_dims(sol.y.T, -1)

    def get_domain(self, **kwargs) -> tuple[float, float]:
        dpts = kwargs[self.t_var.name]
        return dpts.min(), dpts.max()

    @staticmethod
    def grad_func(t, y, trs_matrix):
        return trs_matrix @ y


def identify_expression_kind(sympy_expression: Union[Expr, MatrixBase]) -> str:
    """Find the type of expression

    Not implemented, currently always returns generic

    """

    if isinstance(sympy_expression, MatrixBase):
        # check for gaussian mixture model ...
        ...
        #
        # if all(isinstance(elem, (Number, Parameter)) for elem in sympy_expression):
        #     return "constant"

    return "generic"


def to_numerical(
        expression: Union[NumExprBase, Expr, MatrixBase | Model],
        parameters: Optional[Parameters] = None) -> NumExprBase:
    """Converts sympy expression to slimfit numerical expression

        if the expressions already is an NumExpr; the object is modified in-place by setting
        the parameters

    """
    from slimfit import Model

    if isinstance(expression, Model):
        model_dict = {lhs: to_numerical(rhs) for lhs, rhs in expression.items()}
        from slimfit.models import NumericalModel
        return NumericalModel(model_dict, parameters)
    if isinstance(expression, HadamardProduct):
        from slimfit.operations import Mul
        return Mul(*(to_numerical(arg, parameters) for arg in expression.args), parameters=parameters)
    elif isinstance(expression, MatrixBase):
        return MatrixNumExpr(expression, parameters=parameters)
    elif isinstance(expression, Expr):
        return ScalarNumExpr(expression, parameters=parameters)
    elif isinstance(expression, np.ndarray):
        return ArrayCallable(expression, parameters=parameters)
    elif isinstance(expression, NumExprBase):
        expression.set_parameters(parameters)
        return expression
    else:
        raise TypeError(f"Invalid type {type(expression)!r}")
