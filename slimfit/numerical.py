from __future__ import annotations

import abc
import itertools
from functools import cached_property, reduce
from operator import or_
from typing import (
    Union,
    Callable,
    Optional,
    Any,
    Mapping,
    Iterable,
    TYPE_CHECKING,
    KeysView,
    ItemsView,
    ValuesView,
)

import numpy as np
from scipy.integrate import solve_ivp
from sympy import Expr, MatrixBase, lambdify, HadamardProduct, Matrix, Symbol

from slimfit.base import SymbolicBase

# from slimfit.base import SymbolicBase
from slimfit.parameter import Parameters, Parameter
from slimfit.symbols import FitSymbol
from slimfit.typing import DataType, Shape

if TYPE_CHECKING:
    from slimfit import Model

# @dataclass
class NumExprBase(SymbolicBase):
    """Symbolic expression which allows calling cached lambified expressions
    subclasses must implement `symbols` attribute / property
    """

    def __init__(
        self,
        parameters: Optional[dict[str, Parameter]] = None,
        data: Optional[dict[str, np.ndarray]] = None,
    ):
        self.parameters = parameters or {}
        self.data = data or {}

        # Accepted parameters are a subset of `symbols`
        # #todo property with getter / setter where setter filters parameters?
        # self.parameters = Parameters({name: p for name, p in parameters.items() if name in self.symbols})

    @property
    def parameters(self) -> dict[str, Parameter]:
        """Parameters must be a subset of symbols"""
        return self._parameters

    @parameters.setter
    def parameters(self, value: Mapping[str, Parameter]):
        self._parameters = {name: p for name, p in value.items() if name in self.symbols}

    @property
    def data(self) -> dict[str, np.ndarray]:
        return self._data

    @data.setter
    def data(self, value: Mapping[str, np.ndarray]):
        data_symbols = self.free_symbols.keys() - self.parameters.keys()
        self._data = {k: v for k, v in value.items() if k in data_symbols}

    @property
    def shape(self) -> Shape:
        parameter_shapes = (p.shape for p in self.parameters.values())
        data_shapes = (getattr(v, "shape", tuple()) for v in self.data.values())

        return np.broadcast_shapes(*itertools.chain(parameter_shapes, data_shapes))

    def parse_kwargs(self, **kwargs) -> dict[str, np.ndarray]:
        """Parse kwargs and take only the ones in `free_parameters`"""
        try:
            parameters: dict[str, np.ndarray | float] = {
                k: kwargs[k] for k in self.free_parameters.keys()
            }
        except KeyError as e:
            raise KeyError(f"Missing value for parameter {e}") from e

        return parameters


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


class NumExpr(NumExprBase):
    def __init__(
        self,
        expr: Expr,
        parameters: Optional[dict[str, Parameter]] = None,
        data: Optional[dict[str, np.ndarray]] = None,
    ):
        if not isinstance(expr, (Expr, MatrixBase)):
            # TODO subclass such that typing is correct
            raise TypeError(f"Expression must be an instance of `Expr` or ")
        self.expr = expr

        super().__init__(parameters, data)

    # is same as callablematrix
    @property
    def symbols(self) -> dict[str, Symbol]:
        """all symbols, sorted first by sort priority (variable, probability, parameter), then alphabetized"""
        return {s.name: s for s in sorted(self.expr.free_symbols, key=str)}

    @cached_property
    def lambdified(self) -> Callable:
        ld = lambdify(self.free_symbols.values(), self.expr)

        return ld

    def __call__(self, **kwargs: float) -> np.ndarray | float:
        try:
            parameters: dict[str, np.ndarray | float] = {
                k: kwargs[k] for k in self.parameters.keys()
            }
        except KeyError as e:
            raise KeyError(f"Missing value for parameter {e}") from e

        val = self.lambdified(**parameters, **self.fixed_parameters, **self.data)
        return val


# todo name via kwargs to super
# = composite num expr"?
class MatrixNumExpr(NumExpr):
    def __init__(
        self,
        expr: MatrixBase,
        parameters: Optional[dict[str, Parameter]] = None,
        data: Optional[dict[str, np.ndarray]] = None,
        name: Optional[str] = None,
        kind: Optional[str] = None,
    ):

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

        super().__init__(expr, parameters, data)

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
    def shape(self) -> Shape:
        # again there might be problems depending on how the matrix elements depend on
        # different combinations of parameters and data
        # for now we assume this is the same for all elements

        # Find the shape from broadcasting parameters and data
        base_shape = super().shape

        # squeeze last dim if shape is (1,)
        base_shape = () if base_shape == (1,) else base_shape
        shape = base_shape + self.expr.shape

        return shape

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

        lambdas = np.empty(self.expr.shape, dtype=object)
        # todo might go wrong when not all elements have the same parameters
        for i, j in np.ndindex(self.expr.shape):
            lambdas[i, j] = lambdify(self.free_symbols.values(), self.expr[i, j])

        return lambdas

    def forward(self, **kwargs):
        """forward pass with type checking

        or without?
        """
        ...

    def __call__(self, **kwargs: float) -> np.ndarray:
        # https://github.com/sympy/sympy/issues/5642
        # Prepare kwargs for lambdified
        try:
            parameters: dict[str, np.ndarray | float] = {
                k: kwargs[k] for k in self.parameters.keys()
            }
        except KeyError as e:
            raise KeyError(f"Missing value for parameter {e}") from e

        # check shapes
        # this should move somewhere else
        for p_name, p_value in parameters.items():
            if getattr(p_value, "shape", tuple()) != self.parameters[p_name].shape:
                raise ValueError(f"Shape mismatch for parameter {p_name}")

        out = np.empty(self.shape)
        for i, j in np.ndindex(self.expr.shape):
            out[..., i, j] = self.lambdified[i, j](
                **parameters, **self.fixed_parameters, **self.data
            )

        return out

    @property
    def values(self) -> np.ndarray:
        """

        Returns: Array with elements set to parameter values

        """
        raise DeprecationWarning("Deprecate in favour of `value`")
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
        raise NotImplementedError("Nope")
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
        self,
        x: Symbol,
        m: Matrix,
        parameters: Optional[Parameters] = None,
        kind: Optional[str] = None,
        name: Optional[str] = None,
    ):
        raise NotImplementedError("Nope")
        self.x = x
        super().__init__(m, parameters=parameters, kind=kind, name=name)

    @property
    def symbols(self) -> dict[str, Symbol]:
        symbols = super().symbols | {self.x.name: self.x}
        return {s.name: s for s in sorted(symbols.values(), key=str)}


# different class for Symbolic / Numerical ?
class LambdaNumExpr(NumExprBase):
    def __init__(
        self,
        func,
        symbols: Mapping[str, Symbol] | Iterable[Symbol],
        parameters: Optional[Parameters] = None,
        data: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        self.func = func
        if isinstance(symbols, Mapping):
            self.symbols = dict(symbols)
        else:
            self.symbols = {symbol.name: symbol for symbol in symbols}

        super().__init__(parameters, data)

    @property
    def symbols(self):
        return self._symbols

    @symbols.setter
    def symbols(self, value: dict[str, Symbol]):
        self._symbols = value

    def __call__(self, **kwargs):
        return self.func(**self.parse_kwargs(**kwargs), **self.fixed_parameters, **self.data)


class CompositeExpr(SymbolicBase):
    """Can be both numerical or symbolic """

    def __init__(
        self, expr: dict[str | Symbol, NumExprBase | Expr | CompositeExpr],
    ):
        self.expr = expr

    def __call__(self, **kwargs) -> dict[str, np.ndarray]:
        return {expr_name: expr(**kwargs) for expr_name, expr in self.expr.items()}

    def __getitem__(self, item) -> NumExprBase | Expr:
        return self.expr.__getitem__(item)

    @property
    def numerical(self) -> bool:
        return all(isinstance(v, (NumExprBase, CompositeExpr)) for v in self.values())

    def keys(self) -> KeysView[str]:
        return self.expr.keys()

    def values(self) -> ValuesView[NumExprBase, Expr]:
        return self.expr.values()

    def items(self) -> ItemsView[str, NumExprBase, Expr]:
        return self.expr.items()

    def to_numerical(self, parameters: dict[str, Parameter], data: dict[str, np.ndarray]):
        num_expr = {str(k): to_numerical(expr, parameters, data) for k, expr in self.items()}

        instance = self.__class__(num_expr)
        return instance

    @property
    def symbols(self) -> dict[str, Symbol]:
        """Return symbols in the CompositeNumExpr.
        sorting is by dependent_variables, variables, parameters, then by alphabet
        """

        # this fails because `free_symbols` is a dict on NumExpr but `set` on Expr

        symbols = set()
        for rhs in self.values():
            try:
                # rhs is a sympy `Expr` and has `free_symbols` as a set
                symbols |= rhs.free_symbols
            except TypeError:
                # rhs is a slimfit `NumExpr` and has a `free_symbols` dictionary
                symbols |= set(rhs.free_symbols.values())

        return {s.name: s for s in sorted(symbols, key=str)}

    @property
    def parameters(self) -> dict[str, Parameter]:
        """Parameters must be a subset of symbols"""
        if self.numerical:
            return reduce(or_, (expr.parameters for expr in self.values()))
        else:
            return {}

    # @parameters.setter
    # def parameters(self, value: Mapping[str, Parameter]):
    #     for expr in self.values():
    #         expr.parameters = value

    @property
    def data(self) -> dict[str, np.ndarray]:
        return reduce(or_, (expr.data for expr in self.values()))

    @property
    def shape(self) -> Shape:
        shapes = (expr.shape for expr in self.values())
        return np.broadcast_shapes(*shapes)

    # @data.setter
    # def data(self, value: Mapping[str, np.ndarray]):
    #     for expr in self.values():
    #         expr.data = value


class GMM(CompositeExpr):
    # todo can also be implemented as normal NumExpr but with broadcasting parameter shapes
    # important is that GMM class allows users to find oud positions of parmaeters in mu / sigma
    # matrices for EM GMM optimization.

    def __init__(
        self,
        x: Symbol | NumExpr,
        mu: Matrix | MatrixNumExpr,
        sigma: Matrix | MatrixNumExpr,
        name: Optional[str] = None,
    ):
        if mu.shape[0] != 1:
            raise ValueError(
                "GMM parameter matrices must be of shape (1, N) where N is the number of states."
            )

        # todo dont need these references
        self.x = x
        # todo i guess this should stay symbolic until conversion to numerical
        self.mu = mu  # to_numerical(mu, parameters=parameters, data=data)
        self.sigma = sigma  # to_numerical(sigma, parameters=parameters, data=data)

        expr = {"x": x, "mu": mu, "sigma": sigma}

        name = name or "GMM"  # counter for number of instances?
        super().__init__(expr)

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        # from slimfit.functions import gaussian
        x, mu, sig = result["x"], result["mu"], result["sigma"]
        return 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2) / 2)

    def to_numerical(self, parameters: dict[str, Parameter], data: dict[str, np.ndarray]):
        # num_expr = {k: to_numerical(expr, parameters, data) for k, expr in self.items()}

        # todo
        # GMM(self.x, **num_expr)
        instance = GMM(
            to_numerical(self.x, parameters, data),
            to_numerical(self.mu, parameters, data),
            to_numerical(self.sigma, parameters, data),
        )
        return instance

    # @property
    # def symbols(self) ->  dict[str, Symbol]:
    #     symbols = super().symbols | {self.x.name: self.x}
    #
    #     return {symbol_name: symbols[symbol_name] for symbol_name in sorted(symbols)}


class Mul(CompositeExpr):
    def __init__(self):
        ...


class MatMul(CompositeExpr):
    ...


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
        **ivp_kwargs,
    ):
        super().__init__(parameters)
        self.t_var = t_var
        self.trs_matrix = to_numerical(trs_matrix, parameters)
        self.y0 = to_numerical(y0, parameters)
        self.domain = domain

        ivp_defaults = {"method": "Radau"}
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
            args=(trs_matrix,),
            **self.ivp_defaults,
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


# todo behaviour for `None` ?
def to_numerical(
    expression: Union[NumExprBase, Expr, MatrixBase | Model | CompositeExpr],
    parameters: dict[str, Parameter],
    data: dict[str, np.ndarray],
) -> NumExprBase:
    """Converts sympy expression to slimfit numerical expression

        if the expressions already is an NumExpr; the object is modified in-place by setting
        the parameters

    """
    from slimfit.models import Model

    if hasattr(expression, "to_numerical"):
        return expression.to_numerical(parameters, data)
    elif isinstance(expression, Model):
        model_dict = {lhs: to_numerical(rhs) for lhs, rhs in expression.items()}
        from slimfit.models import NumericalModel

        return NumericalModel(model_dict, parameters, data)
    if isinstance(expression, HadamardProduct):
        raise NotImplementedError("Not yet")
        from slimfit.operations import Mul

        return Mul(
            *(to_numerical(arg, parameters) for arg in expression.args), parameters=parameters
        )
    elif isinstance(expression, MatrixBase):
        return MatrixNumExpr(expression, parameters=parameters, data=data)
    elif isinstance(expression, Expr):
        return NumExpr(expression, parameters=parameters, data=data)
    elif isinstance(expression, np.ndarray):
        raise NotImplementedError("Not yet")
        return ArrayCallable(expression, parameters=parameters, data=data)
    elif isinstance(expression, NumExprBase):
        expression.parameters = parameters
        expression.data = data
        return expression
    else:
        raise TypeError(f"Invalid type {type(expression)!r}")
