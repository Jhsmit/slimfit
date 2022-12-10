from __future__ import annotations

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
from slimfit.typing import Shape

if TYPE_CHECKING:
    from slimfit import Model

# @dataclass
class NumExprBase(SymbolicBase):
    """Symbolic expression which allows calling cached lambified expressions
    subclasses must implement `symbols` attribute / property
    """

    # def __init__(
    #     self,
    # ):
    #
    #     # Accepted parameters are a subset of `symbols`
    #     # #todo property with getter / setter where setter filters parameters?
    #     # self.parameters = Parameters({name: p for name, p in parameters.items() if name in self.symbols})

    @property
    def shape(self) -> Shape:
        shapes = self.shapes.values()

        return np.broadcast_shapes(*shapes)

    def parse_kwargs(self, **kwargs) -> dict[str, np.ndarray]:
        """Parse kwargs and take only the ones in `free_parameters`"""
        try:
            arguments: dict[str, np.ndarray | float] = {k: kwargs[k] for k in self.symbol_names}
        except KeyError as e:
            print(kwargs.keys())
            raise KeyError(f"Missing value for parameter {e}") from e

        return arguments


class DummyNumExpr(NumExprBase):
    """Dummy callable object which returns supplied 'obj' when called
    Has no parameters or symbols
    """

    def __init__(self, obj: Any, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj = obj

    def __call__(self, **kwargs):
        return self.obj

    @property
    def symbols(self) -> set[Symbol]:
        return set()


class ArrayNumExpr(DummyNumExpr):
    @property
    def shape(self) -> Shape:
        return self.obj.shape


# TODO frozen dataclass?
class NumExpr(NumExprBase):
    def __init__(
        self, expr: Expr,
    ):
        if not isinstance(expr, (Expr, MatrixBase)):
            # TODO subclass such that typing is correct
            raise TypeError(f"Expression must be an instance of `Expr` or ")
        self.expr = expr

        # super().__init__()

    @property
    def name(self) -> str:
        # if _name: ... -> superclass
        if isinstance(self.expr, Symbol):
            return self.expr.name

    # is same as callablematrix
    @property
    def symbols(self) -> set[Symbol]:
        return self.expr.free_symbols

    @cached_property
    def lambdified(self) -> Callable:
        ld = lambdify(sorted(self.symbols, key=str), self.expr)

        return ld

    def __call__(self, **kwargs: float) -> np.ndarray | float:

        # try:
        #     parameters: dict[str, np.ndarray | float] = {
        #         k: kwargs[k] for k in self.parameters.keys()
        #     }
        # except KeyError as e:
        #     raise KeyError(f"Missing value for parameter {e}") from e

        val = self.lambdified(**self.parse_kwargs(**kwargs))
        return val

    def __repr__(self):
        return f"NumExpr({self.expr})"


# todo name via kwargs to super
# = composite num expr"?
class MatrixNumExpr(NumExpr):
    def __init__(
        self, expr: MatrixBase, name: Optional[str] = None, kind: Optional[str] = None,
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

        super().__init__(expr)

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        if self.kind == "constant":
            symbol_names = [str(symbol) for symbol in self.expr]
            prefix = [name.split("_")[0] for name in symbol_names]
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
    def element_mapping(self) -> dict[Symbol, tuple[int, int]]:
        """
        Dictionary mapping Symbol to matrix indices
        only returns entries if the matrix element is equal to a `Symbol`, not `Expr`
        """

        element_mapping = {}
        for i, j in np.ndindex(self.expr.shape):
            elem = self.expr[i, j]
            if isinstance(elem, Symbol):
                element_mapping[elem] = (i, j)
        return element_mapping

    @cached_property
    def lambdified(self) -> np.ndarray:
        """Array of lambdified function per matrix element"""
        # TODO scalercallable per element

        lambdas = np.empty(self.expr.shape, dtype=object)
        # todo might go wrong when not all elements have the same parameters
        for i, j in np.ndindex(self.expr.shape):
            lambdas[i, j] = lambdify(sorted(self.symbols, key=str), self.expr[i, j])

        return lambdas

    def forward(self, **kwargs):
        """forward pass with type checking

        or without?
        """
        ...

    def __call__(self, **kwargs: float) -> np.ndarray:
        # https://github.com/sympy/sympy/issues/5642
        # Prepare kwargs for lambdified
        # try:
        #     parameters: dict[str, np.ndarray | float] = {
        #         k: kwargs[k] for k in self.free_parameters.keys()
        #     }
        # except KeyError as e:
        #     raise KeyError(f"Missing value for parameter {e}") from e

        # check shapes
        # this should move somewhere else
        # for p_name, p_value in parameters.items():
        #     if getattr(p_value, "shape", tuple()) != self.parameters[p_name].shape:
        #         raise ValueError(f"Shape mismatch for parameter {p_name}")

        ld_kwargs = self.parse_kwargs(**kwargs)

        # todo precomputed shapes
        # try:
        #     # Shape is given be pre-specified shape
        #     shape = self.shape
        # except AttributeError:
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
        names = sorted(self.symbol_names)
        return f"{self.name}({', '.join(names)})"

    def index(self, symbol: Symbol) -> tuple[int, int]:
        """
        Returns indices of parameter for Matrix Expressions

        Args:
            name: Parameter name to find matrix elements of

        Returns: Tuple of matrix elements ij

        """

        return self.element_mapping[symbol]


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
        self, x: Symbol, m: Matrix, kind: Optional[str] = None, name: Optional[str] = None,
    ):
        raise NotImplementedError("Not implemented")
        self.x = x
        super().__init__(m, kind=kind, name=name)

    @property
    def symbols(self) -> dict[str, Symbol]:
        symbols = super().symbols | {self.x}
        return symbols


# different class for Symbolic / Numerical ?
class LambdaNumExpr(NumExprBase):
    def __init__(self, func, symbols: Iterable[Symbol],) -> None:
        self.func = func
        self._symbols = set(symbols)

    @property
    def symbols(self):
        return self._symbols

    def __call__(self, **kwargs):
        return self.func(**self.parse_kwargs(**kwargs))


# refactor to Hybrid ?
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

    def to_numerical(self):
        num_expr = {str(k): to_numerical(expr) for k, expr in self.items()}

        instance = self.__class__(num_expr)
        return instance

    @cached_property
    def symbols(self) -> set[Symbol]:
        """Return symbols in the CompositeNumExpr.
        sorting is by dependent_variables, variables, parameters, then by alphabet
        """

        # this fails because `free_symbols` is a dict on NumExpr but `set` on Expr

        symbols = set()
        for rhs in self.values():
            if isinstance(rhs, (Expr, MatrixBase)):
                symbols |= rhs.free_symbols
            else:
                try:
                    symbols |= set(rhs.symbols)
                except AttributeError:
                    # RHS doesnt have any symbols; for example might be a numpy array
                    pass

            # symbols = getattr(rhs, 'free_symbols')
            # try:
            #     # rhs is a sympy `Expr` and has `free_symbols` as a set
            #     symbols |= rhs.free_symbols
            # except TypeError:
            #     # rhs is a slimfit `NumExpr`
            #     symbols |= set(rhs.symbols)
            # except AttributeError:
            #     # RHS doesnt have any symbols; for example might be a numpy array
            #     pass
        return symbols

    @property
    def shapes(self) -> dict[str, Shape]:
        """shapes of symbols"""
        return reduce(or_(expr.shapes for expr in self.expr.values()))

    @property
    def shape(self) -> Shape:
        """
        Base class shape is obtained from broadcasting all expressing values together
        """
        shapes = (expr.shape for expr in self.values())
        return np.broadcast_shapes(*shapes)


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
        # check for the correct shape when creating from sympy Matrix
        if isinstance(mu, Matrix) and mu.shape[0] != 1:
            raise ValueError(
                "GMM parameter matrices must be of shape (1, N) where N is the number of states."
            )

        expr = {"x": x, "mu": mu, "sigma": sigma}

        # todo some kind of properties object for this metadata
        name = name or "GMM"  # counter for number of instances?
        self.kind = "gmm"
        super().__init__(expr)

    def __call__(self, **kwargs) -> np.ndarray:
        result = super().__call__(**kwargs)

        x, mu, sig = result["x"], result["mu"], result["sigma"]
        output = 1 / (np.sqrt(2 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2) / 2)

        # Output shape is (datapoints, states, 1) to match output shape of matrix
        # exponentiation model
        return np.expand_dims(output, -1)

    def __repr__(self):
        return f"GMM({self.expr['x']}, {self.expr['mu']}, {self.expr['sigma']})"

    def to_numerical(self) -> GMM:
        # todo probably this is the same as the super class method
        num_expr = {k: to_numerical(expr) for k, expr in self.items()}
        instance = GMM(**num_expr)

        return instance

    @property
    def shape(self) -> Shape:
        shape = super().shape
        return shape + (1,)

    # TODO GMM should have API to find the index of the state given a symbol
    #def state_index(self, ...):

class MarkovIVP(CompositeExpr):
    """Uses scipy.integrate.solve_ivp to numerically find time evolution of a markov process
        given a transition rate matrix.

    Returned shape is (len(t_var), len(y0), 1), or (<datapoints>, <states>, 1)

    """

    def __init__(
        self,
        t: Symbol | NumExpr,
        trs_matrix: Matrix | MatrixNumExpr,
        y0: Matrix | MatrixNumExpr,
        domain: Optional[tuple[float, float]] = None,
        **ivp_kwargs,
    ):

        expr = {"t": t, "trs_matrix": trs_matrix, "y0": y0}

        super().__init__(expr)

        ivp_defaults = {"method": "Radau"}
        self.ivp_defaults = ivp_defaults | ivp_kwargs
        self.domain = domain

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)

        # if `self['t']` does not depend on any parameters; domain can be precomputed and
        # does not have to be determined for every call
        # although its every fast to do so
        domain = self.domain or self.get_domain(result["t"])
        sol = solve_ivp(
            self.grad_func,
            domain,
            y0=result["y0"].squeeze(),
            t_eval=result["t"],
            args=(result["trs_matrix"],),
            **self.ivp_defaults,
        )

        # shape is modified to match the output shape of matrix exponentiation model
        # exp(m*t) @ y0, which is (datapoints, states, 1)
        return np.expand_dims(sol.y.T, -1)

    def to_numerical(self) -> MarkovIVP:
        num_expr = {k: to_numerical(expr) for k, expr in self.items()}
        instance = MarkovIVP(**num_expr, domain=self.domain, **self.ivp_defaults)

        return instance

    def get_domain(self, arr: np.ndarray) -> tuple[float, float]:
        # padding?
        return arr[0], arr[-1]

    @staticmethod
    def grad_func(t, y, trs_matrix):
        return trs_matrix @ y


def identify_expression_kind(sympy_expression: Union[Expr, MatrixBase]) -> str:
    """Find the type of expression

    Only implemented for 'constant' kind, otherwise return is generic

    """

    # check for gaussian mixture model ...

    if isinstance(sympy_expression, MatrixBase):
        ...

        if all(isinstance(elem, Symbol) for elem in sympy_expression):
            # this should also check the symbols for their shapes, for it to be constant
            # the elements should all be scalars
            return "constant"

    return "generic"


def to_numerical(
    expression: Union[NumExprBase, Expr, MatrixBase | Model | CompositeExpr],
) -> NumExprBase | CompositeExpr:
    """Converts sympy expression to slimfit numerical expression

        if the expressions already is an NumExpr; the object is modified in-place by setting
        the parameters

    """
    from slimfit.models import Model

    if hasattr(expression, "to_numerical"):
        return expression.to_numerical()
    # elif isinstance(expression, Model):
    #     model_dict = {lhs: to_numerical(rhs) for lhs, rhs in expression.items()}
    #     from slimfit.models import NumericalModel
    #
    #     return NumericalModel(model_dict, parameters, data)
    if isinstance(expression, HadamardProduct):
        from slimfit.operations import Mul

        return Mul(*(to_numerical(arg) for arg in expression.args))
    elif isinstance(expression, MatrixBase):
        return MatrixNumExpr(expression)
    elif isinstance(expression, Expr):
        return NumExpr(expression)
    elif isinstance(expression, np.ndarray):
        return ArrayNumExpr(expression)
    elif isinstance(expression, NumExprBase):
        return expression
    else:
        raise TypeError(f"Invalid type {type(expression)!r}")
