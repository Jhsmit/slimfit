from __future__ import annotations

import abc
from _operator import or_
from functools import cached_property, reduce
from typing import Optional, KeysView, ValuesView, ItemsView, Callable

from sympy import Symbol, Expr, MatrixBase, HadamardProduct
import numpy as np

from slimfit.parameter import Parameter, Parameters
from slimfit.typing import Shape


class SymbolicBase(metaclass=abc.ABCMeta):
    @cached_property
    @abc.abstractmethod
    def symbols(self) -> set[Symbol]:
        ...

    @cached_property
    def symbol_names(self) -> set[str]:
        return set(s.name for s in self.symbols)

    @property
    def shapes(self) -> dict[str, Shape]:
        """
        dict of symbol shapes
        """

    def filter_parameters(self, parameters: Parameters) -> Parameters:
        """Filters a list of parameters, returning only the ones whose symbols are
        in this model
        """
        return Parameters([p for p in parameters if p.symbol in self.symbols])

    @property
    def T(self):
        return FuncExpr(self, func=np.transpose)

    def __getitem__(self, item):
        from slimfit.operations import Indexer

        return Indexer(self, item)


# refactor to Hybrid ?
class CompositeExpr(SymbolicBase):
    """"""

    def __init__(
        self,
        expr: dict[str | NumExprBase | Expr | CompositeExpr | MatrixBase | np.ndarray, float],
    ):
        if not isinstance(expr, dict):
            raise TypeError(f"{self.__class__.__name__} must be initialized with a dict.")
        # for v in expr.values():
        #     if not isinstance(v, (NumExprBase, Expr, CompositeExpr, MatrixBase)):
        #         raise TypeError(f"Invalid type in expr dict: {v!r}.")

        self.expr = expr

        if self.is_numerical():
            self._call = self._numerical_call
        else:
            self._call = self._symbolic_call

    def _numerical_call(self, **kwargs):
        return {expr_name: expr(**kwargs) for expr_name, expr in self.expr.items()}

    def _symbolic_call(self, **kwargs):
        return self.numerical._call(**kwargs)

    def __call__(self, **kwargs) -> dict[str, np.ndarray]:
        return self._call(**kwargs)

    def __getitem__(self, item) -> NumExprBase | Expr:
        if isinstance(item, str):
            return self.expr.__getitem__(item)
        else:
            return super().__getitem__(item)

    def is_numerical(self) -> bool:
        """Returns `True` if all expressions are numerical expressions."""
        for v in self.values():
            # this should check for all base (non-composite) classes which are allowed and
            # can be converted to numerical
            # todo list this globally and check for this at init time
            if isinstance(v, (Expr, MatrixBase, HadamardProduct, np.ndarray)):
                return False
            if isinstance(v, CompositeExpr):
                # recursively check if composite parts are numerical
                return v.is_numerical()
        return True

    @cached_property
    def numerical(self) -> Optional[CompositeExpr]:
        if self.is_numerical():
            return self
        else:
            return self.to_numerical()

    def keys(self) -> KeysView[str]:
        return self.expr.keys()

    def values(self) -> ValuesView[NumExprBase, Expr]:
        return self.expr.values()

    def items(self) -> ItemsView[str, NumExprBase, Expr]:
        return self.expr.items()

    def to_numerical(self):
        from slimfit.numerical import to_numerical

        num_expr = {str(k): to_numerical(expr) for k, expr in self.items()}

        # TODO **unpack
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
            raise KeyError(f"Missing value for {e}") from e

        return arguments


class CompositeArgsExpr(CompositeExpr):
    """Composite expr which takes *args to init rather than dictionary of expressions"""

    def __init__(self, *args, **kwargs):
        expr = {i: arg for i, arg in enumerate(args)}
        self.kwargs = kwargs
        super().__init__(expr)

    def to_numerical(self):
        from slimfit.numerical import to_numerical

        args = (to_numerical(expr) for expr in self.values())
        instance = self.__class__(*args, **self.kwargs)

        return instance


class CompositeArgExpr(CompositeExpr):
    """Composite expr which single args to init plus additional kwargs"""

    def __init__(self, arg, **kwargs):
        expr = {0: arg}
        self.kwargs = kwargs
        super().__init__(expr)

    def to_numerical(self):
        from slimfit.numerical import to_numerical

        arg = to_numerical(self.expr[0])
        instance = self.__class__(arg, **self.kwargs)

        return instance


class FuncExpr(CompositeArgsExpr):
    def __init__(self, *args, func: Callable, **kwargs):
        super().__init__(*args, func=func, **kwargs)

    def __call__(self, **kwargs):
        result = self._call(**kwargs)
        func = self.kwargs["func"]
        return func(*result.values(), **{k: v for k, v in self.kwargs.items() if k != "func"})

    def __repr__(self) -> str:
        kwds = self.kwargs.copy()
        func = kwds.pop("func")
        arg_rpr = (f"{arg!r}" for arg in self.values())
        kw_rpr = (f"{k}={v!r}" for k, v in kwds.items())

        if func.__module__ == "numpy":
            mod = "np."
        else:
            mod = ""

        return f"{mod}{func.__name__}({', '.join(arg_rpr)}, {', '.join(kw_rpr)})"
