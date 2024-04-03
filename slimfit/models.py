from __future__ import annotations


from sympy import Expr, MatrixBase, Symbol

import slimfit.base
import slimfit.numerical as numerical
import numpy.typing as npt
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from slimfit.parameter import Parameters


class Model(slimfit.base.CompositeExpr):
    def __init__(
        self,
        expr: dict[Symbol | str, Expr | slimfit.base.NumExprBase | MatrixBase],
    ):
        # currently typing has a small problem where keys are expected to be `str`, not symbol
        super().__init__(expr)

    def __repr__(self):
        return f"Model({self.expr.__repr__()})"

    @property
    def dependent_symbols(self) -> dict[str, Symbol]:
        # todo needs to be updated
        """Variables corresponding to dependent (measured) data, given as keys in the model dict"""

        return {symbol.name: symbol for symbol in self.expr.keys()}

    @property
    def components(self) -> dict[str, slimfit.base.NumExprBase]:
        """all NumExprBase components in the model
        keys should be such that their commectivity can be reconstructed ?
        ie {Mul[0]MatMul[1]: <component> ...}
        which tells us that this component is the second element of a matmul whose result
        is the first component in a mul
        """
        raise NotImplementedError("not yet implemented")

        # return {symbol.name: expr for symbol, expr in self.expr.items()}

    def define_parameters(
        self,
        parameters: dict[str, npt.ArrayLike] | Iterable[str] | str = "*",
    ) -> Parameters:
        """
        Defines and initializes parameters for the model.

        This method accepts parameters in various forms (dictionary, iterable, or string)
        and returns an instance of the Parameters class, initialized with the provided
        parameters and the existing symbols of the model. Default value is '*', which
        returns all the model's symbols as parameters.

        Args:
            parameters:
            The parameters to define for the model. Can be a dictionary with parameter
            names as keys and corresponding values, an iterable of parameter names, or a
            single parameter name as a string.

        Returns:
            Parameters: An instance of the Parameters class, initialized with the provided
            parameters and the existing symbols of the model.

        Usage:
            Assuming we have a model instance 'm' and we want to define the symbols 'a' and 'b'
            are parameters:

            ```python
            defined_parameters = m.define_parameters("a b")
            ```

            Use a dictionary to define parameters and provide initial guesses:
            ```python
            guess = {'a': 3., 'b': 10}
            defined_parameters = m.define_parameters(guess)
            ```

        """
        from slimfit.parameter import Parameters
        parameters = Parameters.from_symbols(self.symbols, parameters)

        return parameters


class Eval(slimfit.base.CompositeExpr):
    def __init__(self, expr: Expr | slimfit.base.NumExprBase | MatrixBase):
        super().__init__({"_y": expr})

    def __call__(self, **kwargs):
        ans = super().__call__(**kwargs)
        return ans["_y"]

    def __repr__(self) -> str:
        return f"Eval({self.expr['_y'].__repr__()})"

    def to_numerical(self):
        args = (numerical.to_numerical(expr) for expr in self.values())
        instance = self.__class__(*args)

        return instance
