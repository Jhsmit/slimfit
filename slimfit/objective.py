from __future__ import annotations


from typing import Iterable, Literal

import numpy as np
import numdifftools as nd

from slimfit import Model
from slimfit.loss import Loss
from slimfit.typing import Shape

MIN_PROB = 1e-9  # Minimal probability value (> 0.) to enter into np.log


# py3.10:
# from dataclasses import dataclass, field
# from functools import cached_property
# slots=True,
# kw_only = True
# @dataclass(frozen=True)


class Objective(object):
    """
    Base class for objective functions.

    Args:
        model: Numerical model to call.
        loss: Loss function to use.
        xdata: Input x data.
        ydata: Output y data to calculate loss against.
        negate: Set to `-1` to negate objective return value. Used when minimizers are used for
            maximization problems such as likelihood fitting.
    """

    def __init__(
        self,
        model: Model,
        loss: Loss,
        xdata: dict[str, np.ndarray],
        ydata: dict[str, np.ndarray],
        negate: Literal[1, -1] = 1,
    ):
        self.model = model
        self.loss = loss
        self.xdata = xdata
        self.ydata = ydata
        self.negate = negate


class ScipyObjective(Objective):
    """
    Objective function for use with scipy.optimize.minimize or ScipyMinimizer.

    Args:
        model: Numerical model to call.
        loss: Loss function to use.
        xdata: Input x data.
        ydata: Output y data to calculate loss against.
        shapes: Shapes of the parameters.
        negate: Whether to negate the objective function output (negate for maximization).
    """

    def __init__(
        self,
        model: Model,
        loss: Loss,
        xdata: dict[str, np.ndarray],
        ydata: dict[str, np.ndarray],
        shapes: dict[str, Shape],
        negate: Literal[1, -1] = 1,
    ):
        super().__init__(model=model, loss=loss, xdata=xdata, ydata=ydata, negate=negate)
        self.shapes = shapes

    def __call__(self, x: np.ndarray) -> float:
        """
        Call the objective function.

        Args:
            x: Array of concatenated parameters.

        Returns:
            Objective value.

        """

        parameters = unpack(x, self.shapes)

        y_model = self.model(**parameters, **self.xdata)
        loss_value = self.loss(self.ydata, y_model)

        return self.negate * loss_value

    @property
    def hessian(self) -> Hessian:
        return Hessian(**self.__dict__)


class ScipyEMObjective(Objective):
    def __init__(
        self,
        model: Model,
        loss: Loss,
        xdata: dict[str, np.ndarray],
        posterior: dict[str, np.ndarray],
        shapes: dict[str, Shape],
        negate: Literal[1, -1] = 1,  # TODO: currently this kwarg is not used
    ):
        super().__init__(model=model, loss=loss, xdata=xdata, ydata={}, negate=negate)
        self.posterior = posterior
        self.shapes = shapes

    def __call__(self, x: np.ndarray) -> float:
        parameters = unpack(x, self.shapes)

        probability = self.model(**parameters, **self.xdata)

        # Todo do this in a `loss`
        expectation = {
            lhs: self.posterior[lhs] * np.log(np.clip(prob, a_min=MIN_PROB, a_max=1.0))
            for lhs, prob in probability.items()
        }

        # TODO: LOSS / WEIGHTS

        return -sum(r.sum() for r in expectation.values())


class Hessian(ScipyObjective):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        hess = nd.Hessian(super().__call__)(x)

        return hess


# seperate functions?
def unpack(x: np.ndarray, shapes: dict[str, Shape]) -> dict[str, np.ndarray]:
    """Unpack a ndim 1 array of concatenated parameter values into a dictionary of
    parameter name: parameter_value where parameter values are cast back to their
    specified shapes.
    """
    sizes = [int(np.prod(shape)) for shape in shapes.values()]

    x_split = np.split(x, np.cumsum(sizes))
    p_values = {name: arr.reshape(shape) for (name, shape), arr in zip(shapes.items(), x_split)}

    return p_values


def pack(parameter_values: Iterable[np.ndarray]) -> np.ndarray:
    """Pack a dictionary of parameter_name together as array"""

    return np.concatenate(tuple(param_value.ravel() for param_value in parameter_values))
