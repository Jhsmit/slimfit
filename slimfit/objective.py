from __future__ import annotations


from typing import Iterable

import numpy as np

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


class Objective:
    def __init__(
        self,
        model: Model,
        loss: Loss,
        xdata: dict[str, np.ndarray],
        ydata: dict[str, np.ndarray],
        negate: bool = False,
    ):
        self.model = model
        self.loss = loss
        self.xdata = xdata
        self.ydata = ydata

        self.sign = -1 if negate else 1


class ScipyObjective(Objective):
    def __init__(
        self,
        model: Model,
        loss: Loss,
        xdata: dict[str, np.ndarray],
        ydata: dict[str, np.ndarray],
        shapes: dict[str, Shape],
        negate: bool = False,
    ):
        super().__init__(model=model, loss=loss, xdata=xdata, ydata=ydata, negate=negate)
        self.shapes = shapes

    def __call__(self, x: np.ndarray) -> float:
        parameters = unpack(x, self.shapes)

        y_model = self.model(**parameters, **self.xdata)
        loss = self.loss(self.ydata, y_model)

        return self.sign * loss


class ScipyEMObjective(Objective):
    def __init__(
        self,
        model: Model,
        loss: Loss,
        xdata: dict[str, np.ndarray],
        posterior: dict[str, np.ndarray],
        shapes: dict[str, Shape],
        negate: bool = False,  # todo actually use the negate bool
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


# seperate functions?
def unpack(x: np.ndarray, shapes: dict[str, Shape]) -> dict[str, np.ndarray]:
    """Unpack a ndim 1 array of concatenated parameter values into a dictionary of
        parameter name: parameter_value where parameter values are cast back to their
        specified shapes.
    """
    sizes = [int(np.product(shape)) for shape in shapes.values()]

    x_split = np.split(x, np.cumsum(sizes))
    p_values = {name: arr.reshape(shape) for (name, shape), arr in zip(shapes.items(), x_split)}

    return p_values


def pack(parameter_values: Iterable[np.ndarray]) -> np.ndarray:
    """Pack a dictionary of parameter_name together as array"""

    return np.concatenate(tuple(param_value.ravel() for param_value in parameter_values))
