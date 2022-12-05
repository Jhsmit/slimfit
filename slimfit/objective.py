from __future__ import annotations


from typing import Iterable

import numpy as np

from slimfit import Model
from slimfit.loss import Loss
from slimfit.typing import Shape

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


# class Objective:
#     model: Model
#     loss: Loss
#     xdata: dict[str, np.ndarray]
#     ydata: dict[str, np.ndarray]
#     negate: bool = False
#
#    # sign: int = field(default=1, init=False)
#
#     def __post_init__(self):
#         if not self.model.numerical:
#             raise ValueError("Objective models must be numerical")
#
#         #self.sign = -1 if self.negate else 1


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
        parameters = self.unpack(x)

        y_model = self.model(**parameters, **self.xdata)
        loss = self.loss(self.ydata, y_model)

        return self.sign * loss

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Unpack a ndim 1 array of concatenated parameter values into a dictionary of
            parameter name: parameter_value where parameter values are cast back to their
            specified shapes.
        """
        sizes = [int(np.product(shape)) for shape in self.shapes.values()]

        x_split = np.split(x, np.cumsum(sizes))
        p_values = {
            name: arr.reshape(shape) for (name, shape), arr in zip(self.shapes.items(), x_split)
        }

        return p_values

    def pack(self, parameter_values: Iterable[np.ndarray]) -> np.ndarray:
        """Pack a dictionary of parameter_name together as array"""

        return np.concatenate(tuple(param_value.ravel() for param_value in parameter_values))
