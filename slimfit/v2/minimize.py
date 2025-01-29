from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from slimfit.fitresult import FitResult
from slimfit.objective import pack, unpack
from slimfit.v2.loss import MSELoss
from slimfit.v2.parameter import Parameters


class Minimize:  # = currently only scipy minimize
    def __init__(self, loss: MSELoss, parameters: Parameters, xdata: dict[str, np.ndarray]):
        self.loss = loss
        self.parameters = parameters
        self.xdata = xdata
        self.shapes = {p.name: p.shape for p in self.parameters}

    def func(self, x: np.ndarray):
        parameters = unpack(x, self.shapes)
        return self.loss(**parameters, **self.xdata)

    @property
    def free_parameters(self):
        return [p for p in self.parameters if not p.fixed]

    @property
    def fixed_parameters(self):
        return [p for p in self.parameters if p.fixed]

    def get_bounds(self) -> list[tuple[float | None, float | None]] | None:
        bounds = []
        for p in self.free_parameters:
            size = np.prod(p.shape, dtype=int)
            bounds += [p.bounds] * size

        if all((None, None) == b for b in bounds):
            return None
        else:
            return bounds

    def fit(self):
        x = pack(self.parameters.free.guess.values())
        result = minimize(self.func, x, bounds=self.get_bounds())

        gof_qualifiers = {
            "loss": result["fun"],
        }

        return FitResult(
            fit_parameters=unpack(result.x, self.shapes),
            gof_qualifiers=gof_qualifiers,
            fixed_parameters=self.parameters.fixed.guess,
            guess=self.parameters.free.guess,
            base_result=result,
        )
