from __future__ import annotations

from typing import Optional, Type

import numpy as np
import numpy.typing as npt
from sympy import Expr, MatrixBase

from slimfit.loss import L2Loss, LogLoss, Loss
from slimfit.minimizer import ScipyMinimizer, Minimizer
from slimfit.models import Model
from slimfit.symbols import Variable, Probability
from slimfit.callable import CallableBase


class Fit(object):
    """fit objects take model and data

    their function is to determine minimizer / loss strategies and execute those

    """

    def __init__(
        self,
        model: dict[Variable | Probability, Expr | CallableBase | MatrixBase] | Model,
        **data: npt.ArrayLike,
    ):

        if isinstance(model, dict):
            self.model: Model = Model(model)
        elif isinstance(model, Model):
            self.model: Model = model

        try:
            self.dependent_data: dict[str, np.ndarray] = {
                k: np.asarray(data[k]) for k in self.model.dependent_variables.keys()
            }
        except KeyError as k:
            raise KeyError(f"Missing dependent data: {k}") from k

        try:
            self.independent_data: dict[str, np.ndarray] = {
                k: np.asarray(data[k]) for k in self.model.independent_variables.keys()
            }
        except KeyError as k:
            raise KeyError(f"Missing independent data: {k}") from k

    def execute(
        self,
        guess: Optional[dict[str, float]] = None,
        weights: Optional[dict[str, npt.ArrayLike]] = None,
        minimizer: Optional[Type[Minimizer]] = None,
        loss: Optional[Loss] = None,
        **execute_options,
    ):

        self.model.renew()  # refresh lambdified cached properties
        loss = loss or self.get_loss(weights=weights)
        minimizer_cls = minimizer or self.get_minimizer()
        minimizer_instance = minimizer_cls(
            self.model, self.independent_data, self.dependent_data, loss, guess
        )

        result = minimizer_instance.execute(**execute_options)

        return result

    def get_minimizer(self) -> Type[Minimizer]:
        """Automatically determine which minimizer to use"""
        return ScipyMinimizer

    def get_loss(self, **kwargs):
        if self.model.probabilistic:
            return LogLoss(**kwargs)
        else:
            return L2Loss(**kwargs)
