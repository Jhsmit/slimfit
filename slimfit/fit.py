from __future__ import annotations

from functools import cached_property
from typing import Optional, Type

import numpy as np
import numpy.typing as npt
from sympy import Expr

from slimfit.loss import L2Loss, LogLoss, Loss
from slimfit.minimizer import ScipyMinimizer, Minimizer
from slimfit.models import Model
from slimfit.numerical import to_numerical
from slimfit.parameter import Parameter, Parameters


class Fit(object):
    """fit objects take model and data

    their function is to determine minimizer / loss strategies and execute those

    """

    def __init__(
        self,
        model: Model,
        parameters: Parameters,
        data: dict[str | Expr, npt.ArrayLike],
        loss: Optional[Loss] = L2Loss(),
        # posterior: Optional[CompositeNumExpr],
    ):

        self.symbolic_model = model
        self.parameters = parameters

        data: dict[str, np.ndarray] = {
            getattr(k, "name", k): np.asarray(v) for k, v in data.items()
        }
        self.loss = loss

        # 'independent' data; or 'xdata'; typically chosen measurement points
        self.xdata = {k: v for k, v in data.items() if k in self.symbolic_model.symbols}

        # 'dependent' data; or 'ydata'; typically measurements
        self.ydata = {k: v for k, v in data.items() if k in self.symbolic_model.dependent_symbols}

        # TODO checking if everything is accounted for
        # except KeyError as k:
        #     raise KeyError(f"Missing independent data: {k}") from k

    @cached_property
    def numerical_model(self) -> Model:
        # TODO parameters type
        return to_numerical(self.symbolic_model, self.parameters, self.xdata)

    def execute(
        self,
            minimizer: Optional[Type[Minimizer]] = None,
            # loss = Optional[Loss] = None
            **execute_options,
    ):

        minimizer_cls = minimizer or self.get_minimizer()
        minimizer_instance = minimizer_cls(
            self.numerical_model, self.loss, self.ydata
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
