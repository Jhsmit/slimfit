from __future__ import annotations

from functools import cached_property
from typing import Optional, Type

import numpy as np
import numpy.typing as npt
from sympy import Expr

from slimfit.fitresult import FitResult
from slimfit.loss import SELoss, LogLoss, Loss
from slimfit.minimizers import ScipyMinimizer, Minimizer
from slimfit.models import Model
from slimfit.numerical import to_numerical
from slimfit.parameter import Parameter, Parameters


class Fit:
    """
    Class for fitting a model to data.

    Args:
        model: Model to fit.
        parameters: List or Parameters object of parameters to fit.
        data: Dictionary of data to fit to. Keys correspond to model symbols.
        loss: Loss function to use. Defaults to L2Loss.

    Attributes:
        xdata: Independent data, typically chosen measurement points.
        ydata: Dependent data, typically measurements.
    """

    def __init__(
        self,
        model: Model,
        parameters: list[Parameter] | Parameters,
        data: dict[str | Expr, npt.ArrayLike],
        loss: Optional[Loss] = SELoss(),
    ) -> None:
        self.model = model

        # make a new instance such that external modification does not affect the
        # copy stored here
        self.parameters = Parameters(parameters)

        # Convert data keys to `str` if given as `Symbol; data values as arrays
        data: dict[str, np.ndarray] = {
            getattr(k, "name", k): np.asarray(v) for k, v in data.items()
        }
        self.loss = loss

        # 'independent' data; or 'xdata'; typically chosen measurement points
        self.xdata = {k: v for k, v in data.items() if k in self.model.symbol_names}

        # 'dependent' data; or 'ydata'; typically measurements
        self.ydata = {k: v for k, v in data.items() if k in self.model.dependent_symbols}

    def execute(
        self,
        minimizer: Optional[Type[Minimizer]] = None,
        **execute_options,
    ) -> FitResult:
        """
        Execute the fit.

        Args:
            minimizer: Optional minimizer to use. Defaults to ScipyMinimizer.
            **execute_options:

        Returns:
            Result of the fit as FitResult object.

        """

        minimizer_cls = minimizer or self.get_minimizer()
        minimizer_instance = minimizer_cls(
            self.model, self.parameters, self.loss, self.xdata, self.ydata
        )

        result = minimizer_instance.execute(**execute_options)

        # result.model = self.model

        return result

    def get_minimizer(self) -> Type[Minimizer]:
        """Automatically determine which minimizer to use"""
        return ScipyMinimizer

    def get_loss(self, **kwargs):
        raise NotImplementedError()
        if self.model.probabilistic:
            return LogLoss(**kwargs)
        else:
            return SELoss(**kwargs)
