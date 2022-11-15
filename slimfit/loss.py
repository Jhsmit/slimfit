from __future__ import annotations

import abc
from typing import Protocol, Optional, Literal

import numpy as np
import numpy.typing as npt

from slimfit.reduce import (
    ReductionStrategy,
    mean_reduction,
    sum_reduction,
    concat_reduction,
)


class Loss(object):
    def __init__(
        self,
        weights: Optional[dict[str, npt.ArrayLike]] = None,
        reduction: Literal["mean", "sum", "concat", "none", None] = "mean",
    ):

        self.weights = weights
        if reduction == "mean":
            self.reduce: ReductionStrategy = mean_reduction
        elif reduction == "sum":
            self.reduce: ReductionStrategy = sum_reduction
        elif reduction == "concat":
            self.reduce: ReductionStrategy = concat_reduction
        elif reduction in [None, "none"]:
            self.reduce = lambda x: x

    @abc.abstractmethod
    def __call__(
        self, dependent_data: dict[str, np.ndarray], target_data: dict[str, np.ndarray]
    ) -> np.ndarray | float:
        ...

    # def reduce(
    #     self, residuals: dict[str, np.ndarray]
    # ) -> float | np.ndarray | dict[str, np.ndarray]:
    #     if self.reduction == "mean":
    #         size = sum(arr.size for arr in residuals.values())
    #         total = sum(r.sum() for r in residuals.values())
    #         return total / size
    #     elif self.reduction == "sum":
    #         return sum(r.sum() for r in residuals.values())
    #     elif self.reduction == "concat":
    #         return np.concatenate([a.ravel() for a in residuals.values()])
    #     elif self.reduction in ["none", None]:
    #         return residuals


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
# https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
class L1Loss(Loss):
    """L1 loss """

    def __call__(
        self, dependent_data: dict[str, np.ndarray], target_data: dict[str, np.ndarray]
    ) -> np.ndarray | float:

        if self.weights is None:
            residuals = {k: (target_data[k] - dependent_data[k]) for k in target_data.keys()}
        else:
            residuals = {
                k: (target_data[k] - dependent_data[k]) * self.weights[k]
                for k in target_data.keys()
            }

        return self.reduce(residuals)


class L2Loss(Loss):
    def __call__(
        self, dependent_data: dict[str, np.ndarray], target_data: dict[str, np.ndarray]
    ) -> np.ndarray | float:

        if self.weights is None:
            residuals = {k: (target_data[k] - dependent_data[k]) ** 2 for k in target_data.keys()}
        else:
            residuals = {
                k: ((target_data[k] - dependent_data[k]) * self.weights[k]) ** 2
                for k in target_data.keys()
            }

        return self.reduce(residuals)


# TODO base class for probability-based losses which only take target data in their __call__ (= MLELoss ? )
# then the base should take weights and only do reduction; subclasses to logs / sums etc


class LogLoss(Loss):
    """Takes the elementwise logarithm of predicted input data

    Used in combination with maximum likelihood methods

    returns negative of the reductions are use in combination with minimizers rather than maximizers
    """

    def __init__(
        self,
        weights: Optional[dict[str, npt.ArrayLike]] = None,
        reduction: Literal["mean", "sum", "concat"] = "sum",
    ):
        if reduction not in ["mean", "sum", "concat"]:
            raise ValueError(
                f"LogLoss does not support reduction {reduction!r}, only 'mean', 'sum', 'concat'"
            )
        super().__init__(weights, reduction)

    def __call__(
        self, dependent_data: dict[str, np.ndarray], target_data: dict[str, np.ndarray]
    ) -> np.ndarray | float:
        if self.weights is None:
            log_vals = {k: np.log(target_data[k]) for k in target_data.keys()}

        else:
            log_vals = {k: np.log(target_data[k] * self.weights[k]) for k in target_data.keys()}

        return -self.reduce(log_vals)


class LogSumLoss(Loss):
    """Sums by specified axis, then takes elementwise log

    Used in combination with maximum likelihood methods

    returns negative of the reductions are use in combination with minimizers rather than maximizers
    """

    def __init__(
        self,
        weights: Optional[dict[str, npt.ArrayLike]] = None,
        sum_axis: Optional[int] = 1,
        reduction: Literal["mean", "sum", "concat"] = "sum",
    ):
        self.sum_axis = sum_axis
        if reduction not in ["mean", "sum", "concat"]:
            raise ValueError(
                f"LogSumLoss does not support reduction {reduction!r}, only 'mean', 'sum', 'concat'"
            )
        super().__init__(weights, reduction)

    def __call__(
        self, dependent_data: dict[str, np.ndarray], target_data: dict[str, np.ndarray]
    ) -> np.ndarray | float:
        if self.weights is None:
            log_vals = {
                k: np.log(target_data[k].sum(axis=self.sum_axis)) for k in target_data.keys()
            }

        else:
            log_vals = {
                k: np.log(target_data[k].sum(axis=self.sum_axis)) * self.weights[k]
                for k in target_data.keys()
            }

        return -self.reduce(log_vals)


#
#
# class LogLoss(Loss):
#     """Takes the elementwise logarithm of predicted input data
#
#     Used in combination with maximum likelihood methods
#
#     returns negative of the reductions are use in combination with minimizers rather than maximizers
#     """
#
#     def __init__(
#         self,
#             sum_axis: int,
#             weights: Optional[dict[str, npt.ArrayLike]] = None,
#     ):
#         if reduction not in ["mean", "sum", "concat"]:
#             raise ValueError(
#                 f"LogLoss does not support reduction {reduction!r}, only 'mean', 'sum', 'concat'"
#             )
#         super().__init__(weights, reduction)
#
#     def __call__(
#         self, dependent_data: dict[str, np.ndarray], target_data: dict[str, np.ndarray]
#     ) -> np.ndarray | float:
#         if self.weights is None:
#             log_vals = {k: np.log(target_data[k]) for k in target_data.keys()}
#
#         else:
#             log_vals = {
#                 k: np.log(target_data[k] * self.weights[k]) for k in target_data.keys()
#             }
#
#         return -self.reduce(log_vals)
