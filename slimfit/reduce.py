from __future__ import annotations

from typing import Callable, Union

import numpy as np

ReductionStrategy = Callable[[dict[str, np.array]], Union[float, np.array]]


def mean_reduction(residuals: dict[str, np.array]) -> float:
    size = sum(arr.size for arr in residuals.values())
    total = sum(r.sum() for r in residuals.values())
    return total / size


def sum_reduction(residuals: dict[str, np.array]) -> float:
    return sum(r.sum() for r in residuals.values())


def concat_reduction(residuals: dict[str, np.array]) -> np.array:
    return np.concatenate([a.ravel() for a in residuals.values()])
