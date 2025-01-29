from typing import Optional

from slimfit.reduce import mean_reduction
from slimfit.v2.model import Model


class SELoss:
    """sum/average reduction"""


class MSELoss(SELoss):
    def __init__(self, model: Model, y_data: dict, weights: Optional[dict] = None):
        self.model = model
        self.y_data = y_data
        self.weights = weights or {}

    def __call__(self, **kwargs):
        y_model = self.model(**kwargs)

        residuals = {
            k: ((y_model[k] - self.y_data[k]) * self.weights.get(k, 1)) ** 2
            for k in self.y_data.keys()
        }

        return mean_reduction(residuals)
