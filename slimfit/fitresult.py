from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Optional, Any, Union

import numpy as np
import numdifftools as nd
import yaml

from slimfit import Model
from slimfit.loss import Loss
from slimfit.objective import Objective, pack, unpack
from slimfit.utils import clean_types


@dataclass(frozen=True)
class FitResult:
    """
    Fit result object.
    """

    fit_parameters: dict[str, float | np.ndarray]
    """Fitted parameter values"""

    gof_qualifiers: dict
    """Goodness-of-fit qualifiers"""

    fixed_parameters: dict[str, float] = field(default_factory=dict)
    """Values of the model's fixed parameters"""

    guess: Optional[dict] = None
    """Initial guesses"""

    model: Optional[Model] = None
    """The fitted model"""

    objective: Optional[Objective] = None

    loss: Optional[Loss] = None

    data: Optional[dict] = field(default=None, repr=False)
    """Data on which the fit was performed"""

    metadata: dict = field(default_factory=dict)
    """Additional metadata"""

    base_result: Optional[Any] = field(default=None, repr=False)
    """Source fit result object. Can be dicts of sub results"""

    def __post_init__(self) -> None:
        if "datetime" not in self.metadata:
            now = datetime.now()
            self.metadata["datetime"] = now.strftime("%Y/%m/%d %H:%M:%S")
            self.metadata["timestamp"] = int(now.timestamp())

    def __str__(self):
        s = ""
        try:
            stdev = self.stdev
        except ValueError:
            stdev = {}

        p_size = max(len(k) for k in self.fit_parameters)
        if stdev:
            s += f"{'Parameter':<{p_size}} {'Value':>10} {'Stdv':>10}\n"
        else:
            s += f"{'Parameter':<{p_size}} {'Value':>10}\n"

        for k, v in self.fit_parameters.items():
            s += f"{k:<{max(p_size, 9)}} {v:>10.3g}"
            if stdev:
                s += f" {stdev[k]:>10.3g}"
            s += "\n"

        return s

    def to_dict(self) -> dict:
        """
        Convert the fit result to a dictionary.

        Returns:
            Dictionary representation of the fit result.
        """
        keys = ["gof_qualifiers", "fit_parameters", "fixed_parameters", "guess", "metadata"]
        d = {k: v for k in keys if (v := getattr(self, k)) is not None}

        return d

    def to_yaml(self, path: Union[os.PathLike[str], str], sort_keys: bool = False) -> None:
        """
        Save the fit result as yaml.

        Args:
            path: Path to save to.
            sort_keys: Boolean indicating whether to sort the keys.

        """
        dic = clean_types(self.to_dict())
        Path(path).write_text(yaml.dump(dic, sort_keys=sort_keys))

    def to_pickle(self, path: Union[os.PathLike[str], str]) -> None:
        """
        Save the fit result as pickle.

        Args:
            path: Path to save to.
        """
        try:
            del self.model.numerical
        except AttributeError:
            pass

        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    def __call__(self, **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError("Nope")
        data = self.data or {}
        kwargs = self.parameters | data | kwargs

        return self.model(**kwargs)

    @cached_property
    def hess(self) -> np.ndarray:
        if self.objective is None:
            raise ValueError("No objective found")
        parameter_shapes = {k: v.shape for k, v in self.fit_parameters.items()}

        if list(parameter_shapes.items()) != list(self.objective.shapes.items()):
            raise ValueError("Mismatch between objective and fit parameters")

        # packed parameter values at minium
        sol = pack(self.fit_parameters.values())

        return nd.Hessian(self.objective)(sol)

    @property
    def variance(self) -> dict[str, float | np.ndarray]:
        hess_inv = np.linalg.inv(self.hess)
        var = np.diag(hess_inv)
        parameter_shapes = {k: v.shape for k, v in self.fit_parameters.items()}
        return unpack(var, parameter_shapes)

    @property
    def stdev(self) -> dict[str, float | np.ndarray]:
        return {k: np.sqrt(v) for k, v in self.variance.items()}

    @property
    def parameters(self) -> dict[str, float | np.ndarray]:
        return {**self.fit_parameters, **self.fixed_parameters}
