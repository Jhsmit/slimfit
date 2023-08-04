from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Optional, Any, Union, TYPE_CHECKING

import numpy as np
import numdifftools as nd
import yaml

from slimfit import Model
from slimfit.loss import Loss
from slimfit.objective import Objective, pack, unpack, Hessian
from slimfit.utils import clean_types, rgetattr

if TYPE_CHECKING:
    from slimfit.minimizers import Minimizer


@dataclass
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

    minimizer: Optional[Minimizer] = None

    hessian: Optional[np.ndarray] = None

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
        if any(np.ndim(v) != 0 for v in self.fit_parameters.values()):
            raise ValueError("Cannot print fit result with array values.")

        s = ""
        stdev = self.stdev if self.hessian is not None else None

        p_size = max(len(k) for k in self.fit_parameters)
        if stdev:
            s += f"{'Parameter':<{p_size}} {'Value':>10} {'Stdev':>10}\n"
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
        keys = [
            "gof_qualifiers",
            "fit_parameters",
            "fixed_parameters",
            "guess",
            "metadata",
        ]
        if self.hessian is not None:
            keys += ["stdev"]

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
            del self.minimizer.model.numerical
        except AttributeError:
            pass

        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    def eval_hessian(self, hessian: Optional[Hessian] = None) -> np.ndarray:
        # TODO Hessian as parameter / strategy
        """Evaluate the hessian at the fitted parameters values"""

        hessian = hessian or rgetattr(self.minimizer, "objective.hessian", None)
        if hessian is None:
            raise ValueError("No Hessian available")

        if hasattr(self.minimizer, "objective") and list(hessian.shapes.items()) != list(
            self.minimizer.objective.shapes.items()
        ):
            raise ValueError("Mismatch between objective and fit parameters")

        x = pack(self.fit_parameters.values())
        self.hessian = hessian(x)

        return self.hessian

    @property
    def variance(self) -> dict[str, float | np.ndarray]:
        if self.hessian is None:
            self.eval_hessian()

        hess_inv = np.linalg.inv(self.hessian)
        var = np.diag(hess_inv)
        parameter_shapes = {k: v.shape for k, v in self.fit_parameters.items()}
        return unpack(var, parameter_shapes)

    @property
    def stdev(self) -> dict[str, float | np.ndarray]:
        return {k: np.sqrt(v) for k, v in self.variance.items()}

    @property
    def parameters(self) -> dict[str, float | np.ndarray]:
        return {**self.fit_parameters, **self.fixed_parameters}
