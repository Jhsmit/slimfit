from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Union

import numpy as np
import yaml

from slimfit import Model
from slimfit.utils import clean_types


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

    symbolic_model: Optional[Model] = None
    """The fitted symbolic model"""

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

        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    def __call__(self, **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError("Nope")
        data = self.data or {}
        kwargs = self.parameters | data | kwargs

        return self.symbolic_model(**kwargs)

    @property
    def parameters(self) -> dict[str, float | np.ndarray]:
        return {**self.fit_parameters, **self.fixed_parameters}
