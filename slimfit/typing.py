from __future__ import annotations

from typing import TypeVar, KeysView, ValuesView, ItemsView, Protocol

from typing_extensions import TypeVarTuple

# PEP 646: https://peps.python.org/pep-0646/
Shape = TypeVarTuple("Shape")

DataType = TypeVar("DataType")


# STRATEGY / PROTOCOL?
# class Data(UserDict, Generic[DataType]):
class Data(Protocol[DataType]):

    ...

    def __getitem__(self, item) -> DataType:
        return super().__getitem__(item)

    def __setitem__(self, key: str, value: DataType) -> None:
        return super().__setitem__(key, value)

    def keys(self) -> KeysView[str]:
        return super().keys()

    def values(self) -> ValuesView[DataType]:
        return super().values()

    def items(self) -> ItemsView[str, DataType]:
        return super().items()


if __name__ == "__main__":
    import numpy as np

    d = {"a": np.random.rand(10, 20)}
    data: Data[np.ndarray] = d

    print(data)
