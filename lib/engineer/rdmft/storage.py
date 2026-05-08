from pathlib import Path
from typing import Type, TypeVar, Union

import h5py

StorableType = TypeVar("StorableType", bound="Storable")


class Storable:
    """Base class for storing results."""

    def save(self, path: Union[str, Path]) -> None:
        """Save the datapoint to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix not in (".h5", ".hdf5"):
            path = path.with_suffix(".h5")
        print("check 10")
        with h5py.File(path, "w") as h5:
            self.save_h5(h5)

    @classmethod
    def load(cls: Type[StorableType], path: Union[str, Path]) -> StorableType:
        """Load a datapoint from a file."""
        with h5py.File(path, "r") as h5:
            return cls.load_h5(h5)

    def save_h5(self, h5: h5py.File) -> None:
        raise NotImplementedError(f"save_h5 not implemented {self.__class__.__name__}")

    @classmethod
    def load_h5(cls: Type[StorableType], h5: h5py.File) -> StorableType:
        raise NotImplementedError(f"load_h5 not implemented {cls.__name__}")
