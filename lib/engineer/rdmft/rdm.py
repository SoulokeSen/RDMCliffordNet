import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import h5py
import numpy as np
import numpy.typing as npt
import scipy
from pyscf import ao2mo
from itertools import combinations
from . import utilities

OBType = TypeVar("OBType", bound="OneBody")
OBType2 = TypeVar("OBType2", bound="OneBody")

TBType = TypeVar("TBType", bound="TwoBodyTensor")
TBType2 = TypeVar("TBType2", bound="TwoBodyTensor")

Integrals = namedtuple("Integrals", ["overlap", "h1", "eri"])
"""Integrals in atomic orbital basis set"""


@dataclass
class OneBody:
    u: npt.NDArray[np.float64]
    d: npt.NDArray[np.float64]

    @classmethod
    def load(cls: Type[OBType], u_file: str, d_file: str) -> OBType:
        return cls(np.load(u_file), np.load(d_file))

    @classmethod
    def load_h5(cls: Type[OBType], h5: h5py.File) -> OBType:
        return cls(h5["u"][()], h5["d"][()])

    def save_h5(self, h5: h5py.File) -> None:
        h5.create_dataset("u", data=self.u)
        h5.create_dataset("d", data=self.d)


    def save_h5_singleFile(self,h5: h5py.Group, root:int) -> None:  #souloke
        utilities.tools.appendtohdf5(h5,f"{root}",np.stack([self.u,self.d], axis=0))
#        utilities.tools.appendtohdf5(h5,"d",self.d)


    def apply_function(
        self: OBType, func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    ) -> OBType:
        return self.__class__(func(self.u), func(self.d))

    def __add__(self: OBType, other: Union[int, float, npt.NDArray[np.float64], OBType2]) -> OBType:
        if isinstance(other, OneBody):
            return self.__class__(self.u + other.u, self.d + other.d)
        else:
            return self.__class__(self.u + other, self.d + other)

    def __sub__(self: OBType, other: Union[int, float, npt.NDArray[np.float64], OBType2]) -> OBType:
        if isinstance(other, OneBody):
            return self.__class__(self.u - other.u, self.d - other.d)
        else:
            return self.__class__(self.u - other, self.d - other)

    def __rsub__(
        self: OBType, other: Union[int, float, npt.NDArray[np.float64], OBType2]
    ) -> OBType:
        if isinstance(other, OneBody):
            return self.__class__(other.u - self.u, other.d - self.d)
        else:
            return self.__class__(other - self.u, other - self.d)

    def __mul__(self: OBType, other: Union[int, float, npt.NDArray[np.float64], OBType2]) -> OBType:
        if isinstance(other, OneBody):
            return self.__class__(self.u * other.u, self.d * other.d)
        else:
            return self.__class__(self.u * other, self.d * other)

    __rmul__ = __mul__

    def __truediv__(
        self: OBType, other: Union[int, float, npt.NDArray[np.float64], OBType2]
    ) -> OBType:
        if isinstance(other, OneBody):
            return self.__class__(self.u / other.u, self.d / other.d)
        else:
            return self.__class__(self.u / other, self.d / other)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(u={self.u.shape},d={self.d.shape})"

    # TODO: Make sure that dot works as expected
    def dot(self: OBType, other: Union[npt.NDArray[np.float64], OBType2]) -> Union[OBType, OBType2]:
        if isinstance(other, OneBody):
            return self.__class__(self.u.dot(other.u), self.d.dot(other.d))
        else:
            return self.__class__(self.u.dot(other), self.d.dot(other))

    def ldot(self: OBType, other: Union[OBType2, npt.NDArray[np.float64]]) -> OBType:
        """Performs dot product with other on the left and self on the right"""
        if isinstance(other, OneBody):
            return self.__class__(other.u.dot(self.u), other.d.dot(self.d))
        else:
            return self.__class__(other.dot(self.u), other.dot(self.d))

    def trace(self) -> "OneBodyVector":
        return OneBodyVector(np.trace(self.u), np.trace(self.d))

    def mix_spin(self: OBType, weight: float) -> OBType:
        return self.__class__(
            weight * self.u + (1 - weight) * self.d, weight * self.d + (1 - weight) * self.u
        )

    def spin_sum(self) -> npt.NDArray[np.float64]:
        return self.u + self.d


class OneBodyVector(OneBody):
    @classmethod
    def zeros(cls: Type[OBType], n_orbs: int, dtype: npt.DTypeLike = np.float64) -> OBType:
        return cls(np.zeros(n_orbs, dtype=dtype), np.zeros(n_orbs, dtype=dtype))

    @classmethod
    def empty(cls: Type[OBType], n_orbs: int, dtype: npt.DTypeLike = np.float64) -> OBType:
        return cls(np.empty(n_orbs, dtype=dtype), np.empty(n_orbs, dtype=dtype))

    def diag(self) -> "OneBodyMatrix":
        return OneBodyMatrix(np.diag(self.u).copy(), np.diag(self.d).copy())

    # def dot(self: "OneBodyVector", other: Union[OBType2, npt.NDArray]) -> Union["OneBodyVector", OBType2]:
    #     if isinstance(other, OneBodyVector) or isinstance(other, OneBodyMatrix):
    #         return OneBodyVector(self.u.dot(other.u), self.d.dot(other.d))
    #     elif isinstance(other, OneBody):
    #         if other.u.ndim < 3:
    #             return OneBodyVector(self.u.dot(other.u), self.d.dot(other.d))
    #         elif other.u.ndim == 3:
    #             return OneBodyMatrix(self.u.dot(other.u), self.d.dot(other.d))
    #         else:
    #             return OneBody(self.u.dot(other.u), self.d.dot(other.d))
    #     elif isinstance(other, npt.NDArray):
    #         if other.ndim < 3:
    #             return OneBodyVector(self.u.dot(other), self.d.dot(other))
    #         elif other.ndim == 3:
    #             return OneBodyMatrix(self.u.dot(other), self.d.dot(other))
    #         else:
    #             return OneBody(self.u.dot(other), self.d.dot(other))
    #     else:
    #         return OneBody(self.u.dot(other), self.d.dot(other))


class OneBodyMatrix(OneBody):
    # u and d have shape (n_orbs, n_orbs)

    @classmethod
    def zeros(cls: Type[OBType], n_orbs: int, dtype: npt.DTypeLike = np.float64) -> OBType:
        return cls(np.zeros((n_orbs, n_orbs), dtype=dtype), np.zeros((n_orbs, n_orbs), dtype=dtype))

    @classmethod
    def empty(cls: Type[OBType], n_orbs: int, dtype: npt.DTypeLike = np.float64) -> OBType:
        return cls(np.empty((n_orbs, n_orbs), dtype=dtype), np.empty((n_orbs, n_orbs), dtype=dtype))


    def T(self: OBType) -> OBType:
        return self.__class__(self.u.T, self.d.T)

    def eigh(self) -> Tuple[OneBodyVector, "OneBodyMatrix"]:
        eigvals_u, eigvecs_u = np.linalg.eigh(self.u)
        eigvals_d, eigvecs_d = np.linalg.eigh(self.d)
        return OneBodyVector(eigvals_u, eigvals_d), OneBodyMatrix(eigvecs_u, eigvecs_d)

    def eigvalsh(self) -> OneBodyVector:
        eigvals_u = np.linalg.eigvalsh(self.u)
        eigvals_d = np.linalg.eigvalsh(self.d)
        return OneBodyVector(eigvals_u, eigvals_d)

    def diag(self) -> OneBodyVector:
        return OneBodyVector(np.diag(self.u).copy(), np.diag(self.d).copy())

    def transform(self: OBType, matrix: Union["OneBodyMatrix", npt.NDArray]) -> OBType:
        if isinstance(matrix, OneBodyMatrix):
            return self.__class__(
                matrix.u.T.dot(self.u.dot(matrix.u)), matrix.d.T.dot(self.d.dot(matrix.d))
            )
        else:
            return self.__class__(
                matrix.T.dot(self.u.dot(matrix)), matrix.T.dot(self.d.dot(matrix))
            )

    def contract_with(self, other: Union["OneBodyMatrix", npt.NDArray]) -> float:
        if isinstance(other, OneBodyMatrix):
            return np.trace(self.u.dot(other.u)) + np.trace(self.d.dot(other.d))  # type: ignore
        else:
            return np.trace(self.u.dot(other)) + np.trace(self.d.dot(other))  # type: ignore


class RDM1s(OneBodyMatrix):
    @classmethod
    def load_from_directory(
        cls, directory: str, root1: int = 0, root2: Optional[int] = None
    ) -> "RDM1s":
        if root2 is None:
            return cls.load(
                os.path.join(directory, f"rdm1s_u_root{root1}.npy"),
                os.path.join(directory, f"rdm1s_d_root{root1}.npy"),
            )
        else:
            return cls.load(
                os.path.join(directory, f"trdm1s_u_root{root1}-{root2}.npy"),
                os.path.join(directory, f"trdm1s_d_root{root1}-{root2}.npy"),
            )

    def save(self, directory: str, root1: int = 0, root2: Optional[int] = None) -> None:
        if root2 is None:
            np.save(os.path.join(directory, f"rdm1s_u_root{root1}.npy"), self.u)
            np.save(os.path.join(directory, f"rdm1s_d_root{root1}.npy"), self.d)
        else:
            np.save(os.path.join(directory, f"trdm1s_u_root{root1}-{root2}.npy"), self.u)
            np.save(os.path.join(directory, f"trdm1s_d_root{root1}-{root2}.npy"), self.d)

    def to_NO(
        self, AO_to_MO: npt.NDArray, overlap_AO: npt.NDArray
    ) -> Tuple["RDM1s", "OneBodyMatrix", "OneBodyMatrix"]:
        n_orbs = self.u.shape[0]
#        print("check 1")
        NOONs, MO_to_NO = self.eigh()
        AO_to_NO = MO_to_NO.ldot(AO_to_MO)
        idx = AO_to_NO.apply_function(lambda x: np.argmax(abs(x), axis=0))
        AO_to_NO.u[:, AO_to_NO.u[idx.u, np.arange(n_orbs)] < 0] *= -1
        AO_to_NO.d[:, AO_to_NO.d[idx.d, np.arange(n_orbs)] < 0] *= -1
        MO_to_NO = AO_to_NO.apply_function(lambda x: AO_to_MO.T.dot(overlap_AO.dot(x)))
#        print("check 2")
        return RDM1s(np.diag(NOONs.u).copy(), np.diag(NOONs.d).copy()), AO_to_NO, MO_to_NO


@dataclass
class TwoBodyTensor:
    uu: npt.NDArray  # (n_orbs, n_orbs, n_orbs, n_orbs)
    ud: npt.NDArray  # (n_orbs, n_orbs, n_orbs, n_orbs)
    dd: npt.NDArray  # (n_orbs, n_orbs, n_orbs, n_orbs)

    @classmethod
    def load(cls: Type[TBType], uu_file: str, ud_file: str, dd_file: str) -> TBType:
        return cls(np.load(uu_file), np.load(ud_file), np.load(dd_file))

    @classmethod
    def load_h5(cls: Type[TBType], h5: h5py.File) -> TBType:
        return cls(h5["uu"][()], h5["ud"][()], h5["dd"][()])

    def save_h5(self, h5: h5py.File) -> None:
        h5.create_dataset("uu", data=self.uu)
        h5.create_dataset("ud", data=self.ud)
        h5.create_dataset("dd", data=self.dd)


    def save_h5_singleFile(self, h5: h5py.File, root:int) -> None:
        utilities.tools.appendtohdf5(h5,f"{root}",np.stack([self.uu,self.ud,self.dd], axis=0))
#        utilities.tools.appendtohdf5(h5,"ud",self.ud)
#        utilities.tools.appendtohdf5(h5,"dd",self.dd)

    @classmethod
    def zeros(cls: Type[TBType], n_orbs: int, dtype: npt.DTypeLike = np.float64) -> TBType:
        return cls(
            np.zeros((n_orbs, n_orbs, n_orbs, n_orbs), dtype=dtype),
            np.zeros((n_orbs, n_orbs, n_orbs, n_orbs), dtype=dtype),
            np.zeros((n_orbs, n_orbs, n_orbs, n_orbs), dtype=dtype),
        )

    @classmethod
    def empty(cls: Type[TBType], n_orbs: int, dtype: npt.DTypeLike = np.float64) -> TBType:
        return cls(
            np.empty((n_orbs, n_orbs, n_orbs, n_orbs), dtype=dtype),
            np.empty((n_orbs, n_orbs, n_orbs, n_orbs), dtype=dtype),
            np.empty((n_orbs, n_orbs, n_orbs, n_orbs), dtype=dtype),
        )

    @classmethod
    def from_8s(
        cls: Type[TBType], tensor: npt.NDArray, matrix: Union[npt.NDArray, OneBodyMatrix]
    ) -> TBType:
        """Construct full two-body tensor from 8-fold symmetry tensor and transformation"""

        if isinstance(matrix, OneBodyMatrix):
#            print("Inside from_8s")
#            n_orbs = matrix.u.shape[0]
            n_orbs = matrix.u.shape[1]
#            print("norbs",n_orbs)
            return cls(
                ao2mo.kernel(tensor, matrix.u, aosym=1, compact=False).reshape(
                    n_orbs, n_orbs, n_orbs, n_orbs
                ),
                ao2mo.kernel(
                    tensor,
                    (matrix.u, matrix.u, matrix.d, matrix.d),
                    aosym=1,
                    compact=False,
                ).reshape(n_orbs, n_orbs, n_orbs, n_orbs),
                ao2mo.kernel(tensor, matrix.d, aosym=1, compact=False).reshape(
                    n_orbs, n_orbs, n_orbs, n_orbs
                ),
            )
        else:
#            print("Inside from_8s")
            n_orbs = matrix.shape[0]
#            print("norbs",n_orbs)
            return cls(
                ao2mo.kernel(tensor, matrix, aosym=1, compact=False).reshape(
                    n_orbs, n_orbs, n_orbs, n_orbs
                ),
                ao2mo.kernel(
                    tensor,
                    matrix,
                    aosym=1,
                    compact=False,
                ).reshape(n_orbs, n_orbs, n_orbs, n_orbs),
                ao2mo.kernel(tensor, matrix, aosym=1, compact=False).reshape(
                    n_orbs, n_orbs, n_orbs, n_orbs
                ),
            )

    def __add__(self: TBType, other: Union[TBType2, Union[int, float], npt.NDArray]) -> TBType:
        if isinstance(other, TwoBodyTensor):
            return self.__class__(self.uu + other.uu, self.ud + other.ud, self.dd + other.dd)
        else:
            return self.__class__(self.uu + other, self.ud + other, self.dd + other)

    def __sub__(self: TBType, other: Union[TBType2, Union[int, float], npt.NDArray]) -> TBType:
        if isinstance(other, TwoBodyTensor):
            return self.__class__(self.uu - other.uu, self.ud - other.ud, self.dd - other.dd)
        else:
            return self.__class__(self.uu - other, self.ud - other, self.dd - other)

    def __mul__(self: TBType, other: Union[TBType2, Union[int, float], npt.NDArray]) -> TBType:
        if isinstance(other, TwoBodyTensor):
            return self.__class__(self.uu * other.uu, self.ud * other.ud, self.dd * other.dd)
        else:
            return self.__class__(self.uu * other, self.ud * other, self.dd * other)

    __rmul__ = __mul__

    def __truediv__(self: TBType, other: Union[TBType2, Union[int, float], npt.NDArray]) -> TBType:
        if isinstance(other, TwoBodyTensor):
            return self.__class__(self.uu / other.uu, self.ud / other.ud, self.dd / other.dd)
        else:
            return self.__class__(self.uu / other, self.ud / other, self.dd / other)

    def __str__(self) -> str:
        return f"n_orbs: {self.uu.shape[0]}\n uu: {self.uu}\n ud: {self.ud} \n dd:{self.dd}"

    def transpose(self: TBType, i0: int, i1: int, i2: int, i3: int) -> TBType:
        return self.__class__(
            self.uu.transpose(i0, i1, i2, i3),
            self.ud.transpose(i0, i1, i2, i3),
            self.dd.transpose(i0, i1, i2, i3),
        )

    def mix_spin(self: TBType, weight: float) -> TBType:
        return self.__class__(
            weight * self.uu + (1 - weight) * self.dd,
            weight * self.ud + (1 - weight) * self.ud.transpose(2, 3, 0, 1),
            weight * self.dd + (1 - weight) * self.uu,
        )

    def anti_symmetrize(self: TBType) -> TBType:
        return self.__class__(
#            self.uu - self.uu.transpose(0, 3, 2, 1),
            self.uu - self.uu.transpose(0, 1, 3, 2),
            self.ud,
#            self.dd - self.dd.transpose(0, 3, 2, 1),
            self.dd - self.dd.transpose(0, 1, 3, 2),
        )

    def transform(self: TBType, matrix: Union[OneBodyMatrix, npt.NDArray]) -> TBType:
        """Transform Two-body tensor according to transformation matrix"""
        if isinstance(matrix, OneBodyMatrix):
            return self.__class__(
                ao2mo.kernel(self.uu, matrix.u),
                ao2mo.kernel(self.ud, (matrix.u, matrix.u, matrix.d, matrix.d)),
                ao2mo.kernel(self.dd, matrix.d),
            )
        else:
            return self.__class__(
                ao2mo.kernel(self.uu, matrix),
                ao2mo.kernel(self.ud, matrix),
                ao2mo.kernel(self.dd, matrix),
            )

    def contract_with(self, other: Union[npt.NDArray, TBType]) -> float:
        """Contract Two-body Tensor with an anti-symmetric Two-body Tensor"""
        if isinstance(other, TwoBodyTensor):
            # print("uu cont", 1 / 4 * np.einsum("ijkl,ijkl->", self.uu, other.uu))
            # print("ud cont", np.einsum("ijkl,ijkl->", self.ud, other.ud))
            # print("dd cont", 1 / 4 * np.einsum("ijkl,ijkl->", self.dd, other.dd))
            return (  # type: ignore
                1 / 4 * np.einsum("ijkl,ijkl->", self.uu, other.uu)
                + np.einsum("ijkl,ijkl->", self.ud, other.ud)
                + 1 / 4 * np.einsum("ijkl,ijkl->", self.dd, other.dd)
            )
        else:
            return (  # type: ignore
                1 / 4 * np.einsum("ijkl,ijkl->", self.uu, other)
                + np.einsum("ijkl,ijkl->", self.ud, other)
                + 1 / 4 * np.einsum("ijkl,ijkl->", self.dd, other)
            )


class RDM2s(TwoBodyTensor):
    @classmethod
    def load_from_directory(
        cls, directory: str, root1: int = 0, root2: Optional[int] = None
    ) -> "RDM2s":
        if root2 is None:
            return cls.load(
                uu_file=os.path.join(directory, f"rdm2s_uu_root{root1}.npy"),
                ud_file=os.path.join(directory, f"rdm2s_ud_root{root1}.npy"),
                dd_file=os.path.join(directory, f"rdm2s_dd_root{root1}.npy"),
            )
        else:
            return cls.load(
                uu_file=os.path.join(directory, f"trdm2s_uu_root{root1}-{root2}.npy"),
                ud_file=os.path.join(directory, f"trdm2s_ud_root{root1}-{root2}.npy"),
                dd_file=os.path.join(directory, f"trdm2s_dd_root{root1}-{root2}.npy"),
            )

    @classmethod
    def from_rdm1s(cls, rdm1s: RDM1s) -> "RDM2s":
        return cls(
            np.einsum("pq,rs->pqrs", rdm1s.u, rdm1s.u) - np.einsum("pq,rs->psrq", rdm1s.u, rdm1s.u),
            np.einsum("pq,rs->pqrs", rdm1s.u, rdm1s.d),
            np.einsum("pq,rs->pqrs", rdm1s.d, rdm1s.d) - np.einsum("pq,rs->psrq", rdm1s.d, rdm1s.d),
        )

    def save(self, directory: str, root1: int = 0, root2: Optional[int] = None) -> None:
        if root2 is None:
            np.save(os.path.join(directory, f"rdm2s_uu_root{root1}.npy"), self.uu)
            np.save(os.path.join(directory, f"rdm2s_ud_root{root1}.npy"), self.ud)
            np.save(os.path.join(directory, f"rdm2s_dd_root{root1}.npy"), self.dd)
        else:
            np.save(os.path.join(directory, f"trdm2s_uu_root{root1}-{root2}.npy"), self.uu)
            np.save(os.path.join(directory, f"trdm2s_ud_root{root1}-{root2}.npy"), self.ud)
            np.save(os.path.join(directory, f"trdm2s_dd_root{root1}-{root2}.npy"), self.dd)

    def to_cumulant(self, rdm1s: RDM1s) -> "RDM2s":
        return self - self.from_rdm1s(rdm1s)

    def to_rdm1s(self, tol: float = 10**-12) -> RDM1s:
        """Constructs 1-RDM from 2-RDM

        NOTE: If there are contributions to the wavefunction or ensemble from a one-body state
        then the 2-RDM does *not* contract to the correct 1-RDM!"""
        n_orbs = self.uu.shape[0]

        # Deduce the number of spin up and spin down electrons from the 2-RDM
        n_up_n_up_min_1 = np.einsum("ppqq->", self.uu)  # n_up * (n_up -1)
        n_down_n_down_min_1 = np.einsum("ppqq->", self.dd)  # n_down * (n_down -1)
        n_up_n_down = np.einsum("ppqq->", self.ud)  # n_up * n_down

        # TODO: there are a few unused variables here
        # Depending on whether there are 0, 1 or more than 1 electron of a particular spin
        # we need to choose how to contract the 2-RDM
        if n_up_n_up_min_1 < tol and n_down_n_down_min_1 < tol and n_up_n_down < tol:
            n_up = 0
            n_down = 0
            return RDM1s.zeros(n_orbs)
        elif n_up_n_up_min_1 < tol and n_up_n_down < tol:
            n_up = 0
            n_down = (1 + np.sqrt(1 + 4 * n_down_n_down_min_1)) / 2
            return RDM1s(np.zeros((n_orbs, n_orbs)), np.einsum("pqrr->pq", self.dd) / (n_down - 1))
        elif n_down_n_down_min_1 < tol and n_up_n_down < tol:
            n_up = (1 + np.sqrt(1 + 4 * n_up_n_up_min_1)) / 2
            n_down = 0
            return RDM1s(np.einsum("pqrr->pq", self.uu) / (n_up - 1), np.zeros((n_orbs, n_orbs)))
        elif n_up_n_up_min_1 < tol and n_down_n_down_min_1 < tol:
            n_up = n_down = np.sqrt(n_up_n_down)
            return RDM1s(
                np.einsum("pqrr->pq", self.ud) / n_down, np.einsum("rrpq->pq", self.ud) / n_up
            )
        elif n_up_n_up_min_1 < tol:
            n_down = (1 + np.sqrt(1 + 4 * n_down_n_down_min_1)) / 2
            n_up = n_up_n_down / n_down
            return RDM1s(
                np.einsum("pqrr->pq", self.ud) / n_down,
                1 / 2 * np.einsum("pqrr->pq", self.dd) / (n_down - 1)
                + 1 / 2 * np.einsum("rrpq->pq", self.ud) / n_up,
            )
        elif n_down_n_down_min_1 < tol:
            n_up = (1 + np.sqrt(1 + 4 * n_up_n_up_min_1)) / 2
            n_down = n_up_n_down / n_up
            return RDM1s(
                1 / 2 * np.einsum("pqrr->pq", self.uu) / (n_up - 1)
                + 1 / 2 * np.einsum("pqrr->pq", self.ud) / n_down,
                np.einsum("rrpq->pq", self.ud) / n_up,
            )
        else:
            n_up = (1 + np.sqrt(1 + 4 * n_up_n_up_min_1)) / 2
            n_down = (1 + np.sqrt(1 + 4 * n_down_n_down_min_1)) / 2
            return RDM1s(
                1 / 2 * np.einsum("pqrr->pq", self.uu) / (n_up - 1)
                + 1 / 2 * np.einsum("pqrr->pq", self.ud) / n_down,
                1 / 2 * np.einsum("pqrr->pq", self.dd) / (n_down - 1)
                + 1 / 2 * np.einsum("rrpq->pq", self.ud) / n_up,
            )

    def to_rdm1s_phys(self, tol: float = 10**-12) -> RDM1s:
        """Constructs 1-RDM from 2-RDM

        NOTE: If there are contributions to the wavefunction or ensemble from a one-body state
        then the 2-RDM does *not* contract to the correct 1-RDM!"""
        n_orbs = self.uu.shape[0]

        # Deduce the number of spin up and spin down electrons from the 2-RDM
        n_up_n_up_min_1 = np.einsum("pqpq->", self.uu)  # n_up * (n_up -1)
        n_down_n_down_min_1 = np.einsum("pqpq->", self.dd)  # n_down * (n_down -1)
        n_up_n_down = np.einsum("pqpq->", self.ud)  # n_up * n_down

        # TODO: there are a few unused variables here
        # Depending on whether there are 0, 1 or more than 1 electron of a particular spin
        # we need to choose how to contract the 2-RDM
        if n_up_n_up_min_1 < tol and n_down_n_down_min_1 < tol and n_up_n_down < tol:
            n_up = 0
            n_down = 0
            return RDM1s.zeros(n_orbs)
        elif n_up_n_up_min_1 < tol and n_up_n_down < tol:
            n_up = 0
            n_down = (1 + np.sqrt(1 + 4 * n_down_n_down_min_1)) / 2
            return RDM1s(np.zeros((n_orbs, n_orbs)), np.einsum("prqr->pq", self.dd) / (n_down - 1))
        elif n_down_n_down_min_1 < tol and n_up_n_down < tol:
            n_up = (1 + np.sqrt(1 + 4 * n_up_n_up_min_1)) / 2
            n_down = 0
            return RDM1s(np.einsum("prqr->pq", self.uu) / (n_up - 1), np.zeros((n_orbs, n_orbs)))
        elif n_up_n_up_min_1 < tol and n_down_n_down_min_1 < tol:
            n_up = n_down = np.sqrt(n_up_n_down)
            return RDM1s(
                np.einsum("prqr->pq", self.ud) / n_down, np.einsum("rprq->pq", self.ud) / n_up
            )
        elif n_up_n_up_min_1 < tol:
            n_down = (1 + np.sqrt(1 + 4 * n_down_n_down_min_1)) / 2
            n_up = n_up_n_down / n_down
            return RDM1s(
                np.einsum("prqr->pq", self.ud) / n_down,
                1 / 2 * np.einsum("prqr->pq", self.dd) / (n_down - 1)
                + 1 / 2 * np.einsum("rprq->pq", self.ud) / n_up,
            )
        elif n_down_n_down_min_1 < tol:
            n_up = (1 + np.sqrt(1 + 4 * n_up_n_up_min_1)) / 2
            n_down = n_up_n_down / n_up
            return RDM1s(
                1 / 2 * np.einsum("prqr->pq", self.uu) / (n_up - 1)
                + 1 / 2 * np.einsum("prqr->pq", self.ud) / n_down,
                np.einsum("rprq->pq", self.ud) / n_up,
            )
        else:
            n_up = (1 + np.sqrt(1 + 4 * n_up_n_up_min_1)) / 2
            n_down = (1 + np.sqrt(1 + 4 * n_down_n_down_min_1)) / 2
            return RDM1s(
                1 / 2 * np.einsum("prqr->pq", self.uu) / (n_up - 1)
                + 1 / 2 * np.einsum("prqr->pq", self.ud) / n_down,
                1 / 2 * np.einsum("prqr->pq", self.dd) / (n_down - 1)
                + 1 / 2 * np.einsum("rprq->pq", self.ud) / n_up,
            )



@dataclass
class RDM12s:
    rdm1s: RDM1s
    rdm2s: RDM2s

    @classmethod
    def load_from_directory(
        cls, directory: str, root1: int, root2: Optional[int] = None
    ) -> "RDM12s":
        return cls(
            RDM1s.load_from_directory(directory, root1, root2),
            RDM2s.load_from_directory(directory, root1, root2),
        )

    @classmethod
    def load_h5(cls, h5: h5py.File) -> "RDM12s":
        return cls(RDM1s.load_h5(h5), RDM2s.load_h5(h5))

    def save_h5(self, h5: h5py.File) -> None:
        self.rdm1s.save_h5(h5)
        self.rdm2s.save_h5(h5)

    @classmethod
    def from_pyscf(
        cls,
        rdm1s: Tuple[npt.NDArray, npt.NDArray],
        rdm2s: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    ) -> "RDM12s":
        return cls(RDM1s(rdm1s[0], rdm1s[1]), RDM2s(rdm2s[0], rdm2s[1], rdm2s[2]))

    @classmethod
    def load(cls, u_file: str, d_file: str, uu_file: str, ud_file: str, dd_file: str) -> "RDM12s":
        return cls(RDM1s.load(u_file, d_file), RDM2s.load(uu_file, ud_file, dd_file))

    @classmethod
    def zeros(cls, n_orbs: int, dtype: npt.DTypeLike = np.float64) -> "RDM12s":
        return cls(RDM1s.zeros(n_orbs, dtype), RDM2s.zeros(n_orbs, dtype))

    @classmethod
    def empty(cls, n_orbs: int, dtype: npt.DTypeLike = np.float64) -> "RDM12s":
        return cls(RDM1s.empty(n_orbs, dtype), RDM2s.empty(n_orbs, dtype))

    def save(self, directory: str, root1: int, root2: Optional[int] = None) -> None:
        self.rdm1s.save(directory, root1, root2)
        self.rdm2s.save(directory, root1, root2)
        return

    def to_dict(self) -> Dict[str, npt.NDArray]:
        return {
            "rdm1s_u": self.rdm1s.u,
            "rdm1s_d": self.rdm1s.d,
            "rdm2s_uu": self.rdm2s.uu,
            "rdm2s_ud": self.rdm2s.ud,
            "rdm2s_dd": self.rdm2s.dd,
        }

    def __add__(self, other: "RDM12s") -> "RDM12s":
        return self.__class__(self.rdm1s + other.rdm1s, self.rdm2s + other.rdm2s)

    def __sub__(self, other: "RDM12s") -> "RDM12s":
        return self.__class__(self.rdm1s - other.rdm1s, self.rdm2s - other.rdm2s)

    def __mul__(self, other: Union["RDM12s", float]) -> "RDM12s":
        if isinstance(other, RDM12s):
            return self.__class__(self.rdm1s * other.rdm1s, self.rdm2s * other.rdm2s)
        else:
            return self.__class__(self.rdm1s * other, self.rdm2s * other)

    __rmul__ = __mul__

    def __truediv__(self, other: "RDM12s") -> "RDM12s":
        return self.__class__(self.rdm1s / other.rdm1s, self.rdm2s / other.rdm2s)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(rdm1s={self.rdm1s.u.shape},rdm2s={self.rdm2s.uu.shape})"

    def mix_spin(self, weight: float) -> "RDM12s":
        return self.__class__(self.rdm1s.mix_spin(weight), self.rdm2s.mix_spin(weight))

    def transform(self, matrix: Any) -> "RDM12s":
        return self.__class__(self.rdm1s.transform(matrix), self.rdm2s.transform(matrix))

    def transpose_particle(self) -> "RDM12s":
        return self.__class__(self.rdm1s, self.rdm2s.transpose(2, 3, 0, 1))

    def to_NO(
        self, AO_to_MO: npt.NDArray, overlap_AO: npt.NDArray, phases: "str" = "AO"
    ) -> Tuple["RDM12s", "OneBodyMatrix", "OneBodyMatrix"]:
        rdm1s, AO_to_NO, MO_to_NO = self.rdm1s.to_NO(AO_to_MO, overlap_AO)
        n_orbs = rdm1s.u.shape[0]
        if phases == "random":
            phase_factors = 2 * np.random.randint(2, size=n_orbs) - 1
            AO_to_NO = AO_to_NO * phase_factors[None, :]
            MO_to_NO = MO_to_NO * phase_factors[None, :]

        rdm2s = self.rdm2s.transform(MO_to_NO)
        return self.__class__(rdm1s, rdm2s), AO_to_NO, MO_to_NO

    def to_cumulant(self) -> RDM2s:
        return self.rdm2s.to_cumulant(self.rdm1s)

    def contract_with(
        self, h1: Union["OneBodyMatrix", npt.NDArray], eris: Union["TwoBodyTensor", npt.NDArray]
    ) -> Tuple[float, float]:
        return self.rdm1s.contract_with(h1), self.rdm2s.contract_with(eris)


@dataclass
class EnsembleRDM12s:
    rdm12s: List[RDM12s]
    trdm12s: List[RDM12s]

    @classmethod
    def load(cls, directory: str, n_roots: int) -> "EnsembleRDM12s":
        rdm12s = []
        trdm12s = []

        for root in range(n_roots):
            # rdm12s correspond to the diagonal elements of the density matrix (root, root)
            rdm12s.append(RDM12s.load_from_directory(directory, root))
            for root2 in range(root + 1, n_roots):
                # trdm12s correspond to the off-diagonal elements of the density matrix (root, root2)
                # note that we only store the upper triangle of the density matrix, root < root2
                trdm12s.append(RDM12s.load_from_directory(directory, root, root2))
        return cls(rdm12s, trdm12s)

    def contract_with_dm(self, dm: npt.NDArray) -> RDM12s:
        """Given an N-body density matrix dm, which is symmetric, normalized, positive semi-definite and so on,
        construct the corresponding 2-RDM"""
        rdm12s = RDM12s.zeros(self.rdm12s[0].rdm1s.u.shape[0])
        n_roots = len(self.rdm12s)
        for root in range(n_roots):
            rdm12s += dm[root, root] * self.rdm12s[root]
            for root2 in range(root + 1, n_roots):
                # trdm12s is flattened, so we need to reconstruct the index
                # the index is given by (root, root2) -> root * (n_roots - 2) + root2 - 1
                rdm12s += (
                    dm[root, root2] * self.trdm12s[root * (n_roots - 2) + root2 - 1]
                    + dm[root, root2]
                    * self.trdm12s[root * (n_roots - 2) + root2 - 1].transpose_particle()
                )
        return rdm12s

    def contract_with_equal_occupation(self) -> RDM12s:
        n_roots = len(self.rdm12s)
        dm = np.diag(np.ones(n_roots)) / n_roots
        return self.contract_with_dm(dm)

    def contract_with_random(self) -> Tuple[RDM12s, npt.NDArray]:
        n_roots = len(self.rdm12s)
        w = np.random.rand(n_roots)
        w = w / np.sum(w)
        X = (2 * np.pi * np.random.rand(n_roots * n_roots) - np.pi).reshape(n_roots, n_roots)
        X = X - X.T
        U = scipy.linalg.expm(X)
        dm = U.T.dot(np.diag(w).dot(U))
        return self.contract_with_dm(dm), dm


def to_NO(
    integrals: Integrals,
    ao_mo_transform: npt.NDArray,
    rdm12s: RDM12s,
    phases: str = "AO",
    corehf=None
) -> Tuple[OneBodyMatrix, TwoBodyTensor, RDM12s]:
    rdm12s, AO_to_NO, _ = rdm12s.to_NO(ao_mo_transform, integrals.overlap, phases=phases)

    rdm12s.rdm2s = rdm12s.rdm2s.transpose(0,2,1,3)
#    print("check inside to_NO")
    h1 = AO_to_NO.ldot(AO_to_NO.T().dot(integrals.h1))
    if corehf is not None:
        h1 += AO_to_NO.ldot(AO_to_NO.T().dot(corehf))
    # Transform ERI to NO basis
    eris = TwoBodyTensor.from_8s(integrals.eri, AO_to_NO)
    eris = eris.transpose(0,2,1,3)       #convert to physicists notation
    # Anti-sym ERIs
    eris = eris.anti_symmetrize()
    
    ###perform Cholesky/EVD here ####
#    print(eris.uu.shape)
#    n_orbs = eris.uu.shape[0]
 #   print("eris", eris.uu)
# =============================================================================
#     eri1 = np.zeros(((int(n_orbs*(n_orbs-1)/2)), int((n_orbs*(n_orbs-1)/2))), dtype=float)
#     eri2 = np.zeros(((int(n_orbs*(n_orbs-1)/2)), int((n_orbs*(n_orbs-1)/2))), dtype=float)
#     eri3 = np.zeros(((int(n_orbs*(n_orbs-1)/2)), int((n_orbs*(n_orbs-1)/2))), dtype=float)
#     pairs = list(combinations(range(0, n_orbs), 2))
#     pair_to_index = {pair: idx for idx, pair in enumerate(pairs)}
#     for i, (p, q) in enumerate(pairs):
#         for j, (r, s) in enumerate(pairs):
#             eri1[i, j] = eris.uu[p, q, r, s]   
#             eri2[i, j] = eris.ud[p, q, r, s]  #this one is not symmetric , need full norb^2 x norb^2 
#             eri3[i, j] = eris.dd[p, q, r, s] 
#     eigenvalues1, eigenvectors1 = np.linalg.eigh(eri1)
#     eigenvalues2, eigenvectors2 = np.linalg.eigh(eri2)
#     eigenvalues3, eigenvectors3 = np.linalg.eigh(eri3)
#     print("eigenvalues of down down sector of ERI", eigenvalues3, len(eigenvalues3))         
#     
# =============================================================================
    return h1, eris, rdm12s
