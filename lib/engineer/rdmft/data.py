import functools
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
import scipy
import torch

from .rdm import Integrals, RDM12s, to_NO, TwoBodyTensor, RDM2s, OneBodyVector
from .storage import Storable
from . import utilities

class FancyDict(Dict[str, Any]):
    """
    Fancy dict class with the possibility to access keys as attributes.

    Example
    -------
    >>> dp = FancyDict()
    >>> dp["a"] = 1
    >>> print(dp.a)
    1
    >>> dp.b = 2
    >>> print(dp["b"])
    2
    >>> dp
    {'a': 1, 'b': 2}
    """

    __slots__ = ()

    def to(self, device: torch.device) -> "FancyDict":
        """
        Copy all tensors in the data point to the specified device.

        .. note:: Will always create a copy of the data point.
        """
        return FancyDict(
            **{
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in self.items()
            }
        )

    def __getattr__(self, attr: str) -> torch.Tensor:
        return self[attr]

    def __setattr__(self, attr: str, value: torch.Tensor) -> None:
        self[attr] = value


def add_type_attr(func: Callable):  # type: ignore
    """
    Decorator to add the type attribute to the current HDF5 node.
    """

    @functools.wraps(func)
    def wrapper(self, h5: h5py.File):  # type: ignore
        h5.attrs["type"] = self.__class__.__name__
        return func(self, h5)

    return wrapper


def check_type_attr(func):  # type: ignore
    """
    Decorator to check the type attribute of the current HDF5 node.
    """

    @functools.wraps(func)
    def wrapper(cls, h5: h5py.File):  # type: ignore
        if h5.attrs["type"] != cls.__name__:
            raise ValueError(f"Expected type '{cls.__name__}', got '{h5.attrs['type']}'.")

        return func(cls, h5)

    return wrapper


@dataclass
class Geometry(Storable):
    """
    System information independent of the included states.

    Example
    -------
    >>> geometry = Geometry(
    ...     symbols=["H", "H"],
    ...     nuclear_charges=[1.0, 1.0],
    ...     positions=[
    ...         [0.0, 0.0, +0.701],
    ...         [0.0, 0.0, -0.701],
    ...     ],
    ...     method="fci",
    ...     basis="def2-SVP",
    ... )
    >>> geometry
    Geometry(symbols=['H', 'H'], positions=array([[ 0.   ,  0.   ,  0.701],
           [ 0.   ,  0.   , -0.701]]), nuclear_charges=array([1., 1.]), method='fci', basis='def2-SVP')
    """

    symbols: List[str]
    """Element symbols for each atom"""

    positions: npt.NDArray
    """Nuclear coordinates in Bohr"""

    nuclear_charges: npt.NDArray
    """Fractional nuclear charges"""

    method: Literal["fci"]
    """Quantum chemistry method used"""

    basis: str
    """Atomic orbital basis set"""

    def __init__(
        self,
        symbols: List[str],
        positions: npt.ArrayLike,
        nuclear_charges: npt.ArrayLike,
        method: Any,
        basis: str,
    ):
        self.symbols = symbols
        self.positions = np.ascontiguousarray(positions).reshape(-1, 3)
        self.nuclear_charges = np.ascontiguousarray(nuclear_charges)

        if len(self) != self.positions.shape[0] or len(self) != self.nuclear_charges.shape[0]:
            raise ValueError("Inconsistent number of atoms.")

        self.method = method
        self.basis = basis

    @property
    def atom(self) -> List[Tuple[str, npt.NDArray]]:
        """Return the geometry in pyscf compatible format."""
        return [(self.symbols[at], self.positions[at, :]) for at in range(len(self))]

    def __len__(self) -> int:
        return len(self.symbols)

    @add_type_attr
    def save_h5(self, h5: h5py.File) -> None:
        """Save the datapoint to an HDF5 file."""

        h5.create_dataset("symbols", data=self.symbols)
        h5.create_dataset("positions", data=self.positions)
        h5.create_dataset("nuclear_charges", data=self.nuclear_charges)
        h5.attrs["method"] = self.method
        h5.attrs["basis"] = self.basis

    @classmethod
    @check_type_attr  # type: ignore
    def load_h5(cls, h5: h5py.File) -> "Geometry":
        """Load a datapoint from an HDF5 file."""

        return cls(
            [symbol.decode() for symbol in h5["symbols"]],
            h5["positions"][()],
            h5["nuclear_charges"][()],
            method=h5.attrs["method"],
            basis=h5.attrs["basis"],
        )


State = namedtuple("State", ["charge", "spin", "n_roots"])
"""State information for a given geometry."""


@dataclass
class System(Storable):
    """
    System information for the states computed with a given geometry.

    Example
    -------
    >>> system = System(
    ...     geometry=Geometry(
    ...         symbols=["H", "H"],
    ...         nuclear_charges=[1.0, 1.0],
    ...         positions=[
    ...             [0.0, 0.0, +0.701],
    ...             [0.0, 0.0, -0.701],
    ...         ],
    ...         method="fci",
    ...         basis="def2-SVP",
    ...     ),
    ...     states=[
    ...         State(charge=0, spin=0, n_roots=1),
    ...         {"charge": 0, "spin": 2, "n_roots": 1},
    ...     ]
    ... )
    >>> system.states
    [State(charge=0, spin=0, n_roots=1), State(charge=0, spin=2, n_roots=1)]
    """

    geometry: Geometry
    """System geometry"""

    states: List[State]
    """States with total charge and spin multiplicity"""

    def __init__(self, geometry: Geometry, states: Sequence[Union[Dict[str, int], State]]):
        self.geometry = geometry
        self.states = [state if isinstance(state, State) else State(**state) for state in states]

    @add_type_attr
    def save_h5(self, h5: h5py.File) -> None:
        self.geometry.save_h5(h5.create_group("geometry"))
        h5.create_dataset("states", data=np.array([list(state) for state in self.states]))

    @classmethod
    @check_type_attr  # type: ignore
    def load_h5(cls, h5: h5py.File) -> "System":
        return cls(
            geometry=Geometry.load_h5(h5.require_group("geometry")),
            states=[State(*state) for state in h5["states"][()]],
        )

    def next_state(self, state: State) -> State:
        """Return the next state in the list of states."""

        return self.states[self.states.index(state) + 1 % len(self.states)]


@dataclass
class Result(Storable):
    """Calculation result for a given state."""

    n_elecs: npt.NDArray
    """Number of up and down electrons"""

    nuc_rep_energy : npt.NDArray = None
    """ Nuclear repulsion energy"""
    
    n_dets: int = 0
    """Number of determinants"""

    hf_energy: npt.NDArray = None
    """Hartree-Fock energy"""

    fci_energy: npt.NDArray = None
    """FCI energy"""

    nuclear_repulsion: npt.NDArray = None
    """Nuclear repulsion energy"""

    ao_mo_transform: npt.NDArray = None
    """AO to MO transformation matrix"""

    e_onebody: npt.NDArray = None
    """One-body energy"""

    e_twobody: npt.NDArray = None
    """Two-body energy"""

    spin_square: npt.NDArray = None
    """Expectation value of the spin squared operator"""

    spin_multiplicity: npt.NDArray = None
    """Spin multiplicity"""

    w_correlation: npt.NDArray = None
    """Correlation energy"""

    rdm12s: Dict[int, RDM12s] = None
    """1-RDMs and 2-RDMs for each pair of roots"""

    trdm12s: Dict[Tuple[int, int], RDM12s] = None
    """1-RDMs and 2-RDMs for each pair of roots"""
    
    rdm12s_NO: Dict[int, RDM12s] = None
    
    NOONs: Dict[int, OneBodyVector] = None
    
    rdm2s_NO_phys: Dict[int, RDM12s] = None
    
    eris_NO_phys : Dict[int, TwoBodyTensor] = None

    def contract_with_dm(self, dm: npt.NDArray) -> RDM12s:
        """Given an N-body density matrix dm construct the corresponding 2-RDM"""
        rdm12s = RDM12s.zeros(self.rdm12s[0].rdm1s.u.shape[0])
        n_roots = len(self.rdm12s)
        for root in range(n_roots):
            rdm12s += dm[root, root] * self.rdm12s[root]
            for root2 in range(root + 1, n_roots):
                # trdm12s is flattened, so we need to reconstruct the index
                # the index is given by (root, root2) -> root * (n_roots - 2) + root2 - 1
                rdm12s += (
                    dm[root, root2] * self.trdm12s[(root, root2)]
                    + dm[root, root2] * self.trdm12s[(root, root2)].transpose_particle()
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

    @add_type_attr
    def save_h5(self, h5: h5py.File) -> None:
        print("check 12")
        
        h5.attrs["n_elecs"] = self.n_elecs
        
        if self.n_dets != 0 : h5.attrs["n_dets"] = self.n_dets
        
        if self.hf_energy is not None: h5.create_dataset("hf_energy", data=self.hf_energy)
        
        if self.fci_energy is not None: h5.create_dataset("fci_energy", data=self.fci_energy)
        if self.nuclear_repulsion is not None: h5.create_dataset("nuclear_repulsion", data=self.nuclear_repulsion)
        if self.ao_mo_transform is not None: h5.create_dataset("ao_mo_transform", data=self.ao_mo_transform)
        if self.e_onebody is not None: h5.create_dataset("e_onebody", data=self.e_onebody)
        if self.e_twobody is not None: h5.create_dataset("e_twobody", data=self.e_twobody)
        if self.spin_square is not None: h5.create_dataset("spin_square", data=self.spin_square)
        if self.spin_multiplicity is not None: h5.create_dataset("spin_multiplicity", data=self.spin_multiplicity)
        if self.w_correlation is not None : h5.create_dataset("w_correlation", data=self.w_correlation)

        if self.rdm12s is not None:
            for root, rdm12s in self.rdm12s.items():
                print("check 13")
                rdm12s.save_h5(h5.create_group(f"rdm12s/{root}"))
                
        if self.trdm12s is not None:
            for (root1, root2), trdm12s in self.trdm12s.items():
                print("check 14")
                trdm12s.save_h5(h5.create_group(f"trdm12s/{root1}_{root2}"))
            
         #added by souloke
        if self.rdm12s_NO is not None: 
            for root, rdm12s in self.rdm12s_NO.items():
                print("check 15")
                rdm12s.save_h5(h5.create_group(f"rdm12s_NO/{root}"))    
                
                
        if self.NOONs is not None:
            for root, NOON in self.NOONs.items():
               print("check 16")
               NOON.save_h5(h5.create_group(f"NOONs/{root}"))             
            
        if self.rdm2s_NO_phys is not None:
            for root, rdm12s in self.rdm2s_NO_phys.items():
                print("check 17")
                rdm12s.save_h5(h5.create_group(f"rdm12s_NO_phys/{root}"))    
                      
        if self.eris_NO_phys is not None:   
            for root, eris in self.eris_NO_phys.items():
                print("check 18")
                eris.save_h5(h5.create_group(f"eris_NO_phys/{root}"))   
                

    def save_h5_singleFile(self, h5: h5py.File) -> None:
#        print("check 12")
        
#        h5.attrs["n_elecs"] = self.n_elecs
#        print(type(self.n_elecs))
        utilities.tools.appendtohdf5(h5,"n_elecs",self.n_elecs)


        utilities.tools.appendtohdf5(h5,"e_nuclear_repulsion",self.nuc_rep_energy)
            
        
#        if self.n_dets != 0 : h5.attrs["n_dets"] = self.n_dets
        
# =============================================================================
#         if self.hf_energy is not None: h5.create_dataset("hf_energy", data=self.hf_energy)
#         
#         if self.fci_energy is not None: h5.create_dataset("fci_energy", data=self.fci_energy)
#         if self.nuclear_repulsion is not None: h5.create_dataset("nuclear_repulsion", data=self.nuclear_repulsion)
#         if self.ao_mo_transform is not None: h5.create_dataset("ao_mo_transform", data=self.ao_mo_transform)
#         if self.e_onebody is not None: h5.create_dataset("e_onebody", data=self.e_onebody)
# =============================================================================
        if self.e_twobody is not None: 
            
            utilities.tools.appendtohdf5(h5,"e_twobody",self.e_twobody)

        if self.e_onebody is not None: 
           
           utilities.tools.appendtohdf5(h5,"e_onebody",self.e_onebody)
                       
        if self.fci_energy is not None:
           utilities.tools.appendtohdf5(h5,"fci_energy",self.fci_energy) 
            
#            h5.create_dataset("e_twobody", data=self.e_twobody)
# =============================================================================
#         if self.spin_square is not None: h5.create_dataset("spin_square", data=self.spin_square)
#         if self.spin_multiplicity is not None: h5.create_dataset("spin_multiplicity", data=self.spin_multiplicity)
#         if self.w_correlation is not None : h5.create_dataset("w_correlation", data=self.w_correlation)
# 
# =============================================================================



        if self.NOONs is not None:
            for root, NOON in self.NOONs.items():
#               print("check 16")
               NOON.save_h5_singleFile(h5.require_group("NOONs"),root)
               
            
        if self.rdm2s_NO_phys is not None:
            for root, rdm12s in self.rdm2s_NO_phys.items():
#                print("check 17")
                rdm12s.rdm2s.save_h5_singleFile(h5.require_group("rdm2s"), root)    
                rdm12s.rdm1s.save_h5_singleFile(h5.require_group("rdm1s"), root)    

        if self.eris_NO_phys is not None:   
            for root, eris in self.eris_NO_phys.items():
#                print("check 18")
                eris.save_h5_singleFile(h5.require_group("eris"),root)  

        

    @classmethod
    @check_type_attr  # type: ignore
    def load_h5(cls, h5: h5py.File) -> "Result":
        return cls(
            n_dets=h5.attrs["n_dets"],
            n_elecs=h5.attrs["n_elecs"],
            hf_energy=h5["hf_energy"][()],
            fci_energy=h5["fci_energy"][()],
            nuclear_repulsion=h5["nuclear_repulsion"][()],
            ao_mo_transform=h5["ao_mo_transform"][()],
            e_onebody=h5["e_onebody"][()],
            e_twobody=h5["e_twobody"][()],
            spin_square=h5["spin_square"][()],
            spin_multiplicity=h5["spin_multiplicity"][()],
            w_correlation=h5["w_correlation"][()],
            rdm12s={int(root): RDM12s.load_h5(rdm12s) for root, rdm12s in h5["rdm12s"].items()},
            trdm12s={
                tuple(map(int, root.split("_"))): RDM12s.load_h5(trdm12s)  # type: ignore
                for root, trdm12s in h5["trdm12s"].items()
            }
            if "trdm12s" in h5
            else {},
        )


@dataclass
class Datapoint(Storable):
    """
    Calculation result for a given system.
    """

    system: System
    """System that was calculated"""

    data: Dict[State, Result]
    """Calculation results for each state"""

    integrals: Integrals
    """Integrals in atomic orbital basis"""

    @add_type_attr
    def save_h5(self, h5: h5py.File) -> None:
        self.system.save_h5(h5.create_group("system"))
#        print("check11")
        for name, integral in self.integrals._asdict().items():
            h5.create_dataset(f"data/ao/{name}", data=integral)
        for state, result in self.data.items():
            state_group = h5.create_group(f"data/{state.charge}_{state.spin}")
            result.save_h5(state_group)


    def save_h5_singleFile(self, h5: h5py.File) -> None:

#        print("check11")
#        for name, integral in self.integrals._asdict().items():
#            h5.create_dataset(f"data/ao/{name}", data=integral)
        for state, result in self.data.items():
                state_group = h5.require_group(f"data/{state.charge}_{state.spin}")
#                state_group = h5.create_group(f"data/{state.charge}_{state.spin}")
                result.save_h5_singleFile(state_group)
            
    @classmethod
    @check_type_attr  # type: ignore
    def load_h5(cls, h5: h5py.File) -> "Datapoint":
        system = System.load_h5(h5.require_group("system"))
        integrals = {}
        for name, integral in h5["data/ao"].items():
            integrals[name] = integral[()]
        data = {}
        for state in system.states:
            state_group = h5[f"data/{state.charge}_{state.spin}"]
            data[state] = Result.load_h5(state_group)
        return cls(system=system, data=data, integrals=Integrals(**integrals))

    def select(self, states: List[State]) -> "Datapoint":
        """
        Select a subset of states from the datapoint.
        """
        if not all(state in self.data for state in states):
            missing_states = [state for state in states if state not in self.data]
            raise ValueError(f"States {missing_states} not in {self.data.keys()}")

        return Datapoint(
            system=self.system,
            data={state: self.data[state] for state in states},
            integrals=self.integrals,
        )

    def get(
        self,
        state: State,
        augment_charge: bool = False,
        augment_space: bool = False,
        augment_spin: bool = False,
        augment_phases: bool = False,
        augment_particle_hole: bool = False,
    ) -> FancyDict:
        result = self.data[state]

        if augment_charge:
            next_state = self.system.next_state(state)
            result_ceil = result
            result_floor = self.data[next_state]

            if augment_space:
                rdm12s_ceil, _ = result_ceil.contract_with_random()
                rdm12s_floor, _ = result_floor.contract_with_random()
            else:
                rdm12s_ceil = result_ceil.contract_with_equal_occupation().transform(
                    result_ceil.ao_mo_transform.T
                )
                rdm12s_floor = result_floor.contract_with_equal_occupation().transform(
                    result_floor.ao_mo_transform.T
                )

            if augment_spin:
                rdm12s_ceil = rdm12s_ceil.mix_spin(np.random.rand(1).item())
                rdm12s_floor = rdm12s_floor.mix_spin(np.random.rand(1).item())

            w = np.random.rand(1).item()
            rdm12s = w * rdm12s_ceil + (1 - w) * rdm12s_floor
            rdm12s = rdm12s.transform(self.integrals.overlap.dot(result_ceil.ao_mo_transform))

        else:
            if augment_space:
                rdm12s, _ = result.contract_with_random()
            else:
                rdm12s = result.contract_with_equal_occupation()

            if augment_spin:
                rdm12s = rdm12s.mix_spin(np.random.rand(1).item())

        AO_to_MO = result.ao_mo_transform

        # TODO: Klaas, check if flipping one phase changes anything
        h1, eris, rdm12s = to_NO(
            self.integrals,
            AO_to_MO,
            rdm12s,
            phases="random" if augment_phases else "AO",
        )

        NOONs = rdm12s.rdm1s.diag()
        if augment_particle_hole and np.random.choice([True, False]):
            NOONs = 1 - NOONs

        _, e_twobody = rdm12s.contract_with(h1, eris)
        lambda2s = rdm12s.to_cumulant()
        W_corr_uu = 0.25 * np.einsum("pqrs,pqrs->", lambda2s.uu, eris.uu)
        W_corr_ud = np.einsum("pqrs,pqrs->", lambda2s.ud, eris.ud)
        W_corr_dd = 0.25 * np.einsum("pqrs,pqrs->", lambda2s.dd, eris.dd)

        return FancyDict(
            n_orbs=h1.u.shape[0],  # []
            W=torch.tensor(e_twobody),  # []
            h1_NO_u=torch.tensor(h1.u),  # [n_orbs, n_orbs]
            h1_NO_d=torch.tensor(h1.d),  # [n_orbs, n_orbs]
            NOONs_u=torch.tensor(NOONs.u),  # [n_orbs]
            NOONs_d=torch.tensor(NOONs.d),  # [n_orbs]
            eris_uu=torch.tensor(eris.uu),  # [n_orbs, n_orbs, n_orbs, n_orbs]
            eris_ud=torch.tensor(eris.ud),  # [n_orbs, n_orbs, n_orbs, n_orbs]
            eris_dd=torch.tensor(eris.dd),  # [n_orbs, n_orbs, n_orbs, n_orbs]
            lambda2s_uu=torch.tensor(lambda2s.uu),  # [n_orbs, n_orbs, n_orbs, n_orbs]
            lambda2s_ud=torch.tensor(lambda2s.ud),  # [n_orbs, n_orbs, n_orbs, n_orbs]
            lambda2s_dd=torch.tensor(lambda2s.dd),  # [n_orbs, n_orbs, n_orbs, n_orbs]
            W_corr=torch.tensor(lambda2s.contract_with(eris)),  # []
            W_corr_uu=torch.tensor(W_corr_uu),  # []
            W_corr_ud=torch.tensor(W_corr_ud),  # []
            W_corr_dd=torch.tensor(W_corr_dd),  # []
            N=np.sum(result.n_elecs),  # []
            N_u=result.n_elecs[0],  # []
            N_d=result.n_elecs[1],  # []
        )
