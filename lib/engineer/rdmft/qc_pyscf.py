"""
Entry point for running quantum chemistry reference calculations.
"""

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

import h5py
import numpy as np
import numpy.typing as npt
import yaml
from pyscf import fci, gto, scf, mcscf
from pyscf.fci.direct_spin1 import trans_rdm12s

from .data import Datapoint, Geometry, Result, State, System
from .rdm import Integrals, RDM12s, TwoBodyTensor, to_NO

#import torch
from .utilities import tools
from .configure import Config


def spin_square_orthonormal(rdm12s: RDM12s) -> Tuple[float, float]:
    # Z component of spin
    ssz = 0.25 * (
        np.einsum("iijj->", rdm12s.rdm2s.uu)
        - 2 * np.einsum("iijj->", rdm12s.rdm2s.ud)
        + np.einsum("iijj->", rdm12s.rdm2s.dd)
        + np.trace(rdm12s.rdm1s.u)
        + np.trace(rdm12s.rdm1s.d)
    )

    # XY component of spin
    ssxy = 0.5 * (
        -2 * np.einsum("ijji->", rdm12s.rdm2s.ud)
        + np.trace(rdm12s.rdm1s.u)
        + np.trace(rdm12s.rdm1s.d)
    )
    ss = ssxy + ssz

    s = np.sqrt(ss + 0.25) - 0.5
    spin_multiplicity = s * 2 + 1
    return ss, spin_multiplicity


def get_hcore(mol: gto.Mole, nuclear_charges: npt.NDArray) -> npt.NDArray:
    """
    Get core Hamiltonian with fractional nuclear charges.
    """

    hcore = mol.intor("int1e_kin")

    for i in range(mol.natm):
        mol.set_rinv_origin(mol.atom_coord(i))
        hcore += -mol.intor("int1e_rinv") * nuclear_charges[i]

    return hcore


def create_molecule(geometry: Geometry, state: State, config: Config) -> gto.Mole:
    """
    Create a PySCF compatible representation of the system.

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
    >>> state = State(charge=0, spin=0, n_roots=1)
    >>> mol = create_molecule(geometry, state, Config(verbosity=0))
    >>> mol.atom
    [('H', array([0.   , 0.   , 0.701])), ('H', array([ 0.   ,  0.   , -0.701]))]
    """

    return gto.M(
        atom=geometry.atom,
        unit="Angstrom",
        charge=state.charge,
        spin=state.spin,
        basis=geometry.basis,
        max_memory=config.max_memory,
        verbose=config.verbosity,
    )


def compute_integrals(geometry: Geometry, state: State, config: Config) -> Integrals:
    """
    Compute the integrals for a given system.

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
    >>> state = State(charge=0, spin=0, n_roots=1)
    >>> integrals = compute_integrals(geometry, state, Config(verbosity=0))
    >>> integrals.overlap.shape
    (10, 10)
    >>> integrals.eri.shape
    (1540,)
    """

    mol = create_molecule(geometry, state, config)

    with NamedTemporaryFile() as chkf:
        mf = scf.RHF(mol)
        mf.chkfile = chkf.name
        if mf._eri is None:
            mf._eri = mol.intor("int2e", aosym="s8")

        mf.get_hcore = partial(get_hcore, nuclear_charges=geometry.nuclear_charges)

        h1_ao = mf.get_hcore(mol)
        overlap_ao = mf.get_ovlp()

        return Integrals(
            overlap_ao,
            h1_ao,
            mf._eri,
        )


def compute_state_casci(geometry: Geometry, state: State, integrals: Integrals, config: Config, file_id, method) -> Result:
    """
    Perform a quantum chemistry calculation for a given state.

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
    >>> state = State(charge=0, spin=0, n_roots=1)
    >>> integrals = compute_integrals(geometry, state, Config(verbosity=0))
    >>> result = compute_state(geometry, state, integrals, Config(verbosity=0))
    >>> result.fci_energy
    array([-1.16336172])
    >>> np.allclose(result.fci_energy, result.nuclear_repulsion + result.e_onebody + result.e_twobody)
    True
    >>> geometry.nuclear_charges = np.array([0.5, 1.5])
    >>> integrals = compute_integrals(geometry, state, Config(verbosity=0))
    >>> result = compute_state(geometry, state, integrals, Config(verbosity=0))
    >>> result.fci_energy
    array([-1.57193718])
    >>> np.allclose(result.fci_energy, result.nuclear_repulsion + result.e_onebody + result.e_twobody)
    True
    """

    mol = create_molecule(geometry, state, config)
#    print("mol charge:", mol.charge)
    with NamedTemporaryFile() as chkf:
#        print("check 4")
        mf = scf.RHF(mol)
        mf.chkfile = chkf.name
        if mf._eri is None:
            mf._eri = integrals.eri

        mf.get_hcore = lambda mol=None: integrals.h1

        hf_energy = np.array(mf.kernel())
        # norb = mf.mo_coeff.shape[0]
        # print("norb =", norb)
        nuclear_repulsion = np.array(mol.energy_nuc(charges=geometry.nuclear_charges))
        nuclear_charge_correction = np.array(nuclear_repulsion - mol.energy_nuc())
#        ao_mo_transform = mf.mo_coeff

#        cisolver = fci.FCI(mf)
#        fci_energy, ci_vector = cisolver.kernel(nroots=state.n_roots, davidson_only=True)
#        print("charge correction", nuclear_charge_correction)   
       
        nactive = method["nactive"]
        nelecactive = method["nelecactive"]
        cascisolver = mcscf.CASCI(mf, nactive, nelecactive)
        cascisolver.kernel()
        fci_energy = cascisolver.e_tot
        ci_vector = cascisolver.ci

#        print("fci energy", fci_energy)
        file_id.write(f"{fci_energy:.8f}\n")
#        print("casci energy", fci_energy)
        
        #casci variables
        ao_mo_transform = cascisolver.mo_coeff
        ncas = cascisolver.ncas
        ncore = cascisolver.ncore
        mo_core = ao_mo_transform[:,:ncore]
        mo_cas = ao_mo_transform[:,ncore:ncore+ncas]
        norb = mo_cas.shape[1]
#        print("norb =", norb)

        _,ecore = cascisolver.get_h1cas()
        core_dm = np.dot(mo_core, mo_core.conj().T) * 2
        corevhf = cascisolver.get_veff(cascisolver.mol, core_dm)
       
#        print("mol.nelec", mol.nelec)
#        print("cascisolver.nelec", cascisolver.nelecas)

        # Fix up the nuclear repulsion energy in case of fractional nuclear charges
        hf_energy += nuclear_charge_correction
        fci_energy += nuclear_charge_correction

        fci_energy = np.array([fci_energy] if state.n_roots == 1 else fci_energy)
        ci_vector = np.array([ci_vector] if state.n_roots == 1 else ci_vector)

        n_dets = ci_vector.shape[1]
        e_onebody = np.zeros(state.n_roots)
        e_twobody = np.zeros(state.n_roots)
        W_corr_MO = np.zeros(state.n_roots)
        W_corr_NO = np.zeros(state.n_roots)
        spin_square = np.zeros(state.n_roots)
        spin_multiplicity = np.zeros(state.n_roots)
        rdm12s = {}
        trdm12s = {}
        h1_NO = {}
        eris_NO={}
        eris_NO_phys={}
        rdm12s_NO={}
        rdm2s_NO_phys={}
        NOONs={}
        

        for root in range(state.n_roots):
            # Compute density matrices
            # rdm12s[root] = RDM12s.from_pyscf(
            #     *cisolver.make_rdm12s(ci_vector[root], cisolver.norb, cisolver.nelec)
            # )

            rdm12s[root] = RDM12s.from_pyscf(
                *cascisolver.fcisolver.make_rdm12s(ci_vector[root], cascisolver.ncas, cascisolver.nelecas)
            )

            # Compute expectation value of S^2 as a check
            spin_square[root], spin_multiplicity[root] = spin_square_orthonormal(rdm12s[root])
            if config.verbosity > 0:
                print(
                    f"S^2 of root {root}: {spin_square[root]}, "
                    f"spin multiplicity of root {root}: {spin_multiplicity[root]}"
                )


#calc one-body energy and two-body energy before
            # h1_mo = ao_mo_transform.T @ integrals.h1 @ ao_mo_transform

            # print("one-body energy", rdm12s[root].rdm1s.contract_with(h1_mo))
            # print("two-body energy", fci_energy[root]-rdm12s[root].rdm1s.contract_with(h1_mo)-mol.energy_nuc())

#            h1_NO[root], eris_NO[root], rdm12s_NO[root] = to_NO(integrals, ao_mo_transform, rdm12s[root])
            h1_NO[root], eris_NO[root], rdm12s_NO[root] = to_NO(integrals, mo_cas, rdm12s[root], corehf=corevhf)            

            e_onebody[root], e_twobody[root] = rdm12s_NO[root].contract_with(h1_NO[root], eris_NO[root])
#            print("e-twobody physicist notation", e_twobody[root])
#            print("e-twobody + e-onebody", e_onebody[root]+e_twobody[root])
#            print("one-body energy check", e_onebody[root]) 
#            print("two-body energy check", e_twobody[root])
#            print("total energy check", e_onebody[root]+e_twobody[root]+mol.energy_nuc())
#            print("total energy check", e_onebody[root]+e_twobody[root]+ecore)
# check if everything is PSD
#            print("eris, uu", tools.diag(torch.tensor(eris_NO[root].uu, dtype=torch.float64), upper=True)[0])
#            print("eris, ud", tools.diag(torch.tensor(eris_NO[root].ud, dtype=torch.float64), upper=False)[0]) 
#            print("eris, dd", tools.diag(torch.tensor(eris_NO[root].dd, dtype=torch.float64), upper=True)[0])
            
#            print("rdm2s, uu", tools.diag(torch.tensor(rdm12s_NO[root].rdm2s.uu, dtype=torch.float64), upper=True)[0])
#            print("rdm2s, ud", tools.diag(torch.tensor(rdm12s_NO[root].rdm2s.ud, dtype=torch.float64), upper=False)[0]) 
#            print("rdm2s, dd", tools.diag(torch.tensor(rdm12s_NO[root].rdm2s.dd, dtype=torch.float64), upper=True)[0])
                        
            #check trace
#            tr_uu = np.einsum('ijij->', rdm12s_NO[root].rdm2s.uu)
#            tr_ud = np.einsum('ijij->', rdm12s_NO[root].rdm2s.ud)
#            tr_dd = np.einsum('ijij->', rdm12s_NO[root].rdm2s.dd)
#            print("n_elecs", (tr_uu + tr_dd + 2.0*tr_ud)/(sum(np.array(mol.nelec))-1))

            G_block = tools.compute_G_blocks (rdm12s_NO[root].rdm2s.to_rdm1s_phys(), rdm12s_NO[root].rdm2s) 
            
            Q_block = tools.compute_Q_blocks (rdm12s_NO[root].rdm2s.to_rdm1s_phys(),rdm12s_NO[root].rdm2s) 
            
            
            loss_penalty_D = tools.compute_penalty_function([rdm12s_NO[root].rdm2s.uu, rdm12s_NO[root].rdm2s.ud, rdm12s_NO[root].rdm2s.dd], "D")
            loss_penalty_Q = tools.compute_penalty_function(Q_block,  "Q")       
            loss_penalty_G = tools.compute_penalty_function(G_block, "G")            
            total = loss_penalty_D + loss_penalty_Q + loss_penalty_G
            if abs(total) > 1e-12 :
                print("loss is not zero !!!", total)
                exit()


            NOONs[root] = rdm12s_NO[root].rdm1s.diag()
            rdm1s_check = rdm12s_NO[root].rdm2s.to_rdm1s_phys()
#            print("rdm1s from rdm2s", rdm1s_check.u, rdm1s_check.d)
#            print("rdm1s, ", rdm12s_NO[root].rdm1s.u, rdm12s_NO[root].rdm1s.d)
            
#            print("rdm1s_dim",rdm1s_check.u.shape[0], rdm1s_check.d.shape[0])
            NOONs_error = (
                (rdm12s_NO[root].rdm1s - rdm1s_check)
                .eigvalsh()
                .apply_function(lambda x: np.linalg.norm(x, 1))
            )
            if config.verbosity > 0:
                print(
                    f"RDM1 up error of root {root}: {NOONs_error.u}, "
                    f"RDM1 down error of root {root}: {NOONs_error.d}"
                )

            # Check the computation of the energy

            #S.S check energy by switching to physicists notation
#            eris_NO_phys[root] = eris_NO[root].transpose(0,2,1,3)
#            rdm2s_NO_phys[root] = rdm12s_NO[root].rdm2s.transpose(0,2,1,3)
            
#            tensor = rdm12s_NO[root].rdm2s.ud
#            print("shape of tensor",tensor.shape)
#            matrix = tensor.reshape(tensor.shape[0]**2, tensor.shape[0]**2)
#            print("numpy matrix type",matrix.dtype)
#            print("torch matrix type",torch.tensor(matrix, dtype=torch.float64).dtype)
#            print("RDM2s_ud", torch.linalg.eigh(torch.tensor(matrix,dtype=torch.float64)))
            
#            e_twobody_phy = rdm2s_NO_phys[root].contract_with(eris_NO_phys[root])
#            print("e-twobody physicist notation", e_twobody_phy)
#            print("size of eri tensor :",eris_NO[root].uu.size, eris_NO[root].ud.size, eris_NO[root].dd.size)
#            print("size of 2rdm tensor :",rdm12s_NO[root].rdm2s.uu.size, rdm12s_NO[root].rdm2s.ud.size, 
#                   rdm12s_NO[root].rdm2s.dd.size)
            
            
            if config.verbosity > 0:
                print(
                    f"fullCI energy from NO quantities for root {root}: "
                    f"{nuclear_repulsion + e_onebody[root] + e_twobody[root]}"
                )

            # Check the computation of the correlation energy
#            lambda2s = rdm12s[root].to_cumulant()
#            W_corr_MO[root] = lambda2s.contract_with(
#                TwoBodyTensor.from_8s(mf._eri, ao_mo_transform).anti_symmetrize()
#            )
#            lambda2s_NO = rdm12s_NO[root].to_cumulant()
#            W_corr_NO[root] = lambda2s_NO.contract_with(eris_NO[root])

            if config.verbosity > 0:
                print(f"fullCI correlation energy for root {root} (MO): {W_corr_MO[root]}")
                print(f"fullCI correlation energy for root {root} (NO): {W_corr_NO[root]}")

            for root2 in range(root + 1, state.n_roots):
                trdm12s[(root, root2)] = RDM12s.from_pyscf(
                    *trans_rdm12s(ci_vector[root], ci_vector[root2], norb, mol.nelec)
                )
#        print("check 6")
#        print("e_twobody", e_twobody)
#        print("nuc repulsion energy", np.array([mol.energy_nuc()]))
# =============================================================================
#     return Result(
#         n_dets=n_dets,
#         n_elecs=mol.nelec,
#         hf_energy=hf_energy,
#         fci_energy=fci_energy,
#         nuclear_repulsion=nuclear_repulsion,
#         ao_mo_transform=ao_mo_transform,
#         e_onebody=e_onebody,
#         e_twobody=e_twobody,
#         spin_square=spin_square,
#         spin_multiplicity=spin_multiplicity,
#         w_correlation=W_corr_NO,
#         rdm12s=rdm12s,
#         trdm12s=trdm12s,
#         rdm12s_NO=rdm12s_NO,
#         NOONs=NOONs,
#         rdm2s_phys=rdm2s_NO_phys,
#         eris_NO_phys=eris_NO_phys  # [n_orbs, n_orbs, n_orbs, n_orbs]
#     )
# 
# =============================================================================
    return Result(
#        n_elecs=np.array(mol.nelec),
        n_elecs=np.array(cascisolver.nelecas),
#        nuc_rep_energy=np.array([mol.energy_nuc()]),
        nuc_rep_energy=np.array([ecore]),
        e_onebody = e_onebody,
        e_twobody=e_twobody,
        fci_energy=fci_energy,
        NOONs=NOONs,
        rdm2s_NO_phys=rdm12s_NO,
        eris_NO_phys=eris_NO # [n_orbs, n_orbs, n_orbs, n_orbs]
    )    


def compute_system(system: System, config: Config, filehandle, method) -> Datapoint:
    """
    Perform a quantum chemistry calculation for a given system.

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
    >>> system = System(
    ...     geometry=geometry,
    ...     states=[State(charge=0, spin=0, n_roots=1)],
    ... )
    >>> datapoint = compute_system(system, Config(verbosity=0))
    >>> datapoint.data[system.states[0]].fci_energy
    array([-1.16336172])
    """

    
#    print("check 3")
    integrals = compute_integrals(system.geometry, system.states[0], config)

    data = run_method(method["name"], system, integrals, config, filehandle, method)

    return Datapoint(system=system, data=data, integrals=integrals)   

def run_casci(system, integrals, config, casmethod, fileidx):  
     
    data = {}
    for state in system.states: 
        data[state] = compute_state_casci(system.geometry, state, integrals, config, fileidx, casmethod)
#    print("check 7")    
    
    return data

def run_method(method_name, system, integrals, config, fileid, method):
    
    match method_name:
        case "casci":
            return run_casci(system, integrals, config, method, fileid)
        case _:
            raise ValueError("Unknown method")
            

def get_cli(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="FCI calculation with pyscf")

    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("output", type=str, help="Output file")

    return parser


def main(args: argparse.Namespace) -> None:
    with open(Path(args.input), "r", encoding="utf-8") as fd:
        inp = yaml.safe_load(fd)

    config = Config(**inp.pop("config", {}))
    system = System(Geometry(**inp.pop("geometry")), **inp)

    datapoint = compute_system(system, config)

    datapoint.save(args.output)

    with h5py.File(args.output, "r") as h5:
        h5.visit(lambda name: print(name))


if __name__ == "__main__":
    cli = get_cli()
    main(cli.parse_args())
