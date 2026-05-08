import argparse
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from . import qc_pyscf, configure
from . import data
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import SetBondLength, GetBondLength,SetDihedralDeg, GetDihedralDeg
import shutil
from pathlib import Path

STATE = ['charge', 'spin', 'n_roots']

def single_geometry(*args: Any, **kwargs: Any) -> List[data.Geometry]:
    return [data.Geometry(*args, method="fci", **kwargs)]  # type: ignore[misc]


def positions_interpolation(
    symbols: List[str],
    nuclear_charges: List[float],
    positions_start: List[List[float]],
    positions_end: List[List[float]],
    n_steps: int,
    basis: str,
    dihedral=False,
    butane=False    
) -> List[data.Geometry]:
    weights = np.linspace(1.0, 0.0, n_steps)

    start_array = np.array(positions_start)
    end_array = np.array(positions_end)
    assert start_array.shape == end_array.shape
    assert start_array.shape[-1] == end_array.shape[-1] == 3

    geometries = []
    for weight in weights:
        positions = weight * start_array + (1 - weight) * end_array
#        print("positons:", positions)
        geometry = data.Geometry(
            symbols=symbols,
            nuclear_charges=nuclear_charges,
            positions=positions,
            method="fci",
            basis=basis,
        )
        geometries.append(geometry)

    return geometries

def Getconformer(molecule,atomids, point, dihderal=True):
    
    conf = Chem.Conformer(molecule.GetConformer(0))
    # Set the dihedral
    if dihderal:
        SetDihedralDeg(conf, *atomids, float(point))  
    else:
        SetBondLength(conf, *atomids, float(point)) 
        
    conf_id = molecule.AddConformer(conf, assignId=True)
    new_conf = molecule.GetConformer(conf_id)
    return np.array([list(new_conf.GetAtomPosition(a)) for a in range(molecule.GetNumAtoms())])

# 4️⃣ Find neighboring carbons to define dihedral
def get_ring_neighbor(atom_idx, exclude_idx,mol):
    atom = mol.GetAtomWithIdx(atom_idx)
    for nbr in atom.GetNeighbors():
        if nbr.GetIdx() != exclude_idx and nbr.GetSymbol() == 'C':
            return nbr.GetIdx()


def Getindices(molecule,dihedral,butane):
    
    #for butane
    if dihedral:
        if butane:
            carbon_atoms = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetSymbol() == 'C']
    # For butane, central dihedral is C0–C1–C2–C3
            i, j, k, l = carbon_atoms[0], carbon_atoms[1], carbon_atoms[2], carbon_atoms[3]

            return [i,j,k,l]  
        else:
            central_bond = None
            for bond in molecule.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                
                # both carbons, not aromatic bond (this is the inter-ring bond)
                if (a1.GetSymbol() == 'C' and a2.GetSymbol() == 'C'
                    and not bond.GetIsAromatic()):
                    central_bond = bond
                    break

            j = central_bond.GetBeginAtomIdx()
            k = central_bond.GetEndAtomIdx()

            i = get_ring_neighbor(j, k, molecule)
            l = get_ring_neighbor(k, j, molecule)
            return [i,j,k,l]
    else:
        cc_bond = None
        for bond in molecule.GetBonds():
            if bond.GetBeginAtom().GetSymbol() == 'C' and bond.GetEndAtom().GetSymbol() == 'C':
                cc_bond = bond
                break

        c1 = cc_bond.GetBeginAtomIdx()
        c2 = cc_bond.GetEndAtomIdx()
        return [c1,c2]


def torsion_dist_interpolation(
#    symbols: List[str],
#    nuclear_charges: List[float],
    start_point: float,
    end_point: float,
    smiles_mol : str,
    n_steps: int,
    basis: str,
    dihedral=False,
    CCCC=False
) -> List[data.Geometry]:
    
    setdihedral=dihedral
    butane=CCCC
    
    start = start_point
    end = end_point 
    increment = (end-start)/n_steps
    dihedral_angles = np.arange(start, end, increment)
    mol = Chem.MolFromSmiles(smiles_mol)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    # weights = np.linspace(1.0, 0.0, n_steps)

    # start_array = np.array(positions_start)
    # end_array = np.array(positions_end)
    # assert start_array.shape == end_array.shape
    # assert start_array.shape[-1] == end_array.shape[-1] == 3
    atom_ids = Getindices(mol,setdihedral,butane)
    geometries = []
    nuclear_charges = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    for angles in dihedral_angles:
        
        coords = Getconformer(mol,atom_ids, angles, setdihedral)

        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
#        print("positons:", coords)
        geometry = data.Geometry(
            symbols=symbols,
            nuclear_charges=nuclear_charges,
            positions=coords,
            method="fci",
            basis=basis,
        )
        geometries.append(geometry)

    return geometries


def nuclear_charges_interpolation(
    symbols: List[str],
    nuclear_charges_start: List[List[float]],
    nuclear_charges_end: List[List[float]],
    positions: List[List[float]],
    n_steps: int,
    basis: str,
) -> List[data.Geometry]:
    weights = np.linspace(1.0, 0.0, n_steps)

    start_array = np.array(nuclear_charges_start)
    end_array = np.array(nuclear_charges_end)
    assert start_array.shape == end_array.shape

    geometries = []
    for weight in weights:
        nuclear_charges = weight * start_array + (1 - weight) * end_array
        geometry = data.Geometry(
            symbols=symbols,
            nuclear_charges=nuclear_charges,
            positions=np.array(positions),
            method="fci",  #useless
            basis=basis,
        )
        geometries.append(geometry)

    return geometries


def generate_datapoints_pyscf(
    geometries: List[data.Geometry],
    states: List[data.State],
    config: configure.Config,
    method
) -> List[data.Datapoint]:
    datapoints = []
   
    with open("energy.txt", "w") as f:
        for idx, geometry in enumerate(geometries):
            system = data.System(geometry, states)
#        print("check 2")   
                     
            datapoint = qc_pyscf.compute_system(system, config, f, method)
            datapoints.append(datapoint)
            print("Computed geom", idx, "successfully")
    return datapoints


# here store datapoints list in one hdf5 file as,  {"eri_uu":[tensor_1, tensor2,...tensorN],"rdm_uu":[tensor1, tensor2,...tensorN]}
# create dataset, i.e. key for only the first point and then just append each datapoint to corresponding key afterwards 
def store_datapoints(datapoints: List[data.Datapoint], directory: Optional[str], tag: str) -> str:
    if directory is None:
        tmp_dir = os.path.join("/tmp", f"{tag}-{uuid.uuid1()}")
        os.makedirs(tmp_dir, exist_ok=False)
        directory = tmp_dir

    directory = os.path.expanduser(directory)
    print(f"Writing {len(datapoints)} datapoint(s) to directory: {directory}")
#    print("check 3")
    for idx, datapoint in enumerate(datapoints):
#        print("check 9")
        datapoint.save(path=os.path.join(directory, f"{tag}_{idx}.h5"))

    return directory

def store_datapoints_singleFile(datapoints: List[data.Datapoint], directory: Optional[str], tag: str) -> str:
    if directory is None:
        tmp_dir = os.path.join("/tmp", f"{tag}-{uuid.uuid1()}")
        os.makedirs(tmp_dir, exist_ok=False)
        directory = tmp_dir

    directory = os.path.expanduser(directory)
    print(f"Writing {len(datapoints)} datapoint(s) to directory: {directory}")
#    print("check 3")
    path=os.path.join(directory, f"{tag}.h5")
    with h5py.File(path, "w") as h5:
 #       h5.attrs["n_elecs"] = self.n_elecs
        for idx, datapoint in enumerate(datapoints):
 #           print("check 9")
            datapoint.save_h5_singleFile(h5)

    return directory


def get_cli(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Generate datapoints with PySCF.")

    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Configuration of datapoint generation. "
        "Specify configuration file like so: config=<yaml_file>. "
        "By specifying a.b=x you override settings in the configuration file.",
    )

    return parser


name_to_geometries_generator_dict: Dict[str, Callable[..., List[data.Geometry]]] = {
    "single_geometry": single_geometry,
    "positions_interpolation": positions_interpolation,
    "nuclear_charges_interpolation": nuclear_charges_interpolation,
    "torsional_dist_interpolation":torsion_dist_interpolation
}


def generate_dataset(config: Dict[Any, Any]) -> str:
    geometries_generator_config = config["Task"]
    method = config["Task"]["method"]

    try:
        geometries = name_to_geometries_generator_dict[geometries_generator_config["name"]](
            **geometries_generator_config["specs"]
        )
    except KeyError as e:
        raise RuntimeError(
            f"Unknown generator {e}. "
            f"Choose from: {', '.join(name_to_geometries_generator_dict.keys())}."
        ) from e

#    states = [data.State(**kwargs) for kwargs in config["states"]]
    states = [data.State(**{k: config[k] for k in STATE})]
    qc_config = configure.Config(**config["qc_config"])
  
    
    datapoints = run_program(config["program"], geometries, states, qc_config, method)
#    datapoints = generate_datapoints(geometries, states, qc_config, config["program"])
    

    output_dir = Path(config["directory"])

    if output_dir.exists():
        shutil.rmtree(output_dir)  # remove entire directory

    output_dir.mkdir()
#    print("check 1")
#    directory = store_datapoints(datapoints, directory=config["directory"], tag=config["tag"])
    directory = store_datapoints_singleFile(datapoints, directory=str(output_dir), tag=config["tag"])   
    return directory

def run_program(program_name, geometries, states, qc_config, method):
    match program_name:
        case "pyscf":
            return generate_datapoints_pyscf(geometries, states, qc_config, method)
        case _:
            raise ValueError("Unknown program")
            

def main(config: dict) -> str:
#    config = utilities.get_config(args=args)
    print(" ==== CONFIG ====")
    print(config)
    print("")
#    exit()
    directory = generate_dataset(config)
    return directory


if __name__ == "__main__":
    main(get_cli().parse_args())
