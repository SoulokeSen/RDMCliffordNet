import random
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import itertools

TensorDict = Dict[str, torch.Tensor]


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(np.prod(p.shape) for p in module.parameters()))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_dict_to_device(td: TensorDict, device: torch.device) -> TensorDict:
    return {k: v.to(device) for k, v in td.items()}


def to_numpy(t: torch.Tensor) -> npt.NDArray:
    return t.cpu().detach().numpy()


def init_device(device: str) -> torch.device:
    if device == "cuda":
        assert torch.cuda.is_available(), "No CUDA device available!"
        print(f"CUDA version: {torch.version.cuda}, CUDA device: {torch.cuda.current_device()}")
        return torch.device("cuda")

    return torch.device("cpu")


def flatten_dict_generator(
    d: Dict[str, Any],
    separator: str,
    parent_key: Optional[str] = None,
) -> Generator[Tuple[str, Any], None, None]:
    for key, value in d.items():
        if parent_key is not None:
            key = parent_key + separator + key
        if isinstance(value, dict):
            yield from flatten_dict_generator(value, separator=separator, parent_key=key)
        else:
            yield key, value


def flatten_dict(
    d: Dict[str, Any],
    separator: str,
    parent_key: Optional[str] = None,
) -> Dict[str, Any]:
    return {k: v for k, v in flatten_dict_generator(d, separator, parent_key)}


def get_or_create_resizable_dataset(group, name, shape, dtype):
    """Create dataset if not exists; return it otherwise."""
    if name in group:
        return group[name]
    else:
        maxshape = (None,) + shape  # allow unlimited growth on first axis
        return group.create_dataset(name, shape=(0,) + shape, maxshape=maxshape, dtype=dtype)
        
def appendtohdf5(h5file,dataset_name,data):     
                # Append new data
    dset = get_or_create_resizable_dataset(h5file, dataset_name, data.shape, data.dtype)        
    old_size = dset.shape[0]
    new_size = old_size + 1
    dset.resize((new_size, *data.shape))
    dset[old_size:new_size, ...] = data

def diag(tensor4: torch.Tensor, upper= False) :
    
    if upper:
        tensor_matrix = create_matrix_upper(tensor4)
    else:    
        tensor_matrix = create_matrix_full(tensor4)
    
    eigenvalues, eigenvectors = torch.linalg.eigh(tensor_matrix)
    
    return [eigenvalues, eigenvectors]
  
def create_matrix_full (P:torch.Tensor) -> torch.Tensor :
    
    single_tensor = False
    if P.ndim == 4:
        P = P.unsqueeze(0)  # add batch dimension
        single_tensor = True

    batch_size, n, _, _, _ = P.shape

    N = P.reshape(batch_size, n**2, n**2)
    
    if single_tensor:
        N = N.squeeze(0)  # remove batch dimension    
# Ensure the matrix is Hermitian (or real symmetric)
#    matrix = 0.5 * (matrix + matrix.T)   
    
    return N
    

def create_matrix_upper(P:torch.Tensor) -> torch.Tensor :
    """
    P: (n, n, n, n), antisymmetric in (i<->j) and (k<->l)
    Returns:
        N: (m, m) where m = n(n-1)/2
    """

    single_tensor = False
    if P.ndim == 4:
        P = P.unsqueeze(0)  # add batch dimension
        single_tensor = True

    batch_size, n, _, _, _ = P.shape

    # Build antisymmetric index pairs (i<j)
    pairs = torch.tensor(list(itertools.combinations(range(n), 2)), dtype=torch.long)
    m = pairs.shape[0]

    i_idx = pairs[:, 0]      # (m,)
    j_idx = pairs[:, 1]      # (m,)

    # Build grids of shape (m, m)
    i_grid, k_grid = torch.meshgrid(i_idx, i_idx, indexing='ij')
    j_grid, l_grid = torch.meshgrid(j_idx, j_idx, indexing='ij')

    # Extract the canonical components: P[i_a, j_a, i_b, j_b]
    N = P[:,i_grid, j_grid, k_grid, l_grid]   # → shape (m, m)

    if single_tensor:
        N = N.squeeze(0)  # remove batch dimension


    return N

def compute_G_blocks (RDM1s, RDM2s):
    
    norbs = RDM1s.u.shape[0]
#    batch_size = RDM1s[0].shape[0]

    I = np.eye(norbs)
    # G^{αα, αα}
    G_aaaa = np.einsum('ik,jl->ijkl', RDM1s.u, I) \
            - RDM2s.uu.transpose(0, 3, 2, 1)
#   G_blocks['alphaalpha_alphaalpha'] = G_aaaa

    # G^{ββ, ββ}
    G_bbbb = np.einsum('ik,jl->ijkl', RDM1s.d, I) \
            - RDM2s.dd.transpose(0, 3, 2, 1)
    
    G_aabb =  RDM2s.ud.transpose(0, 2, 3, 1)
#   G_blocks['betabeta_betabeta'] = G_bbbb
    
    # G^{αβ, αβ}
    G_bbaa =  RDM2s.ud.transpose(1, 3, 2, 0)
    
#    print("is same blocks G_aabb and G_bbaa", np.allclose(G_aabb, G_bbaa, atol=1e-12))


    G_abab = np.einsum('ik,jl->ijkl', RDM1s.u, I) \
            - RDM2s.ud.transpose(0, 3, 2, 1)  
  
    G_baba = np.einsum('ik,jl->ijkl', RDM1s.d, I) \
            - RDM2s.ud.transpose(3, 0, 1, 2) 


    return   {"G_aaaa":G_aaaa,"G_bbbb":G_bbbb,"G_aabb":G_aabb,
              "G_bbaa":G_bbaa,"G_abab":G_abab,"G_baba":G_baba}
    
    
def compute_Q_blocks (RDM1s, RDM2s):   
    
    norbs = RDM1s.u.shape[0]
#    batch_size = RDM1s[0].shape[0]
    
    Q_aaaa = np.zeros_like(RDM2s.uu)
    Q_bbbb = np.zeros_like(RDM2s.dd)
    Q_abab = np.zeros_like(RDM2s.ud)
    
  #Q_aaaa
    I = np.eye(norbs)
    Q_aaaa += np.einsum('ik,jl->ijkl', I, I)
    Q_aaaa -= np.einsum('il,jk->ijkl', I, I)

    Q_aaaa -= np.einsum('ik,jl->ijkl', I, RDM1s.u)
    Q_aaaa += np.einsum('il,jk->ijkl', I, RDM1s.u)
    Q_aaaa += np.einsum('jk,il->ijkl', I, RDM1s.u)
    Q_aaaa -= np.einsum('jl,ik->ijkl', I, RDM1s.u)

    # Add Daa^T (note the index reversal for klij)
    Q_aaaa += RDM2s.uu.transpose(2, 3, 0, 1)
    
#Q_bbbb
        
    Q_bbbb += np.einsum('ik,jl->ijkl', I, I)
    Q_bbbb -= np.einsum('il,jk->ijkl', I, I)

    Q_bbbb -= np.einsum('ik,jl->ijkl', I, RDM1s.d)
    Q_bbbb += np.einsum('il,jk->ijkl', I, RDM1s.d)
    Q_bbbb += np.einsum('jk,il->ijkl', I, RDM1s.d)
    Q_bbbb -= np.einsum('jl,ik->ijkl', I, RDM1s.d)

    # Add Daa^T (note the index reversal for klij)
    Q_bbbb += RDM2s.dd.transpose(2, 3, 0, 1)  
    
    
#Q_abab

    Q_abab += np.einsum('ik,jl->ijkl', I, I)
    Q_abab -= np.einsum('ik,jl->ijkl', I, RDM1s.d)
    Q_abab -= np.einsum('jl,ik->ijkl', I, RDM1s.u)

    Q_abab += RDM2s.ud.transpose(2, 3, 0, 1)       
    
    return [Q_aaaa, Q_abab, Q_bbbb]
    

def softened_l1_norm(x, epsilon= 1e-6):
    
    return torch.sqrt(epsilon**2 + x**2) - epsilon
        

def trace_spin_block (RDM2s):
    return torch.einsum('bijij->b', RDM2s)

def compute_penalty_function (M, tensortype, tol=1e-12):
    
    if tensortype == "D" or tensortype == "Q":
        
        eig_ud = diag(torch.tensor(M[1],dtype=torch.float64))[0]
        eig_u = diag (torch.tensor(M[0],dtype=torch.float64),True)[0]
        eig_d = diag (torch.tensor(M[2],dtype=torch.float64),True)[0]
        
        loss_penalty = torch.where(eig_ud < -tol, softened_l1_norm(eig_ud), torch.zeros_like(eig_ud)).sum()
        loss_penalty += torch.where(eig_u < -tol, softened_l1_norm(eig_u), torch.zeros_like(eig_u)).sum()
        loss_penalty += torch.where(eig_d < -tol, softened_l1_norm(eig_d), torch.zeros_like(eig_d)).sum()

        return loss_penalty

    else:

        
        M_uudd = torch.cat([
            torch.cat([create_matrix_full(torch.tensor(M["G_aaaa"],dtype=torch.float64)),
                       create_matrix_full(torch.tensor(M["G_aabb"],dtype=torch.float64))], dim=1),
            torch.cat([create_matrix_full(torch.tensor(M["G_bbaa"],dtype=torch.float64)), 
                       create_matrix_full(torch.tensor(M["G_bbbb"],dtype=torch.float64))], dim=1),
            ], dim=0)        

        eig_uudd, _ = torch.linalg.eigh(M_uudd)
#        print("is symmetric big block", torch.allclose(M_uudd, M_uudd.T, atol=1e-12))
#        print("eigenvalues of the big block", eig_uudd)

#        eig_uuuu = diag(torch.tensor(M["G_aaaa"],dtype=torch.float64))[0]
#        eig_dddd = diag(torch.tensor(M["G_bbbb"],dtype=torch.float64))[0]
#        eig_uudd = diag(torch.tensor(M["G_aabb"],dtype=torch.float64))[0]
#        print("eigenvalues of the G_aaaa", eig_uuuu)
#        print("eigenvalues of the G_bbbb", eig_dddd)
#        print("eigenvalues of the G_aabb", eig_uudd)
        
        eig_udud = diag(torch.tensor(M["G_abab"],dtype=torch.float64))[0]
        eig_dudu = diag(torch.tensor(M["G_baba"],dtype=torch.float64))[0]
        

        loss_penalty = torch.where(eig_uudd < -tol, softened_l1_norm(eig_uudd), torch.zeros_like(eig_uudd)).sum()
        loss_penalty += torch.where(eig_udud < -tol, softened_l1_norm(eig_udud), torch.zeros_like(eig_udud)).sum()
        loss_penalty += torch.where(eig_dudu < -tol, softened_l1_norm(eig_dudu), torch.zeros_like(eig_dudu)).sum()

        return loss_penalty
        