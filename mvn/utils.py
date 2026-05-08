#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 12:22:31 2025

@author: souloke
"""

import os
from typing import List,Dict
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import h5py
import itertools
import matplotlib.pyplot as plt
from pathlib import Path

def diag4tensor(tensor: torch.Tensor) -> Dict[str, List[torch.Tensor]]:

    uu = tensor[0]
    ud = tensor[1]
    dd = tensor[2]    
    
    return {"uu":diag(uu, upper=True), "ud": diag(ud, upper=False), "dd": diag(dd, upper=True)}   
    




def diag(tensor4: torch.Tensor, upper : bool = False) -> List[torch.Tensor] :
    
    if upper:
        tensor_matrix = create_matrix_upper(tensor4)
    else:    
        tensor_matrix = create_matrix_full(tensor4)
    
    eigenvalues, eigenvectors = torch.linalg.eigh(tensor_matrix)
    
    return [eigenvalues, eigenvectors]

    
def Cholesky(tensor4: torch.Tensor, upper : bool = False) -> List[torch.Tensor] :
    
    if upper:
        tensor_matrix = create_matrix_upper(tensor4)
    else:    
        tensor_matrix = create_matrix_full(tensor4)
    
    choleskyvectors = torch.linalg.cholesky(tensor_matrix)
    
    return choleskyvectors

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
    

def create_matrix_upper_1 (tensor:torch.Tensor) -> torch.Tensor :
    
    n_orb = tensor.shape[0]
    ij_pairs = [(i, j) for i in range(n_orb) for j in range(i+1, n_orb)]
    n_pairs = len(ij_pairs)
    matrix = torch.zeros((n_pairs, n_pairs),dtype=torch.float64)

    for p, (i, j) in enumerate(ij_pairs):
        for q, (k, l) in enumerate(ij_pairs):
            matrix[p, q] = tensor[i, j, k, l]    
    
    return matrix


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
    pairs = torch.tensor(list(itertools.combinations(range(n), 2)), dtype=torch.long, device=P.device)
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

#    print("shape of N in create matrix upper", N.shape)
    return N




def expand_symmetric_matrix_to_tensor(N_batch, n):
    """
    Expand a batch of symmetric antisymmetric-pair matrices N_batch to full 4-tensors P in PyTorch.

    Parameters
    ----------
    N_batch : torch.Tensor, shape (batch_size, m, m), m = n*(n-1)//2
        Batch of compressed symmetric matrices.
    n : int
        Dimension of the original tensor space.

    Returns
    -------
    P_batch : torch.Tensor, shape (batch_size, n, n, n, n)
        Batch of expanded 4-tensors with correct symmetries.
    """
    batch_size, m, _ = N_batch.shape
    assert _ == m, f"N_batch should be shape (batch_size, {m}, {m})"
    
    # Generate antisymmetric pairs (i<j)
    pairs = torch.tensor(list(itertools.combinations(range(n), 2)), dtype=torch.long, device=N_batch.device)  # (m,2)
    i_idx, j_idx = pairs[:,0], pairs[:,1]
    
    # Create grids for broadcasting
    i_grid, k_grid = torch.meshgrid(i_idx, i_idx, indexing='ij')
    j_grid, l_grid = torch.meshgrid(j_idx, j_idx, indexing='ij')
    
    # Initialize output tensor
    P_batch = torch.zeros(batch_size, n, n, n, n, device=N_batch.device, dtype=N_batch.dtype)
    
    # Fill all antisymmetries at once
    P_batch[:, i_grid, j_grid, k_grid, l_grid] = N_batch
    P_batch[:, j_grid, i_grid, k_grid, l_grid] = -N_batch
    P_batch[:, i_grid, j_grid, l_grid, k_grid] = -N_batch
    P_batch[:, j_grid, i_grid, l_grid, k_grid] = N_batch
    
    return P_batch


def rdm2s_to_rdm1s(rdm2s, tol: float = 10**-12):
       """Constructs 1-RDM from 2-RDM

       NOTE: If there are contributions to the wavefunction or ensemble from a one-body state
       then the 2-RDM does *not* contract to the correct 1-RDM!"""
       batch_size = rdm2s[0].shape[0]
       n_orbs = rdm2s[0].shape[1]


       # Deduce the number of spin up and spin down electrons from the 2-RDM
       n_up_n_up_min_1 = torch.einsum("bpqpq->b", rdm2s[0])  # n_up * (n_up -1)
       n_down_n_down_min_1 = torch.einsum("bpqpq->b", rdm2s[2])  # n_down * (n_down -1)
       n_up_n_down = torch.einsum("bpqpq->b", rdm2s[1])  # n_up * n_down

       # TODO: there are a few unused variables here
       # Depending on whether there are 0, 1 or more than 1 electron of a particular spin
       # we need to choose how to contract the 2-RDM
       # be careful with the batches, make sure all elements in batch have same n_\alpha n_\beta
       rdm1s_u=[]
       rdm1s_d=[]


 #       n_up_n_up_min_1 = n_up_n_up_min_1_batched.mean()
 #       n_down_n_down_min_1 = n_down_n_down_min_1_batched.mean()
 #       n_up_n_down = n_up_n_down_batched.mean()
 
 
 #this might be slow -> find a vectorized version 
       for i in range(batch_size): 
           if n_up_n_up_min_1[i] < tol and n_down_n_down_min_1[i] < tol and n_up_n_down[i] < tol:
               n_up = 0
               n_down = 0
               rdm1s_u.append(torch.zeros((n_orbs, n_orbs) ,dtype=torch.float64, requires_grad=True, device=rdm2s[0].device)) 
               rdm1s_d.append(torch.zeros((n_orbs, n_orbs) ,dtype=torch.float64, requires_grad=True, device=rdm2s[0].device))
               
           elif n_up_n_up_min_1[i] < tol and n_up_n_down[i] < tol:
               n_up = 0
               n_down = (1 + torch.sqrt(1 + 4 * n_down_n_down_min_1[i])) / 2
               rdm1s_u.append(torch.zeros((n_orbs, n_orbs) ,dtype=torch.float64, requires_grad=True, device=rdm2s[0].device))
               rdm1s_d.append(torch.einsum("prqr->pq", rdm2s[2][i]) / (n_down - 1))
               
           elif n_down_n_down_min_1[i] < tol and n_up_n_down[i] < tol:
               n_up = (1 + torch.sqrt(1 + 4 * n_up_n_up_min_1[i])) / 2
               n_down = 0
               rdm1s_u.append(torch.einsum("prqr->pq", rdm2s[0][i]) / (n_up - 1)) 
               rdm1s_d.append(torch.zeros((n_orbs, n_orbs) ,dtype=torch.float64, requires_grad=True, device=rdm2s[0].device))
        
           elif n_up_n_up_min_1[i] < tol and n_down_n_down_min_1[i] < tol:
               n_up = n_down = torch.sqrt(n_up_n_down[i])
               rdm1s_u.append(torch.einsum("prqr->pq", rdm2s[1][i]) / n_down) 
               rdm1s_d.append(torch.einsum("rprq->pq", rdm2s[1][i]) / n_up)
                   
           elif n_up_n_up_min_1[i] < tol:
               n_down = (1 + torch.sqrt(1 + 4 * n_down_n_down_min_1[i])) / 2
               n_up = n_up_n_down[i] / n_down
               rdm1s_u.append(torch.einsum("prqr->pq", rdm2s[1][i]) / n_down)
               rdm1s_d.append(1 / 2 * torch.einsum("prqr->pq", rdm2s[2][i]) / (n_down - 1)
                   + 1 / 2 * torch.einsum("rprq->pq", rdm2s[1][i]) / n_up)
                              
           elif n_down_n_down_min_1[i] < tol:
               n_up = (1 + torch.sqrt(1 + 4 * n_up_n_up_min_1[i])) / 2
               n_down = n_up_n_down[i] / n_up
               rdm1s_u.append(1 / 2 * torch.einsum("prqr->pq", rdm2s[0][i]) / (n_up - 1)
                   + 1 / 2 * torch.einsum("prqr->pq", rdm2s[1][i]) / n_down)
               rdm1s_d.append(torch.einsum("rprq->pq", rdm2s[1][i]) / n_up)
               
           else:
               n_up = (1 + torch.sqrt(1 + 4 * n_up_n_up_min_1[i])) / 2
               n_down = (1 + torch.sqrt(1 + 4 * n_down_n_down_min_1[i])) / 2
               rdm1s_u.append(1 / 2 * torch.einsum("prqr->pq", rdm2s[0][i]) / (n_up - 1)
                   + 1 / 2 * torch.einsum("prqr->pq", rdm2s[1][i]) / n_down)
               rdm1s_d.append(1 / 2 * torch.einsum("prqr->pq", rdm2s[2][i]) / (n_down - 1)
                   + 1 / 2 * torch.einsum("rprq->pq", rdm2s[1][i]) / n_up)
               
       return [torch.stack(rdm1s_u, dim=0), torch.stack(rdm1s_d, dim=0)]            


def rdm2s_to_rdm1s_4H2(rdm2s, tol: float = 10**-12):
       """Constructs 1-RDM from 2-RDM

       NOTE: If there are contributions to the wavefunction or ensemble from a one-body state
       then the 2-RDM does *not* contract to the correct 1-RDM!"""
       batch_size = rdm2s.shape[0]
       n_orbs = rdm2s.shape[1]


       # Deduce the number of spin up and spin down electrons from the 2-RDM
       n_up_n_down = torch.einsum("bpqpq->b", rdm2s)  # n_up * n_down

       # TODO: there are a few unused variables here
       # Depending on whether there are 0, 1 or more than 1 electron of a particular spin
       # we need to choose how to contract the 2-RDM
       # be careful with the batches, make sure all elements in batch have same n_\alpha n_\beta
       rdm1s_u=[]
       rdm1s_d=[]


 #       n_up_n_up_min_1 = n_up_n_up_min_1_batched.mean()
 #       n_down_n_down_min_1 = n_down_n_down_min_1_batched.mean()
 #       n_up_n_down = n_up_n_down_batched.mean()
 
 
 #this might be slow -> find a vectorized version 
       for i in range(batch_size): 

               n_up = n_down = torch.sqrt(n_up_n_down[i])
               rdm1s_u.append(torch.einsum("prqr->pq", rdm2s[i]) / n_down) 
               rdm1s_d.append(torch.einsum("rprq->pq", rdm2s[i]) / n_up)
                   
       return [torch.stack(rdm1s_u, dim=0), torch.stack(rdm1s_d, dim=0)]            



def return_as_dict(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    
    return {"uu":tensor[0], "ud": tensor[1], "dd": tensor[2]}   


    
def plottrainvalepoch(train_data, val_data, rundir):
    
     outputdir = Path(rundir)
     
     plt.figure(figsize=(8, 5))

     plt.plot(train_data.cpu().numpy(), label="Training Loss", color="blue")
     plt.plot(val_data.cpu().numpy(), label="Validation Loss", color="green")

     plt.xlabel("Epoch")
     plt.ylabel("Loss")
     plt.ylim(0, 0.05)
     plt.title("Training and Validation Loss Over Epochs")
     plt.legend()
#     plt.grid(True)

 # Save as PDF (overwrites if exists)
     plt.savefig(outputdir / "losses_over_epochs.pdf")
#     plt.show()
     plt.close()
     
     
def fixed_eigenvectors(eigvecs):
#    vals, vecs = torch.linalg.eigh(A)
    # Make largest component positive
    max_abs_idx = torch.argmax(eigvecs.abs(), dim=0)
    signs = eigvecs[max_abs_idx, torch.arange(eigvecs.shape[1])].sign()
    eigvecs *= signs
    return eigvecs     