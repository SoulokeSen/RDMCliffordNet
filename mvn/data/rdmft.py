import os
from typing import List,Dict
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import h5py
from itertools import product, combinations
import utils
from collections import defaultdict
from torch.utils.data import Sampler

DATAROOT = os.environ["DATAROOT"]
# For now we store in one hdf5 file all tha data, so you have 3 files, training, val and test
# for each use __getitem__ for lazy loading each point and preprocess persample
# try pre-loading all in __init__ and preprocessing in __init__ all together
# format in which dta is stored here dict of lsts of training points:
# eg. {"eri_uu":[tensor_1, tensor2,...tensorN],"rdm_uu":[tensor1, tensor2,...tensorN]}
# read as dictionary and access from __getitem__ as eri_uu_idx = file["eri_uu"][ix]
# sample eaach point as a dict : sample = {"eri_uu" : tensor1, "rdm_uu": tensor1,..} for point 1
# diagonzalize each spin block of this eri tensor with this sample 
# concatenate the three spin blocks, add zeros in channels wherever necessary
# make node feature tensor and return graph data object. Ask Cong how do i pass these 5 items to the graph data object
# ask about the equivariance thingy also along with the explanation of the arguments and their dimensional requirements in the nbody experiment.

#=============================================================================
class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        keylist = ["rdm2s_uu", "rdm2s_ud", "rdm2s_dd", "rdm1s_u", "rdm1s_d", "eris_uu", "eris_ud", "eris_dd", "n_elecs"]
        if key in keylist:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class RDMFTDataset:

    """
    PyTorch Geometric Dataset reading hierarchical HDF5 data.

    Each lowest-level dataset is treated as a tensor for Data(x=..., y=...).
    """

    def __init__(self, filename, max_samples=None, group_name=None, transform=None,
                  state=None,root=None, num_samples=1):
        """
        Args:
            filename (str): Path to HDF5 file.
            group_name (str, optional): If given, only load datasets under this group (e.g., 'train').
            transform (callable, optional): Applied on Data objects on-the-fly.
            pre_transform (callable, optional): Applied on Data objects at creation time (optional).
        """
 #       super().__init__(None, transform, pre_transform)
        self.filename = filename
        self.group_name = group_name
        self.data_paths = []
        self.length = 0
        self.num_samples = num_samples
#        self.metadata={}
#        self.transform_tensor = transform_tensor
        self.state =state
        self.root=root
        self.rawdata={}
        self.preprocess_data={}
        self.dataname=[]
        # Open file temporarily to collect datasets
        with h5py.File(self.filename, "r") as f:
            group = f[group_name] if group_name else f
            self.data_paths = self._collect_datasets(group)
            # Check all datasets have same length along axis 0
            lengths = [f[path].shape[0] for path in self.data_paths]
            assert len(set(lengths)) == 1, f"All datasets must have the same number of points! Got {lengths}"
            self.length = lengths[0]
            for path in self.data_paths:
                dataset = f[path]
                arr = np.empty(dataset.shape, dtype=np.float64)
                dataset.read_direct(arr)
                tensor = torch.tensor(arr, dtype=torch.float64)
#                key_name = "_".join(path.split("/")[1:])
                key_name = path.split("/")[1]
                self.dataname.append(key_name)
                self.rawdata[key_name] = tensor
         
        temp_dict=defaultdict(list)
        for i in range(self.num_samples):
            data_i = {}
            for name in self.dataname:
                data_i[name] = self.rawdata[name][i]
                
            graphdata_i = self.get_graph_data(data_i) 
                     
            # store results
            for k, v in graphdata_i.items():
                # ensure first dimension exists (batch dim)
                v = v.unsqueeze(0)  
                temp_dict[k].append(v)

                    
        self.preprocess_data = {k: torch.cat(v_list, dim=0) for k, v_list in temp_dict.items()}    


    def _collect_datasets(self, group, prefix=""):
        """Recursively collect all dataset paths under a group."""
        paths = []
        for name, item in group.items():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(item, h5py.Dataset):
                paths.append(path)
            elif isinstance(item, h5py.Group):
                paths.extend(self._collect_datasets(item, prefix=path))
        return paths

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return a single Data object for index idx."""
#        print("Fetching index:", idx)
#        with h5py.File(self.filename, "r") as f:
        data_dict = {}
        for name in self.preprocess_data.keys():

            data_dict[name] = self.preprocess_data[name][idx]

        if isinstance(data_dict['num_nodes'], torch.Tensor) and data_dict['num_nodes'].ndim == 0:
          data_dict['num_nodes'] = data_dict['num_nodes'].item()            


            # Construct PyG Data object
#        return MyData(**graph_data)
        return MyData(**data_dict)

    def get_graph_data(self, datas : Dict):

#pre-process and prepare the input node embeddings    #for now we assume only one charge_spin_root
#1. process the eris and embed channels 
#        prefix = f"{self.state[0]}_{self.state[1]}"
#        suffix = f"{self.root}"

        eris_tmp = datas["eris"]
        
        # eris_uu = datas["eris_cd_uu"]
        # eris_ud = datas["eris_cd_ud"]
        # eris_dd = datas["eris_cd_dd"]
        n_u = eris_tmp[0].shape[0]
        n_d = eris_tmp[2].shape[0]
        # n_u = datas["rdm2s"][0].shape[0]
        # n_d = datas["rdm2s"][2].shape[0]
        num_nodes = n_u + n_d
#        print("num_nodes", num_nodes)
#        eris_evd = {"uu": utils.diag(eris[0], upper=True), 
#                    "ud": utils.diag(eris[1], upper=False), 
#                    "dd": utils.diag(eris[2], upper=True)}
#        embed_eris_eigvecs, embed_eris_eigvals = make_node_embeddings_eris(eris_evd,n_u ,n_d)

        eris_evd_tmp = {"uu": utils.Cholesky(eris_tmp[0], upper=True), 
                    "ud": utils.Cholesky(eris_tmp[1], upper=False), 
                    "dd": utils.Cholesky(eris_tmp[2], upper=True)}

        # eris_evd = {"uu": eris_uu, 
        #             "ud": eris_ud, 
        #             "dd": eris_dd}

##check if these two are same dictionaries
        # for k in eris_evd_tmp:
        #     if not torch.allclose(eris_evd_tmp[k], eris_evd[k], atol=1e-12, rtol=1e-12):
        #         diff = torch.max(torch.abs(eris_evd_tmp[k] - eris_evd[k]))
        #         print(f"Difference at idx={index}, key={k}, max diff={diff}")    
        # for k in eris_evd_tmp:
        #     diff = torch.max(torch.abs(eris_evd_tmp[k] - eris_evd[k]))
        #     print(index, k, diff)
        
#        embed_eris_cholvecs = make_node_embeddings_eris_cholesky(eris_evd,n_u ,n_d)
        embed_eris_cholvecs = make_node_embeddings_eris_cholesky(eris_evd_tmp,n_u ,n_d)

        noons = datas["NOONs"].reshape(n_u+n_d,1)


#        print("n_elecs", datas[prefix+"_n_elecs"].shape)
        return {"noons":noons, 
#            "eigval_uu": embed_eris_eigvals[0],
#            "eigval_ud": embed_eris_eigvals[1],
#            "eigval_dd": embed_eris_eigvals[2],
#            "eigvec_uu": embed_eris_eigvecs[0],
#            "eigvec_ud": embed_eris_eigvecs[1],
 #           "eigvec_dd": embed_eris_eigvecs[2],
                "node_embed_eigevec":  embed_eris_cholvecs,
#                "node_embed_eigeval":  embed_eris_eigvals,
                "rdm2s_uu": datas["rdm2s"][0],
                "rdm2s_ud": datas["rdm2s"][1],
                "rdm2s_dd": datas["rdm2s"][2],
                "rdm1s_u": datas["rdm1s"][0],
                "rdm1s_d": datas["rdm1s"][1],
                "e_twobody": datas["e_twobody"][self.root],
                "e_onebody": datas["e_onebody"][self.root],
                "fci_energy": datas["fci_energy"][self.root],
                "n_elecs" : datas["n_elecs"],
                "e_nuclear_repulsion" : datas["e_nuclear_repulsion"][0],
                "eris_uu": datas["eris"][0],
                "eris_ud": datas["eris"][1],
                "eris_dd": datas["eris"][2],
                "edge_index": fully_connected_edge_index(num_nodes),
                "num_nodes": torch.tensor(num_nodes)
                    }


class RDMFT:
    def __init__(
        self,
        num_samples=100,
        batch_size=10,
        charge=0,
        spin=0,
        root=0,
        pathroot=None,
        filename=None,
        n_train=1,
        n_val=1,
        n_test=1        
    ):
        
        if pathroot is not None:
            dataroot = os.path.join(os.environ["DATAROOT"], pathroot)
        else:
            raise ValueError("pathroot cannot be None")

#        dataroot = os.path.join(os.environ["DATAROOT"], "rdmft/400pts_6_31g/ethylene_stretch/casci_4_4/withprecompChol")

        if filename is not None:
            self.train_dataset = RDMFTDataset(
                    os.path.join(dataroot, filename+"_train.h5"), 
                    max_samples=num_samples, state=[charge, spin],root=root, num_samples=n_train
                    )
            self.valid_dataset = RDMFTDataset(
                os.path.join(dataroot, filename+"_val.h5"), 
                max_samples=num_samples, state=[charge, spin],root=root, num_samples=n_val  
                )

            self.test_dataset = RDMFTDataset(
                    os.path.join(dataroot, filename+"_test.h5"), 
                    max_samples=num_samples, state=[charge, spin],root=root,num_samples=n_test
                    )

            self.traindebug_dataset = RDMFTDataset(
                    os.path.join(dataroot, filename+"_train2debug.h5"), 
                    max_samples=num_samples, state=[charge, spin],root=root,num_samples=n_train
                    )
        else:
            raise ValueError("filename cannot be None")
        
            
        self.batch_size = batch_size

    def train_loader(self):
#        g = torch.Generator()
#        g.manual_seed(42)  # fixed seed for DataLoader shuffle
#        fixed_indices = torch.randperm(280, generator=g).tolist()
#        sampler = FixedSampler(fixed_indices)
        return PyGDataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )


    def val_loader(self):
        
        return PyGDataLoader(
                self.valid_dataset, batch_size=self.batch_size, shuffle=False
            )

    def test_loader(self):
        return PyGDataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False
            )

    def traindebug_loader(self):
          extra_gen = torch.Generator()
          extra_gen.manual_seed(42)
          return PyGDataLoader(
                  self.traindebug_dataset, batch_size=self.batch_size, shuffle=False, generator=extra_gen
              )   

class FixedSampler(Sampler):
    """Sampler that returns elements in a fixed order."""
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

    
    


# =============================================================================
# def global_to_local_index_map():
#     
#     norbs = 20
#     ij_pairs = [(i, j) for i in range(n_orb) for j in range(i+1, n_orb)]
#     n_pairs = len(ij_pairs)
#     
# =============================================================================

def make_node_embeddings_eris(eri_featues,n_u, n_d):
    
    uu_dim = eri_featues["uu"][1].shape[0]
    dd_dim = eri_featues["dd"][1].shape[1]
    ud_dim =  eri_featues["ud"][1].shape[1]   
    c_dim = uu_dim + dd_dim + ud_dim
#    eigvec_concat, eigval_concat = embed_eris(eri_features)
    num_nodes = n_u + n_d  #we assume same number alpha and beta orbitals
    node_eigvec_uu = torch.zeros((num_nodes,uu_dim,uu_dim), dtype=torch.float64)  
    node_eigvec_ud = torch.zeros((num_nodes,ud_dim,ud_dim), dtype=torch.float64)
    node_eigvec_dd = torch.zeros((num_nodes,dd_dim,dd_dim), dtype=torch.float64)
    
    
    node_eigval_uu = eri_featues["uu"][0].unsqueeze(0).repeat(num_nodes, 1)
    node_eigval_ud = eri_featues["ud"][0].unsqueeze(0).repeat(num_nodes, 1)
    node_eigval_dd = eri_featues["dd"][0].unsqueeze(0).repeat(num_nodes, 1)
    
    ij_norb_u = list(combinations([i for i in range(n_u)], 2)) 
    ij_norb_d = list(combinations([i for i in range(n_u,num_nodes)], 2))
    
    #replace with itertools.combinations
    # split this for different \alpha and \beta orbitals
    
    a = list(range(n_u))
    b = list(range(n_u,num_nodes))
    ab = list(product(a, b))

    for i in range(n_u):
        indices_norb = [j for j, t in enumerate(ij_norb_u) if t[0] == i]
        if indices_norb:
            s = slice(indices_norb[0], indices_norb[-1]+1)
            node_eigvec_uu[i,s,:] = eri_featues["uu"][1][s,:]
            
    for i in range(n_d):
       indices_norb = [j for j, t in enumerate(ij_norb_u) if t[0] == i]
       if indices_norb:
           s = slice(indices_norb[0], indices_norb[-1]+1) 
           node_eigvec_dd[n_u+i,s,:] = eri_featues["dd"][1][s,:]

 
    for i in range(n_u): 
        indices_norb = [j for j, t in enumerate(ab) if t[0] == i]
        s = slice(indices_norb[0], indices_norb[-1]+1)
        node_eigvec_ud[i,s,:] = eri_featues["ud"][1][s,:]




    eigvec_uu = torch.transpose(node_eigvec_uu, 1, 2)   # (nodes,channels,bivectors)
    eigvec_ud = torch.transpose(node_eigvec_ud, 1, 2)
    eigvec_dd = torch.transpose(node_eigvec_dd, 1, 2)
    
    eigvec_embed = embed_eigvecs([eigvec_uu, eigvec_ud, eigvec_dd])
    eigval_embed = embed_eigvals([node_eigval_uu, node_eigval_ud, node_eigval_dd])
#symmetrize over nodes now

#    eigvec_embed_sym = torch.zeros((num_nodes,c_dim,c_dim), dtype=torch.float64)

#just loop over the vectors themselves 

    x = [i for i in range(n_u)]
    y = [i for i in range(n_u,num_nodes)]
    nodes = x + y
    x_iter = list(combinations(x, 2))
    y_iter = list(combinations(y, 2))
    xy = list(product(x, y))
    tot_bas=x_iter+xy+y_iter   
    
    for i in nodes:
        for j in range(i):
            col = tot_bas.index((j,i))
            eigvec_embed[i,:,col] = eigvec_embed[i,:,col] + eigvec_embed[j,:,col]


    return eigvec_embed, eigval_embed


def make_node_embeddings_eris_cholesky(eri_featues,n_u, n_d):
    
    uu_dim = eri_featues["uu"].shape[0]
    dd_dim = eri_featues["dd"].shape[1]
    ud_dim =  eri_featues["ud"].shape[1]   
    c_dim = uu_dim + dd_dim + ud_dim
#    eigvec_concat, eigval_concat = embed_eris(eri_features)
    num_nodes = n_u + n_d  #we assume same number alpha and beta orbitals
    node_eigvec_uu = torch.zeros((num_nodes,uu_dim,uu_dim), dtype=torch.float64)  
    node_eigvec_ud = torch.zeros((num_nodes,ud_dim,ud_dim), dtype=torch.float64)
    node_eigvec_dd = torch.zeros((num_nodes,dd_dim,dd_dim), dtype=torch.float64)
    
    
#    node_eigval_uu = eri_featues["uu"][0].unsqueeze(0).repeat(num_nodes, 1)
#    node_eigval_ud = eri_featues["ud"][0].unsqueeze(0).repeat(num_nodes, 1)
#    node_eigval_dd = eri_featues["dd"][0].unsqueeze(0).repeat(num_nodes, 1)
    
    ij_norb_u = list(combinations([i for i in range(n_u)], 2)) 
    ij_norb_d = list(combinations([i for i in range(n_u,num_nodes)], 2))
    
    #replace with itertools.combinations
    # split this for different \alpha and \beta orbitals
    
    a = list(range(n_u))
    b = list(range(n_u,num_nodes))
    ab = list(product(a, b))

    for i in range(n_u):
        indices_norb = [j for j, t in enumerate(ij_norb_u) if t[0] == i]
        if indices_norb:
            s = slice(indices_norb[0], indices_norb[-1]+1)
            node_eigvec_uu[i,s,:] = eri_featues["uu"][s,:]
            
    for i in range(n_d):
       indices_norb = [j for j, t in enumerate(ij_norb_u) if t[0] == i]
       if indices_norb:
           s = slice(indices_norb[0], indices_norb[-1]+1) 
           node_eigvec_dd[n_u+i,s,:] = eri_featues["dd"][s,:]

 
    for i in range(n_u): 
        indices_norb = [j for j, t in enumerate(ab) if t[0] == i]
        s = slice(indices_norb[0], indices_norb[-1]+1)
        node_eigvec_ud[i,s,:] = eri_featues["ud"][s,:]




    eigvec_uu = torch.transpose(node_eigvec_uu, 1, 2)   # (nodes,channels,bivectors)
    eigvec_ud = torch.transpose(node_eigvec_ud, 1, 2)
    eigvec_dd = torch.transpose(node_eigvec_dd, 1, 2)
    
    eigvec_embed = embed_eigvecs([eigvec_uu, eigvec_ud, eigvec_dd])
#    eigval_embed = embed_eigvals([node_eigval_uu, node_eigval_ud, node_eigval_dd])
#symmetrize over nodes now

#    eigvec_embed_sym = torch.zeros((num_nodes,c_dim,c_dim), dtype=torch.float64)

#just loop over the vectors themselves 

    x = [i for i in range(n_u)]
    y = [i for i in range(n_u,num_nodes)]
    nodes = x + y
    x_iter = list(combinations(x, 2))
    y_iter = list(combinations(y, 2))
    xy = list(product(x, y))
    tot_bas=x_iter+xy+y_iter   
    
    for i in nodes:
        for j in range(i):
            col = tot_bas.index((j,i))
            eigvec_embed[i,:,col] = eigvec_embed[i,:,col] + eigvec_embed[j,:,col]


#    return eigvec_embed, eigval_embed
    return eigvec_embed



def embed_eigvecs(tensor_list: List[torch.Tensor]) :
    

    uu_dim = tensor_list[0].shape[2]
    dd_dim = tensor_list[2].shape[2]
    ud_dim =  tensor_list[1].shape[2]
    cdim = uu_dim + dd_dim + ud_dim
    tot_node = tensor_list[0].shape[0]
    
    uu_embed = torch.zeros((tot_node,cdim,uu_dim), dtype=torch.float64)
    dd_embed = torch.zeros((tot_node,cdim,dd_dim), dtype=torch.float64)
    ud_embed = torch.zeros((tot_node,cdim,ud_dim), dtype=torch.float64)
    
    uu_embed[:,0:uu_dim,:] = tensor_list[0]
    ud_embed[:,uu_dim:uu_dim+ud_dim,:] = tensor_list[1]
    dd_embed[:,uu_dim+ud_dim:cdim,:] = tensor_list[2]

    return torch.cat((uu_embed, ud_embed, dd_embed), dim=2) 

def embed_eigvals (tensor_list: List[torch.Tensor]):
    
     
     return torch.cat((tensor_list[0], tensor_list[1], tensor_list[2]), dim=1)    

  
def fully_connected_edge_index(num_nodes):
    row = []
    col = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # skip self-loops
                row.append(i)
                col.append(j)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index


