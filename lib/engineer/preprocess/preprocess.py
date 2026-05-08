#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:00:07 2026

@author: souloke
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import os

class DataPreprocessor ():
    
    def __init__(self, nsample=1, filepath=None, prefix=None, savedir=None, plot_ylim=None, train_frac=0.7, val_frac=0.15):
        self.nsample = nsample
        if filepath is not None:
            self.filepath = filepath
        else:
            raise ValueError("filepath cannot be None")
        if prefix is not None:   
            self.prefix=prefix
            self.trainfile = self.prefix+"_train.h5"
            self.valfile = self.prefix+"_val.h5"
            self.testfile = self.prefix+"_test.h5"
        else:
            raise ValueError("prefix cannot be None")  

        if savedir is not None:    
            self.savedir = savedir
        else:
            self.savedir = os.getcwd()
            
        self.output_dir = Path(self.savedir)
        self.output_dir.mkdir(exist_ok=True)
        self.plot_ylim = plot_ylim
        
        self.train_fraction=train_frac
        self.val_fraction=val_frac
        self.print_header(" SPLITTING DATA ")

    def splittrainvaltest(self):
        
        with h5py.File(self.filepath, "r") as f:
        #    group = f[group_name] if group_name else f
                data_paths=[]
                data_paths = collect_datasets(f)
        #        print(data_paths)
                data_dict = {}
                train_idx, val_idx, test_idx = self.generate_interpolation_indices(self.nsample, train_frac=self.train_fraction, val_frac=self.val_fraction)
                train_id = h5py.File(self.trainfile, 'w')
                val_id = h5py.File(self.valfile, 'w')
                test_id = h5py.File(self.testfile, 'w')
                mean_train = 0.0
                std_train = 0.0
                for path in data_paths:
        #            print("f path shape", f[path].shape)
                    if path == "data/0_0/fci_energy" :
                        fci_energy = f[path][:]
                             
                    trainset =  f[path][train_idx]
                    valset = f[path][val_idx]
                    testset = f[path][test_idx]
        # =============================================================================
                    if path == 'data/0_0/rdm2s/0':
        # #                trainset, mean_train, std_train = standardize_overdataset(trainset)
        # #                valset,_,_ = standardize_overdataset(valset, mean=mean_train, std=std_train)
        # #                testset,_,_ = standardize_overdataset(testset, mean=mean_train, std=std_train)
                         self.plotandanalyze_rdm2s_1(trainset[:,1], "train")  
                         self.plotandanalyze_rdm2s_1(valset[:,1], "val")  
                         self.plotandanalyze_rdm2s_1(testset[:,1], "test")
        #                 
        # =============================================================================
        #            if path == 'data/1_0/eris/0':
        #                plotandanalyze_eris_1(trainset[:,1], "train")   
        #                plotandanalyze_eris_1(valset[:,1], "val")
        #                plotandanalyze_eris_1(testset[:,1], "test")   
                    path=strippath(path)    
                    write_split_file(train_id, path, trainset, mean=mean_train, stdev=std_train)
                    write_split_file(val_id, path, valset, mean=mean_train, stdev=std_train)
                    write_split_file(test_id, path, testset, mean=mean_train, stdev=std_train)


                    
                train_id.close()
                val_id.close()
                test_id.close()

        #check if everythng written properly
        ## Plot the fci energies
        plt.figure(figsize=(8,5))
        plt.plot(fci_energy)
        plt.title("fci_energies")
        plt.savefig(self.output_dir / "fci_energy.pdf")   # saves as PDF
        plt.close()
 
        #exit()
        print(" ==== training file ===")
        total_bytes = 0
        with h5py.File(self.trainfile, "r") as f:
                data_paths=[]
                data_paths = collect_datasets(f)
        #        print(data_paths)    
                for path in data_paths:
                    if path == "data/fci_energy" :
                        fci_energy_2train = f[path][:]
                    # print("f path shape", f[path].shape)
                    # print("f path type", f[path].dtype)
                    total_bytes += np.prod(f[path].shape) * f[path].dtype.itemsize
        total_mb = total_bytes / (1024 * 1024)
        print(f"Total data in memory: {total_bytes} bytes ({total_mb:.2f} MB)")
        #print("fci energies", fci_energy_2train)
        plt.figure(figsize=(8,5))
        plt.plot(fci_energy_2train, color="blue", marker='o')
        plt.title("fci_energies")
        if self.plot_ylim is not None:
            plt.ylim(self.plot_ylim)
        plt.savefig(self.output_dir / "fci_energy2train.pdf")   # saves as PDF
        plt.close()
                    
        print(" ==== validation file ===")
        total_bytes = 0
        with h5py.File(self.valfile, "r") as f:
                data_paths=[]
                data_paths = collect_datasets(f)
        #        print(data_paths)    
                for path in data_paths:
                    if path == "data/fci_energy" :
                        fci_energy_2val = f[path][:]
         #           print("f path shape", f[path].shape)
                    total_bytes += np.prod(f[path].shape) * f[path].dtype.itemsize
        total_mb = total_bytes / (1024 * 1024)                        
        print(f"Total data in memory: {total_bytes} bytes ({total_mb:.2f} MB)")
        #print("fci energies", fci_energy_2val)
        plt.figure(figsize=(8,5))
        plt.plot(fci_energy_2val, color="green", marker='o')
        plt.title("fci_energies")
        if self.plot_ylim is not None:
            plt.ylim(self.plot_ylim)
        plt.savefig(self.output_dir / "fci_energy2val.pdf")   # saves as PDF
        plt.close()

        print(" ==== testing file ===")
        total_bytes = 0
        with h5py.File(self.testfile, "r") as f:
                data_paths=[]
                data_paths = collect_datasets(f)
        #        print(data_paths)    
                for path in data_paths:
                    if path == "data/fci_energy" :
                        fci_energy_2test = f[path][:]
        #            print("f path shape", f[path].shape)
                    total_bytes += np.prod(f[path].shape) * f[path].dtype.itemsize
        total_mb = total_bytes / (1024 * 1024)
        print(f"Total data in memory: {total_bytes} bytes ({total_mb:.2f} MB)")
        #print("fci energies", fci_energy_2test)
        plt.figure(figsize=(8,5))
        plt.plot(fci_energy_2test, color="red",marker='o')
        plt.title("fci_energies")
        if self.plot_ylim is not None:
            plt.ylim(self.plot_ylim)
        plt.savefig(self.output_dir / "fci_energy2test.pdf")   # saves as PDF
        plt.close()

        return str(Path(self.trainfile).resolve()), str(Path(self.valfile).resolve()), str(Path(self.testfile).resolve())


    def generate_interpolation_indices(self, n_samples, train_frac=0.5, val_frac=0.2, random_state=42,movefromval2train=None):
            """
            Generate train, validation, and interpolation test indices for a 1-D sorted dataset.
            
            Parameters:
                n_samples (int): Total number of samples (assume sorted along bond distance)
                train_frac (float): Fraction of samples for training
                val_frac (float): Fraction for validation
                random_state (int): Seed for reproducibility
            
            Returns:
                train_idx, val_idx, test_idx (np.ndarray): Sorted indices
            """
            assert 0 < train_frac < 1, "train_frac must be between 0 and 1"
            assert 0 <= val_frac < 1, "val_frac must be between 0 and 1"
            

        # Compute number of points
            n_train = int(n_samples * train_frac)
            n_val   = int(n_samples * val_frac)
            n_test  = n_samples - n_train - n_val  # remaining points

        # --- Interleaved selection for smooth coverage ---
            indices = np.arange(n_samples)
            train_idx = np.linspace(0, n_samples-1, n_train, dtype=int)
        #    train_idx = train_idx[:n_train]  # trim if needed

            remaining_idx = np.setdiff1d(indices, train_idx)
            val_idx = np.linspace(0, len(remaining_idx)-1, n_val, dtype=int)
            val_idx = remaining_idx[val_idx]
            test_idx  = np.setdiff1d(remaining_idx, val_idx)

        # --- Sort indices for plotting / consistent order ---
            train_idx = np.sort(train_idx)
            val_idx   = np.sort(val_idx)
            test_idx  = np.sort(test_idx)
            ## get validation points and add to train points
            if movefromval2train is not None:
                train_idx, val_idx = move_val_to_train(train_idx,val_idx,movefromval2train)
                train_idx = np.sort(train_idx)
                val_idx   = np.sort(val_idx)


            y_train = np.ones(len(train_idx))
            y_val   = np.ones(len(val_idx)) * 2
            y_test  = np.ones(len(test_idx)) * 3

            plt.figure(figsize=(10,2))
            plt.scatter(train_idx, y_train, label='train', color='blue', marker="x", s=10)
            plt.scatter(val_idx, y_val, label='val', color='green', marker="o", s=20)
            plt.scatter(test_idx, y_test, label='test', color='red', marker="s", s=20)

            plt.yticks([1,2,3], ['train','val','test'])
            plt.xlabel('Index in original array')
            plt.title('Train/Val/Test index distribution')
            plt.legend()
            plt.savefig(self.output_dir / 'train_val_test_indices.pdf', bbox_inches='tight')
            plt.close()
            
            return train_idx, val_idx, test_idx
        
    def plotandanalyze_rdm2s_1(self, batch_data, typestr):

        
        batch_size = batch_data.shape[0]    
        batch_1 = batch_data[:batch_size//2]
        batch_2 = batch_data[batch_size//2:]
        data1 = batch_1.reshape(-1)
        data2 = batch_2.reshape(-1)
    #    print("total non-zero entries", np.sum(np.abs(data) > 1e-12))
        data_nonzero_1 = data1[np.abs(data1) > 1e-8]
        data_nonzero_2 = data2[np.abs(data2) > 1e-8]
    #    print("size of data_nonzero", data_nonzero.size)
        sparsity1 = data_nonzero_1.size/data1.size
        sparsity2 = data_nonzero_2.size/data2.size
        with open(self.output_dir / "sparsity.txt", "w") as f:
            f.write(f"{sparsity1:.8f}\n")
            f.write(f"{sparsity2:.8f}")
            
    #    print("min and max in the whole set of entries", np.abs(data_nonzero).min(), np.abs(data_nonzero).max())
        mean1 = data_nonzero_1.mean()
        var1 = data_nonzero_1.var()
        with open(self.output_dir / "stats1.txt", "w") as f:
            f.write(f"{mean1:.8f}\n")
            f.write(f"{var1:.8f}")       

#        print("mean and variance in the whole set of entries data1", data_nonzero_1.mean(), data_nonzero_1.var())
        exponents = np.arange(np.floor(np.log10(np.abs(data_nonzero_1).min())),
                          np.ceil(np.log10(np.abs(data_nonzero_1).max())) + 1)
        bins = 10**exponents
    # Assign each value to a bin (cluster)
        counts, _ = np.histogram(np.abs(data_nonzero_1), bins=bins)
        bin_indices = np.digitize(np.abs(data_nonzero_1), bins) - 1  # -1 to make 0-based

    # Prepare x positions for plotting clusters
    # Each cluster is at the bin’s log value
        x_positions = [np.log10(bins[i]) for i in bin_indices]

    # Plot using a scatter for visual clusters
        plt.figure(figsize=(10,5))
        plt.scatter(x_positions, data_nonzero_1, s=50, alpha=0.7)
        
    # Overlay counts as a bar plot on the same x-axis
        plt.twinx()  # create a second y-axis
        plt.bar(np.log10(bins[:-1]), counts, width=0.5, alpha=0.2, color='orange', align='edge')
        plt.ylabel('Count in bin')
        
        
        plt.xticks(np.log10(bins), [f'1e{int(e)}' for e in exponents])
        plt.xlabel('Order of magnitude (cluster)')
        plt.ylabel('Value')
        plt.title('Tensor entries clustered by order of magnitude')
        plt.grid(True, alpha=0.3)
        name="scatter_rdm2s_ud_1"+typestr
        plt.savefig(self.output_dir / name)
        plt.close()

        mean2 = data_nonzero_2.mean()
        var2 = data_nonzero_2.var()
        with open(self.output_dir / "stats2.txt", "w") as f:
            f.write(f"{mean2:.8f}\n")
            f.write(f"{var2:.8f}") 

#        print("mean and variance in the whole set of entries data2", data_nonzero_2.mean(), data_nonzero_2.var())

        exponents = np.arange(np.floor(np.log10(np.abs(data_nonzero_2).min())),
                          np.ceil(np.log10(np.abs(data_nonzero_2).max())) + 1)
        bins = 10**exponents
    # Assign each value to a bin (cluster)
        counts, _ = np.histogram(np.abs(data_nonzero_2), bins=bins)
        bin_indices = np.digitize(np.abs(data_nonzero_2), bins) - 1  # -1 to make 0-based

    # Prepare x positions for plotting clusters
    # Each cluster is at the bin’s log value
        x_positions = [np.log10(bins[i]) for i in bin_indices]

    # Plot using a scatter for visual clusters
        plt.figure(figsize=(10,5))
        plt.scatter(x_positions, data_nonzero_2, s=50, alpha=0.7)
        
    # Overlay counts as a bar plot on the same x-axis
        plt.twinx()  # create a second y-axis
        plt.bar(np.log10(bins[:-1]), counts, width=0.5, alpha=0.2, color='orange', align='edge')
        plt.ylabel('Count in bin')
        
        
        plt.xticks(np.log10(bins), [f'1e{int(e)}' for e in exponents])
        plt.xlabel('Order of magnitude (cluster)')
        plt.ylabel('Value')
        plt.title('Tensor entries clustered by order of magnitude')
        plt.grid(True, alpha=0.3)
        name="scatter_rdm2s_ud_2"+typestr
        plt.savefig(self.output_dir / name)
        plt.close()
        
        
    def print_header(self, title, width=40):
        width = int(width)  # ensure it's an int
        print("\n" + "=" * width)
        print(title.center(width))
        print("=" * width)  

def write_split_file(dst, datapath, entries, **kwargs):
 
    if datapath in dst:
        print(f"Dataset '{datapath}' already exists. skipped")
    else:
  
 
        dset = dst.create_dataset(datapath, data=entries)
        dset.attrs["mean"] = kwargs["mean"]
        dset.attrs["std"] = kwargs["stdev"]
        print(f"Dataset '{datapath}' created successfully.")  
        
def collect_datasets(group, prefix=""):
        """Recursively collect all dataset paths under a group."""
        paths = []
        for name, item in group.items():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(item, h5py.Dataset):
                paths.append(path)
            elif isinstance(item, h5py.Group):
                    paths.extend(collect_datasets(item, prefix=path))
        return paths
    
def move_val_to_train(train_idx, val_idx, selected_val_positions):
        # Convert to numpy array if not already
        selected_val_positions = np.array(selected_val_positions)

        # Get actual dataset indices from val_idx using positions
        selected_dataset_indices = val_idx[selected_val_positions]

        # Add selected indices to train_idx
        train_idx = np.concatenate((train_idx, selected_dataset_indices))

        # Create mask to keep only unselected positions
        mask = np.ones(len(val_idx), dtype=bool)
        mask[selected_val_positions] = False

        # Filter val_idx
        val_idx = val_idx[mask]

        return train_idx, val_idx



def strippath(pathtotensor):
        segments = pathtotensor.split("/")

        clean_path = "/".join(
            segment for i, segment in enumerate(segments)
            if i != 1 and not segment.isdigit()
            )

        return clean_path 
    
    