#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:11:31 2026

@author: souloke
"""
import h5py
import numpy as np
import torch
from typing import Dict,List
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import glob
# Example usage

class Mergehdf5 ():    
    
    def __init__(self, filelist=None):
        
        if filelist is not None:
            self.file_list=filelist
        else:
            pathtohdf5 = "*.h5"
            self.file_list = sorted(glob.glob(pathtohdf5))                
            
        with h5py.File(self.file_list[0], "r") as f:
#    group = f[group_name] if group_name else f
            self.data_paths=[]
            self.data_paths = collect_datasets(f)



    def merge(self, filename=None):
        
        if filename is not None:
            filepath = filename
        else:
            filepath = "merged.h5"
        
        with h5py.File(filepath, 'w') as f_out:
            for dname in self.data_paths:
                # List to hold arrays from all files for this dataset
                data_list = []
                for fname in self.file_list:
                    with h5py.File(fname, 'r') as f:
                        data_list.append(f[dname][...])  # read full dataset

                # Concatenate along first axis (axis=0)
                merged_data = np.concatenate(data_list, axis=0)

                # Save to new file
                f_out.create_dataset(dname, data=merged_data)




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





