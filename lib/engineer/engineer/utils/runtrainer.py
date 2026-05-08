#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:59:58 2026

@author: souloke
"""
import os, shutil, subprocess
from pathlib import Path

DATAROOT = os.environ["DATAROOT"]

def runtraining(filestocopy, yamlfile):
    
    for file_path in filestocopy:
        try:
            shutil.copy(file_path, DATAROOT)
        except FileNotFoundError:
            print(f"File not found: {file_path}")  
    # copy the training filefor debug purpose
    for file in Path(DATAROOT).glob("*_train.h5"):
        new_file = file.with_name(file.stem + "2debug" + file.suffix)
        shutil.copy(file, new_file)
        
        
    command = "sweep_local "+ yamlfile
    
    result = subprocess.call(command, shell=True)
    
    print("Training completed successfully")