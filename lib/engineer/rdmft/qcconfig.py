#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:06:28 2026

@author: souloke
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:41:45 2026

@author: souloke
"""

from pathlib import Path
import yaml, os
from copy import deepcopy

# =========================================================
# NODE: core recursive config object
# =========================================================
class Node:
    def __init__(self):
        self.__dict__["_data"] = {}

    # attribute assignment
    def __setattr__(self, key, value):
        self._data[key] = value

    # attribute access (auto-create nested nodes)
    def __getattr__(self, key):
        if key not in self._data:
            node = Node()
            self._data[key] = node
            return node
        return self._data[key]

    # convert to python dict
    def to_dict(self):
        out = {}
        for k, v in self._data.items():
            if isinstance(v, Node):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

# =========================================================
# BUILDER
# =========================================================
class QCConfigBuilder:
    
    
    def __init__(self, program=None):
        

        self.cfg = Node()
        
        if program == "pyscf":
            self.program = "pyscf"
            self._build_defaults()
        else:
            raise ValueError("Not implemented") 
        
        self.print_header(" RUNNING QM CALCULATIONS (" + self.program + ")")
        
    # -----------------------------------------------------
    # DEFAULT CONFIG (fully structured, no CLI strings)
    # -----------------------------------------------------
    def _build_defaults(self):
        cfg = self.cfg

        # -------------------
        # meta
        # -------------------
        cfg.program = self.program
        cfg.charge = 0
        cfg.spin = 0
        cfg.n_roots = 1
#        cfg.Task.name = "positions_interpolation"



    def build_config(self):
           return(self.cfg.to_dict())
     
    def print_header(self,title, width=40):
         width = int(width)  # ensure it's an int
         print("\n" + "=" * width)
         print(title.center(width))
         print("=" * width)   