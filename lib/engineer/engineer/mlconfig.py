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
# FLOW LIST (ONLY for parameter values)
# =========================================================
class FlowList(list):
    pass


def represent_flow_list(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        data,
        flow_style=True
    )


yaml.SafeDumper.add_representer(FlowList, represent_flow_list)
# =========================================================
# BUILDER
# =========================================================
class MLConfigBuilder:
    
    META_KEYS = {"name", "project", "entity", "program", "method", "parameters"}
    
    def __init__(self, program=None):
        self.cfg = Node()
        
        if program == "MultivectorNeurons":
            self.program = program
            self._build_defaults()
        else:
            raise ValueError("Not implemented") 
            
        self.print_header("TRAINING ML MODEL (" + self.program + ")")
    # -----------------------------------------------------
    # DEFAULT CONFIG (fully structured, no CLI strings)
    # -----------------------------------------------------
    def _build_defaults(self):
        cfg = self.cfg

        # -------------------
        # meta
        # -------------------
        cfg.name = "rdmft"
        cfg.project = "rdmft-clifford"
        cfg.entity = "badboyz2000"
        cfg.program = "/Users/souloke/Programs/RDMCliffordNet/mvn/rdmft.py"
        cfg.method = "grid"

        # -------------------
        # trainer
        # -------------------
        cfg.trainer.module = "engineer.trainer.Trainer"
#        cfg.trainer.val_check_interval = 35
        cfg.trainer.train_metrics = ["loss"]
        cfg.trainer.test_metrics = ["loss"]
        cfg.trainer.gradclip = "adaptive"
#        cfg.trainer.log_interval = 35
        cfg.trainer.energyOnly = False
#        cfg.trainer.max_steps = 35000  # from sweep

        # -------------------
        # dataset
        # -------------------
        cfg.dataset.module = "data.rdmft.RDMFT"
#        cfg.dataset.batch_size = 10
        cfg.dataset.pathroot = os.environ["DATAROOT"]
        cfg.dataset.filename = "ethylene"
        cfg.dataset.n_train = 350
        cfg.dataset.n_val = 75
        cfg.dataset.n_test = 75

        # -------------------
        # optimizer
        # -------------------
        cfg.optimizer.module = "torch.optim.AdamW"
        cfg.optimizer.foreach = False
        cfg.optimizer.min_lrs = False

        # -------------------
        # model
        # -------------------
        cfg.model.module = "models.rdmft_clifford_egnn.RDMFTEGNN_C"
        cfg.model.nspatialorbs = 4
        
        # -------------------
        # loss
        # -------------------
        cfg.loss.module = "losses.loss2RDM.RDM2sLoss"
#        cfg.loss.printstep = 35
        cfg.loss.energyOnly = False
#        cfg.model.weight_reg = 1e-4
        cfg.loss.regtype = "L1"
        
#        cfg.model.num_layers = 3

        # -------------------
        # parameters (REAL dotted tree)
        # -------------------
        cfg.parameters.trainer.max_steps = [35000]
        cfg.parameters.optimizer.lr = [5e-3]
        cfg.parameters.optimizer.wlr = [5e-3]
        cfg.parameters.model.num_layers = [3]
        cfg.parameters.seed = [0]
        cfg.parameters.dataset.batch_size = [10]
        cfg.parameters.optimizer.decay_steps = [0.25]
        cfg.parameters.optimizer.weight_decay = [1e-4]
        cfg.parameters.loss.weight_reg = [1e-4]

    # =====================================================
    # COMMAND GENERATION (NO stored CLI strings)
    # =====================================================
    def _flatten_for_command(self, node, prefix=""):
        out = {}

        for k, v in node._data.items():

            if k in self.META_KEYS:
                continue

            key = f"{prefix}.{k}" if prefix else k

            if isinstance(v, Node):
                out.update(self._flatten_for_command(v, key))
            else:
                out[key] = v

        return out

    def _build_command(self):
        flat = self._flatten_for_command(self.cfg)

        cmd = [
            "${env}",
            "${interpreter}",
            "${program}",
            "--dtype=float64",
        ]

        for k, v in flat.items():
            
            if k.startswith("parameters"):
               continue
           
            if isinstance(v, list):
                v = "[" + ",".join(map(str, v)) + "]"
            cmd.append(f"--{k}={v}")

        cmd.append("${args}")
        return cmd

    # =====================================================
    # PARAMETERS EXPORT (dotted keys)
    # =====================================================
    def _build_parameters(self):
        out = {}

        def walk(node, path=""):
            for k, v in node._data.items():
                new_path = f"{path}.{k}" if path else k

                if isinstance(v, Node):
                    walk(v, new_path)
                else:
                    if isinstance(v, list):
                        v = FlowList(v)
                    out[new_path] = {"values": v}

        walk(self.cfg.parameters)
        return out
    # =====================================================
    # FINAL OUTPUT
    # =====================================================
    def _apply_constraintsandpresets(self):
        cfg = self.cfg

        rules = [
            self._set_OnlyenergyinLoss
            ]

        for rule in rules:
            rule(cfg)


    def _set_OnlyenergyinLoss(self, cfg):
        if cfg.trainer.energyOnly is False:
            cfg.loss.energyOnly = False
        
        
    def to_dict(self):
        self._apply_constraintsandpresets()
        return {
            "name": self.cfg.name,
            "project": self.cfg.project,
            "entity": self.cfg.entity,
            "program": self.cfg.program,
            "method": self.cfg.method,
            "command": self._build_command(),
            "parameters": self._build_parameters(),
        }

    def to_yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, Dumper=yaml.SafeDumper)
            
    def print_header(self, title, width=40):
        width = int(width)  # ensure it's an int
        print("\n" + "=" * width)
        print(title.center(width))
        print("=" * width)       
     