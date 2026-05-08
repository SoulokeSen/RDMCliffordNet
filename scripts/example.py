#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:20:56 2026

@author: souloke
"""
import rdmft, preprocess, engineer
import os
## Data generation ##
## Define the task and the .yml file like information here ## eg. Interpolation ###
## a. state here


qcprogram = 'pyscf'
qcbuilder = rdmft.QCConfigBuilder(qcprogram)
qc_cfg = qcbuilder.cfg

qc_cfg.charge = 0
qc_cfg.spin = 0
qc_cfg.n_roots = 1
qc_cfg.Task.name = "torsional_dist_interpolation"
qc_cfg.Task.specs.dihedral = False
qc_cfg.Task.specs.CCCC = False
qc_cfg.Task.specs.start_point = 0.90
qc_cfg.Task.specs.end_point = 5.00
qc_cfg.Task.specs.smiles_mol = "C=C"
qc_cfg.Task.specs.n_steps = 100
qc_cfg.Task.specs.basis = "sto-3g"
#qc_cfg.Task.specs.method = [4,4]
qc_cfg.Task.method.name = "casci"
qc_cfg.Task.method.nactive = 4
qc_cfg.Task.method.nelecactive = 4
qc_cfg.directory="check1"
qc_cfg.tag="ethylene_stretch"
qc_cfg.qc_config.verbosity = 0



cfg = qcbuilder.build_config()
rdmft.generate.main(cfg)


processdata = preprocess.DataPreprocessor (
        nsample=100,
        filepath="check1/ethylene_stretch.h5", 
        prefix="ethylene", 
        savedir="plots", 
        plot_ylim=None, 
        train_frac=0.7, 
        val_frac=0.15
        )
 
#files=[]       
#mergedata = merge.Mergehdf5(files)

## Run QM calculations


#split data into train val and test (additionally plot data)
train, val, test = processdata.splittrainvaltest()
#print(train, val, test)

#mergedata.merge()

#copy files to DATAROOT
mlprogram = "MultivectorNeurons"
mlbuilder = engineer.MLConfigBuilder(mlprogram)

ml_cfg = mlbuilder.cfg
ml_cfg.dataset.module = "data.rdmft.RDMFT"
ml_cfg.model.module = "models.rdmft_clifford_egnn.RDMFTEGNN_C"
ml_cfg.model.nspatialorbs = 4
ml_cfg.loss.module = "losses.loss2RDM.RDM2sLoss"

ml_cfg.trainer.energyOnly = False
#cfg.model.energyOnly = False
ml_cfg.dataset.filename = "ethylene"
ml_cfg.dataset.n_train = 70
ml_cfg.dataset.n_val = 15
ml_cfg.dataset.n_test = 15
# full dotted access everywhere


ml_cfg.parameters.trainer.max_steps = [350]
# cfg.optimizer.lr = 1e-3
ml_cfg.parameters.model.num_layers = [2]

# # modify sweep
# cfg.parameters.trainer.max_steps.values = [35000, 60000]
ymlfile="rdmft_clifford_egnn.yaml"
mlbuilder.to_yaml(ymlfile)

#builder.runtraining("rdmft_clifford_egnn.yaml")

### Run Clifford EGNN ##
engineer.utils.runtraining([train, val, test], ymlfile)

