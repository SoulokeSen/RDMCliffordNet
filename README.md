# [Natural Orbital Funtional Theory Using Geometric Algebra Graph Neural Networks] 


## Requirement and Conda Environment
```
conda create -n mvn
conda activate mvn
conda install python=3.10
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu121.html
pip install pyyaml wandb
conda install -c conda-forge openmm=8.1.1
pip install sidechainnet
pip install pyscf==2.12.1 rdkit numpy==2.2.5  matplotlib h5py==3.15.1
```

Use following commands to set up the folder.
```
cd lib/engineer/
pip install -e .
cd ../../mvn
```

## Code Organization
* `mvn/`: contains the core code snippets.
  * `algebra/`: contains the Clifford Algebra implementation.
  * `configs/`: contains the configuration files. 
  * `data/`: contains necessary (simplicial) data modules.
  * `models/`: contains model and layer implementations.
* `engineer/`: contains the training and evaluation scripts.
* `lib/` contains commands and set up

## Datasets and Experiments

### Datasets
Run `mkdir ./datasets/` to generate a folder for storing datasets of experiments

#### NBody
Download / Generate nbody datasets and move to `./datasets/`

### Experiments
This implementation uses conda environment, change the path of `miniconda/` in `activate.sh` to your local `miniconda/` path and run `sh ../activate.sh`.

#### Instruction to run sweep_local for MVN on pNBody simulation task
* MVN: ```sweep_local configs/nbody_mvn.yaml```
* GVP: ```sweep_local configs/nbody_gvp.yaml```
* CVP: ```sweep_local configs/nbody_cvp.yaml```
* Clifford_EGNN: ```sweep_local configs/nbody_clifford_egnn.yaml```
* EGNN: ```sweep_local configs/nbody_egnn.yaml```


#### Instruction to run sweep_local for MVN on protein structure denoising task
* MVN: ```sweep_local configs/denoise_mvn.yaml```
* GVP: ```sweep_local configs/denoise_gvp.yaml```
* CVP: ```sweep_local configs/denoise_cvp.yaml```
* Clifford_EGNN: ```sweep_local configs/denoise_clifford_egnn.yaml```
* EGNN: ```sweep_local configs/denoise_egnn.yaml```

