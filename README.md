# Target-aware 3D Molecular Generation Based on Guided Equivariant Diffusion Model
Official implementation of **DiffGui**, a guided diffusion model for de novo structure-based drug design and lead optimization, by Qiaoyu Hu, et al.

<p align="center">
  <img src="figures/overview.png" /> 
</p>

## Installation

### Install environment via conda yaml file
```bash
# Create the environment
conda env create -f env.yml
# Activate the environment
conda activate diffgui
```

### Install Vina Docking
```bash
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```
