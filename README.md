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
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```
The package version should be changed according to your need.

## Datasets
The benchmark datasets utilized in this project, PDBbind and CrossDocked, are stored in the Google Drive [data](https://drive.google.com/drive/folders/1pQk1FASCnCLjYRd7yc17WfctoHR50s2r) folder.
### PDBbind
To train the model from scratch, you need to download the preprocessed lmdb file and split file:
* `PDBbind_v2020_pocket10_processed_final.lmdb`
* `PDBbind_pocket10_split.pt`

To process the dataset from scratch, you need to download PDBbind_v2020 from [here](https://drive.google.com/drive/folders/1pQk1FASCnCLjYRd7yc17WfctoHR50s2r), save it in `data`, unzip it, and run the following scripts in `data`:
* [clean_pdbbind.py](data/clean_pdbbind.py) will clean the original dataset, extract the binding affinity and calculate QED, SA, LogP, and TPSA of ligands. It will generate a `index.pkl` file and save it in `data/PDBbind_v2020` folder. *You don't need to do these steps if you have downloaded .lmdb file.*
    ```bash
    python clean_pdbbind.py --source ./PDBbind_v2020
    ```
    
