# Target-aware 3D Molecular Generation Based on Guided Equivariant Diffusion Model
Official implementation of ***DiffGui***, a guided diffusion model for de novo structure-based drug design and lead optimization, by Qiaoyu Hu, et al.

<p align="center">
  <img src="figures/overview.png" /> 
</p>

## Installation

### Install conda environment via yaml file
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

### Install other required softwares
```bash
pip install diffusers==0.21.4 docutils==0.17.1 filelock==3.12.2 fsspec==2023.1.0
pip install softwares/torch_cluster-1.6.1+pt113cu116-cp37-cp37m-linux_x86_64.whl
pip install softwares/torch_scatter-2.1.1+pt113cu116-cp37-cp37m-linux_x86_64.whl
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
* [extract_pockets.py](data/extract_pockets.py) will extract the pocket file from a 10 A region around the binding ligand in the original protein file.
    ```bash
    python extract_pockets.py --source ./PDBbind_v2020 --desti ./PDBbind_v2020_pocket10
    ```
* [split_pl_dataset.py](data/split_pl_dataset.py) will split the train, validation and test set.
    ```bash
    python split_dataset.py --path ./PDBbind_v2020_pocket10 --desti ./PDBbind_pocket10_split.pt --train 17327 --val 1825 --test 100
    ```
