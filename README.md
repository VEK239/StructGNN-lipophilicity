# Lipophilicity Prediction with Graph Convolutions and Molecular Substructures Representation

Lipophilicity is one of the factors determining the permeability of the cell membrane to a drug molecule. Hence, accurate lipophilicity prediction is an essential step in the development of drugs. 

Earlier we introduced a novel approach to encoding additional graph information by extracting molecular substructures. By adding a set of generalized atomic features of substructures to an established Direct Message Passing Neural Network we were able to achieve a new state-of-the-art result in predicting the two main lipophilicity coefficients, namely logP and logD descriptors. 

We further improved our approach, StructGNN, by adding the edges features to substructures encoder of a molecular graph and used the graph convolutional neural network, WeaveNet, approach to improve the embeddings. The WeaveStructGNN gave us a new state-of-the-art result in predicting the logP descriptor.
In addition to the previous approach, we also improved the substructures representation itself in a couple of ways. The first improvement was to add the symmetry feature based on the atom ranking, where two atoms are to be in one equivalence class if the molecule atoms can be enumerated in the same way starting with any of these atoms. Another implemented approach is adding the distance features to differentiate molecules with similar substructures representation, such as *para-xylene* and *ortho-xylene*. The last improvement is the set of features encoding the count of atoms with each hybridization type in a substructure (*s, sp, sp2, sp3, sp3d, sp3d2*). One by one these additional features did not improve the model performance, but they might be useful for the further research.

For a detailed description of SOTA StructGNN we refer the reader to the paper ["Lipophilicity Prediction with Multitask Learning and Molecular Substructures Representation"](https://arxiv.org/abs/2011.12117).

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Structure of this repository](#structure-of-this-repository)
* [Data](#data)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [StructWeave](#structgnn)
* [StructGNNWithImprovements](#structgnnwithimprovements)

## Structure of this repository

There are 2 main branches:

1. SOTA, a branch for StructGNN with improvements model
2. weave-net-encoder, a branch with StructWeaveModel

This repository was built with the help of

* [StructGNN repo](https://github.com/jbr-ai-labs/lipophilicity-prediction)
* [Junction Tree original repo](https://github.com/benatorc/OTGNN)

## Data

The following datasets have been used: 

| Dataset name | Number of Samples | Description | Sources |
| --- | --- | --- | --- |
| logp_wo_logp_json_wo_averaging | 13688 | All logP datasets except logp.json | Diverse1KDataset.csv, NCIDataset.csv, ochem_full.csv, physprop.csv |
| logd_Lip_wo_averaging | 4166 | Merged datasets w/o strange (very soluble) molecules and standardized SMILES. Between duplicated logD for one SMILES the most common value was chosen | Lipophilicity |
| logp_wo_logp_json_logd_Lip_wo_averaging | 17603 | Merged LogP and LogD datasets, 251 molecules have logP and logD values | logp_wo_logp_json_wo_averaging,<br/>logd_Lip_wo_averaging |

## Prerequisites

To use `chemprop` with GPUs, you will need:
 * cuda >= 8.0
 * cuDNN


## Installation

1. `git clone https://github.com//VEK239/StructGNN-lipophilicity.git`
2. `git checkout SOTA`
3. `cd scripts/SOTA/dmpnn`
4. `conda env create -f environment.yml`
5. `conda activate chemprop`
6. `pip install -e .`


<!-- GETTING STARTED -->
## StructWeave

The StructWeave model is based on WeaveNet(https://arxiv.org/abs/1603.00856).

<!-- USAGE EXAMPLES -->
### Training
0. Checkout the WeaveEncoder branch

1. Set `params.yaml`

  ```
  additional_encoder: True # set StructGNN architecture
  gcn_encoder: True # to run the WeaveNet as an additional encoder
  
  file_prefix: <name of dataset without format and train/test/val prefix>
  split_file_prefix: <name of dataset without format and train/test/val prefix for `train_val_data_preparation.py` script>
  input_path: <path to split dataset>
  data_path: <path to train_val dataset>
  separate_test_path: <path to test dataset>
  save_dir: <path to experiments logs>
 
 
  epochs: <number of training epochs>
  patience: <early stopping patience>
  delta: <early stopping delta>
 
  features_generator: [rdkit_wo_fragments_and_counts]
  no_features_scaling: True
  
  target_columns: <name of target column>
 
  split_type: k-fold
  num_folds: <number of folds>
 
  weave_max_atoms: 8 # max distance to consider for the edges
  weave_hidden_size: 900 # the size of WeaveNet vector result 
  hidden_size: 800 # dmpnn ffn hidden size
  ```
A full list of available arguments can be found in args.py

2. Run `python ./scripts/baseline_improvements/chemprop/train_val_data_preparation.py` - to create dataset for cross-validation procedure
3. Run `python ./scripts/baseline_improvements/chemprop/train.py --dataset_type regression --config_path_yaml ./params.yaml` - to train model

<!-- GETTING STARTED -->
## StructGNNWithImprovements
There are three improvaments implemented:
1. Symmetry feature
The symmetry feature is counted as the average symmetry class within a substructure. Two atoms are in the same symmetry class if the molecule atoms can be enumerated in the same way starting with any of these atoms.
2. Distance features
The distance features are included to differentiate the substructures in the figure. They are splitted into three types:
  * in-to-in
  * in-to-out
  * out-to-out
3. Hybridizaion fetures
The hybridization features are counting how many atoms with specific hybridization type are in a substructure. 
<!-- USAGE EXAMPLES -->
### Training
0. Checkout the SOTA branch

1. Set `params.yaml`

  ```
  additional_encoder: True # set StructGNN architecture
  
  substructures_symmetry_feature: True # to use the symmetry feature
  substructures_extra_features: True # to use extra distance features
  substructures_extra_max_in_to_in: 3 # in-to-in features max distance
  substructures_extra_max_in_to_out: 4 # in-to-out features max distance
  substructures_extra_max_out_to_out: 5 # out-to-out features max distance
  use_hybridization_features: True # to use hybridization features
  
  file_prefix: <name of dataset without format and train/test/val prefix>
  split_file_prefix: <name of dataset without format and train/test/val prefix for `train_val_data_preparation.py` script>
  input_path: <path to split dataset>
  data_path: <path to train_val dataset>
  separate_test_path: <path to test dataset>
  save_dir: <path to experiments logs>
 
 
  epochs: <number of training epochs>
  patience: <early stopping patience>
  delta: <early stopping delta>
 
  features_generator: [rdkit_wo_fragments_and_counts]
  no_features_scaling: True
  
  target_columns: <name of target column>
 
  split_type: k-fold
  num_folds: <number of folds>
 
  substructures_hidden_size: 300 # the size of substructures vector result 
  hidden_size: 800 # dmpnn ffn hidden size
  ```
A full list of available arguments can be found in args.py

2. Run `python ./scripts/SOTA/dmpnn/train_val_data_preparation.py` - to create dataset for cross-validation procedure
3. Run `python ./scripts/SOTA/dmpnntrain.py --dataset_type regression --config_path_yaml ./params.yaml` - to train model
