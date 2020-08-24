# Baseline D-MPNN

## Sources

Article - [Analyzing Learned Molecular Representations for Property Prediction](https://arxiv.org/pdf/1904.01561v5.pdf)

Original Github Repo - https://github.com/chemprop/chemprop

## Short description

The algorithm is based on the MPNN (Message Passing Neural Networks), which consist of message passing phase (several NN layers, which propogate hidden states of vertex of molecule graph from node to node in T steps to create final vector representation (molecule descriptor) of the molecule) and readout phase (where the predictions of value of interest are made). As an input MPNN gets set of node (atom) and bonds features.

## Running instructions 

### Training script

Running with best parameters:

```python train.py --data_path ../../../data/raw/baselines/dmpnn/logp_wo_averaging_train.csv --dataset_type regression --save_dir ../../../data/raw/baselines/dmpnn/logs/exp_20 --separate_val_path ../../../data/raw/baselines/dmpnn/logp_wo_averaging_validation.csv --separate_test_path ../../../data/raw/baselines/dmpnn/logp_wo_averaging_test.csv --epochs 100 --depth 6 --features_generator rdkit_2d_normalized --no_features_scaling```

Results are stored on the server in `~/alisa/mol_properties/data/raw/baselines/dmpnn/logs`

| Number of experiment | Parameters | test RMSE | test R2 | Dataset Name |
| --- | --- | --- | --- | --- |
| 201 | DMPNN with RDKit features (include MolLogP) | 0.461395 +/- 0.007687 | 0.936766 +/- 0.002100 | logp_wo_averaging |


## Changes in source code

- [x] Added calculating of R2 score
- [x] Added rdkit_2d_normalized_best features generator (uses only features named in file `mol_properties/data/raw/baselines/dmpnn//RDKitBestfeatures.txt`)
- [x] Added k-fold cross-validation option
- [x] Debug of rdkit features without default normalization scaling (coupling with inf values in calculated features)

