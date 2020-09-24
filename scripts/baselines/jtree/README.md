# Baseline Junction Tree Encoder

## Sources

Article - [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://arxiv.org/pdf/1802.04364%5D)

Original Github Repo - https://github.com/wengong-jin/icml18-jtnn

## Short description

Encodes subgraphs of molecule taken from vocabulary;
Molecule is encoded into 2 vectors: z_T (encoded tree, so to encode clusters with no info about their connections) and z_G (MPNN), then concat.

Original article trains autoencoder, so to decrease Reconstruction loss, here we use encoder to predict logP value.

## Running instructions 

### Data preparation

1. Get SMILES which produces substructeres which are not presented in pre-made JTree Vocabulary with ```mol_properties/notebooks/baselines/jtree/2_encode_molecules.ipynb```. they should be added to ```mol_properties/data/raw/baselines/jtree/train_errs.txt(val_errs.txt, test_errs.txt)```




### Training script

Running with best parameters:

1. Tree and Graph encoder with pretrained weights
```python train_encoder.py --input_model_file "../../../../icml18-jtnn/molvae"  --filename "exp_3" --epochs 500 --model_name "MPNVAE-h450-L56-d3-beta0.005/model.iter-4"```
2. Only Tree encoder with pre-trained weights
```python train_tree_encoder.py --filename "exp_9" --epochs 200 --input_model_file "../../../../icml18-jtnn/molvae" --model_name "MPNVAE-h450-L56-d3-beta0.001/model.iter-4" --patience 35```
3. Tree and Graph encoder with additional initial atom features (atom mass and hybridization)
```python train_encoder_more_atom_feats.py --filename "exp_11" --epochs 200 --patience 35```

Results are stored on the server in `~/alisa/mol_properties/data/raw/baselines/jtree/logs`



### Cross Validation

1. Tree and Graph encoder with pretrained weights
```python train_encoder_CV.py --filename "exp_12" --epochs 200 --patience 35 --input_model_file "../../../../icml18-jtnn/molvae" --model_name "MPNVAE-h450-L56-d3-beta0.005/model.iter-4"```
2. Tree and Graph encoder with additional initial atom features (atom mass and hybridization)
```python train_encoder_more_atom_feats_CV.py --filename "exp_14" --epochs 200 --patience 35```
3. Tree and Graph encoder with additional initial atom features (atom mass and hybridization) and with vocabulary made based on our dataset
```python train_encoder_more_atom_feats_CV.py --filename "exp_14" --epochs 200 --patience 35 --vocab_path '../../../../icml18-jtnn/data/logp/vocab.txt' ```

| Number of experiment | Encoder | Parameters | test RMSE | test R2 | val RMSE | val R2 | train RMSE | train R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14 | MPNN with additional features(atomic mass and hybridization)+TreeEncoder | emb_dim = 56<br/>batch_size  = 32<br/>learn_rate = 0.0007<br/>num_layer = 3<br/>hiddensize = 450<br/>patience = 35<br/><br/>Random weights initialization | 0.5044+-0.0052 | 0.9235+-0.0014 | 0.5194+-0.0087 | 0.9222+-0.0020 |  0.1612+-0.0126 | 0.9738+-0.0040 |



## Changes in source code

- [x] Creating logP predicting model from Encoder part of JTree VAE
- [x] Added calculating of R2 score
- [x] Added logs writing
- [x] Recreate MoleculeDataset generation according to format of our dataset
- [x] Added Early Stopping
- [x] Added Cross-validation
- [x] Added loading pretrained encoder weights
- [x] Added training of only tree encoder
- [x] Added training of encoder with additional atom initial features




