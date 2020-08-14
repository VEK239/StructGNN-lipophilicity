# Baseline Neural Fingerprints

## Sources

Article - [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/pdf/1509.09292v2.pdf)

Original Github Repo - https://github.com/HIPS/neural-fingerprint

## Short description

Differentiable generalization of circular fingerprints.

Circular fingerprints are analogous to convolutional networks in that they apply the same operation 
locally everywhere, and combine information in a global pooling step.

## Running instructions 

### Training script

Running with best parameters:

```python training_with_logs.py -n 200 -f 32 -d 3 -c 10 -s -4 -l -3 -t logp_wo_averaging -b 500 -e 150```

Results are stored on the server in `~/alisa/mol_properties/data/raw/baselines/neuralfingerprint`

### Grid Search

Grid search parameters:

|Parameter|Values|
|---|---|
|Learning rate|[-3; -2] (power of exp)|
|L2 penalty|[-4; -1] (power of exp)|
|Number of convolutions|10,20,30|
|Fingerprint depth (radius)|3,4,6|
|Fingerprint length|32,64,128|
|Dataset name|logp_wo_averaging|
|i|starting number of experiments|

```./Neural_fingerprints_grid_search.sh```

## Changes in source code

- [x] Added calculating of R2 score
- [x] Added saving logs and models
- [x] Provided bash script for grid search over the most important parameters
