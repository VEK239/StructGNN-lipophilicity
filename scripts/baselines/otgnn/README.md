# SOTA Optimal Transport Graph Neural Networks

## Sources

Article - [Optimal Transport Graph Neural Networks](https://arxiv.org/pdf/2006.04804v2.pdf)

Original Github Repo - https://github.com/benatorc/OTGNN

## Short description

Replace the graph embedding aggregated from node (atom) embeddings by Wassershtein distances from node vectors to
basis point clouds (of unaggregated node embeddings).

## Running instructions 

### Training script

Running with best parameters:

```python train_proto.py -data logp_wo_aver -output_dir output/exp_200 -lr 5e-4 -n_epochs 100 -n_hidden 50 -n_ffn_hidden 100 -batch_size 16 -n_pc 20 -pc_size 10 -pc_hidden 5 -distance_metric wasserstein -separate_lr -lr_pc 5e-3 -opt_method emd -mult_num_atoms -nce_coef 0.01```

Results are stored on the server in `~/alisa/mol_properties/data/raw/baselines/otgnn/output`

| Number of experiment | Model | Dataset name | test RMSE | test R2 | val RMSE | val <br/>R2 | train RMSE | train R2 | Best epoch | time<br/>min/epoch | Parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | ProtoW-L2<br/>Wassershtein | logp_wo_averaging | 0.4848 | 0.9356 | 0.5047 | 0.9259 | 0.3740 | 0.9595 | 95/100 | 15 | pc_hidden  =5<br/>n_pc=20<br/>n_ffn_hidden = 100<br/>n_hidden = 50<br/>other default |

### Grid Search

Grid search parameters:

|Parameter|Values|
|---|---|
|n_hidden|50; 200|
|n_ffn_hidden|100, 1000, 10000|
|Number of pointclouds|10,20|
|Point cloud dim|5, 10|
|i|starting number of experiments|

```./grid_search.sh```

## Changes in source code

- [x] Added calculating of R2 score
- [x] Provided bash script for grid search over the most important parameters
