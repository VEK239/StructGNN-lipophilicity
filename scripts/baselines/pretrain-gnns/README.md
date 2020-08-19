# Baseline Pretrain gnns

## Sources

Article - [STRATEGIES FOR PRE-TRAINING GRAPH NEURAL
NETWORKS](https://arxiv.org/pdf/1905.12265v3.pdf)

Original Github Repo - https://github.com/snap-stanford/pretrain-gnns/

## Short description

Training representations in unsupervised(ZINC, node-level)/supervised(CheMBL, graph-level) autoencoder manner and then apply embeddings to prediction tasks.


For node-level pre-training of GNNs, approach is to use easily-accessible unlabeled data to capture domain-specific knowledge/regularities in the graph->self-supervised methods: Context Prediction (r goal is to pre-train a GNN so that it maps nodes appearing in similar structural contexts to nearby embeddings) and Attribute Masking (randomly mask input node/edge attributes, for example atom types in molecular graphs, by replacing them with special masked indicators. We then apply GNNs to obtain the corresponding node/edge embeddings (edge embeddings can be obtained as a sum of node embeddings of the edge’s end nodes). Finally, a linear model is applied on top of embeddings to predict a masked node/edge attribute).

## Running instructions 

### Training script

Running with best parameters:

```python finetune.py --filename infomax.pth --input_model_file ./model_gin --gnn_type gin --epochs 2000```

Results are stored on the server in `~/alisa/mol_properties/data/raw/baselines/pretrain_gnn/logs`

| Pretrained model |test RMSE | test R2 | val RMSE | val R2 | train RMSE | train R2 |
| --- | --- | --- |  --- | --- | --- | --- |
| gin_infomax.pth |  0.6884 | 0.8593 | 0.6734 | 0.8655 | 0.2903 | 0.9157 |

### Grid Search

Grid search over different kinds of pre-trained and random initialized models

```./finetune.sh```

## Changes in source code

- [x] Added calculating of R2 score
- [x] Provided bash script for grid search over various kinds of models and pre-trained weights
