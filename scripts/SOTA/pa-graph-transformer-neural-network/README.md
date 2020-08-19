# SOTA PA-Graph-Transformer

## Authors

The repository of PA-Graph-Transformer: https://github.com/benatorc/PA-Graph-Transformer

The paper: https://arxiv.org/pdf/1905.12712v1.pdf

## Changes in source code

### Grid search

We’ve changed train_prop.py and arguments.py in source code to run grid search over model parameters. The arguments are passed in a list in our version.

### R2 score

We’ve also changed train_utils.py to add R2-score calculation. 

### Experiments directory

A small notebook demonstrating train/valid RMSE for best model’s predictions.

### Symmetry features were added

Adding two atomic and one path feature improved symmetric molecules prediction a little bit.