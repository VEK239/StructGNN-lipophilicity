"""Trains a chemprop model on a dataset."""

import sys
sys.path.append('./chemprop/')
sys.path.append('./')
from chemprop.train import chemprop_train
import pandas as pd
import os
import numpy as np
import yaml

if __name__ == '__main__':
    DATASET_OUTPUT_PATH = './data/raw/baselines/dmpnn'

    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
        
    DATASET_PATH = params['separate_test_path']

    dataset_train = pd.read_csv(os.path.join(DATASET_PATH, params['file_prefix']+'_train.csv'), index_col=0)
    dataset_val = pd.read_csv(os.path.join(DATASET_PATH, params['file_prefix']+'_validation.csv'), index_col=0)
    dataset_train_val = pd.concat([dataset_train, dataset_val], axis = 0).reset_index(drop = True)
    dataset_train_val.to_csv(os.path.join(DATASET_OUTPUT_PATH,  params['file_prefix']+'_train_val_dataset.csv'),index = False)
    
    chemprop_train()
