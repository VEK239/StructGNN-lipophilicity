import pandas as pd
import os
import numpy as np
DATASET_PATH = '../../../data/3_final_data/split_data'
DATASET_OUTPUT_PATH = '../../../data/raw/baselines/dmpnn'

dataset_train = pd.read_csv(os.path.join(DATASET_PATH, 'logp_wo_averaging_train.csv'), index_col=0)
dataset_val = pd.read_csv(os.path.join(DATASET_PATH, 'logp_wo_averaging_validation.csv'), index_col=0)
dataset_train_val = pd.concat([dataset_train, dataset_val], axis = 0).reset_index(drop = True)
dataset_train_val.to_csv(os.path.join(DATASET_OUTPUT_PATH, 'logp_wo_averaging_train_val.csv'),index = False)