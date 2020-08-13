"""Trains a chemprop model on a dataset."""

import sys
sys.path.append('./chemprop/')
sys.path.append('./')
from chemprop.train import chemprop_train


if __name__ == '__main__':
    chemprop_train()
