"""Trains a chemprop model on a dataset."""
import sys
sys.path.append('./')
sys.path.append('./chemprop/')

import os


from train import chemprop_train


if __name__ == '__main__':
    chemprop_train()
