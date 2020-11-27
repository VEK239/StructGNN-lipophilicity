"""Trains a chemprop model on a dataset."""
import sys
sys.path.append('/home/mol/liza/mol_properties')

from train import chemprop_train


if __name__ == '__main__':
    chemprop_train()
