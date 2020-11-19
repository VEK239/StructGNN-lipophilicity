import torch
from torch import nn
from torch.nn import functional
from .embed_atom_id import EmbedAtomID

MAX_ATOMIC_NUM = 170
WEAVE_DEFAULT_NUM_MAX_ATOMS = 170

from typing import List, Union

import numpy as np
from rdkit import Chem

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from args import TrainArgs
from features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph, \
    BatchMolGraphWithSubstructures, get_atom_fdim_with_substructures, \
    mol2graph_with_substructures


WEAVENET_DEFAULT_WEAVE_CHANNELS = [50, ]


class LinearLayer(nn.Module):

    def __init__(self, n_channel, n_layer):
        super(LinearLayer, self).__init__()

        self.layers = [nn.Linear(None, n_channel) for _ in range(n_layer)]
        self.n_output_channel = n_channel

    def forward(self, x):
        n_batch, n_atom, n_channel = x.shape
        x = functional.reshape(x, (n_batch * n_atom, n_channel))
        for l in self.layers:
            x = l(x)
            x = functional.relu(x)
        x = functional.reshape(x, (n_batch, n_atom, self.n_output_channel))
        return x


class AtomToPair(nn.Module):
    def __init__(self, n_channel, n_layer, n_atom):
        super(AtomToPair, self).__init__()
        self.linear_layers = [nn.Linear(None, n_channel) for _ in range(n_layer)]
        self.n_atom = n_atom
        self.n_channel = n_channel

    def forward(self, x):
        n_batch, n_atom, n_feature = x.shape
        atom_repeat = functional.reshape(x, (n_batch, 1, n_atom, n_feature))
        atom_repeat = functional.broadcast_to(
            atom_repeat, (n_batch, n_atom, n_atom, n_feature))
        atom_repeat = functional.reshape(atom_repeat,
                                        (n_batch, n_atom * n_atom, n_feature))

        atom_tile = functional.reshape(x, (n_batch, n_atom, 1, n_feature))
        atom_tile = functional.broadcast_to(
            atom_tile, (n_batch, n_atom, n_atom, n_feature))
        atom_tile = functional.reshape(atom_tile,
                                      (n_batch, n_atom * n_atom, n_feature))

        pair_x0 = torch.cat((atom_tile, atom_repeat), axis=2)
        pair_x0 = functional.reshape(pair_x0,
                                    (n_batch * n_atom * n_atom, n_feature * 2))
        for l in self.linear_layers:
            pair_x0 = l(pair_x0)
            pair_x0 = functional.relu(pair_x0)
        pair_x0 = functional.reshape(pair_x0,
                                    (n_batch, n_atom * n_atom, self.n_channel))

        pair_x1 = torch.cat((atom_repeat, atom_tile), axis=2)
        pair_x1 = functional.reshape(pair_x1,
                                    (n_batch * n_atom * n_atom, n_feature * 2))
        for l in self.linear_layers:
            pair_x1 = l(pair_x1)
            pair_x1 = functional.relu(pair_x1)
        pair_x1 = functional.reshape(pair_x1,
                                    (n_batch, n_atom * n_atom, self.n_channel))
        return pair_x0 + pair_x1


class PairToAtom(nn.Module):
    def __init__(self, n_channel, n_layer, n_atom, mode='sum'):
        super(PairToAtom, self).__init__()
        self.linearLayer = [nn.Linear(None, n_channel) for _ in range(n_layer)]
        self.n_atom = n_atom
        self.n_channel = n_channel
        self.mode = mode

    def forward(self, x):
        n_batch, n_pair, n_feature = x.shape
        a = functional.reshape(
            x, (n_batch * (self.n_atom * self.n_atom), n_feature))
        for l in self.linearLayer:
            a = l(a)
            a = functional.relu(a)
        a = functional.reshape(a, (n_batch, self.n_atom, self.n_atom,
                                  self.n_channel))
        a = functional.sum(a, axis=2)
        return a


class WeaveModule(nn.Module):

    def __init__(self, n_atom, output_channel, n_sub_layer,
                 readout_mode='sum'):
        super(WeaveModule, self).__init__()
        self.atom_layer = LinearLayer(output_channel, n_sub_layer)
        self.pair_layer = LinearLayer(output_channel, n_sub_layer)
        self.atom_to_atom = LinearLayer(output_channel, n_sub_layer)
        self.pair_to_pair = LinearLayer(output_channel, n_sub_layer)
        self.atom_to_pair = AtomToPair(output_channel, n_sub_layer, n_atom)
        self.pair_to_atom = PairToAtom(output_channel, n_sub_layer, n_atom,
                                       mode=readout_mode)
        self.n_atom = n_atom
        self.n_channel = output_channel
        self.readout_mode = readout_mode

    def forward(self, atom_x, pair_x, atom_only=False):
        a0 = self.atom_to_atom.forward(atom_x)
        a1 = self.pair_to_atom.forward(pair_x)
        a = torch.cat([a0, a1], axis=2)
        next_atom = self.atom_layer.forward(a)
        next_atom = functional.relu(next_atom)
        if atom_only:
            return next_atom

        p0 = self.atom_to_pair.forward(atom_x)
        p1 = self.pair_to_pair.forward(pair_x)
        p = torch.cat([p0, p1], axis=2)
        next_pair = self.pair_layer.forward(p)
        next_pair = functional.relu(next_pair)
        return next_atom, next_pair


class WeaveNet(nn.Module):
    """WeaveNet implementation

    Args:
        weave_channels (list): list of int, output dimension for each weave
            module
        hidden_dim (int): hidden dim
        n_atom (int): number of atom of input array
        n_sub_layer (int): number of layer for each `AtomToPair`, `PairToAtom`
            layer
        n_atom_types (int): number of atom id
        readout_mode (str): 'sum' or 'max' or 'summax'
    """

    def __init__(self, weave_channels=None, hidden_dim=16,
                 n_atom=WEAVE_DEFAULT_NUM_MAX_ATOMS,
                 n_sub_layer=1, n_atom_types=MAX_ATOMIC_NUM,
                 readout_mode='sum'):
        weave_channels = weave_channels or WEAVENET_DEFAULT_WEAVE_CHANNELS
        self.weave_module = [
            WeaveModule(n_atom, c, n_sub_layer, readout_mode=readout_mode)
            for c in weave_channels
        ]

        super(WeaveNet, self).__init__()
        # with self.init_scope():
        #     # self.embed = EmbedAtomID(out_size=hidden_dim, in_size=n_atom_types)
        #     self.weave_module = chainer.ChainList(*weave_module)
        #     # self.readout = GeneralReadout(mode=readout_mode)
        self.readout_mode = readout_mode

    def __call__(self, batch):
        atoms, pairs = [], []
        if type(batch) != BatchMolGraphWithSubstructures:
            batch = mol2graph_with_substructures(batch, args=self.args)
        for graph in batch:
            mol = graph.mol
            atom_x = mol.get_atom_features_vector()
            pair_x = mol.construct_pair_feature(num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS)
            # if atom_x.dtype == self.xp.int32:
            #     # atom_array: (minibatch, atom)
            #     atom_x = self.embed(atom_x)

            for i in range(len(self.weave_module)):
                if i == len(self.weave_module) - 1:
                    # last layer, only `atom_x` is needed.
                    atom_x = self.weave_module[i].forward(atom_x, pair_x,
                                                          atom_only=True)
                else:
                    # not last layer, both `atom_x` and `pair_x` are needed
                    atom_x, pair_x = self.weave_module[i].forward(atom_x, pair_x)
            atoms.append(atom_x)
            pairs.append(pair_x)
        return atoms, pairs
