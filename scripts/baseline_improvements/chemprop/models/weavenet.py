import torch
from torch import nn
from torch.nn import functional

MAX_ATOMIC_NUM = 170
WEAVE_DEFAULT_NUM_MAX_ATOMS = 25

import numpy as np

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from features import BatchMolGraphWithSubstructures, mol2graph_with_substructures

WEAVENET_DEFAULT_WEAVE_CHANNELS = [50, ]


class LinearLayer(nn.Module):

    def __init__(self, n_channel, n_layer):
        super(LinearLayer, self).__init__()

        self.layers = nn.Sequential(*([nn.Linear(WEAVE_DEFAULT_NUM_MAX_ATOMS * MAX_ATOMIC_NUM, n_channel)]
                                      + [nn.Linear(n_channel, n_channel) for _ in range(n_layer - 1)]))
        self.n_output_channel = n_channel

    def forward(self, x):
        n_channel, n_atom, n_batch = x.shape
        x = torch.reshape(x, (n_channel * n_atom, n_batch))
        x = torch.transpose(x, 0, 1).float()
        x = self.layers(x)
        return x


class LinearLayerDoubleAtomToAtom(nn.Module):

    def __init__(self, n_channel_old, n_channel, n_layer):
        super(LinearLayerDoubleAtomToAtom, self).__init__()

        self.layers = nn.Sequential(*([nn.Linear((WEAVE_DEFAULT_NUM_MAX_ATOMS + 1) * n_channel_old, n_channel)]
                                      + [nn.Linear(n_channel, n_channel) for _ in range(n_layer - 1)]))
        self.n_output_channel = n_channel

    def forward(self, x):
        x = self.layers(x)
        return x


class PairToAtom(nn.Module):
    def __init__(self, n_channel, n_layer, n_atom, mode='sum'):
        super(PairToAtom, self).__init__()
        self.linearLayer = nn.Sequential(*([nn.Linear(n_atom * n_atom * 20, n_channel * n_atom * n_atom)]
                                           + [nn.Linear(n_channel * n_atom * n_atom, n_channel * n_atom * n_atom) for _
                                              in range(n_layer - 1)]))
        self.n_atom = n_atom
        self.n_channel = n_channel
        self.mode = mode

    def forward(self, x):
        n_feature, n_pair, n_batch = x.shape
        a = torch.reshape(
            x, (n_feature * n_pair, n_batch))
        a = torch.transpose(a, 0, 1).float()
        a = self.linearLayer(a)
        a = torch.reshape(a, (n_batch, self.n_atom, self.n_atom,
                              self.n_channel))
        a = torch.sum(a, 2)
        return a


class WeaveModule(nn.Module):

    def __init__(self, n_atom, output_channel, n_sub_layer,
                 readout_mode='sum', device='cpu'):
        super(WeaveModule, self).__init__()
        self.atom_layer = LinearLayerDoubleAtomToAtom(output_channel, WEAVE_DEFAULT_NUM_MAX_ATOMS * MAX_ATOMIC_NUM,
                                                      n_sub_layer)
        self.atom_to_atom = LinearLayer(output_channel, n_sub_layer)
        self.pair_to_atom = PairToAtom(output_channel, n_sub_layer, n_atom,
                                       mode=readout_mode)
        self.device = device
        self.n_atom = n_atom
        self.n_channel = output_channel
        self.readout_mode = readout_mode

    def forward(self, atom_x, pair_x, atom_only=False):
        a0 = self.atom_to_atom.forward(atom_x).unsqueeze(1)
        a1 = self.pair_to_atom.forward(pair_x)
        a = torch.cat([a0, a1], dim=1)
        n_batch, n_feat, n_channel = a.shape
        a = torch.reshape(a, (n_batch, n_feat * n_channel))
        next_atom = self.atom_layer.forward(a)
        next_atom = functional.relu(next_atom)
        if atom_only:
            return next_atom
        #
        # p0 = self.atom_to_pair.forward(atom_x)
        # p1 = self.pair_to_pair.forward(pair_x)
        # p = torch.cat([p0, p1], 1)
        # next_pair = self.pair_layer.forward(p)
        # next_pair = functional.relu(next_pair)
        # return next_atom, next_pair


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

    def __init__(self, args, weave_channels=None, hidden_dim=16,
                 n_atom=WEAVE_DEFAULT_NUM_MAX_ATOMS,
                 n_sub_layer=1, n_atom_types=MAX_ATOMIC_NUM,
                 readout_mode='sum'):
        global WEAVE_DEFAULT_NUM_MAX_ATOMS
        WEAVE_DEFAULT_NUM_MAX_ATOMS = args.weave_max_atoms
        super(WeaveNet, self).__init__()
        self.args = args
        self.device = args.device
        weave_channels = weave_channels or WEAVENET_DEFAULT_WEAVE_CHANNELS
        self.weave_module = WeaveModule(args.weave_max_atoms, weave_channels[0], n_sub_layer, readout_mode=readout_mode,
                                        device=self.device)
        self.readout_mode = readout_mode

    def forward(self, batch):
        atoms, pairs = [], []
        if type(batch) != BatchMolGraphWithSubstructures:
            batch = mol2graph_with_substructures(batch, args=self.args)
        for graph in batch.mol_graphs:
            mol = graph.mol
            atom_x = mol.get_atom_features_vector(WEAVE_DEFAULT_NUM_MAX_ATOMS)
            pair_x = mol.construct_pair_feature(num_max_atoms=WEAVE_DEFAULT_NUM_MAX_ATOMS)
            atoms.append(atom_x)
            pairs.append(pair_x)

        atom_x = torch.from_numpy(np.dstack(atoms))
        atom_x = atom_x.to(self.device)
        pair_x = torch.from_numpy(np.dstack(pairs))
        pair_x = pair_x.to(self.device)
        atom_x = self.weave_module.forward(atom_x, pair_x,
                                           atom_only=True)
        # else:
        #     # not last layer, both `atom_x` and `pair_x` are needed
        #     atom_x, pair_x = self.weave_module[i].forward(atom_x, pair_x)

        return atom_x
