#https://github.com/XuhanLiu/NGFP/tree/b01b52bab5ef657e681c0f9e305e05891d6d6b20

import torch as T
from torch import nn
from torch.nn import functional as F
from .nf_layer import GraphConv, GraphPool, GraphOutput
import numpy as np
from torch import optim
import time
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from nf_utils import dev

from args import TrainArgs

from features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph, \
    BatchMolGraphWithSubstructures, get_atom_fdim_with_substructures, get_bond_fdim_with_substructures, \
    mol2graph_with_substructures



class QSAR(nn.Module):
    def __init__(self, args:TrainArgs):
        super(QSAR, self).__init__()
        
        self.args = args
        self.device = args.device
        
        self.max_degree = args.substructures_max_degree

        
        hid_dim = args.substructures_hidden_size
        
        self.atom_fdim = get_atom_fdim_with_substructures(use_substructures=args.substructures_use_substructures,
                                                              merge_cycles=args.substructures_merge)
        self.bond_fdim = get_bond_fdim_with_substructures( use_substructures=args.substructures_use_substructures,
                                                              merge_cycles=args.substructures_merge)
        self.gcn1 = GraphConv(input_dim=self.atom_fdim+self.bond_fdim, conv_width=128, max_degree = self.max_degree)
        self.gcn2 = GraphConv(input_dim=141, conv_width=128, max_degree = self.max_degree)
        self.gop = GraphOutput(input_dim=141, output_dim=args.substructures_hidden_size)
        # self.bn = nn.BatchNorm2d(80)
        self.pool = GraphPool()
#         self.fc2 = nn.Linear(hid_dim, n_out)
        self.to(dev)

#     def forward(self, atoms, bonds, edges):
    def forward(self, batch):
        atoms, bonds, edges = batch
        atoms, bonds, edges = atoms.to(self.device), bonds.to(self.device), edges.to(
            self.device)

        
        atoms = self.gcn1(atoms, bonds, edges)
        # atoms = self.bn(atoms)
        atoms = self.pool(atoms, edges)
        atoms = self.gcn2(atoms, bonds, edges)
        # atoms = self.bn(atoms)
        atoms = self.pool(atoms, edges)
        fp = self.gop(atoms, bonds, edges)
#         out = F.sigmoid(self.fc2(fp))
        return fp

#     def fit(self, loader_train, loader_valid, path, epochs=1000, early_stop=100, lr=1e-3):
#         criterion = nn.BCELoss()
#         optimizer = optim.Adam(self.parameters(), lr=lr)
#         best_loss = np.inf
#         last_saving = 0
#         for epoch in range(epochs):
#             t0 = time.time()
#             for Ab, Bb, Eb, yb in loader_train:
#                 Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
#                 optimizer.zero_grad()
#                 y_ = self.forward(Ab, Bb, Eb)
#                 ix = yb == yb
#                 yb, y_ = yb[ix], y_[ix]
#                 loss = criterion(y_, yb)
#                 loss.backward()
#                 optimizer.step()
#             loss_valid = self.evaluate(loader_valid)
#             print('[Epoch:%d/%d] %.1fs loss_train: %f loss_valid: %f' % (
#                 epoch, epochs, time.time() - t0, loss.item(), loss_valid))
#             if loss_valid < best_loss:
#                 T.save(self, path + '.pkg')
#                 print('[Performance] loss_valid is improved from %f to %f, Save model to %s' % (
#                     best_loss, loss_valid, path + '.pkg'))
#                 best_loss = loss_valid
#                 last_saving = epoch
#             else:
#                 print('[Performance] loss_valid is not improved.')
#             if early_stop is not None and epoch - last_saving > early_stop: break
#         return T.load(path + '.pkg')

#     def evaluate(self, loader):
#         loss = 0
#         criterion = nn.BCELoss()
#         for Ab, Bb, Eb, yb in loader:
#             Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
#             y_ = self.forward(Ab, Bb, Eb)
#             ix = yb == yb
#             yb, y_ = yb[ix], y_[ix]
#             loss += criterion(y_, yb).item()
#         return loss / len(loader)

#     def predict(self, loader):
#         score = []
#         for Ab, Bb, Eb, yb in loader:
#             Ab, Bb, Eb, yb = Ab.to(dev), Bb.to(dev), Eb.to(dev), yb.to(dev)
#             y_ = self.forward(Ab, Bb, Eb)
#             score.append(y_.data.cpu())
#         score = T.cat(score, dim=0).numpy()
#         return score
