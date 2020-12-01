import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from .molecule import Molecule, create_molecule_for_smiles, onek_encoding
from typing import List, Tuple, Union

from rdkit import Chem
import torch

import numpy as np
from tqdm import tqdm

from . import featurization_nf as feature

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

MAX_ATOMS=None

BOND_FDIM = 13


def get_atom_fdim_with_substructures(use_substructures=False, merge_cycles=False) -> int:
    """Gets the dimensionality of the atom feature vector."""
    atom_fdim = 160
    if use_substructures:
        atom_fdim += 5
    if merge_cycles:
        atom_fdim += 5
    return atom_fdim


def get_bond_fdim_with_substructures(use_substructures: bool = False,
                                     merge_cycles: bool = False) -> int:
    """
    Gets the dimensionality of the bond feature vector.

    :param atom_messages: Whether atom messages are being used. If atom messages are used,
                          then the bond feature vector only contains bond features.
                          Otherwise it contains both atom and bond features.
    :return: The dimensionality of the bond feature vector.
    """
    return BOND_FDIM #+ (not atom_messages) * get_atom_fdim_with_substructures(use_substructures, merge_cycles)


def atom_features_for_substructures(atom) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :return: A list containing the atom features.
    """
    return atom.get_representation()


def bond_features_for_substructures(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding(int(bond.GetStereo()), 6)
    return fbond


# class MolGraphWithSubstructures:
#     """
#     A :class:`MolGraph` represents the graph structure and featurization of a single molecule.

#     A MolGraph computes the following attributes:

#     * :code:`n_atoms`: The number of atoms in the molecule.
#     * :code:`n_bonds`: The number of bonds in the molecule.
#     * :code:`f_atoms`: A mapping from an atom index to a list of atom features.
#     * :code:`f_bonds`: A mapping from a bond index to a list of bond features.
#     * :code:`a2b`: A mapping from an atom index to a list of incoming bond indices.
#     * :code:`b2a`: A mapping from a bond index to the index of the atom the bond originates from.
#     * :code:`b2revb`: A mapping from a bond index to the index of the reverse bond.
#     """

#     def __init__(self, mol: str, args):
#         """
#         :param mol: A SMILES or an RDKit molecule.
#         """
#         mol = create_molecule_for_smiles(s, args)
#         atoms = mol.GetAtoms()
#         bonds = mol.GetBonds()
        
#         self.f_atoms = [atom_features_for_substructures(atom) for atom in mol.get_atoms()]
#         self.n_atoms = len(self.f_atoms)
        

# #         self.n_atoms = 0  # number of atoms
# #         self.n_bonds = 0  # number of bonds
# #         self.f_atoms = []  # mapping from atom index to atom features
# #         self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
# #         self.a2b = []  # mapping from atom index to incoming bond indices
# #         self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
# #         self.b2revb = []  # mapping from bond index to the index of the reverse bond

# #         # Get atom features
# #         self.f_atoms = [atom_features_for_substructures(atom) for atom in mol.get_atoms()]
# #         self.n_atoms = len(self.f_atoms)

# #         # Initialize atom to bond mapping for each atom
# #         for _ in range(self.n_atoms):
# #             self.a2b.append([])

# #         bonds_set = set()
# #         # Get bond features
# #         for a1 in range(self.n_atoms):
# #             for a2 in range(a1 + 1, self.n_atoms):
# #                 bond = mol.get_bond(a1, a2)
# #                 if bond in bonds_set:
# #                     continue
# #                 else:
# #                     bonds_set.add(bond)

# #                 if bond is None:
# #                     continue

# #                 f_bond = bond_features_for_substructures(bond.get_rdkit_bond())
# #                 self.f_bonds.append(self.f_atoms[a1] + f_bond)
# #                 self.f_bonds.append(self.f_atoms[a2] + f_bond)

# #                 # Update index mappings
# #                 b1 = self.n_bonds
# #                 b2 = b1 + 1
# #                 self.a2b[a2].append(b1)  # b1 = a1 --> a2
# #                 self.b2a.append(a1)
# #                 self.a2b[a1].append(b2)  # b2 = a2 --> a1
# #                 self.b2a.append(a2)
# #                 self.b2revb.append(b2)
# #                 self.b2revb.append(b1)
# #                 self.n_bonds += 2
    


# class BatchMolGraphWithSubstructures:
#     """
#     A :class:`BatchMolGraph` represents the graph structure and featurization of a batch of molecules.

#     A BatchMolGraph contains the attributes of a :class:`MolGraph` plus:

#     * :code:`atom_fdim`: The dimensionality of the atom feature vector.
#     * :code:`bond_fdim`: The dimensionality of the bond feature vector (technically the combined atom/bond features).
#     * :code:`a_scope`: A list of tuples indicating the start and end atom indices for each molecule.
#     * :code:`b_scope`: A list of tuples indicating the start and end bond indices for each molecule.
#     * :code:`max_num_bonds`: The maximum number of bonds neighboring an atom in this batch.
#     * :code:`b2b`: (Optional) A mapping from a bond index to incoming bond indices.
#     * :code:`a2a`: (Optional): A mapping from an atom index to neighboring atom indices.
#     """

#     def __init__(self, mol_graphs: List[MolGraphWithSubstructures], args):
#         r"""
#         :param mol_graphs: A list of :class:`MolGraph`\ s from which to construct the :class:`BatchMolGraph`.
#         """
        
#         self.n = len(mol_graphs)
    
#         self.atom_fdim = get_atom_fdim_with_substructures(use_substructures=args.substructures_use_substructures,
#                                                           merge_cycles=args.substructures_merge)
#         self.bond_fdim = get_bond_fdim_with_substructures(use_substructures=args.substructures_use_substructures,
#                                                           merge_cycles=args.substructures_merge)

#         # Start n_atoms and n_bonds at 1 b/c zero padding
# #         self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
# #         self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        
#         atom_tensor = np.zeros((n, max_atoms or 1, self.atom_fdim))
#         bond_tensor = np.zeros((n, max_atoms or 1, max_degree or 1, self.bond_fdim))
#         edge_tensor = -np.ones((n, max_atoms or 1, max_degree or 1), dtype=int)
        
#         # All start with zero padding so that indexing with zero padding returns zeros
# #         f_atoms = [[0] * self.atom_fdim]  # atom features
# #         f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        
#         for mol_graph in mol_graphs:
#             f_atoms.extend(mol_graph.f_atoms)
#             f_bonds.extend(mol_graph.f_bonds)

#             for a in range(mol_graph.n_atoms):
#                 a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

#             for b in range(mol_graph.n_bonds):
#                 b2a.append(self.n_atoms + mol_graph.b2a[b])
#                 b2revb.append(self.n_bonds + mol_graph.b2revb[b])

#             self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
#             self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
#             self.n_atoms += mol_graph.n_atoms
#             self.n_bonds += mol_graph.n_bonds

#         self.max_num_bonds = max(1, max(
#             len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

#         self.f_atoms = torch.FloatTensor(f_atoms)
#         self.f_bonds = torch.FloatTensor(f_bonds)
        
#     def get_components(self, args) -> Tuple[torch.FloatTensor, torch.FloatTensor,
#                                             torch.LongTensor, torch.LongTensor, torch.LongTensor,
#                                             List[Tuple[int, int]], List[Tuple[int, int]]]:
#         """
#         Returns the components of the :class:`BatchMolGraph`.

#         The returned components are, in order:

#         * :code:`f_atoms`
#         * :code:`f_bonds`
#         * :code:`a2b`
#         * :code:`b2a`
#         * :code:`b2revb`
#         * :code:`a_scope`
#         * :code:`b_scope`

#         :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond feature
#                               vector to contain only bond features rather than both atom and bond features.
#         :return: A tuple containing PyTorch tensors with the atom features, bond features, graph structure,
#                  and scope of the atoms and bonds (i.e., the indices of the molecules they belong to).
#         """
#         if args.substructures_atom_messages:
#             f_bonds = self.f_bonds[:, :get_bond_fdim_with_substructures(atom_messages=args.substructures_atom_messages,
#                                                                         use_substructures=args.substructures_use_substructures)]
#         else:
#             f_bonds = self.f_bonds

#         return self.f_atoms, f_bonds
    
def padaxis(array, new_size, axis, pad_value=0, pad_right=True):
    """ Padds one axis of an array to a new size
    This is just a wrapper for np.pad, more useful when only padding a single axis
    # Arguments:
        array: the array to pad
        new_size: the new size of the specified axis
        axis: axis along which to pad
        pad_value: pad value,
        pad_right: boolean, pad on the right or left side
    # Returns:
        padded_array: np.array
    """
    add_size = new_size - array.shape[axis]
    pad_width = [(0, 0)] * len(array.shape)

    if pad_right:
        pad_width[axis] = (0, add_size)
    else:
        pad_width[axis] = (add_size, 0)

    return np.pad(array, pad_width=pad_width, mode='constant', constant_values=pad_value)


def tensorise_smiles(smiles, args, max_atoms=MAX_ATOMS):
    """Takes a list of smiles and turns the graphs in tensor representation.
    # Arguments:
        smiles: a list (or iterable) of smiles representations
        max_atoms: the maximum number of atoms per molecule (to which all
            molecules will be padded), use `None` for auto
        max_degree: max_atoms: the maximum number of neigbour per atom that each
            molecule can have (to which all molecules will be padded), use `None`
            for auto
        **NOTE**: It is not recommended to set max_degree to `None`/auto when
            using `NeuralGraph` layers. Max_degree determines the number of
            trainable parameters and is essentially a hyperparameter.
            While models can be rebuilt using different `max_atoms`, they cannot
            be rebuild for different values of `max_degree`, as the architecture
            will be different.
            For organic molecules `max_degree=5` is a good value (Duvenaud et. al, 2015)
    # Returns:
        atoms: np.array, An atom feature np.array of size `(molecules, max_atoms, atom_features)`
        bonds: np.array, A bonds np.array of size `(molecules, max_atoms, max_neighbours)`
        edges: np.array, A connectivity array of size `(molecules, max_atoms, max_neighbours, bond_features)`
    """

    # import sizes
    max_degree = args.substructures_max_degree
    n = len(smiles)
    
    n_atom_features = get_atom_fdim_with_substructures(use_substructures=args.substructures_use_substructures,
                                                          merge_cycles=args.substructures_merge)
    n_bond_features = get_bond_fdim_with_substructures(use_substructures=args.substructures_use_substructures,
                                                          merge_cycles=args.substructures_merge)
#     print(n_bond_features)
    
#     self.n_atoms = 0  # number of atoms
#     self.n_bonds = 0  # number of bonds
#     self.f_atoms = []  # mapping from atom index to atom features
#     self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
#     self.a2b = []  # mapping from atom index to incoming bond indices
#     self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
#     self.b2revb = []  # mapping from bond index to the index of the reverse bond
    


    # preallocate atom tensor with 0's and bond tensor with -1 (because of 0 index)
    # If max_degree or max_atoms is set to None (auto), initialise dim as small
    #   as possible (1)
    atom_tensor = np.zeros((n, max_atoms or 1, n_atom_features))
    bond_tensor = np.zeros((n, max_atoms or 1, max_degree or 1, n_bond_features))
    edge_tensor = -np.ones((n, max_atoms or 1, max_degree or 1), dtype=int)

    for mol_ix, s in enumerate(tqdm(smiles)):
#         mol = Chem.MolFromSmiles(s)
        mol = create_molecule_for_smiles(s, args)
        atoms = mol.get_atoms()
#         bonds = mol.GetBonds()

        # If max_atoms is exceeded, resize if max_atoms=None (auto), else raise
        if len(atoms) > atom_tensor.shape[1]:
            atom_tensor = padaxis(atom_tensor, len(atoms), axis=1)
            bond_tensor = padaxis(bond_tensor, len(atoms), axis=1)
            edge_tensor = padaxis(edge_tensor, len(atoms), axis=1, pad_value=-1)
        rdkit_ix_lookup = {}

        for atom_ix, atom in enumerate(atoms):
            # write atom features
            atom_tensor[mol_ix, atom_ix, : n_atom_features] = atom_features_for_substructures(atom) #feature.atom_features(atom)

            # store entry in idx
            rdkit_ix_lookup[atom.idx] = atom_ix

        # preallocate array with neighbor lists (indexed by atom)
        connectivity_mat = [[] for _ in atoms]

        
        bonds_set = set()
        # Get bond features
        for a1_ix in range(len(atoms)):
            for a2_ix in range(a1_ix + 1, len(atoms)):
                bond = mol.get_bond(a1_ix, a2_ix)
                if bond in bonds_set:
                    continue
                else:
                    bonds_set.add(bond)

                if bond is None:
                    continue
                    
                a1_neigh = len(connectivity_mat[a1_ix])
                a2_neigh = len(connectivity_mat[a2_ix])
                
                new_degree = max(a1_neigh, a2_neigh) + 1
                if new_degree > bond_tensor.shape[2]:
                    assert max_degree is None, 'too many neighours ({0}) in molecule: {1}'.format(new_degree, s)
                    bond_tensor = padaxis(bond_tensor, new_degree, axis=2)
                    edge_tensor = padaxis(edge_tensor, new_degree, axis=2, pad_value=-1)
                    
                bond_features = bond_features_for_substructures(bond.get_rdkit_bond())#np.array(feature.bond_features(bond), dtype=int)
#                 print(len(bond_features))
                bond_tensor[mol_ix, a1_ix, a1_neigh, :] = bond_features
                bond_tensor[mol_ix, a2_ix, a2_neigh, :] = bond_features

                # add to connectivity matrix
                connectivity_mat[a1_ix].append(a2_ix)
                connectivity_mat[a2_ix].append(a1_ix)

        # store connectivity matrix
        for a1_ix, neighbours in enumerate(connectivity_mat):
            degree = len(neighbours)
            edge_tensor[mol_ix, a1_ix, : degree] = neighbours
                

    return torch.from_numpy(atom_tensor).float(), \
           torch.from_numpy(bond_tensor).float(), \
           torch.from_numpy(edge_tensor).long()



# def mol2graph_with_substructures(mols: Union[List[str]], args) -> BatchMolGraphWithSubstructures:
#     """
#     Converts a list of SMILES or RDKit molecules to a :class:`BatchMolGraph` containing the batch of molecular graphs.

#     :param mols: A list of SMILES or a list of RDKit molecules.
#     :return: A :class:`BatchMolGraph` containing the combined molecular graph for the molecules.
#     """
#     return BatchMolGraphWithSubstructures(
#         [MolGraphWithSubstructures(mol, args) for mol in mols], args=args)
