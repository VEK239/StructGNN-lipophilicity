from collections import defaultdict
import numpy

from rdkit import Chem

BOND_FDIM = 13
ATOM_FDIM = 170

BT_MAPPING_CHAR = {
    Chem.rdchem.BondType.SINGLE: 'S',
    Chem.rdchem.BondType.DOUBLE: 'D',
    Chem.rdchem.BondType.TRIPLE: 'T',
    Chem.rdchem.BondType.AROMATIC: 'A',
}
BT_MAPPING_INT = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 1.5,
}
STRUCT_TO_NUM = {
    'ATOM': 0,
    'RING': 1,
    'ACID': 2,
    'AMIN': 3,
    'ESTER': 4,
    'SULFONAMID': 5
}


def get_cycles_for_molecule(mol):
    """
    Finds all the cycles in a given RDKit mol
    :param mol: The given RDKit molecule
    :return: A list of cycles in a mol
    """
    all_cycles = Chem.GetSymmSSSR(mol)
    all_cycles = [set(ring) for ring in all_cycles]
    return all_cycles


def merge_substructures(rings, acids, esters, amins, sulfonamids):
    """
    Merges intersecting substructures
    :param rings: A list of all rings in mol
    :param acids: A list of all acids in mol
    :param esters: A list of all esters in mol
    :param amins: A list of all amins in mol
    :param sulfonamids: A list of all sulfonamids in mol
    :return: A list of cycles in a mol
    """
    atom_to_ring = defaultdict(set)
    for cycle_idx, cycle in enumerate(rings):
        for atom in cycle:
            atom_to_ring[atom].add(cycle_idx)
    rings_to_merge = [1]
    while rings_to_merge:
        rings_to_merge = None
        for atom, atom_cycles in atom_to_ring.items():
            if len(atom_cycles) > 1:
                rings_to_merge = atom_cycles.copy()
        if rings_to_merge:
            ring_new_idx = min(rings_to_merge)
            for ring_idx in rings_to_merge:
                for atom in rings[ring_idx]:
                    rings[ring_new_idx].add(atom)
                    atom_to_ring[atom].remove(ring_idx)
                    atom_to_ring[atom].add(ring_new_idx)
            for ring_idx in rings_to_merge:
                if ring_idx != ring_new_idx:
                    rings[ring_idx] = []
    all_cycles = [set(cycle) for cycle in rings if len(cycle) > 2]

    remaining_substructures = [all_cycles]
    for other_substructures in [acids, esters, amins, sulfonamids]:
        not_intersecting_substr = []
        for substructure in other_substructures:
            substructure_add = True
            for cycle in all_cycles:
                if len(cycle & substructure) > 0:
                    substructure_add = False
            if substructure_add:
                not_intersecting_substr.append(substructure)
        remaining_substructures.append([substr for substr in not_intersecting_substr])
    remaining_substructures = ([list(e) for e in substr_type_list] for substr_type_list in remaining_substructures)
    return remaining_substructures


def get_acids_for_molecule(mol):
    """
    Finds all acid parts in a given RDKit mol
    :param mol: The given RDKit molecule
    :return: A list of acid parts in a mol
    """
    acid_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    acids = mol.GetSubstructMatches(acid_pattern)
    return [set(acid) for acid in acids]


def get_esters_for_molecule(mol):
    """
    Finds all ester parts in a given RDKit mol
    :param mol: The given RDKit molecule
    :return: A list of ester parts in a mol
    """
    ester_pattern = Chem.MolFromSmarts('[#6][CX3](=O)[OX2H0][#6]')
    esters = mol.GetSubstructMatches(ester_pattern)
    esters = [list(ester) for ester in esters]
    ester_atoms = []
    for ester in esters:
        atoms = []
        for atom in ester:
            if mol.GetAtomWithIdx(atom).GetSymbol() == 'O':
                atoms.append(atom)
        for atom in ester:
            if mol.GetBondBetweenAtoms(atom, atoms[0]) and mol.GetBondBetweenAtoms(atom, atoms[1]):
                atoms.append(atom)
        ester_atoms.append(set(atoms))
    return ester_atoms


def get_amins_for_molecule(mol):
    """
    Finds all amino parts in a given RDKit mol
    :param mol: The given RDKit molecule
    :return: A list of amino parts in a mol
    """
    sulfonamids = get_sulfonamids_for_molecule(mol)
    sulfonamid_atoms = []
    for sulfo in sulfonamids:
        for atom in sulfo:
            sulfonamid_atoms.append(atom)
    amin_pattern = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
    amins = mol.GetSubstructMatches(amin_pattern)
    amins = [list(amin) for amin in amins]
    amins = [[amin[0] if mol.GetAtomWithIdx(amin[0]).GetSymbol() == 'N' else amin[1]] for amin in amins]
    return [set(amin) for amin in amins if amin[0] not in sulfonamid_atoms]


def get_sulfonamids_for_molecule(mol):
    """
    Finds all sulfonamid parts in a given RDKit mol
    :param mol: The given RDKit molecule
    :return: A list of sulfonamid parts in a mol
    """
    sulphoneamid_pattern = Chem.MolFromSmarts(
        '[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]')
    sulfonamids = mol.GetSubstructMatches(sulphoneamid_pattern)
    sulfonamid_atoms = []
    for sulfonamid in sulfonamids:
        sulfonamid = list(sulfonamid)
        atoms = [sulfonamid[0]]
        for neighbor in mol.GetAtomWithIdx(atoms[0]).GetNeighbors():
            if neighbor.GetSymbol() != 'C':
                atoms.append(neighbor.GetIdx())
        sulfonamid_atoms.append(set(atoms))
    return sulfonamid_atoms


def structure_encoding(atoms):
    """
    Generates one-hot mapping for molecule structure
    :param atoms: A list of atoms to encode
    :return: A vector with encoding
    """
    enc = [0 for _ in range(55)]
    for atom in atoms:
        enc[atom.GetAtomicNum()] += 1
    return enc


def onek_encoding(value, choices_len):
    """
    Creates a one-hot encoding.
    :param value: The value for which the encoding should be one.
    :param choices_len: A list of possible values.
    :return: A one-hot encoding of the value.
    """
    encoding = [0] * choices_len
    encoding[int(value)] = 1
    return encoding


def generate_substructure_sum_vector_mapping(substruct, mol, structure_type, args):
    """
    Generates a vector with mapping for a substructure
    :param substruct: The given substructure
    :param mol: RDKit molecule
    :param structure_type: The type of a structure (one of STRUCT_TO_NUM)
    :return: An encoding vector
    """
    atoms = [mol.GetAtomWithIdx(i) for i in substruct]

    substruct_atomic_encoding = structure_encoding(atoms)

    implicit_substruct_valence = 0
    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            bond = mol.GetBondBetweenAtoms(substruct[i], substruct[j])
            if bond:
                implicit_substruct_valence += BT_MAPPING_INT[
                    mol.GetBondBetweenAtoms(substruct[i], substruct[j]).GetBondType()]
    substruct_valence = sum(atom.GetExplicitValence() for atom in atoms) - 2 * implicit_substruct_valence
    substruct_valence_array = onek_encoding(substruct_valence, 40)

    substruct_formal_charge = sum(atom.GetFormalCharge() for atom in atoms)

    substruct_num_Hs = sum(atom.GetTotalNumHs() for atom in atoms)
    substruct_Hs_array = onek_encoding(substruct_num_Hs, 65 if args.substructures_merge else 60)

    substruct_is_aromatic = 1 if sum(atom.GetIsAromatic() for atom in atoms) > 0 else 0

    substruct_mass = sum(atom.GetMass() for atom in atoms)

    substruct_edges_sum = implicit_substruct_valence

    if args.substructures_use_substructures:
        substruct_type = onek_encoding(STRUCT_TO_NUM[structure_type], len(STRUCT_TO_NUM))
    else:
        substruct_type = [1 if structure_type == 'RING' else 0]

    features = substruct_atomic_encoding + substruct_valence_array + substruct_Hs_array + substruct_type + \
               [substruct_formal_charge, substruct_is_aromatic, substruct_mass * 0.01, substruct_edges_sum * 0.1]
    return tuple(features)


class Atom:
    def __init__(self, idx, atom_representation, atom_type, atom_list, symbol=''):
        self.symbol = symbol
        self.idx = idx
        self.atom_representation = atom_representation
        self.atom_type = atom_type
        self.atom_list = atom_list
        self.bonds = []

    def add_bond(self, bond):
        self.bonds.append(bond)

    def get_representation(self):
        return list(self.atom_representation)


class Bond:
    def __init__(self, rdkit_bond, idx, out_atom_idx, in_atom_idx, bond_type='fictitious'):
        self.rdkit_bond = rdkit_bond
        self.idx = idx
        self.out_atom_idx = out_atom_idx
        self.in_atom_idx = in_atom_idx
        self.bond_type = bond_type
        if self.bond_type == 'fictitious':
            self.weight = -1

    def get_rdkit_bond(self):
        return self.rdkit_bond


class Molecule:
    def __init__(self, atoms, bonds, rdkit_mol, custom_atom_idx_to_usual_idx):
        self.atoms = atoms
        self.bonds = bonds
        self.rdkit_mol = rdkit_mol
        self.distance_matrix = None
        self.custom_atom_idx_to_usual_idx = custom_atom_idx_to_usual_idx

    def bond_features_for_substructures(self, i, j):
        """
        Builds a feature vector for a bond.

        :param bond: An RDKit bond.
        :return: A list containing the bond features.
        """
        bond = self.get_bond(i, j)
        if bond is None:
            fbond = [1] + [0] * (BOND_FDIM - 1)
        else:
            bt = bond.rdkit_bond.GetBondType()
            fbond = [
                0,  # bond is not None
                bt == Chem.rdchem.BondType.SINGLE,
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE,
                bt == Chem.rdchem.BondType.AROMATIC,
                (bond.rdkit_bond.GetIsConjugated() if bt is not None else 0),
                (bond.rdkit_bond.IsInRing() if bt is not None else 0)
            ]
            fbond += onek_encoding(int(bond.rdkit_bond.GetStereo()), 6)
        return numpy.array(numpy.array(fbond))

    def get_bond(self, atom_1_idx, atom_2_idx):
        # If bond does not exist between atom_1 and atom_2, return None
        if atom_1_idx >= len(self.atoms) or atom_2_idx >= len(self.atoms):
            return None
        for bond in self.atoms[atom_1_idx].bonds:
            if atom_2_idx == bond.out_atom_idx or atom_2_idx == bond.in_atom_idx:
                return bond
        return None

    def get_real_indices_for_atom(self, atom_idx):
        return self.custom_atom_idx_to_usual_idx[atom_idx]

    def get_rdkit_mol(self):
        return self.rdkit_mol

    def get_atoms(self):
        return self.atoms

    def get_atom_features_vector(self, num_max_atoms):
        atoms = [numpy.array(atom.get_representation()) for atom in self.atoms]
        atoms += [numpy.zeros(ATOM_FDIM) for _ in range(num_max_atoms - len(atoms))]
        atoms = atoms[:num_max_atoms]
        return numpy.stack(atoms)

    def get_atom(self, atom_idx):
        return self.atoms[atom_idx]

    def get_num_atoms(self):
        return len(self.atoms)

    def construct_distance_vec(self, i, j, max_distance):
        distance_matrix = self.get_distance_matrix()
        if i >= len(distance_matrix) or j >= len(distance_matrix):
            return numpy.zeros((max_distance,), dtype=numpy.float32)
        distance = min(max_distance, distance_matrix[i][j])
        distance_feature = numpy.zeros((max_distance,), dtype=numpy.float32)
        distance_feature[:distance] = 1.0
        return distance_feature

    def get_distance_matrix(self):
        if self.distance_matrix is None:
            n = self.get_num_atoms()
            self.distance_matrix = [[1e12 for _ in range(n)] for _ in range(n)]
            for bond in self.bonds:
                self.distance_matrix[bond.out_atom_idx][bond.in_atom_idx] = 1
                self.distance_matrix[bond.in_atom_idx][bond.out_atom_idx] = 1
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        self.distance_matrix[i][j] = min(self.distance_matrix[i][j], self.distance_matrix[i][k] +
                                                         self.distance_matrix[k][j])
        return self.distance_matrix

    def construct_pair_feature(self, num_max_atoms=10, max_distance=7):
        """construct pair feature

        Args:
            mol (Mol): mol instance
            num_max_atoms (int): number of max atoms

        Returns (numpy.ndarray): 2 dimensional array. First axis size is
            `num_max_atoms` ** 2, representing index of each atom pair.
            Second axis for feature.

        """
        n_atom = self.get_num_atoms()
        distance_feature = numpy.zeros((num_max_atoms ** 2, max_distance,),
                                       dtype=numpy.float32)
        for i in range(num_max_atoms):
            for j in range(num_max_atoms):
                distance_feature[i * num_max_atoms + j] = self.construct_distance_vec(i, j, max_distance)
        bond_feature = numpy.zeros((num_max_atoms ** 2, BOND_FDIM,), dtype=numpy.float32)
        for i in range(num_max_atoms):
            for j in range(num_max_atoms):
                bond_feature[i * num_max_atoms + j] = self.bond_features_for_substructures(i, j)
        # ring_feature = construct_ring_feature_vec(mol, num_max_atoms=num_max_atoms)
        # feature = numpy.hstack((distance_feature, bond_feature, ring_feature))
        feature = numpy.hstack((distance_feature, bond_feature))
        return feature

    def prnt(self):
        for atom in self.atoms:
            print(atom.symbol, atom.idx, atom.bonds, atom.atom_representation)
        for bond in self.bonds:
            print(bond.out_atom_idx, bond.in_atom_idx)


def create_molecule_for_smiles(smiles, args):
    mol = Chem.MolFromSmiles(smiles)

    rings = get_cycles_for_molecule(mol)
    if args.substructures_use_substructures:
        acids = get_acids_for_molecule(mol)
        esters = get_esters_for_molecule(mol)
        amins = get_amins_for_molecule(mol)
        sulfonamids = get_sulfonamids_for_molecule(mol)
    else:
        acids = []
        esters = []
        amins = []
        sulfonamids = []

    if args.substructures_merge:
        rings, acids, esters, amins, sulfonamids = merge_substructures(rings, acids, esters, amins, sulfonamids)
    else:
        rings = [list(e) for e in rings]
        acids = [list(e) for e in acids]
        esters = [list(e) for e in esters]
        amins = [list(e) for e in amins]
        sulfonamids = [list(e) for e in sulfonamids]

    used_atoms = set()
    mol_bonds = []
    mol_atoms = []
    idx_to_atom = defaultdict(set)
    custom_atom_idx_to_idx = defaultdict(set)
    min_not_used_idx = 0

    for structure_type in [[rings, 'RING'], [acids, 'ACID'], [esters, 'ESTER'], [amins, 'AMIN'],
                           [sulfonamids, 'SULFONAMID']]:
        substructure_type_string = structure_type[1]
        substructures = structure_type[0]
        for substruct in substructures:
            mapping = generate_substructure_sum_vector_mapping(substruct, mol, substructure_type_string, args)
            substruct_atom = Atom(idx=min_not_used_idx,
                                  atom_representation=mapping, atom_type=substructure_type_string, atom_list=substruct)
            min_not_used_idx += 1
            mol_atoms.append(substruct_atom)
            for idx in substruct:
                idx_to_atom[idx].add(substruct_atom)
                used_atoms.add(idx)
                custom_atom_idx_to_idx[min_not_used_idx - 1].add(idx)

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom_idx not in used_atoms:
            atom_repr = generate_substructure_sum_vector_mapping([atom_idx], mol, 'ATOM', args)
            custom_atom = Atom(idx=min_not_used_idx, atom_representation=atom_repr, symbol=atom.GetSymbol(),
                               atom_type='ATOM', atom_list=[atom_idx])
            min_not_used_idx += 1
            mol_atoms.append(custom_atom)
            idx_to_atom[atom_idx].add(custom_atom)
            custom_atom_idx_to_idx[min_not_used_idx - 1].add(atom_idx)

    for idx, bond in enumerate(mol.GetBonds()):
        start_atoms = idx_to_atom[bond.GetBeginAtomIdx()]
        end_atoms = idx_to_atom[bond.GetEndAtomIdx()]
        if len(start_atoms & end_atoms) == 0:
            custom_bond = Bond(bond, idx, list(start_atoms)[0].idx, list(end_atoms)[0].idx, bond.GetBondType())
            mol_bonds.append(custom_bond)
            for start_atom in start_atoms:
                start_atom.add_bond(custom_bond)
            for end_atom in end_atoms:
                end_atom.add_bond(custom_bond)
        elif args.fictitious_edges and len(start_atoms & end_atoms) == 2:
            cur_atoms = start_atoms & end_atoms
            custom_bond = Bond(bond, idx, list(cur_atoms)[0].idx, list(cur_atoms)[0].idx)
            mol_bonds.append(custom_bond)
            for start_atom in start_atoms:
                start_atom.add_bond(custom_bond)
            for end_atom in end_atoms:
                end_atom.add_bond(custom_bond)
        elif args.fictitious_edges and len(start_atoms & end_atoms) == 3:
            pass
            # TODO
    custom_mol = Molecule(mol_atoms, mol_bonds, mol, custom_atom_idx_to_idx)
    return custom_mol
