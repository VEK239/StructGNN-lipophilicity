from collections import defaultdict

import pandas as pd
from rdkit import Chem


def get_cycles_for_molecule(mol, merging_cycles=False):
    all_cycles = Chem.GetSymmSSSR(mol)
    all_cycles = [set(ring) for ring in all_cycles]
    if merging_cycles:
        atom_to_ring = defaultdict(set)
        for cycle_idx, cycle in enumerate(all_cycles):
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
                    for atom in all_cycles[ring_idx]:
                        all_cycles[ring_new_idx].add(atom)
                        atom_to_ring[atom].remove(ring_idx)
                        atom_to_ring[atom].add(ring_new_idx)
                for ring_idx in rings_to_merge:
                    if ring_idx != ring_new_idx:
                        all_cycles[ring_idx] = []
    all_cycles = [list(cycle) for cycle in all_cycles if len(cycle) > 2]
    return all_cycles


def get_acids_for_molecule(mol):
    acid_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    acids = mol.GetSubstructMatches(acid_pattern)
    return [list(acid) for acid in acids]


def get_amins_for_molecule(mol):
    amin_pattern = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
    amins = mol.GetSubstructMatches(amin_pattern)
    amins = [list(amin) for amin in amins]
    return [[amin[0] if mol.GetAtomWithIdx(amin[0]).GetSymbol() == 'N' else amin[1]] for amin in amins]


def get_esters_for_molecule(mol):
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
        ester_atoms.append(atoms)
    return ester_atoms


def get_sulfoneamids_for_molecule(mol):
    sulphoneamid_pattern = Chem.MolFromSmarts(
        '[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]')
    sulfoneamids = mol.GetSubstructMatches(sulphoneamid_pattern)
    sulfoneamid_atoms = []
    for sulfoneamid in sulfoneamids:
        sulfoneamid = list(sulfoneamid)
        atoms = [sulfoneamid[0]]
        for neighbor in mol.GetAtomWithIdx(atoms[0]).GetNeighbors():
            if neighbor.GetSymbol() != 'C':
                atoms.append(neighbor.GetIdx())
        sulfoneamid_atoms.append(atoms)
    return sulfoneamid_atoms


class SubstructureDictionaryHolder:
    def __init__(self):
        self.atoms_set_dict = {}
        self.BT_MAPPING_CHAR = {
            Chem.rdchem.BondType.SINGLE: 'S',
            Chem.rdchem.BondType.DOUBLE: 'D',
            Chem.rdchem.BondType.TRIPLE: 'T',
            Chem.rdchem.BondType.AROMATIC: 'A',
        }
        self.BT_MAPPING_INT = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 1.5,
        }
        self.STRUCT_TO_NUM = {
            'ATOM': 0,
            'RING': 1,
            'ACID': 2,
            'AMIN': 3,
            'ESTER': 4,
            'SULFONEAMID': 5
        }

    def get_mapping_for_substructure(self, substructure, mol, structure_type):
        return self.generate_substructure_sum_vector_mapping(substructure, mol, structure_type)

    def structure_encoding(self, atoms):
        enc = [0 for _ in range(55)]
        for atom in atoms:
            enc[atom.GetAtomicNum()] += 1
        return enc

    def onek_encoding_unk(self, value, choices_len):
        encoding = [0] * choices_len
        encoding[int(value)] = 1
        return encoding

    def generate_substructure_sum_vector_mapping(self, substruct, mol, structure_type):
        atoms = [mol.GetAtomWithIdx(i) for i in substruct]

        substruct_atomic_encoding = self.structure_encoding(atoms)

        implicit_substruct_valence = 0
        for i in range(len(atoms)):
            for j in range(i, len(atoms)):
                bond = mol.GetBondBetweenAtoms(substruct[i], substruct[j])
                if bond:
                    implicit_substruct_valence += self.BT_MAPPING_INT[
                        mol.GetBondBetweenAtoms(substruct[i], substruct[j]).GetBondType()]
        substruct_valence = sum(atom.GetExplicitValence() for atom in atoms) - 2 * implicit_substruct_valence
        substruct_valence_array = self.onek_encoding_unk(substruct_valence, 40)

        substruct_formal_charge = sum(atom.GetFormalCharge() for atom in atoms)

        substruct_num_Hs = sum(atom.GetTotalNumHs() for atom in atoms)
        substruct_Hs_array = self.onek_encoding_unk(substruct_num_Hs, 60)

        substruct_is_aromatic = 1 if sum(atom.GetIsAromatic() for atom in atoms) > 0 else 0

        substruct_mass = sum(atom.GetMass() for atom in atoms)

        substruct_edges_sum = implicit_substruct_valence

        substruct_type = self.onek_encoding_unk(self.STRUCT_TO_NUM[structure_type], len(self.STRUCT_TO_NUM))

        features = substruct_atomic_encoding + substruct_valence_array + substruct_Hs_array + substruct_type + \
                   [substruct_formal_charge, substruct_is_aromatic, substruct_mass * 0.01, substruct_edges_sum * 0.1]
        return tuple(features)


if __name__ == "__main__":
    """
        Generates mapping for all the cycles from the input file.
        Encodes a substruct into a vector with features
    """
    data = pd.read_csv('../../data/3_final_data/logP.csv')
    substructs_dictionary_holder = SubstructureDictionaryHolder()
