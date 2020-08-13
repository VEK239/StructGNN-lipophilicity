import itertools
import json
from collections import defaultdict
from functools import reduce

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def get_cycles_for_molecule(mol):
    all_cycles = Chem.GetSymmSSSR(mol)
    all_cycles = [set(ring) for ring in all_cycles]
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


def generate_submol(ring, mol):
    bonds = set()
    for atom_idx_1 in ring:
        for atom_idx_2 in ring:
            bond = mol.GetBondBetweenAtoms(atom_idx_1, atom_idx_2)
            if bond:
                bonds.add(bond.GetIdx())
    return Chem.PathToSubmol(mol, list(bonds))


class RingsDictionaryHolder:
    def __init__(self, filename, representation_type):
        self.filename = filename
        self.type = representation_type
        self.rings_dict = {}
        self.atoms_set_dict = {}
        self.ring_to_num = {'B': 0,
                            'C': 1,
                            'N': 2,
                            'O': 3,
                            'F': 4,
                            'Na': 6,
                            'P': 7,
                            'S': 8,
                            'Cl': 9,
                            'K': 10,
                            'Ca': 11,
                            'Fe': 12,
                            'Br': 13,
                            'I': 14}
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

    def get_mapping_for_ring(self, ring, mol):
        ring_string, _ = self.generate_ring_symbol(ring, mol)
        if self.type.startswith('one-hot'):
            one_hot_mapping = [0 for _ in range(self.get_max_class_dict())]
            one_hot_mapping[self.rings_dict[ring_string]] = 1
            return one_hot_mapping
        return self.rings_dict[ring_string]

    def save_rings_to_json(self):
        with open(self.filename, 'w') as fp:
            json.dump(self.rings_dict, fp)

    def read_json_rings_dict(self):
        with open(self.filename, 'r') as fp:
            self.rings_dict = json.load(fp)
            return self.rings_dict

    def generate_ring_mapping(self, data):
        for smi in data.smiles:
            mol = Chem.MolFromSmiles(smi)
            # print(smi)
            rings = get_cycles_for_molecule(mol)
            for ring in rings:
                ring_string, found_in_dict = self.generate_ring_symbol(ring, mol)
                if not found_in_dict:
                    if self.type == 'mass':
                        self.rings_dict[ring_string] = self.create_hash_for_ring(ring, mol)
                    elif self.type == 'one-hot':
                        self.rings_dict[ring_string] = self.generate_ring_one_hot_class_mapping(ring, mol)
                    elif self.type == 'one-hot-aromatic':
                        self.rings_dict[ring_string] = self.generate_ring_one_hot_class_mapping(ring, mol,
                                                                                                aroma=True)
                    elif self.type == 'sum-vector':
                        self.rings_dict[ring_string] = self.generate_ring_sum_vector_mapping(ring, mol)

    def generate_ring_symbol(self, ring, mol):
        submol = generate_submol(ring, mol)
        ring_symbol = Chem.MolToSmiles(submol)
        return ring_symbol, ring_symbol in self.rings_dict

    def create_hash_for_ring(self, ring, mol):
        bonds = [self.BT_MAPPING_INT[mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]).GetBondType()] for i in
                 range(len(ring))]
        atoms = [mol.GetAtomWithIdx(i).GetMass() for i in ring]
        bond_pairs = [bonds[i - 1] * bonds[i] for i in range(len(bonds))]
        return sum(bond_pairs) + sum(atoms)

    def get_max_class_dict(self):
        return len(self.atoms_set_dict) + 60  # 60 for single atoms

    def generate_ring_one_hot_class_mapping(self, ring, mol, aroma=False):
        atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring]
        atoms.sort()
        if aroma:
            arom = [mol.GetBondBetweenAtoms(ring[i - 1], ring[i]).GetIsAromatic() for i
                    in range(len(ring))]
            atoms.append(
                reduce(lambda x, y: x or y, arom))
        atoms = tuple(atoms)
        if atoms not in self.ring_to_num.keys():
            self.ring_to_num[atoms] = len(self.ring_to_num.keys())
        return self.ring_to_num[atoms]

    def structure_encoding(self, atoms):
        enc = [0 for _ in range(55)]
        for atom in atoms:
            enc[atom.GetAtomicNum()] += 1
        return enc

    def get_valence(self, value):
        enc = [0 for _ in range(40)]
        enc[int(value)] = 1
        return enc

    def hydrogens_count_encoding(self, value):
        enc = [0 for _ in range(65)]
        enc[int(value)] = 1
        return enc

    def create_one_hot_for_ring(self, value):
        enc = [0 for _ in range(130)]
        enc[int(value)] = 1
        return enc

    def generate_ring_sum_vector_mapping(self, ring, mol):
        submol = generate_submol(ring, mol)
        atoms = [mol.GetAtomWithIdx(i) for i in ring]

        ring_atomic_encoding = self.structure_encoding(atoms)

        implicit_ring_valence = 0
        for i in range(len(atoms)):
            for j in range(i, len(atoms)):
                bond = mol.GetBondBetweenAtoms(ring[i], ring[j])
                if bond:
                    implicit_ring_valence += self.BT_MAPPING_INT[
                        mol.GetBondBetweenAtoms(ring[i], ring[j]).GetBondType()]
        ring_valence = sum(atom.GetExplicitValence() for atom in atoms) - 2 * implicit_ring_valence
        ring_valence_array = self.get_valence(ring_valence)

        ring_formal_charge = sum(atom.GetFormalCharge() for atom in atoms)

        ring_num_Hs = sum(atom.GetTotalNumHs() for atom in atoms)
        print(ring_num_Hs)
        ring_Hs_array = self.hydrogens_count_encoding(ring_num_Hs)

        # arom = submol.GetIsAromatic()
        ring_is_aromatic = 1 if len(submol.GetAromaticAtoms()) > 0 else 0

        ring_mass = Descriptors.ExactMolWt(submol)

        ring_edges_sum = implicit_ring_valence

        features = ring_atomic_encoding + ring_valence_array + ring_Hs_array + \
                   [ring_formal_charge, ring_is_aromatic, ring_mass * 0.01, 1, ring_edges_sum * 0.1]
        return tuple(features)


if __name__ == "__main__":
    """
        Generates mapping for all the cycles from the input file.

        type=mass: 
        Encodes a ring into a number by summing ring's atomic mass and bond pairs composition.
        type=one-hot: 
        Encodes a ring into a one-hot vector where rings with similar atom composition are in the same class        
        type=one-hot-aromatic: 
        Encodes a ring into a one-hot vector where rings with similar atom composition and aromaticy are in the same class
        type=sum-vector:
        Encodes a ring into a vector with features
    """
    data = pd.read_csv('../../data/3_final_data/logP.csv')
    rings_dictionary_holder = RingsDictionaryHolder('rings_logp_features.json', 'sum-vector')
    rings_dictionary_holder.generate_ring_mapping(data)
    rings_dictionary_holder.save_rings_to_json()
    # mol = Chem.MolFromSmiles('C1OC1C1CO1')
    # print(mol.GetNumAtoms())
    # cycles = get_cycles_for_molecule(mol)
    # print(cycles)
    # for cycle in cycles:
    #     print(rings_dictionary_holder.generate_ring_symbol(cycle, mol))
