import itertools
import json
from functools import reduce

import pandas as pd
from rdkit import Chem


def get_cycles_for_molecule(mol):
    all_cycles = Chem.GetSymmSSSR(mol)
    all_cycles = [list(ring) for ring in all_cycles]
    return all_cycles


class RingsDictionaryHolder:
    def __init__(self, filename, representation_type):
        self.filename = filename
        self.type = representation_type
        self.rings_dict = {}
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
        bonds = [self.BT_MAPPING_CHAR[mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]).GetBondType()] for i
                 in range(len(ring))]
        atoms = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring]
        found_in_dict = None

        assert len(atoms) > 2
        for i in range(len(atoms)):
            cur_atoms = atoms[i:] + atoms[:i]
            cur_bonds = bonds[i:] + bonds[:i]
            elements = [cur_atoms, cur_bonds]
            ring_tuple = (i for i in itertools.chain.from_iterable(elements))
            ring_string = ''.join(ring_tuple)
            if ring_string in self.rings_dict.keys():
                found_in_dict = self.rings_dict[ring_string]
                break
        return ring_string, found_in_dict

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
        if atoms not in self.atoms_set_dict.keys():
            self.atoms_set_dict[atoms] = self.get_max_class_dict()
        return self.atoms_set_dict[atoms]

    def generate_ring_sum_vector_mapping(self, ring, mol):
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        ring_atomic_charge = sum(atom.GetAtomicNum() for atom in atoms)
        ring_valence = sum(atom.GetExplicitValence() for atom in atoms) - 2 * sum(
            [self.BT_MAPPING_INT[mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]).GetBondType()] for i in
             range(len(ring))])
        ring_formal_charge = sum(atom.GetFormalCharge() for atom in atoms)
        ring_num_Hs = sum(atom.GetTotalNumHs() for atom in atoms)
        arom = [mol.GetBondBetweenAtoms(ring[i - 1], ring[i]).GetIsAromatic() for i
                in range(len(ring))]
        ring_is_aromatic = int(reduce(lambda x, y: x or y, arom))
        return ring_atomic_charge, ring_valence, ring_formal_charge, ring_num_Hs, ring_is_aromatic


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
    data = pd.read_csv('../../data/3_final_data/zinc_dataset.csv')
    rings_dictionary_holder = RingsDictionaryHolder('rings_zinc_features.json', 'sum-vector')
    rings_dictionary_holder.generate_ring_mapping(data)
    rings_dictionary_holder.save_rings_to_json()
