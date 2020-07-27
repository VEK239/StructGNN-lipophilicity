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
    def __init__(self):
        self.rings_dict = {}
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

    def save_rings_to_json(self):
        with open('rings.json', 'w') as fp:
            json.dump(self.rings_dict, fp)

    def read_json_rings_dict(self):
        with open('rings.json', 'r') as fp:
            self.rings_dict = json.load(fp)
            return self.rings_dict

    def generate_ring_symbol(self, ring, mol):
        bonds = [self.BT_MAPPING_CHAR[mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]).GetBondType()] for i
                 in
                 range(len(ring))]
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

    def generate_ring_mapping(self, data):
        for smi in data.smiles:
            mol = Chem.MolFromSmiles(smi)
            rings = get_cycles_for_molecule(mol)
            for ring in rings:
                ring_string, found_in_dict = self.generate_ring_symbol(ring, mol)
                if not found_in_dict:
                    self.rings_dict[ring_string] = self.create_hash_for_ring(ring, mol)


if __name__ == "__main__":
    """
        Generates mapping for all the cycles from the input file.
        Encodes a ring into a number by summing ring's atomic mass and bond pairs composition.
    """
    data = pd.read_csv('../../data/3_final_data/logP.csv')
    rings_dictionary_holder = RingsDictionaryHolder()
    rings_dictionary_holder.generate_ring_mapping(data)
    rings_dictionary_holder.save_rings_to_json()
