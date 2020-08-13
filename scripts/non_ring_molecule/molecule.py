from collections import defaultdict

from rdkit import Chem
import pandas as pd
from ring_dictionary_creating import RingsDictionaryHolder, get_cycles_for_molecule
from tqdm import tqdm


class Atom:
    def __init__(self, idx, symbol, atom_representation, is_ring=False):
        self.symbol = symbol
        self.idx = idx
        self.atom_representation = atom_representation
        self.is_ring = is_ring
        self.bonds = []

    def add_bond(self, bond):
        self.bonds.append(bond)

    def get_representation(self):
        return list(self.atom_representation)


class Bond:
    def __init__(self, rdkit_bond, idx, out_atom_idx, in_atom_idx, bond_type):
        self.rdkit_bond = rdkit_bond
        self.idx = idx
        self.out_atom_idx = out_atom_idx
        self.in_atom_idx = in_atom_idx
        self.bond_type = bond_type

    def get_rdkit_bond(self):
        return self.rdkit_bond


class Molecule:
    def __init__(self, atoms, bonds, rdkit_mol):
        self.atoms = atoms
        self.bonds = bonds
        self.rdkit_mol = rdkit_mol

    def get_bond(self, atom_1, atom_2):
        # If bond does not exist between atom_1 and atom_2, return None
        for bond in self.atoms[atom_1].bonds:
            if atom_2 == bond.out_atom_idx or atom_2 == bond.in_atom_idx:
                return bond
        return None

    def get_atoms(self):
        return self.atoms

    def get_atom(self, atom_idx):
        return self.atoms[atom_idx]

    def get_num_atoms(self):
        return len(self.atoms)

    def prnt(self):
        for atom in self.atoms:
            print(atom.symbol, atom.idx, atom.bonds, atom.atom_representation)
        for bond in self.bonds:
            print(bond.out_atom_idx, bond.in_atom_idx)


def create_molecule_for_smiles(rings_dictionary_holder, smiles, representation_type):
    mol = Chem.MolFromSmiles(smiles)
    rings = get_cycles_for_molecule(mol)
    ring_atoms = set()
    for ring in rings:
        for atom in ring:
            ring_atoms.add(atom)
    used_atoms = set()
    mol_bonds = []
    mol_atoms = []
    idx_to_atom = {}

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom_idx not in used_atoms:
            if atom_idx not in ring_atoms:
                if representation_type == 'mass':
                    atom_repr = atom.GetMass()
                elif representation_type == 'one-hot' or representation_type == 'one-hot-aromatic':
                    atom_repr = [0 for _ in range(rings_dictionary_holder.get_max_class_dict())]
                    atom_repr[atom.GetAtomicNum()] = 1
                elif representation_type == 'sum-vector':
                    atom_repr = tuple(
                        rings_dictionary_holder.structure_encoding([atom]) + rings_dictionary_holder.get_valence(
                            atom.GetExplicitValence()) + rings_dictionary_holder.hydrogens_count_encoding(atom.GetTotalNumHs()) + [atom.GetFormalCharge(),
                                                          int(atom.GetIsAromatic()), atom.GetMass() * 0.01, 0, 0])

                custom_atom = Atom(idx=atom_idx, atom_representation=atom_repr, symbol=atom.GetSymbol())
                mol_atoms.append(custom_atom)
                idx_to_atom[atom_idx] = custom_atom

    for ring in rings:
        ring_mapping = rings_dictionary_holder.get_mapping_for_ring(ring, mol)
        ring_symbol, _ = rings_dictionary_holder.generate_ring_symbol(ring, mol)
        ring_atom = Atom(idx=min(*ring), symbol=ring_symbol, atom_representation=ring_mapping, is_ring=True)
        mol_atoms.append(ring_atom)
        for idx in ring:
            idx_to_atom[idx] = ring_atom

    for idx, bond in enumerate(mol.GetBonds()):
        start_atom = idx_to_atom[bond.GetBeginAtomIdx()]
        end_atom = idx_to_atom[bond.GetEndAtomIdx()]
        if start_atom != end_atom:
            custom_bond = Bond(bond, idx, start_atom, end_atom, bond.GetBondType())
            mol_bonds.append(custom_bond)
            start_atom.add_bond(custom_bond)
            end_atom.add_bond(custom_bond)

    custom_mol = Molecule(mol_atoms, mol_bonds, mol)
    return custom_mol


def create_custom_mols_for_data(data, representation_type, rings_data_file):
    custom_mols = []
    smiles = list(data.smiles)
    for i in tqdm(range(len(smiles))):
        smi = smiles[i]
        assert representation_type in ['mass', 'one-hot', 'one-hot-aromatic', 'sum-vector']
        rings_dictionary_holder = RingsDictionaryHolder(rings_data_file, representation_type)
        rings_dictionary_holder.read_json_rings_dict()
        custom_mols.append(create_molecule_for_smiles(rings_dictionary_holder, smi, representation_type))
    return custom_mols

if __name__ == "__main__":
    representation_type = 'sum-vector'
    rings_data_file = 'rings_logp_features.json'
    data = pd.read_csv('../../data/3_final_data/logP.csv')
    assert representation_type in ['mass', 'one-hot', 'one-hot-aromatic', 'sum-vector']
    mols = create_custom_mols_for_data(data, representation_type, rings_data_file)
    atoms_count = pd.DataFrame({'custom_count': [mol.get_num_atoms() for mol in mols], 'rdkit_count':
        [mol.rdkit_mol.GetNumAtoms() for mol in mols]})
    atoms_count.to_csv('atoms_count.csv')