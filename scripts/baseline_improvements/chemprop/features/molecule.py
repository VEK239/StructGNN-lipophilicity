from collections import defaultdict

from rdkit import Chem
import pandas as pd
from scripts.baseline_improvements.chemprop.features.substructure_dictionary_creating import \
    SubstructureDictionaryHolder, get_cycles_for_molecule, \
    get_acids_for_molecule, get_esters_for_molecule, get_amins_for_molecule, get_sulfoneamids_for_molecule
from tqdm import tqdm


class Atom:
    def __init__(self, idx, atom_representation, symbol=''):
        self.symbol = symbol
        self.idx = idx
        self.atom_representation = atom_representation
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


def create_molecule_for_smiles(substruct_dictionary_holder, smiles):
    mol = Chem.MolFromSmiles(smiles)

    rings = get_cycles_for_molecule(mol)
    acids = get_acids_for_molecule(mol)
    esters = get_esters_for_molecule(mol)
    amins = get_amins_for_molecule(mol)
    sulfoneamids = get_sulfoneamids_for_molecule(mol)

    used_atoms = set()
    mol_bonds = []
    mol_atoms = []
    idx_to_atom = defaultdict(set)

    for structure_type in [[rings, 'RING'], [acids, 'ACID'], [esters, 'ESTER'], [amins, 'AMIN'],
                           [sulfoneamids, 'SULFONEAMID']]:
        substructure_type_string = structure_type[1]
        substructures = structure_type[0]
        for substruct in substructures:
            mapping = substruct_dictionary_holder.get_mapping_for_substructure(substruct, mol, substructure_type_string)
            substruct_atom = Atom(idx=(min(*substruct) if len(substruct) > 1 else substruct[0]),
                                  atom_representation=mapping)
            mol_atoms.append(substruct_atom)
            for idx in substruct:
                idx_to_atom[idx].add(substruct_atom)
                used_atoms.add(idx)

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom_idx not in used_atoms:
            atom_repr = substruct_dictionary_holder.get_mapping_for_substructure([atom_idx], mol, 'ATOM')
            custom_atom = Atom(idx=atom_idx, atom_representation=atom_repr, symbol=atom.GetSymbol())
            mol_atoms.append(custom_atom)
            idx_to_atom[atom_idx].add(custom_atom)

    for idx, bond in enumerate(mol.GetBonds()):
        start_atoms = idx_to_atom[bond.GetBeginAtomIdx()]
        end_atoms = idx_to_atom[bond.GetEndAtomIdx()]
        if len(start_atoms & end_atoms) > 0:
            custom_bond = Bond(bond, idx, start_atoms, end_atoms, bond.GetBondType())
            mol_bonds.append(custom_bond)
            for start_atom in start_atoms:
                start_atom.add_bond(custom_bond)
            for end_atom in end_atoms:
                end_atom.add_bond(custom_bond)

    custom_mol = Molecule(mol_atoms, mol_bonds, mol)
    return custom_mol


def create_custom_mols_for_data(data):
    custom_mols = []
    smiles = list(data.smiles)
    for i in tqdm(range(len(smiles))):
        smi = smiles[i]
        substruct_dictionary_holder = SubstructureDictionaryHolder()
        custom_mols.append(create_molecule_for_smiles(substruct_dictionary_holder, smi))
    return custom_mols


if __name__ == "__main__":
    data = pd.read_csv('../../data/3_final_data/logP.csv')
    mols = create_custom_mols_for_data(data)
    atoms_count = pd.DataFrame({'custom_count': [mol.get_num_atoms() for mol in mols], 'rdkit_count':
        [mol.rdkit_mol.GetNumAtoms() for mol in mols]})
    atoms_count.to_csv('atoms_count.csv')
