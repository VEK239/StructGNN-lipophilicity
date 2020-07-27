from rdkit import Chem
import pandas as pd
from ring_dictionary_creating import RingsDictionaryHolder, get_cycles_for_molecule


class Atom:
    def __init__(self, idx, symbol, atomic_mass, is_ring=False):
        self.symbol = symbol
        self.idx = idx
        self.atomic_mass = atomic_mass
        self.is_ring = is_ring
        self.bonds = []

    def add_bond(self, bond):
        self.bonds.append(bond)


class Bond:
    def __init__(self, idx, out_atom_idx, in_atom_idx, bond_type):
        self.idx = idx
        self.out_atom_idx = out_atom_idx
        self.in_atom_idx = in_atom_idx
        self.bond_type = bond_type


class Molecule:
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds

    def get_bond(self, atom_1, atom_2):
        # If bond does not exist between atom_1 and atom_2, return None
        for bond in self.atoms[atom_1].bonds:
            if atom_2 == bond.out_atom_idx or atom_2 == bond.in_atom_idx:
                return bond
        return None

    def get_atom(self, atom_idx):
        return self.atoms[atom_idx]

    def get_num_atoms(self):
        return len(self.atoms)

    def prnt(self):
        for atom in self.atoms:
            print(atom.symbol, atom.idx, atom.bonds)
        for bond in self.bonds:
            print(bond.out_atom_idx, bond.in_atom_idx)


def create_molecule_for_smiles(rings_dictionary_holder, smiles):
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
                custom_atom = Atom(idx=atom_idx, atomic_mass=atom.GetMass(), symbol=atom.GetSymbol())
                mol_atoms.append(custom_atom)
                idx_to_atom[atom_idx] = custom_atom
    for ring in rings:
        ring_symbol, ring_mass = rings_dictionary_holder.generate_ring_symbol(ring, mol)
        ring_atom = Atom(idx=min(*ring), symbol=ring_symbol, atomic_mass=ring_mass, is_ring=True)
        mol_atoms.append(ring_atom)
        for idx in ring:
            idx_to_atom[idx] = ring_atom

    for idx, bond in enumerate(mol.GetBonds()):
        start_atom = idx_to_atom[bond.GetBeginAtomIdx()]
        end_atom = idx_to_atom[bond.GetEndAtomIdx()]
        if start_atom != end_atom:
            bond = Bond(idx, start_atom, end_atom, bond.GetBondType())
            mol_bonds.append(bond)
            start_atom.add_bond(bond)
            end_atom.add_bond(bond)

    custom_mol = Molecule(mol_atoms, mol_bonds)
    return custom_mol


def create_custom_mols_for_data(data):
    custom_mols = []
    for smi in data.smiles:
        rings_dictionary_holder = RingsDictionaryHolder()
        rings_dictionary_holder.read_json_rings_dict()
        custom_mols.append(create_molecule_for_smiles(rings_dictionary_holder, smi))


if __name__ == "__main__":
    rings_dictionary_holder = RingsDictionaryHolder()
    rings_dictionary_holder.read_json_rings_dict()
    custom_mol = create_molecule_for_smiles(rings_dictionary_holder, 'CSc1scc(-c2cccs2)c1C#N')
    print(custom_mol.get_num_atoms())
    custom_mol.prnt()
