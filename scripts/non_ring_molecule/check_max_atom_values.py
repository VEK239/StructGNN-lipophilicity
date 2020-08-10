import pandas as pd
import json
import itertools
from rdkit import Chem
from ring_dictionary_creating import get_cycles_for_molecule

BT_MAPPING_INT = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 1.5,
        }

if __name__ == "__main__":
    data = pd.read_csv('../../data/3_final_data/zinc_dataset.csv')
    max_valence = 0
    atom_symbols = set()
    ring_charges = set()
    atom_charges = set()
    max_ring_formal_charge = 0
    for smi in data.smiles:
        mol = Chem.MolFromSmiles(smi)
        for atom in mol.GetAtoms():
            atom_symbols.add(atom.GetSymbol())
        rings = get_cycles_for_molecule(mol)
        for ring in rings:
            atoms = [mol.GetAtomWithIdx(i) for i in ring]
            for atom in atoms:
                atom_charges.add(atom.GetAtomicNum())
            ring_charges.add(sum(atom.GetAtomicNum() for atom in atoms))
            max_valence = max(max_valence, sum(atom.GetExplicitValence() for atom in atoms) - 2 * sum([BT_MAPPING_INT[mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)]).GetBondType()] for i in
                 range(len(ring))]))
            if max_valence == 39:
                print(smi)
                break
            max_ring_formal_charge = max(max_ring_formal_charge, sum(atom.GetFormalCharge() for atom in atoms))
    print('max valence:', max_valence)
    print('ring charges: ', ring_charges)
    print('atom charges: ', atom_charges)
    print('atoms:', atom_symbols)