import unittest
from scripts.baseline_improvements.chemprop.features.molecule import create_molecule_for_smiles
from scripts.baseline_improvements.chemprop.args import TrainArgs
from rdkit import Chem


class MoleculeTest(unittest.TestCase):

    def test_atom_count_wo_substructures(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = False

        smi = "Nc1ccc(O)c2ncccc12"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.atoms), 4)

    def test_bond_count_wo_substructures(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = False

        smi = "Nc1ccc(O)c2ncccc12"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.bonds), 2)  # no bond between cycles

    def test_atom_count_with_esters(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "CC#CC#CC#CC=CC(=O)OC"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.atoms), 11)

    def test_bond_count_with_esters(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "CC#CC#CC#CC=CC(=O)OC"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.bonds), 10)  # no bonds inside ester


    def test_atom_count_with_amins(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "CCC(C)CCCN"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.atoms), 8)

    def test_bond_count_with_amins(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "CCC(C)CCCN"

        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.bonds), 7)  # all bonds are present

    def test_atom_count_with_acids(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "CC(=O)O"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.atoms), 2)

        smi = "CC(=O)OC"
        mol = create_molecule_for_smiles(smi, args)
        self.assertNotIn('ACID', [atom.atom_type for atom in mol.atoms])
        self.assertIn('ESTER', [atom.atom_type for atom in mol.atoms])

    def test_bond_count_with_acids(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "CC(=O)O"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.bonds), 1)


    def test_atom_count_with_sulfonamids(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "O=S(=O)(c1ccccc1)N"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.atoms), 2)
        self.assertNotIn('AMIN', [atom.atom_type for atom in mol.atoms])
        self.assertIn('SULFONAMID', [atom.atom_type for atom in mol.atoms])

    def test_bond_count_with_sulfonamids(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "O=S(=O)(c1ccccc1)N"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.bonds), 1)

    def test_atom_types_with_substructures(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "O=S(=O)(c1cc(N)cc(CCC(=O)O)c1)N"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.atoms), 6)
        self.assertIn('AMIN', [atom.atom_type for atom in mol.atoms])
        self.assertIn('SULFONAMID', [atom.atom_type for atom in mol.atoms])
        self.assertIn('RING', [atom.atom_type for atom in mol.atoms])
        self.assertIn('ACID', [atom.atom_type for atom in mol.atoms])
        self.assertEqual(2, len([atom for atom in mol.atoms if atom.atom_type == 'ATOM']))

    def test_bond_count_with_substructures(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "O=S(=O)(c1cc(N)cc(CCC(=O)O)c1)N"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.bonds), 5)
        self.assertEqual(5, len([bond for bond in mol.bonds if bond.bond_type == Chem.rdchem.BondType.SINGLE]))

    def test_bond_count_for_duplicated_bonds(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True
        smi = "C1CCC2C(C1)C3C2CCCC3"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.bonds), 0)

    def test_atom_count_for_duplicated_bonds(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True
        smi = "C1CCC2C(C1)C3C2CCCC3"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(len(mol.atoms), 3)
        self.assertEqual(3, len([atom for atom in mol.atoms if atom.atom_type == 'RING']))



if __name__ == '__main__':
    unittest.main()
