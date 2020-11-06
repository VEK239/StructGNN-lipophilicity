import unittest

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from args import TrainArgs
from features import MolGraphWithSubstructures


class MolGraphWithSubstructuresTest(unittest.TestCase):

    def test_bonds_count(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "Nc1ccc(O)c2ncccc12"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(len(mol.f_bonds), 4)

    def test_bonds_count_for_big_molecule(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "C1=CC=C(C(=C1CC(C)C)CCCCNCCN)C(=O)O"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(len(mol.f_bonds), 26)

    def test_atoms_count(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "Nc1ccc(O)c2ncccc12"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(len(mol.f_atoms), 4)

    def test_atoms_count_for_big_molecule(self):
        args = TrainArgs()
        args.substructures_merge = False
        args.substructures_use_substructures = True

        smi = "C1=CC=C(C(=C1CC(C)C)CCCCNCCN)C(=O)O"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(len(mol.f_atoms), 14)


if __name__ == '__main__':
    unittest.main()
