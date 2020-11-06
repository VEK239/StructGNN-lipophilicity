import inspect
import os
import sys
import unittest

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from args import TrainArgs
from features import MolGraphWithSubstructures
from features.molecule import create_molecule_for_smiles


class MoleculeTestWithSubstructuresMergeTest(unittest.TestCase):

    def test_bonds_count_merge(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "Nc1ccc(O)c2ncccc12"
        mol = create_molecule_for_smiles(smi, args)
        print(mol.bonds)
        self.assertEqual(2, len(mol.bonds))

    def test_atoms_count_merge(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "Nc1ccc(O)c2ncccc12"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(3, len(mol.f_atoms))

    def test_bonds_count_for_big_molecule_merge(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "C1CC3C2C(CCC1)C=CC(=C2C3)CC(C)C"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(4, len(mol.bonds))

    def test_atoms_count_for_big_molecule_merge(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "C1CC3C2C(CCC1)C=CC(=C2C3)CC(C)C"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(5, len(mol.f_atoms))

    def test_many_merged_cycles(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "C1=CC%11=C9C2=C1C=C3C(=C2)C=C4C(=C3)C=CC5=C4C6=C(C=C5)C=CC7=C6C=C8C(=C7)C=C%10C(=C8)C9=C(C=C%10)C=C%11"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(1, len(mol.f_atoms))
        self.assertEqual(0, len(mol.f_bonds))

    def test_adamantan(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "C1C2CC3CC1CC(C2)C3"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(1, len(mol.f_atoms))
        self.assertEqual(0, len(mol.f_bonds))

    def test_substructures_intersection(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "C1=CC(=CC=C1N)S(=O)(=O)N"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(3, len(mol.atoms))
        self.assertEqual(2, len(mol.bonds))

    def test_many_substructures_intersect(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "C1=CC=NC(=C1)NS(=O)(=O)C2=CC=C(C=C2)NN=C3C=CC(=O)C(=C3)C(=O)O"
        mol = create_molecule_for_smiles(smi, args)
        self.assertEqual(8, len(mol.atoms))
        self.assertEqual(7, len(mol.bonds))

    def test_big_rings_merging_with_substructures_inside(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "C1CCC3C(CC1)CC(C2C(CCCC2)N[S](C(C)C4=CC=CC3=C4)(=C)=O)C5CCCCC5"
        mol = create_molecule_for_smiles(smi, args)
        print(mol.atoms)
        self.assertEqual(5, len(mol.atoms))  # TODO: change this to firstly sulfonamid or ring. One of two
        self.assertEqual(4, len(mol.bonds))
