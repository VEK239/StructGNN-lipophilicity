import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from scripts.baseline_improvements.chemprop.args import TrainArgs
from scripts.baseline_improvements.chemprop.features import MolGraphWithSubstructures
from scripts.baseline_improvements.chemprop.test.MolGraphWithSubstructuresTest import MolGraphWithSubstructuresTest
from scripts.baseline_improvements.chemprop.test.MoleculeTest import MoleculeTest


class MoleculeTestWithSubstructuresMergeTest(MolGraphWithSubstructuresTest, MoleculeTest):

    def test_bonds_count_merge(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "Nc1ccc(O)c2ncccc12"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(len(mol.f_bonds), 2)

    def test_atoms_count_merge(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "Nc1ccc(O)c2ncccc12"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(len(mol.f_bonds), 3)

    def test_bonds_count_for_big_molecule_merge(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "C1CC3C2C(CCC1)C=CC(=C2C3)CC(C)C"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(len(mol.f_bonds), 4)

    def test_atoms_count_for_big_molecule_merge(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "C1CC3C2C(CCC1)C=CC(=C2C3)CC(C)C"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(len(mol.f_atoms), 5)

    def test_many_merged_cycles(self):
        args = TrainArgs()
        args.substructures_merge = True
        args.substructures_use_substructures = True

        smi = "C1=CC%11=C9C2=C1C=C3C(=C2)C=C4C(=C3)C=CC5=C4C6=C(C=C5)C=CC7=C6C=C8C(=C7)C=C%10C(=C8)C9=C(C=C%10)C=C%11"
        mol = MolGraphWithSubstructures(smi, args)
        self.assertEqual(len(mol.f_atoms), 1)
        self.assertEqual(len(mol.f_bonds), 0)
