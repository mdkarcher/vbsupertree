import unittest
import sys
sys.path.append("..")
from classes import *


class MyTreeTestCase(unittest.TestCase):
    def test_root_clade(self):
        taxon_set = set("ABCDE")
        tree = MyTree.random(taxon_set)
        self.assertEqual(tree.root_clade(), Clade(taxon_set))

    def test_copy(self):
        taxon_set = set("ABCDE")
        tree1 = MyTree.random(taxon_set)
        tree2 = tree1.copy()
        self.assertEqual(tree1, tree2)
        subset = set("ABCD")
        tree2 = tree2.restrict(subset)
        self.assertNotEqual(tree1, tree2)


if __name__ == '__main__':
    unittest.main()
