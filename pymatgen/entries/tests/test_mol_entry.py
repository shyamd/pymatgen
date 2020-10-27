from pymatgen import Molecule
from pymatgen.entries.mol_entry import MoleculeEntry
import pytest
try:
    import openbabel as ob
except ImportError:
    ob = None

def make_a_mol_entry():
    r"""
    Make a symmetric (fake) molecule with ring.
                O(0)
               / \
              /   \
      H(1)--C(2)--C(3)--H(4)
             |     |
            H(5)  H(6)
    """
    species = ["O", "H", "C", "C", "H", "H", "H"]
    coords = [
        [0.0, 1.0, 0.0],
        [-1.5, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [-0.5, -1.0, 0.0],
        [0.5, -1.0, 0.0],
    ]

    m = Molecule(species, coords)
    entry = MoleculeEntry(m, energy=0.0)

    return entry

class TestMolEntry:

    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_property():
        """
        Most of them are tested in pymatgen.Molecule, so here we only test `new` stuff.
        """
        entry = make_a_mol_entry()
        assert set(entry.edges) == {(0, 2), (0, 3), (1, 2), (2, 3), (3, 4), (2, 5), (3, 6)}


    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_get_fragments():
        entry = make_a_mol_entry()
        fragments = entry.get_fragments()

        assert len(fragments) == 7

        # break bond yields 1 fragment
        assert len(fragments[(0, 2)]) == 1
        assert len(fragments[(0, 3)]) == 1
        assert len(fragments[(2, 3)]) == 1

        # break bond yields 2 fragments
        assert len(fragments[(1, 2)]) == 2
        assert len(fragments[(2, 5)]) == 2
        assert len(fragments[(3, 4)]) == 2
        assert len(fragments[(3, 6)]) == 2


    @staticmethod
    @pytest.mark.skipif(not ob, reason="OpenBabel not present. Skipping...")
    def test_get_isomorphic_bonds():
        entry = make_a_mol_entry()
        iso_bonds = entry.get_isomorphic_bonds()

        # sort iso_bonds for easier comparison
        iso_bonds = sorted([sorted(group) for group in iso_bonds])

        assert iso_bonds == [[(0, 2), (0, 3)], [(1, 2), (2, 5), (3, 4), (3, 6)], [(2, 3)]]
