# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

__author__ = "Kyle Bystrom"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyam Dwarakanth"
__email__ = "shyamd@lbl.gov"
__date__ = "11/29/2019"

import unittest
from pymatgen.analysis.interface import (
    Interface,
    CoherentInterfaceBuilder,
    get_rot_3d_for_2d,
    get_2d_transform,
    from_2d_to_3d,
    match_strain
)
from pymatgen.analysis.substrate_analyzer import ZSLGenerator
from pymatgen.util.testing import PymatgenTest
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
import random
import numpy as np


class InterfaceTest(PymatgenTest):
    def setUp(self):
        si_struct = self.get_structure("Si")
        sio2_struct = self.get_structure("SiO2")

        sga = SpacegroupAnalyzer(si_struct)
        si_conventional = sga.get_conventional_standard_structure()
        sga = SpacegroupAnalyzer(sio2_struct)
        sio2_conventional = sga.get_conventional_standard_structure()
        self.builder = CoherentInterfaceBuilder(
            substrate_structure=si_conventional,
            film_structure=sio2_conventional,
            substrate_miller=(1, 0, 0),
            film_miller=(1, 0, 0),
        )
        termination = self.builder.terminations[0]
        self.interface = next(
            self.builder.get_interfaces(
                termination=termination,
                gap=2.0,
                vacuum_over_film=20.0,
                film_thickness=1,
                substrate_thickness=1,
            )
        )

    def test_basic_props(self):
        props = ["film_indices", "film_sites", "film"]

        interface = self.interface
        assert len(interface.substrate_indicies) == 52
        assert len(interface.film_indices) == 63
        assert len(interface.film_sites) == len(interface.film_indices)
        assert len(interface.substrate_sites) == len(interface.substrate_indicies)
        assert interface.gap == 2.0
        assert np.allclose(interface.in_plane_offset, [0, 0])
        assert interface.vacuum_over_film == 20.0
        assert interface.structure_properties["film_thickness"] == 1
        assert interface.structure_properties["substrate_thickness"] == 1

    def test_gap(self):
        interface = self.interface
        init_lattice = interface.lattice.matrix.copy()

        assert np.allclose(interface.gap, 2.0)

        max_sub_c = np.max(np.array([s.frac_coords for s in interface.substrate])[:, 2])
        min_film_c = np.min(np.array([f.frac_coords for f in interface.film])[:, 2])
        gap = (min_film_c - max_sub_c) * interface.lattice.c
        assert np.allclose(interface.gap, gap)

        interface.gap += 1

        assert np.allclose(interface.gap, 3.0)

        max_sub_c = np.max(np.array([s.frac_coords for s in interface.substrate])[:, 2])
        min_film_c = np.min(np.array([f.frac_coords for f in interface.film])[:, 2])
        gap = (min_film_c - max_sub_c) * interface.lattice.c
        assert np.allclose(interface.gap, gap)

    def test_in_plane_offset(self):

        interface = self.interface
        init_coords = self.interface.frac_coords
        interface.in_plane_offset += np.array([0.2, 0.2])

        assert np.allclose(interface.in_plane_offset, np.array([0.2, 0.2]))

        test_coords = np.array(init_coords)
        for i in interface.film_indices:
            test_coords[i] += [0.2, 0.2, 0]
        assert np.allclose(np.mod(test_coords, 1.0), np.mod(interface.frac_coords, 1.0))

    def test_vacuum_over_film(self):

        interface = self.interface
        init_coords = self.interface.cart_coords

        assert interface.vacuum_over_film == 20

        interface.vacuum_over_film += 10

        assert interface.vacuum_over_film == 30
        assert np.allclose(init_coords, interface.cart_coords)


class CoherentInterfaceBuilderTest(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        si_struct = cls.get_structure("Si")
        sio2_struct = cls.get_structure("SiO2")

        sga = SpacegroupAnalyzer(si_struct)
        si_conventional = sga.get_conventional_standard_structure()
        sga = SpacegroupAnalyzer(sio2_struct)
        sio2_conventional = sga.get_conventional_standard_structure()

        cls.builder = CoherentInterfaceBuilder(
            substrate_structure=si_conventional,
            film_structure=sio2_conventional,
            substrate_miller=(1, 0, 0),
            film_miller=(1, 0, 0),
        )

    def test_init(self):
        assert len(self.builder.zsl_matches) == 20
        assert len(self.builder.terminations) == 2

    def test_get_interfaces(self):
        termination = self.builder.terminations[0]
        interfaces = list(self.builder.get_interfaces(termination))
        assert len(interfaces) == len(self.builder.zsl_matches)


class InterfaceUtilsTest(PymatgenTest):
    def test_get_rot_3d_for_2d(self):
        film_matrix = np.eye(3)
        sub_matrix = np.array(film_matrix)
        sub_matrix[0][1] = 1.0
        two_d_rot = [
            [0.89442719, -0.4472136, 0.0],
            [0.4472136, 0.89442719, 0.0],
            [0.0, 0.0, 1.0],
        ]
        assert np.allclose(get_rot_3d_for_2d(film_matrix, sub_matrix), two_d_rot)

    def test_get_2d_transform(self):

        film_matrix = np.eye(3)[:2]
        sub_matrix = np.eye(3)
        sub_matrix[0][1] = 1.0
        sub_matrix = sub_matrix[:2]
        two_d_rot = [[1, 1], [0, 1]]

        assert np.allclose(get_2d_transform(film_matrix, sub_matrix), two_d_rot)

    def test_from_2d_to_3d(self):

        test_matrix = np.ones((2, 2))
        assert np.allclose(
            from_2d_to_3d(test_matrix), [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
        )
        
    def test_match_strain(self):

        assert np.isclose(
            match_strain([[1.0, 0, 0], [0, 1, 0]], [[1.0, 0, 0], [0, 1, 0]]), 0.0
        )

        assert np.isclose(
            match_strain([[1, 0, 0], [0, 1, 0]], [[0.95, 0, 0], [0, 1, 0]]), 0.0325
        )
        assert np.isclose(
            match_strain([[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 0.95, 0]]), 0.0325
        )
        assert np.isclose(
            match_strain([[1, 0, 0], [0, 1, 0]], [[0.95, 0, 0], [0, 0.95, 0]]), 0.0325
        )
        assert np.isclose(
            match_strain([[1, 0, 0], [0, 1, 0]], [[0, 0.95, 0], [0.95, 0, 0]]), 0.0325
        )


if __name__ == "__main__":
    unittest.main()
