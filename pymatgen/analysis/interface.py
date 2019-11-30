# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
from __future__ import annotations
import numpy as np
from itertools import product, combinations
from typing import List, Dict

from scipy.linalg import polar
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from pymatgen import Lattice, Structure, Site
from pymatgen.core.surface import Slab, SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.substrate_analyzer import fast_norm, ZSLGenerator


"""
This module provides classes to store, generate, and manipulate material interfaces.
"""

__author__ = "Eric Sivonxay, Shyam Dwaraknath, and Kyle Bystrom"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyam Dwaraknath"
__email__ = "shyamd@lbl.gov"
__date__ = "5/29/2019"
__status__ = "Prototype"


class Interface(Structure):
    """
    This class stores data for defining an interface between two structures.
    It is a subclass of pymatgen.core.structure.Structure.
    """

    def __init__(
        self,
        sub_slab: Slab,
        film_slab: Slab,
        in_plane_offset: List = (0, 0),
        gap: float = 1.6,
        vacuum_over_film: float = 0.0,
        structure_properties: Dict = {},
        center_slab: bool = True,
        **kwargs,
    ):
        """
        Makes an interface structure by merging a substrate and film slab
        The top of the film slab will be put ontop of the top of the
        substrate slab the two slabs are expected to be commensurate
        IE. the a,b, and c lattice vectors for the film and slab correspond

        For now, it's suggested to use one of the factory methods rather
        than directly calling this constructor

        Args:
            sub_slab: slab for the substrate
            film_slab: slab for the film
            in_plane_offset: fractional shift in plane
                for the film with respect to the substrate
            gap: gap between substrate and film in Angstroms
            vacuum_over_film: vacuum space above the film in Angstroms
            structure_properties: dictionary of misc properties for this structure
            center_slab: center the slab


        """
        # save the originals
        self.sub_slab = sub_slab.copy()
        self.film_slab = film_slab.copy()
        self.structure_properties = dict(structure_properties)
        self._in_plane_offset = list(in_plane_offset)
        self._gap = gap
        self._vacuum_over_film = vacuum_over_film
        self._center_slab = center_slab
        self._kwargs = dict(kwargs)

        # Ensure c-axis is orthogonal to a/b plane
        sub_slab = sub_slab.get_orthogonal_c_slab()
        film_slab = film_slab.get_orthogonal_c_slab()
        assert np.allclose(film_slab.lattice.alpha, 90, 0.1)
        assert np.allclose(film_slab.lattice.beta, 90, 0.1)
        assert np.allclose(sub_slab.lattice.alpha, 90, 0.1)
        assert np.allclose(sub_slab.lattice.beta, 90, 0.1)

        # Ensure sub is right-handed
        # IE sub has surface facing "up"
        sub_vecs = sub_slab.lattice.matrix.copy()
        if np.dot(np.cross(*sub_vecs[:2]), sub_vecs[2]) < 0:
            sub_vecs[2] *= -1.0
            sub_slab.lattice = Lattice(sub_vecs)

        # Find the limits of C-coords
        sub_coords = sub_slab.frac_coords
        film_coords = film_slab.frac_coords
        sub_min_c = np.min(sub_coords[:, 2]) * sub_slab.lattice.c
        sub_max_c = np.max(sub_coords[:, 2]) * sub_slab.lattice.c
        film_min_c = np.min(film_coords[:, 2]) * film_slab.lattice.c
        film_max_c = np.max(film_coords[:, 2]) * film_slab.lattice.c
        min_height = np.abs(film_max_c - film_min_c) + np.abs(sub_max_c - sub_min_c)

        # construct new lattice
        abc = sub_slab.lattice.abc[:2] + (min_height + gap + vacuum_over_film,)
        angles = sub_slab.lattice.angles
        lattice = Lattice.from_parameters(*abc, *angles)

        # Get the species
        species = sub_slab.species + film_slab.species

        # Get the coords
        # Shift substrate to bottom in new lattice
        sub_coords = np.subtract(sub_coords, [0, 0, np.min(sub_coords[:, 2])])
        sub_coords[:, 2] *= sub_slab.lattice.c / lattice.c

        # Flip the film over
        film_coords[:, 2] *= -1.0
        film_coords[:, 2] *= film_slab.lattice.c / lattice.c

        # Shift the film coords to right over the substrate + gap
        film_coords = np.subtract(film_coords, [0, 0, np.min(film_coords[:, 2])])
        film_coords = np.add(
            film_coords, [0, 0, gap / lattice.c + np.max(sub_coords[:, 2])]
        )

        # Build coords
        coords = np.concatenate([sub_coords, film_coords])

        # Shift coords to center
        if self._center_slab:
            coords = np.add(coords, [0, 0, 0.5 - np.average(coords[:, 2])])

        # Only merge site properties in both slabs
        site_properties = {}
        site_props_in_both = set(sub_slab.site_properties.keys()) & set(
            film_slab.site_properties.keys()
        )

        for key in site_props_in_both:
            site_properties[key] = [
                *sub_slab.site_properties[key],
                *film_slab.site_properties[key],
            ]

        site_properties["interface_label"] = ["substrate"] * len(sub_slab) + [
            "film"
        ] * len(film_slab)

        super().__init__(
            lattice,
            species,
            coords,
            to_unit_cell=False,
            coords_are_cartesian=False,
            site_properties=site_properties,
            **kwargs,
        )

    @property
    def in_plane_shift(self) -> List[float]:
        return self._in_plane_shift

    @in_plane_shift.setter
    def in_plane_shift(self, *new_shift: float) -> None:
        """
        Given two floats da and db, adjust the shift vector
        by da * (first lattice vector) + db * (second lattice vector).
        This shift is in the plane of the interface.
        I.e. da and db are fractional coordinates.

        Args:
            new_shift - fractional shift in a and b
        """
        if len(new_shift) != 2:
            raise Exception("In-plane shifts require two floats for a and b vectors")
        delta = new_shift - self.in_plane_shift
        self._in_plane_shift = new_shift
        self.translate_sites(
            self.film_indicies, [delta[0], delta[1], 0], to_unit_cell=True
        )

    @property
    def gap(self) -> float:
        return self._gap

    @gap.setter
    def gap(self, new_gap) -> None:
        # TODO: Should this change the lattice as well to be consistent with the constructor?
        delta = new_gap - self.gap
        self._gap = new_gap

        self.translate_sites(
            self.film_indices, [0, 0, delta], frac_coords=False, to_unit_cell=True
        )

    @property
    def vacuum_over_film(self) -> float:
        return self._vacuum_over_film

    @vacuum_over_film.setter
    def vacuum_over_film(self, new_vacuum) -> None:
        raise NotImplemented("This is not trivial and might not be reasonable")

    @property
    def substrate_indicies(self) -> List[int]:
        sub_indicies = [
            i
            for i, tag in enumerate(self.site_properties["interface_label"])
            if "substrate" in tag
        ]
        return sub_indicies

    @property
    def substrate_sites(self) -> List[Site]:
        """
        Return the substrate sites of the interface.
        """
        sub_sites = [
            site
            for site, tag in zip(self, self.site_properties["interface_label"])
            if "substrate" in tag
        ]
        return sub_sites

    @property
    def substrate(self) -> Structure:
        """
        Return the substrate (Structure) of the interface.
        """
        return Structure.from_sites(self.substrate_sites)

    @property
    def film_indices(self) -> List[int]:
        """
        Retrieve the indices of the film sites
        """
        film_indicies = [
            i
            for i, tag in enumerate(self.site_properties["interface_label"])
            if "film" in tag
        ]
        return film_indicies

    @property
    def film_sites(self) -> List[Site]:
        """
        Return the film sites of the interface.
        """
        film_sites = [
            site
            for site, tag in zip(self, self.site_properties["interface_label"])
            if "film" in tag
        ]
        return film_sites

    @property
    def film(self) -> Structure:
        """
        Return the film (Structure) of the interface.
        """
        return Structure.from_sites(self.film_sites)

    def copy(self, site_properties=None) -> Interface:
        """
        Convenience method to get a copy of the structure, with options to add
        site properties.

        Returns:
            A copy of the Interface.
        """

        return Interface(
            sub_slab=self.sub_slab.copy(),
            film_slab=self.film_slab.copy(),
            in_plane_offset=self.in_plane_offset,
            gap=self.gap,
            vacuum_over_film=self.vacuum_over_film,
            **self._kwargs,
        )

    def get_sorted_structure(self, key=None, reverse=False) -> Structure:
        """
        Get a sorted copy of the structure. The parameters have the same
        meaning as in list.sort. By default, sites are sorted by the
        electronegativity of the species.

        Args:
            key: Specifies a function of one argument that is used to extract
                a comparison key from each list element: key=str.lower. The
                default value is None (compare the elements directly).
            reverse (bool): If set to True, then the list elements are sorted
                as if each comparison were reversed.
        """
        struct_copy = Structure.from_sites(self)
        struct_copy.sort(key=key, reverse=reverse)
        return struct_copy

    def __update_c(self, new_c: float) -> None:
        """
        Modifies the c-direction of the lattice without changing the site cartesian coordinates
        """
        assert new_c > 0, "New c-length must be greater than 0"
        new_latice = Lattice(self.lattice.matrix[:2].tolist() + [0, 0, new_c])
        self._lattice = new_latice

        for site, c_coords in zip(self, self.cart_coords):
            site.lattice = new_latice  # Update the lattice
            site.coords = c_coords  # Put back into original cartesian space


class CoherentInterfaceBuilder:
    """
    This class constructs the epitaxially matched interfaces between two crystalline slabs
    """

    def __init__(
        self,
        substrate_structure,
        film_structure,
        film_miller,
        substrate_miller,
        zslgen=None,
        **kwargs,
    ):
        """
        Args:
            substrate_structure (Structure): structure of substrate
            film_structure (Structure): structure of film
        """

        # Bulk structures
        self.substrate_structure = substrate_structure
        self.film_structure = film_structure
        self.zsl_matches = []
        self.film_miller = film_miller
        self.substrate_miller = substrate_miller
        self.zslgen = zslgen or ZSLGenerator()

        self.get_slabs()
        self.get_matches()

    def get_matches(self):

        film_sg = SlabGenerator(
            self.film_structure,
            self.film_miller,
            min_slab_size=1,
            min_vacuum_size=3,
            in_unit_planes=True,
            center_slab=True,
            primitive=True,
            reorient_lattice=False,  # This is necessary to not screw up the lattice
        )

        sub_sg = SlabGenerator(
            self.substrate_structure,
            self.substrate_miller,
            min_slab_size=1,
            min_vacuum_size=3,
            in_unit_planes=True,
            center_slab=True,
            primitive=True,
            reorient_lattice=False,  # This is necessary to not screw up the lattice
        )

        film_slab = film_sg.get_slab(shift=0)
        sub_slab = sub_sg.get_slab(shift=0)

        film_vecs = film_slab.lattice.matrix
        sub_vecs = sub_slab.lattice.matrix

        # Generate all possible interface matches
        self.zsl_matches = list(self.zslgen(film_vecs[:2], sub_vecs[:2], lowest=False))

        for match in self.zsl_matches:
            xform = get_2d_transform(film_vecs, match["film_vecs"])
            strain, rot = polar(xform)
            assert np.allclose(
                strain, np.round(strain)
            ), "Film lattice vectors changed during ZSL match, check your ZSL Generator parameters"

            xform = get_2d_transform(sub_vecs, match["sub_vecs"])
            strain, rot = polar(xform)
            assert np.allclose(
                strain, strain.astype(int)
            ), "Substrate lattice vectors changed during ZSL match, check your ZSL Generator parameters"

    def get_slabs(self):

        film_sg = SlabGenerator(
            self.film_structure,
            self.film_miller,
            min_slab_size=1,
            min_vacuum_size=3,
            in_unit_planes=True,
            center_slab=True,
            primitive=True,
            reorient_lattice=False,  # This is necessary to not screw up the lattice
        )

        sub_sg = SlabGenerator(
            self.substrate_structure,
            self.substrate_miller,
            min_slab_size=1,
            min_vacuum_size=3,
            in_unit_planes=True,
            center_slab=True,
            primitive=True,
            reorient_lattice=False,  # This is necessary to not screw up the lattice
        )

        film_slabs = film_sg.get_slabs()
        sub_slabs = sub_sg.get_slabs()

        film_shits = [s.shift for s in film_slabs]
        film_terminations = [label_termination(s) for s in film_slabs]

        sub_shifts = [s.shift for s in sub_slabs]
        sub_terminations = [label_termination(s) for s in sub_slabs]

        self._terminations = {
            (film_label, sub_label): (film_shift, sub_shift)
            for (film_label, film_shift), (sub_label, sub_shift) in product(
                zip(film_terminations, film_shits), zip(sub_terminations, sub_shifts)
            )
        }
        self.terminations = list(self._terminations.keys())

    def get_interfaces(
        self, termination, gap=2.0, vacuum_over_film=20.0, film_layers=1, sub_layers=1
    ):
        film_sg = SlabGenerator(
            self.film_structure,
            self.film_miller,
            min_slab_size=film_layers,
            min_vacuum_size=3,
            in_unit_planes=True,
            center_slab=True,
            primitive=True,
            reorient_lattice=False,  # This is necessary to not screw up the lattice
        )

        sub_sg = SlabGenerator(
            self.substrate_structure,
            self.substrate_miller,
            min_slab_size=sub_layers,
            min_vacuum_size=3,
            in_unit_planes=True,
            center_slab=True,
            primitive=True,
            reorient_lattice=False,  # This is necessary to not screw up the lattice
        )

        film_shift, sub_shift = self._terminations[termination]

        film_slab = film_sg.get_slab(shift=film_shift)
        sub_slab = sub_sg.get_slab(shift=sub_shift)

        self.interfaces = []

        for match in self.zsl_matches:
            # Build film superlattice
            super_film_transform = np.round(
                from_2d_to_3d(
                    get_2d_transform(
                        film_slab.lattice.matrix[:2], match["film_sl_vecs"]
                    )
                )
            ).astype(int)
            film_sl_slab = film_slab.copy()
            film_sl_slab.make_supercell(super_film_transform)
            assert np.allclose(
                film_sl_slab.lattice.matrix[2], film_slab.lattice.matrix[2]
            ), "2D transformation affected C-axis for Film transformation"
            assert np.allclose(
                film_sl_slab.lattice.matrix[:2], match["film_sl_vecs"]
            ), "Transformation didn't make proper supercell for film"

            # Build substrate superlattice
            super_sub_transform = np.round(
                from_2d_to_3d(
                    get_2d_transform(sub_slab.lattice.matrix[:2], match["sub_sl_vecs"])
                )
            ).astype(int)
            sub_sl_slab = sub_slab.copy()
            sub_sl_slab.make_supercell(super_sub_transform)
            assert np.allclose(
                sub_sl_slab.lattice.matrix[2], sub_slab.lattice.matrix[2]
            ), "2D transformation affected C-axis for Film transformation"
            assert np.allclose(
                sub_sl_slab.lattice.matrix[:2], match["sub_sl_vecs"]
            ), "Transformation didn't make proper supercell for substrate"

            # Add extra info
            match["strain"] = match_strain(
                match["film_sl_vecs"].tolist(), match["sub_sl_vecs"].tolist()
            )
            match["termination"] = termination
            match["film_layers"] = film_layers
            match["sub_layers"] = sub_layers

            yield (
                Interface(
                    sub_slab=sub_sl_slab,
                    film_slab=film_sl_slab,
                    gap=gap,
                    vacuum_over_film=vacuum_over_film,
                    validate_proximity=True,
                    structure_properties=match,
                )
            )


def get_rot_3d_for_2d(film_matrix, sub_matrix):
    """
    Finds a trasnformation matrix that will rotate and strain the film to the subtrate while preserving the c-axis
    """
    film_matrix = np.array(film_matrix)
    film_matrix = film_matrix.tolist()[:2]
    film_matrix.append(np.cross(film_matrix[0], film_matrix[1]))

    # Generate 3D lattice vectors for substrate super lattice
    # Out of plane substrate super lattice has to be same length as
    # Film out of plane vector to ensure no extra deformation in that
    # direction
    sub_matrix = np.array(sub_matrix)
    sub_matrix = sub_matrix.tolist()[:2]
    temp_sub = np.cross(sub_matrix[0], sub_matrix[1])
    temp_sub = temp_sub / fast_norm(temp_sub)
    temp_sub = temp_sub * fast_norm(film_matrix[2])
    sub_matrix.append(temp_sub)

    transform_matrix = np.transpose(np.linalg.solve(film_matrix, sub_matrix))

    rot, strain = polar(transform_matrix)

    return rot


def get_2d_transform(start, end):
    """
    Gets a 2d transformation matrix
    that converts start to end
    """
    return np.dot(end, np.linalg.pinv(start))


def from_2d_to_3d(mat):
    new_mat = np.diag([1.0, 1.0, 1.0])
    new_mat[:2, :2] = mat
    return new_mat


def label_termination(slab):
    frac_coords = slab.frac_coords
    n = len(frac_coords)

    if n == 1:
        # Clustering does not work when there is only one data point.
        form = slab.composition.reduced_formula
        sp_symbol = SpacegroupAnalyzer(slab, symprec=0.1).get_space_group_symbol()
        return f"{form}_{sp_symbol}_{len(slab)}"

    dist_matrix = np.zeros((n, n))
    h = slab.lattice.c
    # Projection of c lattice vector in
    # direction of surface normal.
    for i, j in combinations(list(range(n)), 2):
        if i != j:
            cdist = frac_coords[i][2] - frac_coords[j][2]
            cdist = abs(cdist - round(cdist)) * h
            dist_matrix[i, j] = cdist
            dist_matrix[j, i] = cdist

    condensed_m = squareform(dist_matrix)
    z = linkage(condensed_m)
    clusters = fcluster(z, 0.25, criterion="distance")

    clustered_sites = {c: [] for c in clusters}
    for i, c in enumerate(clusters):
        clustered_sites[c].append(slab[i])

    plane_heights = {
        np.average(np.mod([s.frac_coords[2] for s in sites], 1)): c
        for c, sites in clustered_sites.items()
    }
    top_plane_cluster = sorted(plane_heights.items(), key=lambda x: x[0])[-1][1]
    top_plane_sites = clustered_sites[top_plane_cluster]
    top_plane = Structure.from_sites(top_plane_sites)

    sp_symbol = SpacegroupAnalyzer(top_plane, symprec=0.1).get_space_group_symbol()
    form = top_plane.composition.reduced_formula
    return f"{form}_{sp_symbol}_{len(top_plane)}"


def match_strain(film_sl_vecs, sub_sl_vecs):

    # Generate 3D lattice vectors
    film_sl_vecs.append(np.cross(film_sl_vecs[0], film_sl_vecs[1]))

    # Generate 3D lattice vectors for substrate super lattice
    # Out of plane substrate super lattice has to be same length as
    # Film out of plane vector to ensure no extra deformation in that
    # direction
    temp_sub = np.cross(sub_sl_vecs[0], sub_sl_vecs[1])
    temp_sub = temp_sub * fast_norm(film_sl_vecs[2]) / fast_norm(temp_sub)
    sub_sl_vecs.append(temp_sub)

    transform_matrix = np.transpose(np.linalg.solve(film_sl_vecs, sub_sl_vecs))

    dfm = Deformation(transform_matrix)

    return dfm.green_lagrange_strain.von_mises_strain
