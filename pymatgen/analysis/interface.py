# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Generator, Union
from itertools import product, combinations, chain
import numpy as np

from scipy.linalg import polar
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from pymatgen import Lattice, Structure, Site
from pymatgen.core.surface import Slab, SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.elasticity.strain import Deformation
from pymatgen.analysis.substrate_analyzer import fast_norm, ZSLGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


"""
This module provides classes to store, generate, and manipulate material interfaces.
"""

__author__ = "Shyam Dwaraknath, Eric Sivonxay, and Kyle Bystrom"
__copyright__ = "Copyright 2019, The Materials Project"
__maintainer__ = "Shyam Dwaraknath"
__email__ = "shyamd@lbl.gov"
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
        in_plane_offset: Tuple[float, float] = (0, 0),
        gap: float = 1.6,
        vacuum_over_film: float = 0.0,
        structure_properties: Dict = {},
        center_slab: bool = True,
        **kwargs,
    ):
        """
        Makes an interface structure by merging a substrate and film slabs
        The film a- and b-vectors will be forced to be the substrate slab's
        a- and b-vectors.

        For now, it's suggested to use a factory method that will ensure the
        appropriate interface structure is already met.

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
        self.center_slab = center_slab
        self._kwargs = dict(kwargs)

        # Ensure c-axis is orthogonal to a/b plane
        if isinstance(sub_slab, Slab):
            sub_slab = sub_slab.get_orthogonal_c_slab()
        if isinstance(film_slab, Slab):
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
        if self.center_slab:
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

        self.sort()

    @property
    def in_plane_offset(self) -> Tuple[float, float]:
        """
        The shift between the film and substrate in fractional
        coordinates
        """
        return self._in_plane_offset

    @in_plane_offset.setter
    def in_plane_offset(self, new_shift: Tuple[float, float]) -> None:
        if len(new_shift) != 2:
            raise ValueError("In-plane shifts require two floats for a and b vectors")
        new_shift = np.mod(new_shift, 1)
        delta = new_shift - np.array(self.in_plane_offset)
        self._in_plane_offset = new_shift.tolist()
        self.translate_sites(
            self.film_indices, [delta[0], delta[1], 0], to_unit_cell=True
        )

    @property
    def gap(self) -> float:
        """
        The gap in cartesian units between the film and the substrate
        """
        return self._gap

    @gap.setter
    def gap(self, new_gap: float) -> None:
        if new_gap < 0:
            raise ValueError("Can't reduce interface gap below 0")

        delta = new_gap - self.gap
        self._gap = new_gap

        self.__update_c(self.lattice.c + delta)
        self.translate_sites(
            self.film_indices, [0, 0, delta], frac_coords=False, to_unit_cell=True
        )

    @property
    def vacuum_over_film(self) -> float:
        """
        The vacuum space over the film in cartesian units
        """
        return self._vacuum_over_film

    @vacuum_over_film.setter
    def vacuum_over_film(self, new_vacuum: float) -> None:
        if new_vacuum < 0:
            raise ValueError("The vacuum over the film can not be less then 0")

        delta = new_vacuum - self.vacuum_over_film
        self._vacuum_over_film = new_vacuum

        self.__update_c(self.lattice.c + delta)

    @property
    def substrate_indicies(self) -> List[int]:
        """
        Site indicies for the substrate atoms
        """
        sub_indicies = [
            i
            for i, tag in enumerate(self.site_properties["interface_label"])
            if "substrate" in tag
        ]
        return sub_indicies

    @property
    def substrate_sites(self) -> List[Site]:
        """
        The site objects in the substrate
        """
        sub_sites = [
            site
            for site, tag in zip(self, self.site_properties["interface_label"])
            if "substrate" in tag
        ]
        return sub_sites

    @property
    def substrate_layers(self) -> int:
        """
        Total number of substrate unit slab layers in this interface
        """
        return self.sub_slab.lattice.c / self.sub_slab.oriented_unit_cell.lattice.c

    @property
    def substrate(self) -> Structure:
        """
        A pymatgen Structure for just the substrate
        """
        return Structure.from_sites(self.substrate_sites)

    @property
    def film_indices(self) -> List[int]:
        """
        Site indices of the film sites
        """
        f_indicies = [
            i
            for i, tag in enumerate(self.site_properties["interface_label"])
            if "film" in tag
        ]
        return f_indicies

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

    def copy(self) -> Interface:
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
            structure_properties=self.structure_properties,
            center_slab=self.center_slab,
            **self._kwargs,
        )

    def get_sorted_structure(self, key=None, reverse=False) -> Structure:
        """
        Get a sorted structure for the interface. The parameters have the same
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

    def shifts_based_on_adsorbate_sites(self, tolerance=0.1):
        """
        Computes possible in-plane shifts based on adsorbate finding algorithm

        Args:
            tolerance: tolerance for "uniqueness" for shifts in Cartesian unit
                This is usually Angstroms.
        """
        sub_slab = self.sub_slab.copy()
        film_slab = self.film_slab.copy()

        if isinstance(sub_slab, Slab):
            sub_slab = sub_slab.get_orthogonal_c_slab()
        if isinstance(film_slab, Slab):
            film_slab = film_slab.get_orthogonal_c_slab()

        substrate_surface_sites = np.dot(
            list(
                chain.from_iterable(
                    AdsorbateSiteFinder(sub_slab).find_adsorption_sites().values()
                )
            ),
            sub_slab.lattice.inv_matrix,
        )

        # Film gets forced into substrate lattice anyways, so shifts can be computed in fractional coords
        film_surface_sites = np.dot(
            list(
                chain.from_iterable(
                    AdsorbateSiteFinder(film_slab).find_adsorption_sites().values()
                )
            ),
            film_slab.lattice.inv_matrix,
        )
        pos_shift = np.array(
            [
                np.add(np.multiply(-1, film_shift), sub_shift)
                for film_shift, sub_shift in product(
                    film_surface_sites, substrate_surface_sites
                )
            ]
        )

        def _base_round(x,base=0.05):
            return (base * (np.array(x) / base).round())

        # Round shifts to tolerance
        pos_shift[:, 0] = _base_round(
            pos_shift[:, 0], base=tolerance / sub_slab.lattice.a
        )
        pos_shift[:, 1] = _base_round(
            pos_shift[:, 1], base=tolerance / sub_slab.lattice.b
        )
        # C-axis is not usefull
        pos_shift = pos_shift[:, 0:2]

        return np.unique(pos_shift, axis=0)

    def __update_c(self, new_c: float) -> None:
        """
        Modifies the c-direction of the lattice without changing the site cartesian coordinates
        Be carefull you can mess up the interface by setting a c-length that can't accomodate all the sites
        """
        if new_c <= 0:
            raise ValueError("New c-length must be greater than 0")

        new_latt_matrix = self.lattice.matrix[:2].tolist() + [[0, 0, new_c]]
        new_latice = Lattice(new_latt_matrix)
        self._lattice = new_latice

        for site, c_coords in zip(self, self.cart_coords):
            site._lattice = new_latice  # Update the lattice
            site.coords = c_coords  # Put back into original cartesian space


class CoherentInterfaceBuilder:
    """
    This class constructs the epitaxially matched interfaces between two crystalline slabs
    """

    def __init__(
        self,
        substrate_structure: Structure,
        film_structure: Structure,
        film_miller: Tuple[int, int, int],
        substrate_miller: Tuple[int, int, int],
        zslgen: Optional[ZSLGenerator] = None,
    ):
        """
        Args:
            substrate_structure: structure of substrate
            film_structure: structure of film
            film_miller: miller index of the film layer
            substrate_miller: miller index for the substrate layer
            zslgen: ZSLGenerator if you want custom lattice matching tolerances for coherency
        """

        # Bulk structures
        self.substrate_structure = substrate_structure
        self.film_structure = film_structure
        self.film_miller = film_miller
        self.substrate_miller = substrate_miller
        self.zslgen = zslgen or ZSLGenerator()

        self._find_matches()
        self._find_terminations()

    def _find_matches(self) -> None:
        """
        Finds and stores the ZSL matches
        """
        self.zsl_matches = []

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

    def _find_terminations(self):
        """
        Finds all terminations
        """

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
        self,
        termination: Tuple[str, str],
        gap: float = 2.0,
        vacuum_over_film: float = 20.0,
        film_thickness: Union[float, int] = 1,
        substrate_thickness: Union[float, int] = 1,
        in_layers: bool = True,
    ) -> Generator[Interface]:
        """
        Generates interface structures given the film and substrate structure
        as well as the desired terminations


        Args:
            terminations: termination from self.termination list
            gap: gap between film and substrate
            vacuum_over_film: vacuum over the top of the film
            film_thickness: the film thickness
            substrate_thickness: substrate thickness 
            in_layers: set the thickness in layer units
        """
        film_sg = SlabGenerator(
            self.film_structure,
            self.film_miller,
            min_slab_size=film_thickness,
            min_vacuum_size=3,
            in_unit_planes=in_layers,
            center_slab=True,
            primitive=True,
            reorient_lattice=False,  # This is necessary to not screw up the lattice
        )

        sub_sg = SlabGenerator(
            self.substrate_structure,
            self.substrate_miller,
            min_slab_size=substrate_thickness,
            min_vacuum_size=3,
            in_unit_planes=in_layers,
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
            match["film_thickness"] = film_thickness
            match["substrate_thickness"] = substrate_thickness

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


Vector3D = Tuple[float, float, float]
Matrix3D = Tuple[Vector3D, Vector3D, Vector3D]
Matrix2D = Tuple[Vector3D, Vector3D]


def get_rot_3d_for_2d(film_matrix: Matrix3D, sub_matrix: Matrix3D):
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


def get_2d_transform(start: Matrix2D, end: Matrix2D):
    """
    Gets a 2d transformation matrix
    that converts start to end
    """
    return np.dot(end, np.linalg.pinv(start))


def from_2d_to_3d(mat: Matrix2D):
    new_mat = np.diag([1.0, 1.0, 1.0])
    new_mat[:2, :2] = mat
    return new_mat


def label_termination(slab: Slab):
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


def match_strain(film_sl_vecs: Matrix2D, sub_sl_vecs: Matrix2D):

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
