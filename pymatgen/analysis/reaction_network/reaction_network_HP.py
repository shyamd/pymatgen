from abc import ABCMeta, abstractproperty, abstractmethod, abstractclassmethod
from abc import ABC, abstractmethod
from time import time
from gunicorn.util import load_class
from monty.json import MSONable
import logging
import copy
import itertools
import heapq
import numpy as np
from monty.json import MSONable, MontyDecoder
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
import networkx as nx
from networkx.algorithms import bipartite
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.core.composition import CompositionError
from typing import List, Dict, Tuple, Generator

MappingDict = Dict[str, Dict[int, Dict[int, List[MoleculeEntry]]]]
Mapping_Energy_Dict = Dict[str, int]
Mapping_ReactionType_Dict = Dict[str, str]
Mapping_Record_Dict = Dict[str, List[str]]



class Reaction(MSONable, metaclass=ABCMeta):
    """
       Abstract class for subsequent types of reaction class

       Args:
           reactants ([MoleculeEntry]): A list of MoleculeEntry objects of len 1.
           products ([MoleculeEntry]): A list of MoleculeEntry objects of max len 2
       """

    def __init__(self, reactants: List[MoleculeEntry], products: List[MoleculeEntry]):
        self.reactants = reactants
        self.products = products
        self.entry_ids = {e.entry_id for e in self.reactants}

    def __in__(self, entry: MoleculeEntry):
        return entry.entry_id in self.entry_ids

    def __len__(self):
        return len(self.reactants)

    @classmethod
    @abstractmethod
    def generate(cls, entries: MappingDict):
        pass

    @abstractmethod
    def graph_representation(self) -> nx.DiGraph:
        pass

    @abstractmethod
    def reaction_type(self) -> Mapping_ReactionType_Dict:
        pass

    @abstractmethod
    def energy(self) -> Mapping_Energy_Dict:
        pass

    @abstractmethod
    def free_energy(self) -> Mapping_Energy_Dict:
        pass

    @abstractmethod
    def rate(self):
        pass

#     @abstractmethod
#     def num_entries(self):
#         """
# 		number of molecule entries that are intertacting
# 		num reactants + num products not including electron
# 		"""
#
#     @abstractmethod
#     def virtual_edges(self, from_entry):
#         """
# 		Returns the virtual networkx edge that goes from the molecule node to the PR reaction node
# 		Virtual edge - (from_node,to_node)
# 		"""
#         pass
#
#     @abstractmethod
#     def update_weights(self, preq_costs: Dict[str, float]) -> Dict[Tuple(str, str), Dict[str, float]]:
#         """
#
# 		Arguments:
# 			prereq_costs: dictionary mapping entry_id to the new/delta cost
#
# 		Returns:
# 			Dictionary to update weights of edges in networkx graph
# 			Dictionary key is a tuple for the two nodes that define the edge
# 			Dictionary value is a dictionary with key of the weight function(s)
# 				and value of the new weight in new/delta cost
# # 		"""


def graph_rep_1_2(reaction: Reaction) -> nx.DiGraph:

    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 1 reactant -> 2
    products

    Args:
       reactant (any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction):
       :param reaction:
    """

    if len(reaction.reactants) != 1 or len(reaction.products) != 2:
        raise ValueError("Must provide reaction with 1 reactant and 2 products for graph_rep_1_2")

    reactant_0 = reaction.reactants[0]
    product_0 = reaction.products[0]
    product_1 = reaction.products[1]
    graph = nx.DiGraph()

    if product_0.parameters["ind"] <= product_1.parameters["ind"]:
        two_mol_name = str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
        two_mol_name_entry_ids = str(product_0.entry_id) + "+" + str(product_1.entry_id)
    else:
        two_mol_name = str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
        two_mol_name_entry_ids = str(product_1.entry_id) + "+" + str(product_0.entry_id)

    two_mol_name0 = str(product_0.parameters["ind"]) + "+PR_" + str(product_1.parameters["ind"])
    two_mol_name1 = str(product_1.parameters["ind"]) + "+PR_" + str(product_0.parameters["ind"])
    node_name_A = str(reactant_0.parameters["ind"]) + "," + two_mol_name
    node_name_B0 = two_mol_name0 + "," + str(reactant_0.parameters["ind"])
    node_name_B1 = two_mol_name1 + "," + str(reactant_0.parameters["ind"])

    two_mol_entry_ids0 = str(product_0.entry_id) + "+PR_" + str(product_1.entry_id)
    two_mol_entry_ids1 = str(product_1.entry_id) + "+PR_" + str(product_0.entry_id)
    entry_ids_name_A = str(reactant_0.entry_id) + "," + two_mol_name_entry_ids
    entry_ids_name_B0 = two_mol_entry_ids0 + "," + str(reactant_0.entry_id)
    entry_ids_name_B1 = two_mol_entry_ids1 + "," + str(reactant_0.entry_id)

    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]

    graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A)

    graph.add_edge(reactant_0.parameters["ind"],
                   node_name_A,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A,
                   product_1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B0)
    graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B1)

    graph.add_edge(node_name_B0,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B1,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_edge(product_0.parameters["ind"],
                   node_name_B0,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0
                   )
    graph.add_edge(product_1.parameters["ind"],
                   node_name_B1,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0)
    return graph


def graph_rep_1_1(reaction: Reaction) -> nx.DiGraph:

    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 1 reactant -> 1
    product

    Args:
       reactant (any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction):
       :param reaction:
    """

    if len(reaction.reactants) != 1 or len(reaction.products) != 1:
        raise ValueError("Must provide reaction with 1 reactant and product for graph_rep_1_1")

    reactant_0 = reaction.reactants[0]
    product_0 = reaction.products[0]
    graph = nx.DiGraph()
    node_name_A = str(reactant_0.parameters["ind"]) + "," + str(product_0.parameters["ind"])
    node_name_B = str(product_0.parameters["ind"]) + "," + str(reactant_0.parameters["ind"])
    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]
    entry_ids_A = str(reactant_0.entry_id) + "," + str(product_0.entry_id)
    entry_ids_B = str(product_0.entry_id) + "," + str(reactant_0.entry_id)

    graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_A)
    graph.add_edge(reactant_0.parameters["ind"],
                   node_name_A,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   weight=1.0)
    graph.add_edge(node_name_A,
                   product_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0)
    graph.add_node(node_name_B, rxn_type=rxn_type_B, bipartite=1,energy_B=energy_B,free_energy=free_energy_B,
                   entry_ids=entry_ids_B)
    graph.add_edge(product_0.parameters["ind"],
                   node_name_B,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0)
    graph.add_edge(node_name_B,
                   reactant_0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0)

    return graph


class RedoxReaction(Reaction):

    """
    A class to define redox reactions as follows:
    One electron oxidation / reduction without change to bonding
        A^n ±e- <-> A^n±1
        Two entries with:
        identical composition
        identical number of edges
        a charge difference of 1
        isomorphic molecule graphs

    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of single molecular entry
    """

    def __init__(self, reactant: MoleculeEntry, product: MoleculeEntry):
        self.reactant = reactant
        self.product = product
        self.electron_free_energy = None
        super().__init__([self.reactant], [self.product])

    def graph_representation(self) -> nx.DiGraph:
        return graph_rep_1_1(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:
        reactions = []
        for formula in entries:
            for Nbonds in entries[formula]:
                charges = list(entries[formula][Nbonds].keys())
                if len(charges) > 1:
                    for ii in range(len(charges) - 1):
                        charge0 = charges[ii]
                        charge1 = charges[ii + 1]
                        if charge1 - charge0 == 1:
                            for entry0 in entries[formula][Nbonds][charge0]:
                                for entry1 in entries[formula][Nbonds][charge1]:
                                    if entry0.mol_graph.isomorphic_to(entry1.mol_graph):
                                        r = cls(entry0, entry1)
                                        reactions.append(r)

        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        if self.product.charge < self.reactant.charge:
            rxn_type_A = "One electron reduction"
            rxn_type_B = "One electron oxidation"
        else:
            rxn_type_A = "One electron oxidation"
            rxn_type_B = "One electron reduction"

        reaction_type = {"class": "RedoxReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:

        if self.product.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product.free_energy

            if self.reaction_type()["rxn_type_A"] == "One electron reduction":
                free_energy_A += -self.electron_free_energy
                free_energy_B += self.electron_free_energy
            else:
                free_energy_A += self.electron_free_energy
                free_energy_B += -self.electron_free_energy
        else:
            free_energy_A = None
            free_energy_B = None
        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:

        if self.product.energy is not None and self.reactant.energy is not None:
            energy_A = self.product.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product.energy
        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class IntramolSingleBondChangeReaction(Reaction):

    """
    A class to define intramolecular single bond change as follows:
        Intramolecular formation / breakage of one bond
        A^n <-> B^n
        Two entries with:
            identical composition
            number of edges differ by 1
            identical charge
            removing one of the edges in the graph with more edges yields a graph isomorphic to the other entry

    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of single molecular entry
    """

    def __init__(self, reactant: MoleculeEntry, product: MoleculeEntry):
        self.reactant = reactant
        self.product = product
        super().__init__([self.reactant], [self.product])

    def graph_representation(self) -> nx.DiGraph:
        return graph_rep_1_1(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:

        reactions = []
        for formula in entries:
            Nbonds_list = list(entries[formula].keys())
            if len(Nbonds_list) > 1:
                for ii in range(len(Nbonds_list) - 1):
                    Nbonds0 = Nbonds_list[ii]
                    Nbonds1 = Nbonds_list[ii + 1]
                    if Nbonds1 - Nbonds0 == 1:
                        for charge in entries[formula][Nbonds0]:
                            if charge in entries[formula][Nbonds1]:
                                for entry1 in entries[formula][Nbonds1][charge]:
                                    for edge in entry1.edges:
                                        mg = copy.deepcopy(entry1.mol_graph)
                                        mg.break_edge(edge[0], edge[1], allow_reverse=True)
                                        if nx.is_weakly_connected(mg.graph):
                                            for entry0 in entries[formula][Nbonds0][charge]:
                                                if entry0.mol_graph.isomorphic_to(mg):
                                                    r = cls(entry0, entry1)
                                                    reactions.append(r)
                                                    break

        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:

        if self.product.charge < self.reactant.charge:
            rxn_type_A = "Intramolecular single bond breakage"
            rxn_type_B = "Intramolecular single bond formation"
        else:
            rxn_type_A = "Intramolecular single bond formation"
            rxn_type_B = "Intramolecular single bond breakage"

        reaction_type = {"class": "IntramolSingleBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:

        if self.product.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product.free_energy
        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:

        if self.product.energy is not None and self.reactant.energy is not None:
            energy_A = self.product.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class IntermolecularReaction(Reaction):

    """
    A class to define intermolecular bond change as follows:
        Intermolecular formation / breakage of one bond
        A <-> B + C aka B + C <-> A
        Three entries with:
            comp(A) = comp(B) + comp(C)
            charge(A) = charge(B) + charge(C)
            removing one of the edges in A yields two disconnected subgraphs that are isomorphic to B and C

    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of two molecular entries
    """

    def __init__(self, reactant: MoleculeEntry, product: List[MoleculeEntry]):
        self.reactant = reactant
        self.product_0 = product[0]
        self.product_1 = product[1]
        super().__init__([self.reactant], [self.product_0, self.product_1])

    def graph_representation(self) -> nx.DiGraph:  # temp here, use graph_rep_1_2 instead
        return graph_rep_1_2(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:

        reactions = []
        for formula in entries:
            for Nbonds in entries[formula]:
                if Nbonds > 0:
                    for charge in entries[formula][Nbonds]:
                        for entry in entries[formula][Nbonds][charge]:
                            for edge in entry.edges:
                                bond = [(edge[0], edge[1])]
                                try:
                                    frags = entry.mol_graph.split_molecule_subgraphs(bond, allow_reverse=True)
                                    formula0 = frags[0].molecule.composition.alphabetical_formula
                                    Nbonds0 = len(frags[0].graph.edges())
                                    formula1 = frags[1].molecule.composition.alphabetical_formula
                                    Nbonds1 = len(frags[1].graph.edges())
                                    if formula0 in entries and formula1 in entries:
                                        if Nbonds0 in entries[formula0] and Nbonds1 in entries[formula1]:
                                            for charge0 in entries[formula0][Nbonds0]:
                                                for entry0 in entries[formula0][Nbonds0][charge0]:
                                                    if frags[0].isomorphic_to(entry0.mol_graph):
                                                        charge1 = charge - charge0
                                                        if charge1 in entries[formula1][Nbonds1]:
                                                            for entry1 in entries[formula1][Nbonds1][charge1]:
                                                                if frags[1].isomorphic_to(entry1.mol_graph):
                                                                    #r1 = ReactionEntry([entry], [entry0, entry1])
                                                                    r = cls(entry, [entry0, entry1])
                                                                    reactions.append(r)
                                                                    break
                                                        break
                                except MolGraphSplitError:
                                    pass

        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
        rxn_type_B = "Molecular formation from one new bond A+B -> C"

        reaction_type = {"class": "IntermolecularReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:

        if self.product_1.free_energy is not None and self.product_0.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product_0.free_energy + self.product_1.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product_0.free_energy - self.product_1.free_energy

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:

        if self.product_1.energy is not None and self.product_0.energy is not None and self.reactant.energy is not None:
            energy_A = self.product_0.energy + self.product_1.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product_0.energy - self.product_1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class CoordinationBondChangeReaction(Reaction):

    """
    A class to define coordination bond change as follows:
        Simultaneous formation / breakage of multiple coordination bonds
        A + M <-> AM aka AM <-> A + M
        Three entries with:
            M = Li or Mg
            comp(AM) = comp(A) + comp(M)
            charge(AM) = charge(A) + charge(M)
            removing two M-containing edges in AM yields two disconnected subgraphs that are isomorphic to B and C

    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of two molecular entries
    """

    def __init__(self, reactant: MoleculeEntry, product: List[MoleculeEntry]):
        self.reactant = reactant
        self.product_0 = product[0]
        self.product_1 = product[1]
        super().__init__([self.reactant],[self.product_0, self.product_1])

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:
        reactions = []
        M_entries = {}
        for formula in entries:
            if formula == "Li1" or formula == "Mg1":
                if formula not in M_entries:
                    M_entries[formula] = {}
                for charge in entries[formula][0]:
                    assert (len(entries[formula][0][charge]) == 1)
                    M_entries[formula][charge] = entries[formula][0][charge][0]
        if M_entries != {}:
            for formula in entries:
                if "Li" in formula or "Mg" in formula:
                    for Nbonds in entries[formula]:
                        if Nbonds > 2:
                            for charge in entries[formula][Nbonds]:
                                for entry in entries[formula][Nbonds][charge]:
                                    nosplit_M_bonds = []
                                    for edge in entry.edges:
                                        if str(entry.molecule.sites[edge[0]].species) in M_entries or str(
                                                entry.molecule.sites[edge[1]].species) in M_entries:
                                            M_bond = (edge[0], edge[1])
                                            try:
                                                frags = entry.mol_graph.split_molecule_subgraphs([M_bond],
                                                                                                 allow_reverse=True)
                                            except MolGraphSplitError:
                                                nosplit_M_bonds.append(M_bond)
                                    bond_pairs = itertools.combinations(nosplit_M_bonds, 2)
                                    for bond_pair in bond_pairs:
                                        try:
                                            frags = entry.mol_graph.split_molecule_subgraphs(bond_pair,
                                                                                             allow_reverse=True)
                                            M_ind = None
                                            M_formula = None
                                            for ii, frag in enumerate(frags):
                                                frag_formula = frag.molecule.composition.alphabetical_formula
                                                if frag_formula in M_entries:
                                                    M_ind = ii
                                                    M_formula = frag_formula
                                                    break
                                            if M_ind != None:
                                                for ii, frag in enumerate(frags):
                                                    if ii != M_ind:
                                                        # nonM_graph = frag.graph
                                                        nonM_formula = frag.molecule.composition.alphabetical_formula
                                                        nonM_Nbonds = len(frag.graph.edges())
                                                        if nonM_formula in entries:
                                                            if nonM_Nbonds in entries[nonM_formula]:
                                                                for nonM_charge in entries[nonM_formula][
                                                                    nonM_Nbonds]:
                                                                    M_charge = entry.charge - nonM_charge
                                                                    if M_charge in M_entries[M_formula]:
                                                                        for nonM_entry in \
                                                                                entries[nonM_formula][nonM_Nbonds][
                                                                                    nonM_charge]:
                                                                            if frag.isomorphic_to(nonM_entry.mol_graph):
                                                                                r = cls(entry,[nonM_entry,
                                                                                                    M_entries[
                                                                                                        M_formula][
                                                                                                        M_charge]])
                                                                                reactions.append(r)
                                                                                break
                                        except MolGraphSplitError:
                                            pass
        return reactions

    def graph_representation(self) -> nx.DiGraph:
        return graph_rep_1_2(self)

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        rxn_type_A = "Coordination bond breaking AM -> A+M"
        rxn_type_B = "Coordination bond forming A+M -> AM"

        reaction_type = {"class": "CoordinationBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:

        if self.product_1.free_energy is not None and self.product_0.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product_0.free_energy + self.product_1.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product_0.free_energy - self.product_1.free_energy

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:

        if self.product_1.energy is not None and self.product_0.energy is not None and self.reactant.energy is not None:
            energy_A = self.product_0.energy + self.product_1.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product_0.energy - self.product_1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class ReactionPath(MSONable):

    def __init__(self, path):
        self.path = path
        self.byproducts = []
        self.unsolved_prereqs = []
        self.solved_prereqs = []
        self.all_prereqs = []
        self.cost = 0.0
        self.overall_free_energy_change = 0.0
        self.hardest_step = None
        self.description = ""
        self.pure_cost = 0.0
        self.full_path = None
        self.hardest_step_deltaG = None
        self.path_dict = {"byproducts": self.byproducts, "unsolved_prereqs": self.unsolved_prereqs,
                          "solved_prereqs": self.solved_prereqs, "all_prereqs": self.all_prereqs, "cost": self.cost,
                          "path": self.path, "overall_free_energy_change": self.overall_free_energy_change,
                          "hardest_step":self.hardest_step,"description": self.description, "pure_cost": self.pure_cost,
                          "hardest_step_deltaG":self.hardest_step_deltaG, "full_path":self.full_path}



    @classmethod
    def characterize_path(cls, path:List[str], weight:str, min_cost:Dict[str,float], graph:nx.DiGraph, PR_paths={}): #-> ReactionPath
        if path is None:
            class_instance = cls(None)
        else:
            class_instance = cls(path)
            for ii, step in enumerate(path):
                if ii != len(path) - 1:
                    class_instance.cost += graph[step][path[ii + 1]][weight]
                    if ii % 2 == 1:
                        rxn = step.split(",")
                        if "+PR_" in rxn[0]:
                            PR = int(rxn[0].split("+PR_")[1])
                            class_instance.all_prereqs.append(PR)
                        if "+" in rxn[1]:
                            desired_prod_satisfied = False
                            prods = rxn[1].split("+")
                            for prod in prods:
                                if int(prod) != path[ii + 1]:
                                    class_instance.byproducts.append(int(prod))
                                elif desired_prod_satisfied:
                                    class_instance.byproducts.append(int(prod))
                                else:
                                    desired_prod_satisfied = True
            for PR in class_instance.all_prereqs:
                if PR in class_instance.byproducts:
                    # Note that we're ignoring the order in which BPs are made vs they come up as PRs...
                    class_instance.all_prereqs.remove(PR)
                    class_instance.byproducts.remove(PR)

                    if PR in min_cost:
                        class_instance.cost -= min_cost[PR]
                    else:
                        print("Missing PR cost to remove:", PR)
            for PR in class_instance.all_prereqs:
                if PR in PR_paths:
                    class_instance.solved_prereqs.append(PR)
                else:
                    class_instance.unsolved_prereqs.append(PR)

            class_instance.path_dict = {"byproducts": class_instance.byproducts,
                                        "unsolved_prereqs": class_instance.unsolved_prereqs,
                                        "solved_prereqs": class_instance.solved_prereqs,
                                        "all_prereqs": class_instance.all_prereqs, "cost": class_instance.cost,
                                        "path": class_instance.path,
                                        "overall_free_energy_change": class_instance.overall_free_energy_change,
                                        "hardest_step": class_instance.hardest_step,
                                        "description": class_instance.description,
                                        "pure_cost": class_instance.pure_cost,
                                        "hardest_step_deltaG": class_instance.hardest_step_deltaG,
                                        "full_path": class_instance.full_path}
        return class_instance


    @classmethod
    def characterize_path_final(cls, path: List[str], weight:str, min_cost:Dict[str,float], graph:nx.DiGraph, PR_paths): #Mapping_PR_Dict): -> ReactionPath
        class_instance = cls.characterize_path(path, weight, min_cost, graph, PR_paths)
        if path is None:
            class_instance = cls(None)
        else:
            assert (len(class_instance.solved_prereqs)==len(class_instance.all_prereqs))
            assert (len(class_instance.unsolved_prereqs)==0)

            PRs_to_join = copy.deepcopy(class_instance.all_prereqs)
            full_path = copy.deepcopy(path)
            while len(PRs_to_join) > 0:
                new_PRs = []
                for PR in PRs_to_join:
                    PR_path = None
                    PR_min_cost = 1000000000000000.0
                    for start in PR_paths[PR]:
                        if PR_paths[PR][start].path != None:
                            if PR_paths[PR][start].cost < PR_min_cost:
                                PR_min_cost = PR_paths[PR][start].cost
                                PR_path = PR_paths[PR][start]
                    assert (len(PR_path.solved_prereqs) == len(PR_path.all_prereqs))
                    for new_PR in PR_path.all_prereqs:
                        new_PRs.append(new_PR)
                        class_instance.all_prereqs.append(new_PR)
                    for new_BP in PR_path.byproducts:
                        class_instance.byproducts.append(new_BP)
                    full_path = PR_path.path + full_path
                PRs_to_join = copy.deepcopy(new_PRs)

            for PR in class_instance.all_prereqs:
                if PR in class_instance.byproducts:
                    print("WARNING: Matching prereq and byproduct found!", PR)

            for ii, step in enumerate(full_path):
                if graph.nodes[step]["bipartite"] == 1:
                    if weight == "softplus":
                        class_instance.pure_cost += ReactionNetwork.softplus(graph.nodes[step]["free_energy"])
                    elif weight == "exponent":
                        class_instance.pure_cost += ReactionNetwork.exponent(graph.nodes[step]["free_energy"])

                    class_instance.overall_free_energy_change += graph.nodes[step]["free_energy"]

                    if class_instance.description == "":
                        class_instance.description += graph.nodes[step]["rxn_type"]
                    else:
                        class_instance.description += ", " + graph.nodes[step]["rxn_type"]

                    if class_instance.hardest_step is None:
                        class_instance.hardest_step = step
                    elif graph.nodes[step]["free_energy"] > graph.nodes[class_instance.hardest_step]["free_energy"]:
                        class_instance.hardest_step = step

            class_instance.full_path = full_path

            if class_instance.hardest_step is None:
                class_instance.hardest_step_deltaG = None
            else:
                class_instance.hardest_step_deltaG = graph.nodes[class_instance.hardest_step]["free_energy"]


        class_instance.path_dict = {"byproducts": class_instance.byproducts, "unsolved_prereqs": class_instance.unsolved_prereqs,
                          "solved_prereqs": class_instance.solved_prereqs, "all_prereqs": class_instance.all_prereqs, "cost": class_instance.cost,
                          "path": class_instance.path, "overall_free_energy_change": class_instance.overall_free_energy_change,
                          "hardest_step":class_instance.hardest_step,"description": class_instance.description, "pure_cost": class_instance.pure_cost,
                          "hardest_step_deltaG": class_instance.hardest_step_deltaG, "full_path":class_instance.full_path}

        return class_instance


Mapping_PR_Dict = Dict[int, Dict[int, ReactionPath]]

class ReactionNetwork(MSONable):
    """
       Class to build a reaction network from entries

       Args:
           input_entries ([MoleculeEntry]): A list of MoleculeEntry objects.
           electron_free_energy (float): The Gibbs free energy of an electron.
               Defaults to -2.15 eV, the value at which the LiEC SEI forms.
       """


    def __init__(self, input_entries: List[MoleculeEntry], electron_free_energy=-2.15):
        print("CLEANED VERSION")
        self.graph = nx.DiGraph()
        self.reactions = []
        self.input_entries = input_entries
        self.entry_ids = {e.entry_id for e in self.input_entries}
        self.electron_free_energy = electron_free_energy
        self.entries = {}
        self.entries_list = []
        self.num_starts = 0
        self.weight = None
        self.PR_record = None
        self.Reactant_record = None
        self.min_cost = {}

        print(len(self.input_entries), "input entries")

        connected_entries = []
        for entry in self.input_entries:
            if len(entry.molecule) > 1:
                if nx.is_weakly_connected(entry.graph):
                    connected_entries.append(entry)
            else:
                connected_entries.append(entry)
        print(len(connected_entries), "connected entries")

        get_formula = lambda x: x.formula
        get_Nbonds = lambda x: x.Nbonds
        get_charge = lambda x: x.charge
        get_free_energy = lambda x: x.free_energy

        sorted_entries_0 = sorted(connected_entries, key=get_formula)
        for k1, g1 in itertools.groupby(sorted_entries_0, get_formula):
            sorted_entries_1 = sorted(list(g1), key=get_Nbonds)
            self.entries[k1] = {}

            for k2, g2 in itertools.groupby(sorted_entries_1, get_Nbonds):
                sorted_entries_2 = sorted(list(g2), key=get_charge)
                self.entries[k1][k2] = {}
                for k3, g3 in itertools.groupby(sorted_entries_2, get_charge):
                    sorted_entries_3 = list(g3)
                    sorted_entries_3.sort(key=get_free_energy)
                    if len(sorted_entries_3) > 1:
                        unique = []
                        for entry in sorted_entries_3:
                            isomorphic_found = False
                            for ii, Uentry in enumerate(unique):
                                if entry.mol_graph.isomorphic_to(Uentry.mol_graph):
                                    isomorphic_found = True
                                    # print("Isomorphic entries with equal charges found!")
                                    if entry.free_energy != None and Uentry.free_energy != None:
                                        if entry.free_energy < Uentry.free_energy:
                                            unique[ii] = entry
                                            # if entry.energy > Uentry.energy:
                                            #     print("WARNING: Free energy lower but electronic energy higher!")
                                    elif entry.free_energy != None:
                                        unique[ii] = entry
                                    elif entry.energy < Uentry.energy:
                                        unique[ii] = entry
                                    break
                            if not isomorphic_found:
                                unique.append(entry)
                        self.entries[k1][k2][k3] = unique
                    else:
                        self.entries[k1][k2][k3] = sorted_entries_3
                    for entry in self.entries[k1][k2][k3]:
                        self.entries_list.append(entry)

        print(len(self.entries_list), "unique entries")
        for ii, entry in enumerate(self.entries_list):
            if "ind" in entry.parameters.keys():
                pass
            else:
                entry.parameters["ind"] = ii

        self.entries_list = sorted(self.entries_list, key=lambda x: x.parameters["ind"])

    @staticmethod
    def softplus(free_energy: int) -> int:
        return np.log(1 + (273.0 / 500.0) * np.exp(free_energy))

    @staticmethod
    def exponent(free_energy: int) -> int:
        return np.exp(free_energy)

    def build(self, reaction_types={"RedoxReaction", "IntramolSingleBondChangeReaction", "IntermolecularReaction", "CoordinationBondChangeReaction"}) -> nx.DiGraph:

        self.graph.add_nodes_from(range(len(self.entries_list)), bipartite=0)
        reaction_types = [load_class(str(self.__module__)+"."+s) for s in reaction_types]
        self.reactions = [r.generate(self.entries) for r in reaction_types]
        self.reactions = [i for i in self.reactions if i]
        self.reactions = list(itertools.chain.from_iterable(self.reactions))

        for r in self.reactions:
            if r.reaction_type()["class"] == "RedoxReaction":
                r.electron_free_energy = self.electron_free_energy
            self.add_reaction(r.graph_representation())

        self.PR_record = self.build_PR_record()
        self.Reactant_record = self.build_reactant_record()

        return self.graph

    def add_reaction(self, graph_representation: nx.DiGraph):
        self.graph.add_nodes_from(graph_representation.nodes(data=True))
        self.graph.add_edges_from(graph_representation.edges(data=True))

    def build_PR_record(self) -> Mapping_Record_Dict:
        PR_record = {}
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0:
                PR_record[node] = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 1:
                if "+PR_" in node.split(",")[0]:
                    PR = int(node.split(",")[0].split("+PR_")[1])
                    PR_record[PR].append(node)
        return PR_record

    def build_reactant_record(self) -> Mapping_Record_Dict:
        Reactant_record = {}
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0:
                Reactant_record[node] = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 1:
                non_PR_reactant = node.split(",")[0].split("+PR_")[0]
                Reactant_record[int(non_PR_reactant)].append(node)
        return Reactant_record

    def solve_prerequisites(self, starts: List[int], target: int, weight: str, max_iter=20) -> Mapping_Record_Dict:
        PRs = {}
        old_solved_PRs = []
        new_solved_PRs = ["placeholder"]
        old_attrs = {}
        new_attrs = {}
        self.weight = weight

        if len(self.graph.nodes) == 0:
            self.build()
        orig_graph = copy.deepcopy(self.graph)

        for start in starts:
            PRs[start] = {}

        for PR in PRs:
            for start in starts:
                if start == PR:
                    PRs[PR][start] = ReactionPath.characterize_path([start], weight, self.min_cost, self.graph)
                else:
                    PRs[PR][start] = ReactionPath(None)

            old_solved_PRs.append(PR)
            self.min_cost[PR] = PRs[PR][PR].cost
        for node in self.graph.nodes():
            if self.graph.nodes[node]["bipartite"] == 0 and node != target:
                if node not in PRs:
                    PRs[node] = {}

        ii = 0

        while (len(new_solved_PRs) > 0 or old_attrs != new_attrs) and ii < max_iter:
            min_cost = {}
            cost_from_start = {}
            for PR in PRs:
                cost_from_start[PR] = {}
                min_cost[PR] = 10000000000000000.0
                for start in PRs[PR]:
                    if PRs[PR][start].path == None:#"no_path":
                        cost_from_start[PR][start] = "no_path"
                    else:
                        cost_from_start[PR][start] = PRs[PR][start].cost
                        if PRs[PR][start].cost < min_cost[PR]:
                            min_cost[PR] = PRs[PR][start].cost
                for start in starts:
                    if start not in cost_from_start[PR]:
                        cost_from_start[PR][start] = "unsolved"

            for node in self.graph.nodes():
                if self.graph.nodes[node]["bipartite"] == 0 and node not in old_solved_PRs and node != target:
                    for start in starts:
                        if start not in PRs[node]:
                            path_exists = True
                            try:
                                length, dij_path = nx.algorithms.simple_paths._bidirectional_dijkstra(
                                    self.graph,
                                    source=hash(start),
                                    target=hash(node),
                                    ignore_nodes=self.find_or_remove_bad_nodes([target, node]),
                                    weight=self.weight)
                            except nx.exception.NetworkXNoPath:
                                PRs[node][start] = ReactionPath(None)
                                path_exists = False
                                cost_from_start[node][start] = "no_path"
                            if path_exists:
                                if len(dij_path) > 1 and len(dij_path) % 2 == 1:
                                    path_class = ReactionPath.characterize_path(dij_path, weight, self.min_cost, self.graph, old_solved_PRs)
                                    cost_from_start[node][start] = path_class.cost
                                    assert (path_class.cost == path_class.cost)
                                    if len(path_class.unsolved_prereqs) == 0:
                                        PRs[node][start] = path_class
                                    if path_class.cost < min_cost[node]:
                                        min_cost[node] = path_class.cost
                                else:
                                    print("Does this ever happen?")
            solved_PRs = copy.deepcopy(old_solved_PRs)
            new_solved_PRs = []
            for PR in PRs:
                if PR not in solved_PRs:
                    if len(PRs[PR].keys()) == self.num_starts:
                        solved_PRs.append(PR)
                        new_solved_PRs.append(PR)
                    else:
                        best_start_so_far = [None, 10000000000000000.0]
                        for start in PRs[PR]:
                            if PRs[PR][start] is not None: #ALWAYS TRUE
                                # if PRs[PR][start] == "unsolved": #### DOES THIS EVER HAPPEN ---- NEED TO FIX
                                #     print("ERROR: unsolved should never be encountered here!")
                                if PRs[PR][start].cost < best_start_so_far[1]:
                                    best_start_so_far[0] = start
                                    best_start_so_far[1] = PRs[PR][start].cost
                        if best_start_so_far[0] is not None:
                            num_beaten = 0
                            for start in cost_from_start[PR]:
                                if start != best_start_so_far[0]:
                                    if cost_from_start[PR][start] == "no_path":
                                        num_beaten += 1
                                    elif cost_from_start[PR][start] > best_start_so_far[1]:
                                        num_beaten += 1
                            if num_beaten == self.num_starts - 1:
                                solved_PRs.append(PR)
                                new_solved_PRs.append(PR)

            print(ii, len(old_solved_PRs), len(new_solved_PRs))

            attrs = self.update_edge_weights(min_cost, orig_graph)

            self.min_cost = copy.deepcopy(min_cost)
            old_solved_PRs = copy.deepcopy(solved_PRs)
            old_attrs = copy.deepcopy(new_attrs)
            new_attrs = copy.deepcopy(attrs)

            ii += 1

        self.final_PR_check(PRs)

        return PRs

    def update_edge_weights(self, min_cost:Dict[int, float], orig_graph: nx.DiGraph) -> Dict[Tuple[int, str],Dict[str,float]]:#, solved_PRs: List[int], new_attrs:Dict[Tuple[int, str],Dict[str,float]]):
        attrs = {}
        for PR_ind in min_cost:
            for rxn_node in self.PR_record[PR_ind]:
                non_PR_reactant_node = int(rxn_node.split(",")[0].split("+PR_")[0])
                attrs[(non_PR_reactant_node, rxn_node)] = {
                        self.weight: orig_graph[non_PR_reactant_node][rxn_node][self.weight] + min_cost[PR_ind]}
        nx.set_edge_attributes(self.graph, attrs)

        return attrs

    def final_PR_check(self, PRs:Mapping_PR_Dict):
        for PR in PRs:
            path_found = False
            if PRs[PR] != {}:
                for start in PRs[PR]:
                    if PRs[PR][start].path != None:
                        path_found = True
                        path_dict_class = ReactionPath.characterize_path_final(PRs[PR][start].path, self.weight, self.min_cost, self.graph, PRs)
                        if abs(path_dict_class.cost - path_dict_class.pure_cost) > 0.0001:
                            print("WARNING: cost mismatch for PR", PR, path_dict_class.cost, path_dict_class.pure_cost,
                                  path_dict_class.full_path)
                if not path_found:
                    print("No path found from any start to PR", PR)
            else:
                print("Unsolvable path from any start to PR", PR)

    def find_or_remove_bad_nodes(self, nodes:List[int], remove_nodes=False) -> List[str] or nx.DiGraph:
        bad_nodes = []
        for node in nodes:
            for bad_node in self.PR_record[node]:
                bad_nodes.append(bad_node)
            for bad_nodes2 in self.Reactant_record[node]:
                bad_nodes.append(bad_nodes2)
        if remove_nodes:
            pruned_graph = copy.deepcopy(self.graph)
            pruned_graph.remove_nodes_from(bad_nodes)
            return pruned_graph
        else:
            return bad_nodes

    def valid_shortest_simple_paths(self, start:int, target:int, PRs=[]):# -> Generator[List[str]]:????
        bad_nodes = PRs
        bad_nodes.append(target)
        valid_graph = self.find_or_remove_bad_nodes(bad_nodes, remove_nodes=True)
        return nx.shortest_simple_paths(valid_graph, hash(start), hash(target), weight=self.weight)

    def find_paths(self, starts, target, weight, num_paths=10): #-> ??
        """
        Args:
            starts ([int]): List of starting node IDs (ints).
            target (int): Target node ID.
            weight (str): String identifying what edge weight to use for path finding.
            num_paths (int): Number of paths to find. Defaults to 10.
        """
        self.weight = weight
        paths = []
        c = itertools.count()
        my_heapq = []

        print("Solving prerequisites...")
        self.num_starts = len(starts)
        PR_paths = self.solve_prerequisites(starts, target, weight)

        print("Finding paths...")
        for start in starts:
            ind = 0
            for path in self.valid_shortest_simple_paths(start, target):
                if ind == num_paths:
                    break
                else:
                    ind += 1
                    path_dict_class2 = ReactionPath.characterize_path_final(path, self.weight, self.min_cost, self.graph, PR_paths)
                    heapq.heappush(my_heapq, (path_dict_class2.cost, next(c), path_dict_class2))

        while len(paths) < num_paths and my_heapq:
            # Check if any byproduct could yield a prereq cheaper than from starting molecule(s)?
            (cost_HP, _x, path_dict_HP_class) = heapq.heappop(my_heapq)
            print("!!",len(paths), cost_HP, len(my_heapq), path_dict_HP_class.path_dict)
            paths.append(path_dict_HP_class.path_dict) ### ideally just append the class, but for now dict for easy printing

        print(PR_paths)
        print(paths)

        return PR_paths, paths


