from abc import ABCMeta, abstractproperty, abstractmethod, abstractclassmethod
from abc import ABC, abstractmethod
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


class Reaction(MSONable, metaclass=ABCMeta):
    """
       Abstract class for subsequent types of reaction class

       Args:
           reactants ([MoleculeEntry]): A list of MoleculeEntry objects of len 1.
           products ([MoleculeEntry]): A list of MoleculeEntry objects of max len 2
       """

    def __init__(self, reactants, products):
        self.reactants = reactants
        self.products = products
        self.entry_ids = {e.entry_id for e in self.reactants}

    def __in__(self, entry):
        return entry.entry_id in self.entry_ids

    def __len__(self):
        return len(self.reactants)

    @classmethod
    @abstractmethod
    def generate(cls, entries) -> list:
        pass

    @abstractmethod
    def graph_representation(self) -> nx.DiGraph:
        pass

    @abstractmethod
    def reaction_type(self):
        pass

    @abstractmethod
    def energy(self):
        pass

    @abstractmethod
    def free_energy(self):
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


def graph_rep_1_2(reaction) -> nx.DiGraph:

    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 1 reactant -> 2
    products

    Args:
       reactant (any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction):
    """

    entry = reaction.reactant
    entry0 = reaction.product0
    entry1 = reaction.product1
    graph = nx.DiGraph()

    if entry0.parameters["ind"] <= entry1.parameters["ind"]:
        two_mol_name = str(entry0.parameters["ind"]) + "+" + str(entry1.parameters["ind"])
        two_mol_name_entry_ids = str(entry0.entry_id) + "+" + str(entry1.entry_id)
    else:
        two_mol_name = str(entry1.parameters["ind"]) + "+" + str(entry0.parameters["ind"])
        two_mol_name_entry_ids = str(entry1.entry_id) + "+" + str(entry0.entry_id)

    two_mol_name0 = str(entry0.parameters["ind"]) + "+PR_" + str(entry1.parameters["ind"])
    two_mol_name1 = str(entry1.parameters["ind"]) + "+PR_" + str(entry0.parameters["ind"])
    node_name_A = str(entry.parameters["ind"]) + "," + two_mol_name
    node_name_B0 = two_mol_name0 + "," + str(entry.parameters["ind"])
    node_name_B1 = two_mol_name1 + "," + str(entry.parameters["ind"])

    two_mol_entry_ids0 = str(entry0.entry_id) + "+PR_" + str(entry1.entry_id)
    two_mol_entry_ids1 = str(entry1.entry_id) + "+PR_" + str(entry0.entry_id)
    entry_ids_name_A = str(entry.entry_id) + "," + two_mol_name_entry_ids
    entry_ids_name_B0 = two_mol_entry_ids0 + "," + str(entry.entry_id)
    entry_ids_name_B1 = two_mol_entry_ids1 + "," + str(entry.entry_id)

    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]

    graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_name_A)

    graph.add_edge(entry.parameters["ind"],
                   node_name_A,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   weight=1.0
                   )

    graph.add_edge(node_name_A,
                   entry0.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_A,
                   entry1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B0)
    graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                   entry_ids=entry_ids_name_B1)

    graph.add_edge(node_name_B0,
                   entry.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )
    graph.add_edge(node_name_B1,
                   entry.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0
                   )

    graph.add_edge(entry0.parameters["ind"],
                   node_name_B0,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0
                   )
    graph.add_edge(entry1.parameters["ind"],
                   node_name_B1,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0)
    return graph


def graph_rep_1_1(reaction) -> nx.DiGraph:

    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 1 reactant -> 1
    product

    Args:
       reactant (any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction):
    """

    entry0 = reaction.reactant
    entry1 = reaction.product
    graph = nx.DiGraph()
    node_name_A = str(entry0.parameters["ind"]) + "," + str(entry1.parameters["ind"])
    node_name_B = str(entry1.parameters["ind"]) + "," + str(entry0.parameters["ind"])
    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]
    entry_ids_A = str(entry0.entry_id) + "," + str(entry1.entry_id)
    entry_ids_B = str(entry1.entry_id) + "," + str(entry0.entry_id)

    graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                   entry_ids=entry_ids_A)
    graph.add_edge(entry0.parameters["ind"],
                   node_name_A,
                   softplus=ReactionNetwork.softplus(free_energy_A),
                   exponent=ReactionNetwork.exponent(free_energy_A),
                   weight=1.0)
    graph.add_edge(node_name_A,
                   entry1.parameters["ind"],
                   softplus=0.0,
                   exponent=0.0,
                   weight=1.0)
    graph.add_node(node_name_B, rxn_type=rxn_type_B, bipartite=1,energy_B=energy_B,free_energy=free_energy_B,
                   entry_ids=entry_ids_B)
    graph.add_edge(entry1.parameters["ind"],
                   node_name_B,
                   softplus=ReactionNetwork.softplus(free_energy_B),
                   exponent=ReactionNetwork.exponent(free_energy_B),
                   weight=1.0)
    graph.add_edge(node_name_B,
                   entry0.parameters["ind"],
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

    def __init__(self, reactant, product):
        if len(reactant) != 1 or len(product) != 1:
            raise RuntimeError("One electron redox requires two lists that each contain one entry!")
        self.reactant = reactant[0]
        self.product = product[0]
        self.electron_free_energy = None
        super().__init__([self.reactant], [self.product])

    def graph_representation(self) -> nx.DiGraph:
        pass

    def graph_representation_0(self) -> nx.DiGraph:
        entry0 = self.reactant
        entry1 = self.product
        graph = nx.DiGraph()

        val0 = entry0.charge
        val1 = entry1.charge
        if val1 < val0:
            rxn_type_A = "One electron reduction"
            rxn_type_B = "One electron oxidation"
        else:
            rxn_type_A = "One electron oxidation"
            rxn_type_B = "One electron reduction"

        node_name_A = str(entry0.parameters["ind"]) + "," + str(entry1.parameters["ind"])
        node_name_B = str(entry1.parameters["ind"]) + "," + str(entry0.parameters["ind"])
        energy_A = entry1.energy - entry0.energy
        energy_B = entry0.energy - entry0.energy

        if entry1.free_energy != None and entry0.free_energy != None:
            free_energy_A = entry1.free_energy - entry0.free_energy
            free_energy_B = entry0.free_energy - entry1.free_energy
            if rxn_type_A == "One electron reduction":
                free_energy_A += -self.electron_free_energy
                free_energy_B += self.electron_free_energy
            else:
                free_energy_A += self.electron_free_energy
                free_energy_B += -self.electron_free_energy
        else:
            free_energy_A = None
            free_energy_B = None

        graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                       entry_ids=str(entry0.entry_id) + "," + str(entry1.entry_id))
        graph.add_edge(entry0.parameters["ind"],
                       node_name_A,
                       softplus=ReactionNetwork.softplus(self, free_energy_A),
                       exponent=ReactionNetwork.exponent(self, free_energy_A),
                       weight=1.0)
        graph.add_edge(node_name_A,
                       entry1.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0)
        graph.add_node(node_name_B, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                       entry_ids=str(entry1.entry_id) + "," + str(entry0.entry_id))
        graph.add_edge(entry1.parameters["ind"],
                       node_name_B,
                       softplus=ReactionNetwork.softplus(self, free_energy_B),
                       exponent=ReactionNetwork.exponent(self, free_energy_B),
                       weight=1.0)
        graph.add_edge(node_name_B,
                       entry0.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0)

        return graph

    @classmethod
    def generate(cls, entries):
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
                                        r = cls([entry0], [entry1])
                                        reactions.append(r)

        return reactions

    def reaction_type(self):
        val0 = self.reactant.charge
        val1 = self.product.charge
        if val1 < val0:
            rxn_type_A = "One electron reduction"
            rxn_type_B = "One electron oxidation"
        else:
            rxn_type_A = "One electron oxidation"
            rxn_type_B = "One electron reduction"

        reaction_type = {"class": "RedoxReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self):
        entry0 = self.reactant
        entry1 = self.product
        if entry1.free_energy is not None and entry0.free_energy is not None:
            free_energy_A = entry1.free_energy - entry0.free_energy
            free_energy_B = entry0.free_energy - entry1.free_energy

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

    def energy(self):
        entry0 = self.reactant
        entry1 = self.product
        if entry1.energy is not None and entry0.energy is not None:
            energy_A = entry1.energy - entry0.energy
            energy_B = entry0.energy - entry1.energy
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

    def __init__(self, reactant, product):
        if len(reactant) != 1 or len(product) != 1:
            raise RuntimeError("Intramolecular single bond change requires two lists that each contain one entry!")
        self.reactant = reactant[0]
        self.product = product[0]
        super().__init__([self.reactant], [self.product])

    def graph_representation(self) -> nx.DiGraph:  ## temp here, use graph_rep_1_1() method
        entry0 = self.reactant
        entry1 = self.product
        graph = nx.DiGraph()
        val0 = entry0.Nbonds
        val1 = entry1.Nbonds
        if val1 < val0:
            rxn_type_A = "Intramolecular single bond breakage"
            rxn_type_B = "Intramolecular single bond formation"
        else:
            rxn_type_A = "Intramolecular single bond formation"
            rxn_type_B = "Intramolecular single bond breakage"

        node_name_A = str(entry0.parameters["ind"]) + "," + str(entry1.parameters["ind"])
        node_name_B = str(entry1.parameters["ind"]) + "," + str(entry0.parameters["ind"])
        energy_A = entry1.energy - entry0.energy
        energy_B = entry0.energy - entry1.energy
        if entry1.free_energy != None and entry0.free_energy != None:
            free_energy_A = entry1.free_energy - entry0.free_energy
            free_energy_B = entry0.free_energy - entry1.free_energy
        else:
            free_energy_A = None
            free_energy_B = None

        graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                       entry_ids=str(entry0.entry_id) + "," + str(entry1.entry_id))
        graph.add_edge(entry0.parameters["ind"],
                       node_name_A,
                       softplus=self.softplus(free_energy_A),
                       exponent=self.exponent(free_energy_A),
                       weight=1.0)
        graph.add_edge(node_name_A,
                       entry1.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0)
        graph.add_node(node_name_B, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                       entry_id=str(entry1.entry_id) + "," + str(entry0.entry_id))
        graph.add_edge(entry1.parameters["ind"],
                       node_name_B,
                       softplus=self.softplus(free_energy_B),
                       exponent=self.exponent(free_energy_B),
                       weight=1.0)
        graph.add_edge(node_name_B,
                       entry0.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0)

        return graph

    @classmethod
    def generate(cls, entries):

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
                                                    r = cls([entry0], [entry1])
                                                    reactions.append(r)
                                                    break

        return reactions

    def reaction_type(self):
        val0 = self.reactant.charge
        val1 = self.product.charge
        if val1 < val0:
            rxn_type_A = "Intramolecular single bond breakage"
            rxn_type_B = "Intramolecular single bond formation"
        else:
            rxn_type_A = "Intramolecular single bond formation"
            rxn_type_B = "Intramolecular single bond breakage"

        reaction_type = {"class": "IntramolSingleBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self):
        entry0 = self.reactant
        entry1 = self.product
        if entry1.free_energy != None and entry0.free_energy != None:
            free_energy_A = entry1.free_energy - entry0.free_energy
            free_energy_B = entry0.free_energy - entry1.free_energy
        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self):
        entry0 = self.reactant
        entry1 = self.product
        if entry1.energy is not None and entry0.energy is not None:
            energy_A = entry1.energy - entry0.energy
            energy_B = entry0.energy - entry1.energy

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


    def __init__(self, reactant, product):
        self.reactant = reactant[0]
        self.product0 = product[0]
        self.product1 = product[1]
        super().__init__([self.reactant], [self.product0, self.product1])

    def graph_representation(self) -> nx.DiGraph:  # temp here, use graph_rep_1_2 instead
        entry = self.reactant
        entry0 = self.product0
        entry1 = self.product1
        graph = nx.DiGraph()

        if entry0.parameters["ind"] <= entry1.parameters["ind"]:
            two_mol_name = str(entry0.parameters["ind"]) + "+" + str(entry1.parameters["ind"])
            two_mol_name_entry_ids = str(entry0.entry_id) + "+" + str(entry1.entry_id)
        else:
            two_mol_name = str(entry1.parameters["ind"]) + "+" + str(entry0.parameters["ind"])
            two_mol_name_entry_ids = str(entry1.entry_id) + "+" + str(entry0.entry_id)

        two_mol_name0 = str(entry0.parameters["ind"]) + "+PR_" + str(entry1.parameters["ind"])
        two_mol_entry_ids0 = str(entry0.entry_id) + "+PR_" + str(entry1.entry_id)
        two_mol_name1 = str(entry1.parameters["ind"]) + "+PR_" + str(entry0.parameters["ind"])
        two_mol_entry_ids1 = str(entry1.entry_id) + "+PR_" + str(entry0.entry_id)

        node_name_A = str(entry.parameters["ind"]) + "," + two_mol_name
        node_name_B0 = two_mol_name0 + "," + str(entry.parameters["ind"])
        node_name_B1 = two_mol_name1 + "," + str(entry.parameters["ind"])
        entry_ids_name_A = str(entry.entry_id) + "," + two_mol_name_entry_ids
        entry_ids_name_B0 = two_mol_entry_ids0 + "," + str(entry.entry_id)
        entry_ids_name_B1 = two_mol_entry_ids1 + "," + str(entry.entry_id)
        rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
        rxn_type_B = "Molecular formation from one new bond A+B -> C"
        energy_A = entry0.energy + entry1.energy - entry.energy
        energy_B = entry.energy - entry0.energy - entry1.energy
        if entry1.free_energy != None and entry0.free_energy != None and entry.free_energy != None:
            free_energy_A = entry0.free_energy + entry1.free_energy - entry.free_energy
            free_energy_B = entry.free_energy - entry0.free_energy - entry1.free_energy

        graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                       entry_ids=entry_ids_name_A)

        graph.add_edge(entry.parameters["ind"],
                       node_name_A,
                       softplus=self.softplus(free_energy_A),
                       exponent=self.exponent(free_energy_A),
                       weight=1.0
                       )

        graph.add_edge(node_name_A,
                       entry0.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0
                       )
        graph.add_edge(node_name_A,
                       entry1.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0
                       )

        graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                       entry_ids=entry_ids_name_B0)
        graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                       entry_ids=entry_ids_name_B1)

        graph.add_edge(node_name_B0,
                       entry.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0
                       )
        graph.add_edge(node_name_B1,
                       entry.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0
                       )

        graph.add_edge(entry0.parameters["ind"],
                       node_name_B0,
                       softplus=self.softplus(free_energy_B),
                       exponent=self.exponent(free_energy_B),
                       weight=1.0
                       )
        graph.add_edge(entry1.parameters["ind"],
                       node_name_B1,
                       softplus=self.softplus(free_energy_B),
                       exponent=self.exponent(free_energy_B),
                       weight=1.0)
        return graph

    @classmethod
    def generate(cls, entries):

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
                                                                    r = cls([entry], [entry0, entry1])
                                                                    reactions.append(r)
                                                                    break
                                                        break
                                except MolGraphSplitError:
                                    pass

        return reactions

    def reaction_type(self):
        rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
        rxn_type_B = "Molecular formation from one new bond A+B -> C"

        reaction_type = {"class": "IntermolecularReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self):
        entry = self.reactant
        entry0 = self.product0
        entry1 = self.product1

        if entry1.free_energy is not None and entry0.free_energy is not None and entry.free_energy is not None:
            free_energy_A = entry0.free_energy + entry1.free_energy - entry.free_energy
            free_energy_B = entry.free_energy - entry0.free_energy - entry1.free_energy

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self):
        entry = self.reactant
        entry0 = self.product0
        entry1 = self.product1

        if entry1.energy is not None and entry0.energy is not None and entry.energy is not None:
            energy_A = entry0.energy + entry1.energy - entry.energy
            energy_B = entry.energy - entry0.energy - entry1.energy

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

    def __init__(self, reactant, product):
        self.reactant = reactant[0]
        self.product0 = product[0]
        self.product1 = product[1]
        super().__init__([self.reactant],[self.product0, self.product1])

    @classmethod
    def generate(cls, entries):
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
                                                                                r = cls([entry],[nonM_entry,
                                                                                                    M_entries[
                                                                                                        M_formula][
                                                                                                        M_charge]])
                                                                                reactions.append(r)
                                                                                break
                                        except MolGraphSplitError:
                                            pass
        return reactions

    def graph_representation(self) -> nx.DiGraph:
        entry = self.reactant
        entry0 = self.product0
        entry1 = self.product1
        graph = nx.DiGraph()

        if entry0.parameters["ind"] <= entry1.parameters["ind"]:
            two_mol_name = str(entry0.parameters["ind"]) + "+" + str(entry1.parameters["ind"])
            two_mol_name_entry_ids = str(entry0.entry_id) + "+" + str(entry1.entry_id)
        else:
            two_mol_name = str(entry1.parameters["ind"]) + "+" + str(entry0.parameters["ind"])
            two_mol_name_entry_ids = str(entry1.entry_id) + "+" + str(entry0.entry_id)

        two_mol_name0 = str(entry0.parameters["ind"]) + "+PR_" + str(entry1.parameters["ind"])
        two_mol_entry_ids0 = str(entry0.entry_id) + "+PR_" + str(entry1.entry_id)
        two_mol_name1 = str(entry1.parameters["ind"]) + "+PR_" + str(entry0.parameters["ind"])
        two_mol_entry_ids1 = str(entry1.entry_id) + "+PR_" + str(entry0.entry_id)

        node_name_A = str(entry.parameters["ind"]) + "," + two_mol_name
        node_name_B0 = two_mol_name0 + "," + str(entry.parameters["ind"])
        node_name_B1 = two_mol_name1 + "," + str(entry.parameters["ind"])
        entry_ids_name_A = str(entry.entry_id) + "," + two_mol_name_entry_ids
        entry_ids_name_B0 = two_mol_entry_ids0 + "," + str(entry.entry_id)
        entry_ids_name_B1 = two_mol_entry_ids1 + "," + str(entry.entry_id)
        rxn_type_A = "Coordination bond breaking AM -> A+M"
        rxn_type_B = "Coordination bond forming A+M -> AM"

        energy_A = entry0.energy + entry1.energy - entry.energy
        energy_B = entry.energy - entry0.energy - entry1.energy
        if entry1.free_energy != None and entry0.free_energy != None and entry.free_energy != None:
            free_energy_A = entry0.free_energy + entry1.free_energy - entry.free_energy
            free_energy_B = entry.free_energy - entry0.free_energy - entry1.free_energy

        graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
                       entry_ids=entry_ids_name_A)

        graph.add_edge(entry.parameters["ind"],
                       node_name_A,
                       softplus=self.softplus(free_energy_A),
                       exponent=self.exponent(free_energy_A),
                       weight=1.0
                       )

        graph.add_edge(node_name_A,
                       entry0.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0
                       )
        graph.add_edge(node_name_A,
                       entry1.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0
                       )

        graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                       entry_ids=entry_ids_name_B0)
        graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
                       entry_ids=entry_ids_name_B1)

        graph.add_edge(node_name_B0,
                       entry.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0
                       )
        graph.add_edge(node_name_B1,
                       entry.parameters["ind"],
                       softplus=0.0,
                       exponent=0.0,
                       weight=1.0
                       )

        graph.add_edge(entry0.parameters["ind"],
                       node_name_B0,
                       softplus=self.softplus(free_energy_B),
                       exponent=self.exponent(free_energy_B),
                       weight=1.0
                       )
        graph.add_edge(entry1.parameters["ind"],
                       node_name_B1,
                       softplus=self.softplus(free_energy_B),
                       exponent=self.exponent(free_energy_B),
                       weight=1.0)

        return graph

    def reaction_type(self):
        rxn_type_A = "Coordination bond breaking AM -> A+M"
        rxn_type_B = "Coordination bond forming A+M -> AM"

        reaction_type = {"class": "CoordinationBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self):
        entry = self.reactant
        entry0 = self.product0
        entry1 = self.product1

        if entry1.free_energy is not None and entry0.free_energy is not None and entry.free_energy is not None:
            free_energy_A = entry0.free_energy + entry1.free_energy - entry.free_energy
            free_energy_B = entry.free_energy - entry0.free_energy - entry1.free_energy

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self):
        entry = self.reactant
        entry0 = self.product0
        entry1 = self.product1

        if entry1.energy is not None and entry0.energy is not None and entry.energy is not None:
            energy_A = entry0.energy + entry1.energy - entry.energy
            energy_B = entry.energy - entry0.energy - entry1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class ReactionNetwork:
    """
       Class to build a reaction network from entries

       Args:
           input_entries ([MoleculeEntry]): A list of MoleculeEntry objects.
           electron_free_energy (float): The Gibbs free energy of an electron.
               Defaults to -2.15 eV, the value at which the LiEC SEI forms.
       """


    def __init__(self, input_entries, electron_free_energy=-2.15):
        self.graph = nx.DiGraph()
        self.reactions = []
        self.input_entries = input_entries
        self.entry_ids = {e.entry_id for e in self.input_entries}
        self.electron_free_energy = electron_free_energy
        self.entries = {}
        self.entries_list = []

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
    def softplus(free_energy):
        return np.log(1 + (273.0 / 500.0) * np.exp(free_energy))

    @staticmethod
    def exponent(free_energy):
        return np.exp(free_energy)


    def build(self, reaction_types={"RedoxReaction", "IntramolSingleBondChangeReaction", "IntermolecularReaction", "CoordinationBondChangeReaction"}):
        reaction_types = [load_class(str(self.__module__)+"."+s) for s in reaction_types]

        self.reactions = [r.generate(self.entries) for r in reaction_types]
        self.reactions = [i for i in self.reactions if i]
        self.reactions = list(itertools.chain.from_iterable(self.reactions))
        self.graph.add_nodes_from(range(len(self.entries_list)), bipartite=0)
        for r in self.reactions:
            if r.reaction_type()["class"] == "RedoxReaction" or r.reaction_type()["class"] == "IntramolSingleBondChangeReaction":
                if r.reaction_type()["class"] == "RedoxReaction":
                    r.electron_free_energy = self.electron_free_energy
                self.add_reaction(graph_rep_1_1(r))
            else:
                self.add_reaction(graph_rep_1_2(r))

        return self.graph

    def add_reaction(self, graph_representation):
        self.graph.add_nodes_from(graph_representation.nodes(data=True))
        self.graph.add_edges_from(graph_representation.edges(data=True))


    #
    # def generate_concerted_reactions(self, order: int = 2, entries=None):
    #     entries = entries if entries else [e.entry_id for self.entries]
    #
    #     #...

# class ConcertedReaction(Reaction):
#
#     def generate(cls, entries) -> List[Reaction]:
#         raise RunTimeError("Concerted reactions only be generated by a reaction network")
#
#
# OneElectronRedox
# IntermolecularBond
# IntramolecularBond
# Coordination - removes
# a
# single
# atom
# from a coordination
#
# environment
