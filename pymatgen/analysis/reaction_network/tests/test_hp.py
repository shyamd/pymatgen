import os
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymongo import MongoClient
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.analysis.fragmenter import Fragmenter, metal_edge_extender
from crystal_toolkit.renderables import *
from crystal_toolkit.helpers.pythreejs_renderer import view
import os
import unittest
import time
import copy
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.util.testing import PymatgenTest
#from pymatgen.analysis.reaction_network.reaction_network import ReactionNetwork
from pymatgen.analysis.fragmenter import Fragmenter, metal_edge_extender
from pymatgen.entries.mol_entry import MoleculeEntry

from monty.serialization import loadfn,dumpfn
import openbabel as ob
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph
from ast import literal_eval as make_tuple
import itertools
#from pymatgen.analysis.reaction_network.reaction_network_HP import Reaction, RedoxReaction, ReactionNetwork, IntermolecularReaction, IntramolSingleBondChangeReaction, CoordinationBondChangeReaction
#from pymatgen.analysis.reaction_network.test_hetal.temp2 import graph_rep_1_1, Reaction, RedoxReaction, IntermolecularReaction, IntramolSingleBondChangeReaction, CoordinationBondChangeReaction
from pymatgen.analysis.reaction_network.reaction_network import ReactionNetwork as RN_sam
from pymatgen.analysis.reaction_network.reaction_network_HP import ReactionNetwork as RN_hetal
#from pymatgen.analysis.reaction_network.test_hetal.temp2 import ReactionNetwork as RN_hetal



database = "mp_hp_sei"
collection = "tasks"
admin_user = "hpatel_lbl.gov_readWrite"
admin_password = "blushfully-diremption-availability-gardy-anacusis"
host = "mongodb03.nersc.gov"
port = 27017
client = MongoClient()
client = MongoClient(host, port)
db = client[database]
database_sam = "sb_qchem"
collection_sam = "molecules"
admin_user_sam = "smblau_lbl.gov_readWrite"
admin_password_sam = "baffler-underranger-sanguinely-distent-flukeworm"
host_sam = "mongodb03.nersc.gov"
port_sam = 27017
client_sam = MongoClient()
client_sam = MongoClient(host_sam, port_sam)
db_sam = client[database_sam]
db_sam.authenticate(admin_user_sam, admin_password_sam)
Li_1 = 116001
FEC = 137146
FEC_minus = 137196
LiFEC_1_mono = 140692 ####
LiFEC_0_mono = 173616 ####
LiFEC_0_bi = 140687
LiFEC_0_RO2 = 223972
LiFEC_0_RO4 = 197363

#ind = [223972, 173616]
#ind = [173616, 223972]
#ind = [137146, 137196, 140692, 173616, 140687, 223972]
#ind = [140692, 173616, 137146, 137196]# redox
#ind_r = [137146]
#ind_p = [137196]
ind = [FEC, FEC_minus, LiFEC_1_mono, LiFEC_0_mono, LiFEC_0_bi, LiFEC_0_RO4]
entries = []
for index in ind:
    for x in db_sam.get_collection("tasks").find({"task_id": index}):
        mol = Molecule.from_dict(x["output"]["optimized_molecule"])
        energy = x["output"]["final_energy"]
        enthalpy = x["output"]["enthalpy"]
        entropy = x["output"]["entropy"]
        task_id = x["task_id"]
        entry = MoleculeEntry(molecule=mol,
                              energy=energy,
                              enthalpy=enthalpy,
                              entropy=entropy,
                              entry_id=task_id)
        entries.append(entry)


# for index in ind:
#     for x in db_sam.get_collection("tasks").find({"task_id": index}):
#         mol = Molecule.from_dict(x["output"]["optimized_molecule"])
#         energy = x["output"]["final_energy"]
#         enthalpy = x["output"]["enthalpy"]
#         entropy = x["output"]["entropy"]
#         task_id = x["task_id"]
#         entry = MoleculeEntry(molecule=mol,
#                               energy=energy,
#                               enthalpy=enthalpy,
#                               entropy=entropy,
#                               entry_id=task_id)
#         entries.append(entry)

for x in db_sam.get_collection("tasks").find({"task_id": 116001}):
    mol = Molecule.from_dict(x["output"]["initial_molecule"])
    energy = x["output"]["final_energy"]
    enthalpy = x["output"]["enthalpy"]
    entropy = x["output"]["entropy"]
    task_id = x["task_id"]
    entry = MoleculeEntry(molecule=mol,
                          energy=energy,
                          enthalpy=enthalpy,
                          entropy=entropy,
                          entry_id=task_id)
    entries.append(entry)


entries_HP = entries
#RN = ReactionNetwork(entries_HP, electron_free_energy=-2.15)
#RN.find_paths([5], 0, weight="softplus", num_paths=10)
#RN.solve_prerequisites([6, 5], 0, weight="softplus")
#PR_paths, paths = RN.find_paths([5],0, weight="softplus", num_paths=10)
# print(PR_paths)
# print(paths)

#print(ReactionNetwork(entries_HP))

#RN = ReactionNetwork(entries_HP).build()

#Reaction(reactant_entries, product_entries)
# r = RedoxReaction(reactant_entries, product_entries)
# graph_rep_1_1(r)


#print(entries)
#print(reactant_entries, product_entries)
#print(Reaction(entries_HP))
######
#r1 = Reaction(entries_HP)
# print("#######")
# print(r1.entries_list)
# print("#######")
#r2 = CoordinationBondChangeReaction.generate(r1.entries)
#print(r2)
#print(entries_HP)
#print(r1.entries)
#r2 = RedoxReaction.generate(r1.entries)
# # print(RedoxReaction.generate(r1.entries))
# for r in r2:
#     r.graph_representation()
# print("#########")
#ReactionNetwork(entries_HP).build()

# r2 = RedoxReaction.generate(r1.entries)

#r2 = IntermolecularReaction.generate(r1.entries)
# for r in r2:
#     r.graph_representation()

#ReactionNetwork(entries_HP).build()
#print(r2)

#for d in r2:
 #   print(d.reaction_entry.reactants[0], d.reaction_entry.products[0])
    #print(d.reactants, d.products)
  #  print(d.graph_representation().edges)
  #  print(d.graph_representation().nodes)
    #print(d.reactant.parameters["ind"], d.product)
    #print(type(d.reactant))
##############
#ReactionNetwork(entries_HP).built()

# print(r2.__len__())
# print(r2)
# r4 = RedoxReaction(RedoxReaction.generate(r1.entries))
# r3 = r4.graph_representation()
#print(r3)
#r3 = RedoxReaction.generate(entries_HP)
#r2 = RedoxReaction()
#print(r1)
#
# G=nx.DiGraph()
# G.add_node(1, bipartite=1)
# G.add_node(2, bipartite=1)
# G.add_edge(1, 2)
# G.add_edge(2,1)
# print(G.nodes, G.edges)
# H=nx.DiGraph()
# H.add_node(1, bipartite=1)
# H.add_node(4, bipartite=1)
# H.add_edge(1, 4)
# H.add_edge(4, 1)
# print(H.nodes, H.edges)
# U = nx.DiGraph()
# U.add_nodes_from(G.nodes())
# U.add_edges_from(G.edges())
# U.add_nodes_from(H.nodes())
# U.add_edges_from(H.edges())
#print(U.nodes, U.edges)
#U.add_edges_from(G.edges+H.edges)
#U.add_nodes_from(G.nodes+H.nodes)
#
# reaction_types=["RedoxReaction", "IntramolSingleBondChangeReaction", "IntermolecularReaction",
#                                 "CoordinationBondChangeReaction"]
#
# s = "RedoxReaction"
# print("pymatgen.analysis.reaction_network.Reaction_Network_HP.{}".format(s))
#
# for k in reaction_types:
#     print(k)
# #reaction_types = [load_class("pymatgen.analysis.reaction_network.Reaction_Network_HP.{}".format(s)) for s in
#                      # reaction_types]
# l = ["pymatgen.analysis.reaction_network.Reaction_Network_HP.{}".format(s) for s in reaction_types]
#                      # reaction_types
# print(l)
# print(list(l))


#RN = ReactionNetwork(entries_HP).build()
#print(RN.neighbors(1))
# print(list(RN.neighbors(1)))
# print(RN.nodes())
# print(list(RN.nodes(data=True)))

#
# print(RN.nodes(data=True))
# left, right = nx.bipartite.sets(RN)
# print(list(left))
# print(list(right))
# print(RN.nodes(data=True))
#
# G = nx.DiGraph()
#
# G.add_edge("1", "2")
# print(G.nodes(data=True))
# print("***")
# for neighbor in list(RN.neighbors(1)):
#     print(neighbor)
#
#     for key in RN.nodes[neighbor]:
#         print(key,":",RN.nodes[neighbor][key])
#     print()

# colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#
sb = RN_sam(entries_HP)
hp = RN_hetal(entries_HP)
#print(type(sb), type(hp))
print(sb.graph.nodes)
print(hp.build().nodes)
print(hp.entries_list)
print(nx.is_isomorphic(sb.graph, hp.build()))
# import matplotlib.pyplot as plt
# import networkx as nx
# nx.draw(hp.build())
#print(CoordinationBondChangeReaction.generate(RN_hetal(entries_HP).entries))

# def plot_all_exothermic_pathways(file_name, pathway_nodes, pathway_edges, colors):
#
#     u = Digraph(file_name, filename=file_name)
#     #u.attr(size='6,6')
#     for i, node in enumerate(pathway_nodes):
#         u.attr('node', style='filled', colorscheme='purples9',
#                shape='circle')
#         u.node(str(node))
#
#     _edge = '\t%s -> %s%s'
#     for (i, j) in pathway_edges:
#         u.edge(str(i), str(j))
#     u.render('test_output/'+file_name, view=True)
#     return
#
#
# plot_all_exothermic_pathways("testgraph_hp", hp.build().nodes(), hp.build().edges(), colors)
# plot_all_exothermic_pathways("testgraph_sam", sb.graph.nodes(), sb.graph.edges(), colors)

#entries_SB = loadfn("LiEC_reextended_entries.json")