import numpy as np
import math
import random
from pymatgen.reaction_network.reaction_network import ReactionNetwork
import time
import matplotlib.pyplot as plt
import pickle
from scipy.constants import N_A
from numba import jit
from pymatgen.reaction_network.reaction_propagator import *
import os

mypath = os.path.dirname(__file__)

"""Simulation parameters setup"""
# List the IDs of initial species present in electrolyte
li_id = 2335
ec_id = 2606
emc_id = 1877
h2o_id = 3306
# id_list = [li_id, ec_id, emc_id, h2o_id]

# Concentrations of electrolyte species in molarity
li_conc = 1.0
    # 1M LiPF6 in 3:7 EC:EMC, a standard, proven Li-ion electrolyte formulation
ec_conc = 3.57
emc_conc = 7.0555
h2o_conc = 1.665*10**-4
# conc_list = [li_conc, ec_conc, emc_conc, h2o_conc]

initial_conditions = dict()
initial_conditions[li_id] = li_conc
initial_conditions[ec_id] = ec_conc
initial_conditions[emc_id] = emc_conc
initial_conditions[h2o_id] = h2o_conc

num_label = 12 # number of top species listed in simulation plot legend

# # Testing parameters
# iterations = 3  # can conduct multiple iterations for statistical analysis of results and runtime
# volumes_list = [10**-24]
# timesteps_list = [19450] # number of time steps in simulation

# Load reaction network
pickle_in = open("pickle_rxnnetwork_Li-limited", "rb")
reaction_network = pickle.load(pickle_in)
num_iterations = 5

[state, initial_state_dict, species_rxn_mapping, reactant_array, product_array, coord_array,
    rate_constants, propensities, molid_index_mapping] = initialize_simulation(reaction_network,
                                                                                initial_conditions)

# kmc_rxns_out = list()
# kmc_tau_out = list()
#
# for i in range(num_iterations):
#     print('iteration ', i + 1)
#     # state = np.array(initial_state, dtype=int)
#     # propensities = initial_propensities
#     t0 = time.time()
#     [state, initial_state_dict, species_rxn_mapping, reactant_array, product_array, coord_array,
#      rate_constants, propensities, molid_index_mapping] = initialize_simulation(reaction_network,
#                                                                                         initial_conditions)
#     sim_data = kmc_simulate(19450, coord_array, rate_constants, propensities, species_rxn_mapping, reactant_array,
#                             product_array, state)
#     t1 = time.time()
#     print("KMC simulation time (min) = ", (t1-t0)/60)
#     kmc_rxns_out.append(sim_data[0, :])
#     kmc_tau_out.append(sim_data[1, :])
#
# pickle.dump(kmc_rxns_out, open('kmc_rxn_5.pickle', 'wb'))
# pickle.dump(kmc_tau_out, open('kmc_tau_5.pickle', 'wb'))

pickle_in = open("kmc_rxn_5.pickle", "rb")
kmc_rxns_out = pickle.load(pickle_in)

pickle_in = open('kmc_tau_5.pickle', 'rb')
kmc_tau_out = pickle.load(pickle_in)

print('num_sims: ', len(kmc_tau_out), len(kmc_rxns_out))
print('sim_length: ', len(kmc_tau_out[0]))

analyzer = KMC_data_analyzer(reaction_network, molid_index_mapping, species_rxn_mapping, initial_state_dict,
                             product_array, reactant_array, kmc_rxns_out, kmc_tau_out)

profiles = analyzer.generate_time_dep_profiles()
analyzer.plot_species_profiles(profiles['species_profiles'], profiles['final_states'], num_label=10,
                               filename="KMC_testing")
# intermed_analysis = analyzer.analyze_intermediates(profiles['species_profiles'])
# for mol_ind in intermed_analysis:
#     print(intermed_analysis[mol_ind])
# print('number of intermediates: ', len(intermed_analysis))
# ranked_rxns = analyzer.quantify_rank_reactions(num_rxns=20)
# correlate_rxns = list()
# for i, (rxn_ind, data) in enumerate(ranked_rxns):
#     print('rxn ', rxn_ind, '   -    ', data)
#     if i < 6:
#         correlate_rxns.append(rxn_ind)
# correlations = analyzer.correlate_reactions(correlate_rxns[:2])
# print(correlations)

# frequency_data = analyzer.frequency_analysis(rxn_inds=[122385], spec_inds=[11], partitions=20)
# for rxn_ind in frequency_data:
#     print(frequency_data[rxn_ind])
