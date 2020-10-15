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

pickle_in = open("ledc_network_pickle_nomono_rightbarrier", "rb")
reaction_network = pickle.load(pickle_in)

"""Simulation parameters setup"""
# Find the IDs of initial species present in electrolyte for li, ec, emc, h2o
li_id = "Li_plus"
ec_id = "EC"
h2o_id = "H2O"


# Concentrations of electrolyte species in molarity
li_conc = 1.0
# 1M LiPF6 in EC solvent, with some h2o impurities
ec_conc = 13.037
h2o_conc = 1.665*10**-4
# conc_list = [li_conc, ec_conc, emc_conc, h2o_conc]

initial_conditions = dict()
initial_conditions[li_id] = li_conc
initial_conditions[ec_id] = ec_conc
initial_conditions[h2o_id] = h2o_conc

num_label = 12 # number of top species listed in simulation plot legend

num_iterations = 1

kmc_rxns_out = list()
kmc_tau_out = list()
[initial_state, initial_state_dict, species_rxn_mapping, reactant_array, product_array, coord_array, rate_constants,
 propensities, molid_index_mapping] = initialize_simulation(reaction_network, initial_conditions)
state = np.array(initial_state, dtype=int)

for i in range(num_iterations):
    t0 = time.time()
    sim_data = kmc_simulate(10000, coord_array, rate_constants, propensities, species_rxn_mapping, reactant_array,
                            product_array, state)
    t1 = time.time()
    print("KMC simulation time (min) = ", (t1-t0)/60)
    kmc_rxns_out.append(sim_data[0, :])
    kmc_tau_out.append(sim_data[1, :])
#
pickle.dump(kmc_rxns_out, open('small_net_kmc_rxn_1.pickle', 'wb'))
pickle.dump(kmc_tau_out, open('small_net_kmc_tau_1.pickle', 'wb'))

# pickle_in = open("kmc_rxn_1.pickle", "rb")
# kmc_rxns_out = pickle.load(pickle_in)

# pickle_in = open('kmc_tau_1.pickle', 'rb')
# kmc_tau_out = pickle.load(pickle_in)

analyzer = KMC_data_analyzer(reaction_network, molid_index_mapping, species_rxn_mapping, initial_state_dict,
                             product_array, reactant_array, kmc_rxns_out, kmc_tau_out)

profiles = analyzer.generate_time_dep_profiles()
analyzer.plot_species_profiles(profiles['species_profiles'], profiles['final_states'], num_label=15,
                               filename="small_net_KMC")
intermed_analysis = analyzer.analyze_intermediates(profiles['species_profiles'])
for mol_ind in intermed_analysis:
    print(intermed_analysis[mol_ind])
print('number of intermediates: ', len(intermed_analysis))
ranked_rxns = analyzer.quantify_rank_reactions(num_rxns=20)
correlate_rxns = list()
for (rxn_ind, data) in ranked_rxns[:2]:
    print('rxn ', rxn_ind, '   -    ', data)
    correlate_rxns.append(rxn_ind)
correlations = analyzer.correlate_reactions(correlate_rxns)
print(correlations)

# frequency_data = analyzer.frequency_analysis(rxn_inds=[122385], spec_inds=[11], partitions=20)
# for rxn_ind in frequency_data:
#     print(frequency_data[rxn_ind])


