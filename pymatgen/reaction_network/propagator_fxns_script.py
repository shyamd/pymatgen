import numpy as np
import math
import random
from pymatgen.reaction_network.reaction_network import ReactionNetwork
import time
import matplotlib.pyplot as plt
import pickle
from scipy.constants import N_A
from numba import jit
from pymatgen.reaction_network.reaction_propagator_fxns import *

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

num_label = 15 # number of top species listed in simulation plot legend

# # Testing parameters
# iterations = 3  # can conduct multiple iterations for statistical analysis of results and runtime
# volumes_list = [10**-24]
# timesteps_list = [19450] # number of time steps in simulation

# Load reaction network
pickle_in = open("pickle_rxnnetwork_Li-limited", "rb")
reaction_network = pickle.load(pickle_in)

[initial_state, initial_state_dict, species_rxn_mapping, reactant_array, product_array, coord_array, rate_constants, propensities,
    molid_index_mapping] = initialize_simulation(reaction_network, initial_conditions)

state = np.array(initial_state, dtype = int)
t0 = time.time()
sim_data = kmc_simulate(19450, coord_array, rate_constants, propensities, species_rxn_mapping, reactant_array, product_array, state)
t1 = time.time()
print("KMC simulation time (min) = ", (t1-t0)/60)
reaction_history = sim_data[0, :]
times = sim_data[1, :]
plot_trajectory(reaction_network, molid_index_mapping, initial_state_dict, product_array, reactant_array, reaction_history, times, num_label, "test_plot3")