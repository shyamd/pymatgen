import numpy as np
import math
import random
from pymatgen.reaction_network.reaction_network import ReactionNetwork
import time
import matplotlib.pyplot as plt
import pickle
from scipy.constants import N_A
from numba import jit

__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"
__credit__ = "Xiaowei Xie"

"""Function-based KMC simulation for a reaction network, assuming spatial homogeneity. Simulation is performed without objects, 
such is required to improve performance with Numba. Initial part of this script is to set up parameters needed in the functions.

Inputs required: 
    reaction_network (Reaction Network): load from a json or pickle file
    mol_ids (int): molecule ids (corresponding to reaction network) of initially present electrolyte species
    mol_conc (float): molecule concentration (M) of initially present electrolyte species
    num_species (int): number of entries initially used to generate reaction_network
    num_label (int): number of ids listed in the legend of simulation plot 
"""

"""Simulation parameters setup"""
# List the IDs of initial species present in electrolyte
li_id = 2335
ec_id = 2606
emc_id = 1877
h2o_id = 3306
id_list = [li_id, ec_id, emc_id, h2o_id]

# Concentrations of electrolyte species in molarity
li_conc = 1.0
    # 1M LiPF6 in 3:7 EC:EMC, a standard, proven Li-ion electrolyte formulation
ec_conc = 3.57
emc_conc = 7.0555
h2o_conc = 1.665*10**-4
conc_list = [li_conc, ec_conc, emc_conc, h2o_conc]

num_species = 5732
num_label = 15 # number of top species listed in simulation plot legend

# Testing parameters
iterations = 1  # can conduct multiple iterations for statistical analysis of results and runtime
volumes_list = [10**-24]
timesteps_list = [19450] # number of time steps in simulation

# Load reaction network
pickle_in = open("pickle_rxnnetwork_Li-limited", "rb")
reaction_network = pickle.load(pickle_in)

def initialize_simulation(reaction_network, num_entries, spec_ids, initial_conc, volume = 10**-24):
    """Initial loop through reactions to create product/reactant id arrays, mapping of each species to the reactions it participates in. Required
    to eliminate reaction network object usage during actual simulation.

    Args:
        reaction_network (ReactionNetwork): Fully generated reaction network
        num_entries (int): amount of species initially used to generate reaction network
        file_name (str): File name to save simulation results as
        initial_conc (list of floats): initial concentrations
        volume (float)

    """
    num_rxns = len(reaction_network.reactions)
    num_initial_species = len(spec_ids)
    if num_initial_species != len(initial_conc):
        raise RuntimeError("The number of initial species must be consistent!")

    conc_to_amt = lambda c: int(c * volume * N_A * 1000)
    initial_state_dict = dict()

    for ind in range(num_initial_species):
        initial_state_dict[spec_ids[ind]] = conc_to_amt(initial_conc[ind])
    initial_state = [0 for i in range(num_entries)]

    for mol_id in initial_state_dict:
        initial_state[mol_id] = initial_state_dict[mol_id]

    species_rxn_mapping_list = [[] for j in range(num_entries)]
    reactant_array = -1 * np.ones((num_rxns, 2), dtype = int)
    product_array = -1 * np.ones((num_rxns, 2), dtype = int)
    coord_array = np.zeros(2 * num_rxns)
    rate_constants = np.zeros(2 * num_rxns)

    for id, reaction in enumerate(reaction_network.reactions):
        num_reactants_for = list()
        num_reactants_rev = list()
        rate_constants[2 * id] = reaction.rate_constant()["k_A"]
        rate_constants[2 * id + 1] = reaction.rate_constant()["k_B"]
        for idx, react in enumerate(reaction.reactants):
            reactant_array[id, idx] = react.entry_id
            species_rxn_mapping_list[react.entry_id].append(2 * id)
            num_reactants_for.append(initial_state_dict.get(react.entry_id, 0))
        for idx, prod in enumerate(reaction.products):
            product_array[id, idx] = prod.entry_id
            species_rxn_mapping_list[prod.entry_id].append(2*id + 1)
            num_reactants_rev.append(initial_state_dict.get(prod.entry_id, 0))
        if len(reaction.reactants) == 1:
            coord_array[2 * id] = num_reactants_for[0]
        elif (len(reaction.reactants) == 2) and (reaction.reactants[0] == reaction.reactants[1]):
            coord_array[2 * id] = num_reactants_for[0] * (num_reactants_for[0] - 1)
        elif (len(reaction.reactants) == 2) and (reaction.reactants[0] != reaction.reactants[1]):
            coord_array[2 * id] = num_reactants_for[0] * num_reactants_for[1]
        else:
            raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
        # For reverse reaction
        if len(reaction.products) == 1:
            coord_array[2 * id + 1] = num_reactants_rev[0]
        elif (len(reaction.products) == 2) and (reaction.products[0] == reaction.products[1]):
            coord_array[2 * id + 1] = num_reactants_rev[0] * (num_reactants_rev[0] - 1)
        elif (len(reaction.products) == 2) and (reaction.products[0] != reaction.products[1]):
            coord_array[2 * id + 1] = num_reactants_rev[0] * num_reactants_rev[1]
        else:
            raise RuntimeError("Only single and bimolecular reactions supported by this simulation")

    rxn_mapping_lengths = [len(rxn_list) for rxn_list in species_rxn_mapping_list]
    max_mapping_length = max(rxn_mapping_lengths)
    species_rxn_mapping = -1 * np.ones((num_entries, max_mapping_length), dtype=int)

    for index, rxn_list in enumerate(species_rxn_mapping_list):
        this_map_length = rxn_mapping_lengths[index]
        if this_map_length == max_mapping_length:
            species_rxn_mapping[index, :] = rxn_list
        else:
            species_rxn_mapping[index, : this_map_length - max_mapping_length] = rxn_list
    return [initial_state_dict, initial_state, species_rxn_mapping, reactant_array, product_array, coord_array, rate_constants]

@jit(nopython = True, parallel = True)
def simulate(time_steps, coord_array, rate_constants, propensity_array,
             total_propensity, species_rxn_mapping, reactants, products, state):
    """ KMC Simulation of reaction network and specified initial conditions.

    Args:
         time_steps (int): Number of time steps/iterations desired to run.
         coord_array (array): Numpy array containing coordination numbers of forward and reverse reactions. [h1f, h1r, h2f, h2r, ...]
         rate_constants (array): Numpy array containing rate constants of forward and reverse reactions.
         propensity_array (array): Numpy array containing propensities of for and rev reactions.
         total_propensity (float): Sum of all reaction propensities.
         species_rxn_mapping (2d array): Contains all the reaction indexes that each species takes part in
         reactants (2d array): Species IDs corresponding to the reactants of each forward reaction
         products (2d array): Species IDs corresponding to products of each forward reaction
         state (array): Array containing molecular amounts of each species in the reaction network

    Returns:
        A (2 x time_steps) Numpy array. First row contains the indeces of reactions that occurred. Second row are the time steps generated at each iterations.
    """
    t = 0.0
    reaction_history = [0 for step in range(time_steps)]
    times = [0.0 for step in range(time_steps)]
    relevant_ind = np.where(propensity_array > 0)[0]
    for step_counter in range(time_steps):
        r1 = random.random()
        r2 = random.random()
        tau = -np.log(r1) / total_propensity
        random_propensity = r2 * total_propensity
        abrgd_reaction_choice_ind = np.where(np.cumsum(propensity_array[relevant_ind]) >= random_propensity)[0][0]
        reaction_choice_ind = relevant_ind[abrgd_reaction_choice_ind]
        converted_rxn_ind = math.floor(reaction_choice_ind / 2)
        if reaction_choice_ind % 2:
            reverse = True
        else:
            reverse = False
        state = update_state(reactants, products, state, converted_rxn_ind, reverse)
        # Log the reactions that need to be altered after reaction is performed, for the coordination array
        reactions_to_change = list()
        for reactant_id in reactants[converted_rxn_ind, :]:
            if reactant_id == -1:
                continue
            else:
                reactions_to_change.extend(list(species_rxn_mapping[reactant_id, :]))
        for product_id in products[converted_rxn_ind, :]:
            if product_id == -1:
                continue
            else:
                reactions_to_change.extend(list(species_rxn_mapping[product_id, :]))
        reactions_to_change = set(reactions_to_change)
        for rxn_ind in reactions_to_change:
            if rxn_ind == -1:
                continue
            elif rxn_ind % 2:
                this_reverse = True
            else:
                this_reverse = False
            this_h = get_coordination(reactants, products, state, math.floor(rxn_ind/2), this_reverse)
            coord_array[rxn_ind] = this_h

        propensity_array = np.multiply(rate_constants, coord_array)
        relevant_ind = np.where(propensity_array > 0)[0]
        total_propensity = np.sum(propensity_array[relevant_ind])
        reaction_history[step_counter] = reaction_choice_ind
        times[step_counter] = tau
    return np.vstack((np.array(reaction_history), np.array(times)))

@jit(nopython = True)
def update_state(reactants, products, state, rxn_ind, reverse):
    """ Update the system based on the reaction chosen
            Args:
                reaction (Reaction)
                reverse (bool): If True, let the reverse reaction proceed.
                    Otherwise, let the forwards reaction proceed.

            Returns:
                None
            """
    if rxn_ind == -1:
        raise RuntimeError("Incorrect reaction index when updating state")
    if reverse:
        for reactant_id in products[rxn_ind, :]:
            if reactant_id == -1:
                continue
            else:
                state[reactant_id] -= 1
                if state[reactant_id] < 0:
                    raise ValueError("State invalid! Negative specie: {}!")
        for product_id in reactants[rxn_ind, :]:
            if product_id == -1:
                continue
            else:
                state[product_id] += 1
    else:
        for reactant_id in reactants[rxn_ind, :]:
            if reactant_id == -1:
                continue
            else:
                state[reactant_id] -= 1
                if state[reactant_id] < 0:
                    raise ValueError("State invalid! Negative specie: {}!")
        for product_id in products[rxn_ind, :]:
            if product_id == -1:
                continue
            else:
                state[product_id] += 1
    return state

@jit(nopython = True)
def get_coordination(reactants, products, state, rxn_id, reverse):
    """
    Calculate the coordination number for a particular reaction, based on the reaction type
    number of molecules for the reactants.

    Args:
        rxn_id (int): index of the reaction chosen in the array [R1f, R1r, R2f, R2r, ...]
        reverse (bool): If True, give the propensity for the reverse
            reaction. If False, give the propensity for the forwards
            reaction.

    Returns:
        propensity (float)
    """
    if reverse:
        reactant_array = products[rxn_id, :] # Numpy array of reactant molecule IDs
        num_reactants = len(np.where(reactant_array != -1)[0])
    else:
        reactant_array = reactants[rxn_id, :]
        num_reactants = len(np.where(reactant_array != -1)[0])

    num_mols_list = list()
    for reactant_id in reactant_array:
        num_mols_list.append(state[reactant_id])

    if num_reactants == 1:
        h_prop = num_mols_list[0]
    elif (num_reactants == 2) and (reactant_array[0] == reactant_array[1]):
        h_prop = num_mols_list[0] * (num_mols_list[0] - 1) / 2
    elif (num_reactants == 2) and (reactant_array[0] != reactant_array[1]):
        h_prop = num_mols_list[0] * num_mols_list[1]
    else:
        raise RuntimeError("Only single and bimolecular reactions supported by this simulation")

    return h_prop

def plot_trajectory(initial_state_dict, products, reactants, reaction_history, times, num_label, filename, iteration):
    """ Given lists of reaction history and time steps, iterate through and plot the data """
    cumulative_time = list(np.cumsum(np.array(times)))
    state = initial_state_dict
    state_to_plot = dict()
    for mol_id in initial_state_dict:
        state_to_plot[mol_id] = [(0.0, initial_state_dict[mol_id])]
    total_iterations = len(reaction_history)

    for iter in range(total_iterations):
        this_rxn_ind = reaction_history[iter]
        converted_ind = math.floor(this_rxn_ind/2)
        t = cumulative_time[iter]

        if this_rxn_ind % 2: # update state dicts for reverse reaction
            for rid in products[converted_ind, :]:
                if rid == -1:
                    continue
                else:
                    try:
                        state[rid] -= 1
                        if state[rid] < 0:
                            raise ValueError("State invalid: negative specie: {}".format(rid))
                        state_to_plot[rid].append((t, state[rid]))
                    except KeyError:
                        raise ValueError("Reactant specie {} given is not in state!".format(rid))
            for pid in reactants[converted_ind, :]:
                if pid == -1:
                    continue
                else:
                    if (pid in state) and (pid in state_to_plot):
                        state[pid] += 1
                        state_to_plot[pid].append((t, state[pid]))
                    else:
                        state[pid] = 1
                        state_to_plot[pid] = [(0.0, 0), (t, state[pid])]

        else: # Update state dicts for forward reaction
            for rid in reactants[converted_ind, :]:
                if rid == -1:
                    continue
                else:
                    try:
                        state[rid] -= 1
                        if state[rid] < 0:
                            raise ValueError("State invalid: negative specie: {}".format(rid))
                        state_to_plot[rid].append((t, state[rid]))
                    except KeyError:
                        raise ValueError("Specie {} given is not in state!".format(rid))
            for pid in products[converted_ind, :]:
                if pid == -1:
                    continue
                elif (pid in state) and (pid in state_to_plot):
                    state[pid] += 1
                    state_to_plot[pid].append((t, state[pid]))
                else:
                    state[pid] = 1
                    state_to_plot[pid] = [(0.0, 0), (t, state[pid])]

    t_end = t
    # Sorting and plotting:
    fig, ax = plt.subplots()

    sorted_ids = sorted([(k, v) for k, v in state.items()], key = lambda x: x[1], reverse = True)
    sorted_ids = [mol_tuple[0] for mol_tuple in sorted_ids]

    colors = plt.cm.get_cmap('hsv', num_label)
    this_id = 0

    for mol_id in state_to_plot:
        ts = np.append(np.array([e[0] for e in state_to_plot[mol_id]]), t_end)

        nums = np.append(np.array([e[1] for e in state_to_plot[mol_id]]), state_to_plot[mol_id][-1][1])
        if mol_id in sorted_ids[0:num_label]:
            for entry in reaction_network.entries_list:
                if mol_id == entry.entry_id:
                    this_composition = entry.molecule.composition.alphabetical_formula
                    this_charge = entry.molecule.charge
                    this_label = this_composition + " " + str(this_charge)
                    this_color = colors(this_id)
                    this_id += 1
                    break

            ax.plot(ts, nums, label=this_label, color=this_color)
        else:
            ax.plot(ts, nums)

    title = "KMC simulation, total time {}".format(cumulative_time[-1])
    ax.set(title=title,
           xlabel="Time (s)",
           ylabel="# Molecules")
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1),
              ncol=2, fontsize="small")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def time_analysis(time_array):
    time_dict = dict()
    time_dict["t_avg"] = np.average(time_array)
    time_dict["t_std"] = np.std(time_array)
    time_dict["steps"] = len(time_array)
    time_dict["total_t"] = (time_array)[-1] # minutes
    return time_dict

"""Script to run the simulation"""
for vol in volumes_list:
    for t_steps in timesteps_list:
        for iter in range(iterations):
            file_name = "li_limited_t_" + str(t_steps) + "_V_" + str(vol) + "_ea_10000_Numba_Run" + str(iter + 1)
            [initial_state_dict, initial_state, species_rxn_mapping, reactants, products, coord_array, rate_constants] = initialize_simulation(reaction_network,
                                                                                                                        num_species, id_list, conc_list, vol)
            state = np.array(initial_state, dtype = int)
            propensity_array = np.multiply(coord_array, rate_constants)
            total_propensity = np.sum(propensity_array)
            print("Initial total propensity = ", total_propensity)
            t1 = time.time()
            data = simulate(t_steps, coord_array, rate_constants, propensity_array,
                 total_propensity, species_rxn_mapping, reactants, products, state) # Data in lists
            t2 = time.time()
            sim_time = (t2 - t1)/60
            reaction_history = data[0, :]
            times = data[1, :]
            t_end = np.sum(times)
            print("Final simulated time: ", t_end)
            print("Simulation time (min): ", sim_time)
            t3 = time.time()
            plot_trajectory(initial_state_dict, products, reactants, reaction_history, times, num_label, file_name, iter)
            t4 = time.time()
            print("Plotting time (sec): ", t4 - t3)