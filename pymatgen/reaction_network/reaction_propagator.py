# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import N_A
import time
from numba import jit
from numba import int64, float64
from numba.experimental import jitclass
import os

__author__ = "Ronald Kam, Evan Spotte-Smith, Xiaowei Xie"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

spec = [('products', int64[:, :]),
        ('reactants', int64[:, :]),
        ('initial_state', int64[:]),
        ('state', int64[:]),
        ('rate_constants', float64[:]),
        ('coord_array', float64[:]),
        ('species_rxn_mapping', int64[:, :]),
        ('volume', float64),
        ('num_species', int64),
        ('num_rxns', int64),
        ('rxn_ind', int64[:]),
        ('propensity_array', float64[:]),
        ('total_propensity', float64),
        ('state', int64[:]),
        ('times', float64[:]),
        ('reaction_history', int64[:])
        ]

"""Function-based KMC simulation for a reaction network, assuming spatial homogeneity. Simulation is performed without 
objects, such is required to improve performance with Numba. Algorithm is described by Gillespie (1976).
"""


def initialize_simulation(reaction_network, initial_cond, volume=10**-24):
    """Initial loop through reactions to create lists, mappings, and initial states needed for simulation without
    reaction network object.

    Args:
        reaction_network (ReactionNetwork): Fully generated reaction network
        initial_cond (dict): [mol_id: initial_conc [M] (float)]
        volume [m^3] (float)

    Returns:
        initial_state (list): Initial molecule amounts. Species indexing corresponds to reaction_network.entries_list
        initial_state_dict (dict): convert initial_cond to a dict of [initial_mol_ind: #molecules...]
        species_rxn_mapping (list of list): each species has a list of reaction (indexes) which they take part in
        molid_index_mapping (dict): mapping between species index and its Molecule entry id
                                    [molecule_index: molecule_entry_id]
        reactant_array (array) (n_rxns x 2): each row contains the reactant indexes of forward reaction
        products_array (array) (n_rxns x 2): each row contains the product indexes of forward reaction
        coord_array (array) (2*n_rxns x 1): coordination number for each forward and reverse reaction
                                            [c1_f, c1_r, c2_f, c2_r ...]
        rate_constants (array) (2*n_rxns x 1): rate constant of each for and rev reaction [k1_f, k1_r, k2_f, k2_r ...]
        propensities (array) (2*n_rxns x 1): reaction propensities, obtained by element-wise multiplication of
                                            coord_array and rate_constants
        molid_index_mapping (dict): [mol_id: mol_index (int) ... ]

    """
    num_rxns = len(reaction_network.reactions)
    num_species = len(reaction_network.entries_list)
    molid_index_mapping = dict()
    initial_state = [0 for i in range(num_species)]
    initial_state_dict = dict()

    for ind, mol in enumerate(reaction_network.entries_list):
        molid_index_mapping[mol.entry_id] = ind
        this_c = initial_cond.get(mol.entry_id, 0)
        this_mol_amt = int(volume * N_A * 1000 * this_c)
        initial_state[ind] = this_mol_amt
        if mol.entry_id in initial_cond:
            initial_state_dict[ind] = this_mol_amt

    species_rxn_mapping_list = [[] for j in range(num_species)]
    reactant_array = -1 * np.ones((num_rxns, 2), dtype=int)
    product_array = -1 * np.ones((num_rxns, 2), dtype=int)
    coord_array = np.zeros(2 * num_rxns)
    rate_constants = np.zeros(2 * num_rxns)
    for id, reaction in enumerate(reaction_network.reactions):
        num_reactants_for = list()
        num_reactants_rev = list()
        rate_constants[2 * id] = reaction.rate_constant()["k_A"]
        rate_constants[2 * id + 1] = reaction.rate_constant()["k_B"]
        for idx, react in enumerate(reaction.reactants):
            # for each reactant, need to find the corresponding mol_id with the index
            mol_ind = molid_index_mapping[react.entry_id]
            reactant_array[id, idx] = mol_ind
            species_rxn_mapping_list[mol_ind].append(2 * id)
            num_reactants_for.append(initial_state[mol_ind])

        for idx, prod in enumerate(reaction.products):
            mol_ind = molid_index_mapping[prod.entry_id]
            product_array[id, idx] = mol_ind
            species_rxn_mapping_list[mol_ind].append(2*id + 1)
            num_reactants_rev.append(initial_state[mol_ind])

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
    species_rxn_mapping = -1 * np.ones((num_species, max_mapping_length), dtype=int)

    for index, rxn_list in enumerate(species_rxn_mapping_list):
        this_map_length = rxn_mapping_lengths[index]
        if this_map_length == max_mapping_length:
            species_rxn_mapping[index, :] = rxn_list
        else:
            species_rxn_mapping[index, : this_map_length - max_mapping_length] = rxn_list
    propensities = np.multiply(coord_array, rate_constants)
    return [initial_state, initial_state_dict, species_rxn_mapping, reactant_array, product_array,
            coord_array, rate_constants, propensities, molid_index_mapping]


@jit(nopython=True, parallel=True)
def kmc_simulate(time_steps, coord_array, rate_constants, propensity_array,
                 species_rxn_mapping, reactants, products, state):

    """ KMC Simulation of reaction network and specified initial conditions.

    Args:
         time_steps (int): Number of time steps/iterations desired to run.
         coord_array (array): Numpy array containing coordination numbers of forward and reverse reactions.
                                [h1f, h1r, h2f, h2r, ...]
         rate_constants (array): Numpy array containing rate constants of forward and reverse reactions.
         propensity_array (array): Numpy array containing propensities of for and rev reactions.
         species_rxn_mapping (2d array): Contains all the reaction indexes that each species takes part in
         reactants (2d array): Species IDs corresponding to the reactants of each forward reaction
         products (2d array): Species IDs corresponding to products of each forward reaction
         state (array): Array containing molecular amounts of each species in the reaction network

    Returns:
        A (2 x time_steps) Numpy array. First row contains the indeces of reactions that occurred.
        Second row are the time steps generated at each iteration.
    """
    total_propensity = np.sum(propensity_array)
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
        reaction_history[step_counter] = int(reaction_choice_ind)
        times[step_counter] = tau

    return np.vstack((np.array(reaction_history), np.array(times)))


@jit(nopython=True)
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


@jit(nopython=True)
def get_coordination(reactants, products, state, rxn_id, reverse):
    """ Calculate the coordination number for a particular reaction, based on the reaction type
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
        reactant_array = products[rxn_id, :]
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


class KMC_data_analyzer:
    """
    Functions to analyze (function-based) KMC outputs from many simulation runs.

    """
    def __init__(self, reaction_network, molid_ind_mapping, initial_state_dict, products, reactants,
                 reaction_history, time_history):
        self.reaction_network = reaction_network
        self.molid_ind_mapping = molid_ind_mapping
        self.initial_state_dict = initial_state_dict
        self.products = products
        self.reactants = reactants
        self.reaction_history = reaction_history
        self.time_history = time_history

        self.num_sims = len(self.reaction_history)
        if self.num_sims != len(self.time_history):
            raise RuntimeError('Number of datasets for rxn history and time step history should be same!')
        print('products: ', self.products)
        print('reactants: ', self.reactants)

    def generate_time_dep_profiles(self):
        """
        Generate plottable time-dependent profiles of species and rxns from raw KMC output, obtain final states.
        :return dict containing species profiles, reaction profiles, and final states from each simulation.
                species_profiles: [ {mol_ind1: [(t0, n(t0)), (t1, n(t1)...], mol_ind2: ... ,  ... }, {...}, ... ]
                reaction_profiles: [ {rxn_ind1: [t0, t1, ...], rxn_ind2: ..., ...}, {...}, ...]
                final_states: [ {mol_ind1: n1, mol_ind2: ..., ...}, {...}, ...]

        """
        species_profiles = list()
        reaction_profiles = list()
        final_states = list()
        for n_sim in range(self.num_sims):
            sim_time_history = self.time_history[n_sim]  # array
            sim_rxn_history = self.reaction_history[n_sim]

            sim_species_profile = dict()
            sim_rxn_profile = dict()

            cumulative_time = list(np.cumsum(np.array(sim_time_history)))

            state = self.initial_state_dict

            for mol_ind in self.initial_state_dict:
                sim_species_profile[mol_ind] = [(0.0, self.initial_state_dict[mol_ind])]
            total_iterations = len(sim_rxn_history)

            for iter in range(total_iterations):
                rxn_ind = sim_rxn_history[iter]
                t = cumulative_time[iter]
                if rxn_ind not in sim_rxn_profile:
                    sim_rxn_profile[rxn_ind] = [t]
                else:
                    sim_rxn_profile[rxn_ind].append(t)
                converted_ind = math.floor(rxn_ind/2)

                if rxn_ind % 2:
                    for r_ind in self.products[converted_ind, :]:
                        if r_ind == -1:
                            continue
                        else:
                            try:
                                state[r_ind] -= 1
                                if state[r_ind] < 0:
                                    raise ValueError("State invalid: negative specie: {}".format(r_ind))
                                sim_species_profile[r_ind].append((t, state[r_ind]))
                            except KeyError:
                                raise ValueError("Reactant specie {} given is not in state!".format(r_ind))
                    for p_ind in self.reactants[converted_ind, :]:
                        if p_ind == -1:
                            continue
                        else:
                            if (p_ind in state) and (p_ind in sim_species_profile):
                                state[p_ind] += 1
                                sim_species_profile[p_ind].append((t, state[p_ind]))
                            else:
                                state[p_ind] = 1
                                sim_species_profile[p_ind] = [(0.0, 0), (t, state[p_ind])]

                else:
                    for r_ind in self.reactants[converted_ind, :]:
                        if r_ind == -1:
                            continue
                        else:
                            try:
                                state[r_ind] -= 1
                                if state[r_ind] < 0:
                                    raise ValueError("State invalid: negative specie: {}".format(r_ind))
                                sim_species_profile[r_ind].append((t, state[r_ind]))
                            except KeyError:
                                raise ValueError("Specie {} given is not in state!".format(r_ind))
                    for p_ind in self.products[converted_ind, :]:
                        if p_ind == -1:
                            continue
                        elif (p_ind in state) and (p_ind in sim_species_profile):
                            state[p_ind] += 1
                            sim_species_profile[p_ind].append((t, state[p_ind]))
                        else:
                            state[p_ind] = 1
                            sim_species_profile[p_ind] = [(0.0, 0), (t, state[p_ind])]

                species_profiles.append(sim_species_profile)
                reaction_profiles.append(sim_rxn_profile)
                final_states.append(state)

            return {'species_profiles': species_profiles, 'reaction_profiles': reaction_profiles,
                    'final_states': final_states}

    def plot_species_profiles(self, species_profiles, final_states, num_label, filename='KMC', file_dir = None):
        """
        Sorting and plotting species profiles for a specified number of simulations. The profiles might be very similar,
        so may not need to plot all of the runs for good understanding of results.

        Args:
            species_profiles (list of dicts): species as function of time, for each simulation
            final_states (list of dicts): final states of each simulation
            num_label (int): number of species in the legend
            filename (str)
            file_dir (str)
            num_plots (int): number of simulations desired to plot. If None, plot all the data.

        """
        num_plots = self.num_sims
        for n_sim in range(num_plots):
            # Sorting and plotting:
            fig, ax = plt.subplots()
            sorted_state = sorted([(k, v) for k, v in final_states[n_sim].items()], key=lambda x: x[1], reverse=True)
            sorted_inds = [mol_tuple[0] for mol_tuple in sorted_state]
            sorted_ind_id_mapping = dict()
            iter_counter = 0

            for id, ind in self.molid_ind_mapping.items():
                if ind in sorted_inds[:num_label]:
                    sorted_ind_id_mapping[ind] = id
                    iter_counter += 1
                if iter_counter == num_label:
                    break

            colors = plt.cm.get_cmap('hsv', num_label)
            this_id = 0
            t_end = sum(self.time_history[n_sim])

            for mol_ind in species_profiles[n_sim]:
                ts = np.append(np.array([e[0] for e in species_profiles[n_sim][mol_ind]]), t_end)
                nums = np.append(np.array([e[1] for e in species_profiles[n_sim][mol_ind]]),
                                 species_profiles[n_sim][mol_ind][-1][1])
                if mol_ind in sorted_inds[:num_label]:
                    mol_id = sorted_ind_id_mapping[mol_ind]
                    for entry in self.reaction_network.entries_list:
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

            title = "KMC simulation, total time {}".format(t_end)
            ax.set(title=title,
                   xlabel="Time (s)",
                   ylabel="# Molecules")
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1),
                      ncol=2, fontsize="small")

            sim_filename = filename + '_run_' + str(n_sim+1)
            if file_dir == None:
                plt.savefig(sim_filename)
            else:
                plt.savefig(file_dir + '/' + sim_filename)

    def identify_intermediates(self, species_profile, cutoff=0.1):
        """
        Identify intermediates from species vs time profile of one simulation. It is an intermediate if its final state
        contains less than 1/10 of its maximal amount. User can adjust this fraction.
        :param species_profile: (Dict of list of tuple)
        :param cutoff: (float) fraction to adjust definition of intermediate
        :return: list of identified intermediates, by molecule index
        """
        intermediates = list()
        for mol_ind in species_profile:
            history = [t[1] for t in species_profile[mol_ind]]
            if (max(history) > 20) and (history[-1] < cutoff*max(history)):
                intermediates.append(mol_ind)
        return intermediates

    def quantify_specific_reaction(self, reaction_profile, reaction_index):

        if reaction_index not in reaction_profile:
            raise RuntimeError('Reaction did not occur in this simulation run.')
        reaction_count = len(reaction_profile[reaction_index])

        return reaction_count

    def quantify_rank_reactions(self, reaction_profiles, reaction_type=None, num_rxns=None):
        """
        Given reaction history of a simulation, identify the most commonly occurring reactions.
        Can rank generally, or by reactions of a certain type.

         Args:
             reaction_profiles (list of dicts): reactions fired as a function of time
             reaction_type (string)
             num_rxns (int): the amount of reactions interested in collecting data on. If None, record for all.

         Returns:
             reaction_data (list of list of dict): each dict contains reaction index, count, reactant ids, product ids.
                                                    Each sim contains the list of dicts for each reaction fired, sorted
                                                    from highest count to lowest count.
         """

        if reaction_type != None:
            rxns_of_type = list()
            if (reaction_type != 'One electron reduction') or (reaction_type != 'One electron oxidation') or \
                    (reaction_type != 'Intramolecular single bond breakage') or \
                    (reaction_type != 'Intramolecular single bond formation') or \
                    (reaction_type != 'Coordination bond breaking AM -> A+M') or \
                    (reaction_type != 'Coordination bond forming A+M -> AM') or \
                    (reaction_type != 'Molecular decomposition breaking one bond A -> B+C') or \
                    (reaction_type != 'Molecular formation from one new bond A+B -> C') or \
                    (reaction_type != 'Concerted'):
                raise RuntimeError('This reaction type does not (yet) exist in our reaction networks.')

            for ind, rxn in enumerate(self.reaction_network.reactions):
                if rxn.reaction_type()['rxn_type_A'] == reaction_type:
                    rxns_of_type.append(2*ind)
                elif rxn.reaction_type()['rxn_type_B'] == reaction_type:
                    rxns_of_type.append(2*ind + 1)

        reaction_data = list()
        for n_sim in range(self.num_sims):
            num_fired_rxns = len(reaction_profiles[n_sim])
            if reaction_type != None:
                reaction_counts = [(int(rxn_ind), len(reaction_profiles[n_sim][rxn_ind]))
                                for rxn_ind in reaction_profiles[n_sim] if rxn_ind in rxns_of_type]
            else:
                reaction_counts = [(int(rxn_ind), len(reaction_profiles[n_sim][rxn_ind]))
                                   for rxn_ind in reaction_profiles[n_sim]]
            sorted_reaction_counts = sorted(reaction_counts, key=lambda x: x[1], reverse=True)
            print('sorted_rxn_count: ', sorted_reaction_counts)
            sim_reaction_data = list()

            if num_rxns == None:
                num_rxns = num_fired_rxns
            elif num_rxns > num_fired_rxns:
                num_rxns = num_fired_rxns

            for rxn_ind, count in sorted_reaction_counts[:num_rxns]:
                this_data = dict()
                this_data['reaction_index'] = rxn_ind
                this_data['count'] = count
                this_data['reactants'] = list()
                this_data['products'] = list()
                if rxn_ind % 2:
                    react_inds = self.products[rxn_ind, :]
                    prod_inds = self.reactants[rxn_ind, :]
                else:
                    react_inds = self.reactants[rxn_ind, :]
                    prod_inds = self.products[rxn_ind, :]

                for r_ind in react_inds:
                    this_data['reactants'].append(r_ind)

                for p_ind in prod_inds:
                    this_data['products'].append(p_ind)

                sim_reaction_data.append(this_data)

            reaction_data.append(sim_reaction_data)

            # Then run averages and std dev of all runs, and rank each reaction
        return reaction_data

    def time_step_analysis(self, sim_time_array):
        t_avg = np.average(sim_time_array)
        t_std = np.std(sim_time_array)
        log_time = np.log10(sim_time_array)
        logt_avg = np.average(log_time)
        logt_std = np.std(log_time)

        # loop through and identify "initial cascade"

        for step_num, log_tau in enumerate(log_time):
            if log_tau >= logt_avg - logt_std:
                if all(log_time[step_num-5:step_num] >= logt_avg - logt_std):
                    # ensure we are in steady region after cascade
                    t_cascade = 10**log_tau
                    break
        for log_tau in log_time[step_num:]:
            if log_tau > 2*logt_std + logt_avg:
                t_steady_state = 10**log_tau
                break

        return {'t_avg': t_avg, 't_std': t_std, 'logt_avg': logt_avg, 'logt_std': logt_std, 't_cascade': t_cascade,
                't_steady_state': t_steady_state, 'steps': len(sim_time_array), 'total_t': sim_time_array[-1]}

    def reaction_frequency_analysis(self, time_step_analysis, rxn_ind, n=30):
        """
        Calculate the frequency of reaction as a function of time. Simulation data is discretized into size of n steps,
        and probability of reaction occurring in this set is obtained.

        :param time_step_analysis:
        :param rxn_ind:
        :param n: discretizing time by the number of steps n, in calculating reaction frequencies
        :return:

        """
        t_avg = time_step_analysis['t_avg']
        t_cascade = time_step_analysis['t_cascade']
        frequency_data = list()
        for n_sim in self.num_sims:
            cascade_end = np.where(self.time_history[n_sim] >= t_cascade)[0][0]
            cascade_freq = np.count_nonzero(self.reaction_history[n_sim][:cascade_end] == rxn_ind) / cascade_end
            intervals = np.arange(0, len(self.time_history[n_sim]), step=n)  # discretize time into sizes of n steps
            sim_rxn_frequencies = dict()
            freq_list = list()
            for i in range(len(intervals[1:])):
                rxn_freq = np.count_nonzero(self.reaction_history[n_sim][intervals[i] - intervals[i-1]]) / n
                t_mdpt = (self.time_history[n_sim][intervals[i]] + self.time_history[n_sim][intervals[i-1]]) / 2
                freq_list.append((t_mdpt, rxn_freq))

            last_rxn_freq = np.count_nonzero(self.reaction_history[n_sim][intervals[i]:])
            last_t_mdpt = (self.time_history[n_sim][intervals[i]] + self.time_history[n_sim][intervals[-1]]) / 2
            freq_list.append((last_t_mdpt, last_rxn_freq))

            sim_rxn_frequencies['freq_data'] = freq_list
            sim_rxn_frequencies['cascade_frequency'] = cascade_freq
            frequency_data.append(sim_rxn_frequencies)

        return frequency_data

    def find_rxn_index(self, reaction, reverse):

        for ind, rxn in enumerate(self.reaction_network.reactions):
            if rxn == reaction:
                if reverse == True:
                    rxn_ind = 2*ind + 1
                else:
                    rxn_ind = 2*ind
                break

        return rxn_ind


class KineticMonteCarloSimulator:
    """
    Class for stochastic kinetic Monte Carlo simulation of ReactionNetwork.
    Object-oriented version of the KMC simulator; higher computational cost compared to function-based, but the
    algorithm is identical (Gillespie 1976). Involves less use of reference Numpy arrays, as ReactionNetwork objects
    can be called.

    Args:
        reaction_network (ReactionNetwork)
        initial_state (dict): {Molecule ID (int): concentration (float)}}
        volume (float): Volume in Liters (default = 1 nm^3 = 1 * 10^-24 L)
        temperature (float): Temperature in Kelvin

    """
    def __init__(self, reaction_network, initial_state, volume=1.0*10**-24,
                 temperature=298.15):
        self.reaction_network = reaction_network

        self.num_rxns = len(self.reaction_network.reactions)

        self.volume = volume
        self.temperature = temperature

        self._state = dict()
        self.initial_state = dict()

        # Convert initial state from concentrations to numbers of molecules
        for molecule_id, concentration in initial_state.items():
            num_mols = int(concentration * self.volume * N_A * 1000)  # volume in m^3
            self.initial_state[molecule_id] = num_mols
            self._state[molecule_id] = num_mols

        # Initialize arrays for propensity calculation.
        # The rate constant array [k1f k1r k2f k2r ... ], other arrays indexed in same fashion.
        # Also create a "mapping" of each species to reactions it is involved in, for future convenience
        self.reactions = dict()

        self.rate_constants = np.zeros(2 * self.num_rxns)
        self.coord_array = np.zeros(2 * self.num_rxns)
        self.rxn_ind = np.arange(2 * self.num_rxns)

        self.species_rxn_mapping = dict()  # associating reaction index to each molecule
        for rid, reaction in enumerate(self.reaction_network.reactions):
            self.reactions[rid] = reaction
            self.rate_constants[2 * rid] = reaction.rate_constant(temperature=temperature)["k_A"]
            self.rate_constants[2 * rid + 1] = reaction.rate_constant(temperature=temperature)["k_B"]
            num_reactants_for = list()
            num_reactants_rev = list()
            for reactant in reaction.reactants:
                num_reactants_for.append(self.initial_state.get(reactant.entry_id, 0))
                if reactant.entry_id not in self.species_rxn_mapping:
                    self.species_rxn_mapping[reactant.entry_id] = [2 * rid]
                else:
                    self.species_rxn_mapping[reactant.entry_id].append(2 * rid)
            for product in reaction.products:
                num_reactants_rev.append(self.initial_state.get(product.entry_id, 0))
                if product.entry_id not in self.species_rxn_mapping:
                    self.species_rxn_mapping[product.entry_id] = [2 * rid + 1]
                else:
                    self.species_rxn_mapping[product.entry_id].append(2 * rid + 1)

            # Obtain coordination value for forward reaction
            self.coord_array[2 * rid] = self.get_coordination(reaction, False)

            # For reverse reaction
            self.coord_array[2 * rid + 1] = self.get_coordination(reaction, True)

        self.propensity_array = np.multiply(self.rate_constants, self.coord_array)
        self.total_propensity = np.sum(self.propensity_array)
        print("Initial total propensity = ", self.total_propensity)
        self.data = {"times": list(),
                     "reactions": list(),
                     "state": dict()}

    @property
    def state(self):
        return self._state

    def get_coordination(self, reaction, reverse):
        """
        Calculate the coordination number for a particular reaction, based on the reaction type
        number of molecules for the reactants.

        Args:
            reaction (Reaction)
            reverse (bool): If True, give the propensity for the reverse
                reaction. If False, give the propensity for the forwards
                reaction.

        Returns:
            propensity (float)
        """

        if reverse:
            num_reactants = len(reaction.products)
            reactants = reaction.products
        else:
            num_reactants = len(reaction.reactants)
            reactants = reaction.reactants

        num_mols_list = list()
        for reactant in reactants:
            reactant_num_mols = self.state.get(reactant.entry_id, 0)
            num_mols_list.append(reactant_num_mols)
        if num_reactants == 1:
            h_prop = num_mols_list[0]
        elif (num_reactants == 2) and (reactants[0].entry_id == reactants[1].entry_id):
            h_prop = num_mols_list[0] * (num_mols_list[0] - 1) / 2
        elif (num_reactants == 2) and (reactants[0].entry_id != reactants[1].entry_id):
            h_prop = num_mols_list[0] * num_mols_list[1]
        else:
            raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
        return h_prop

    def update_state(self, reaction, reverse):
        """ Update the system state dictionary based on a chosen reaction

        Args:
            reaction (Reaction)
            reverse (bool): If True, let the reverse reaction proceed.
                Otherwise, let the forwards reaction proceed.

        Returns:
            None
        """
        if reverse:
            for reactant in reaction.products:
                try:
                    self._state[reactant.entry_id] -= 1
                    if self._state[reactant.entry_id] < 0:
                        raise ValueError("State invalid! Negative specie: {}!".format(reactant.entry_id))
                except KeyError:
                    raise ValueError("Specie {} given is not in state!".format(reactant.entry_id))
            for product in reaction.reactants:
                p_id = product.entry_id
                if p_id in self.state:
                    self._state[p_id] += 1
                else:
                    self._state[p_id] = 1
        else:
            for reactant in reaction.reactants:
                try:
                    self._state[reactant.entry_id] -= 1
                    if self._state[reactant.entry_id] < 0:
                        raise ValueError("State invalid! Negative specie: {}!".format(reactant.entry_id))
                except KeyError:
                    raise ValueError("Specie {} given is not in state!".format(reactant.entry_id))
            for product in reaction.products:
                p_id = product.entry_id
                if p_id in self.state:
                    self._state[p_id] += 1
                else:
                    self._state[p_id] = 1
        return self._state

    def choose_reaction(self, rando):
        """
        Based on a random factor (between 0 and 1), select a reaction for the
            next time step.

        Args:
            rando: (float) Random number in the interval (0, 1)

        Return:
            ind: (int) index of the reaction to be chosen
        """

        random_propensity = rando * self.total_propensity
        ind = self.rxn_ind[np.where(np.cumsum(self.propensity_array) >= random_propensity)[0][0]]
        return ind

    def simulate(self, t_end):
        """
        Main body code of the KMC simulation. Propagates time and updates species amounts.
        Store reactions, time, and time step for each iteration

        Args:
            t_end: (float) ending time of simulation

        Return:
            self.data: (dict) complete state after simulation is complete
        """
        # If any change have been made to the state, revert them
        self._state = self.initial_state
        t = 0.0
        self.data = {"times": list(),
                     "reaction_ids": list(),
                     "state": dict()}

        for mol_id in self._state.keys():
            self.data["state"][mol_id] = [(0.0, self._state[mol_id])]

        step_counter = 0
        while t < t_end:
            step_counter += 1

            # Obtain reaction propensities, on which the probability distributions of
            # time and reaction choice depends.

            # drawing random numbers on uniform (0,1) distrubution
            r1 = random.random()
            r2 = random.random()
            # Obtaining a time step tau from the probability distrubution
            tau = -np.log(r1) / self.total_propensity

            # Choosing a reaction mu
            # Discrete probability distrubution of reaction choice
            reaction_choice_ind = self.choose_reaction(r2)
            reaction_mu = self.reactions[math.floor(reaction_choice_ind / 2)]

            if reaction_choice_ind % 2:
                reverse = True
            else:
                reverse = False

            # Update state
            self.update_state(reaction_mu, reverse)

            # Update reaction propensities only as needed
            reactions_to_change = list()
            for reactant in reaction_mu.reactants:
                reactions_to_change.extend(self.species_rxn_mapping[reactant.entry_id])
            for product in reaction_mu.products:
                reactions_to_change.extend(self.species_rxn_mapping[product.entry_id])

            reactions_to_change = set(reactions_to_change)

            for rxn_ind in reactions_to_change:
                if rxn_ind % 2:
                    this_reverse = True
                else:
                    this_reverse = False
                this_h = self.get_coordination(self.reactions[math.floor(rxn_ind / 2)],
                                               this_reverse)

                self.coord_array[rxn_ind] = this_h

            self.propensity_array = np.multiply(self.rate_constants, self.coord_array)
            self.total_propensity = np.sum(self.propensity_array)

            self.data["times"].append(tau)

            self.data["reaction_ids"].append(reaction_choice_ind)

            t += tau

            # Update data with the time step where the change in state occurred
            # Useful for subsequent analysis
            if reverse:
                for reactant in reaction_mu.products:
                    self.data["state"][reactant.entry_id].append((t, self._state[reactant.entry_id]))
                for product in reaction_mu.reactants:
                    if product.entry_id not in self.data["state"]:
                        self.data["state"][product.entry_id] = [(0.0, 0),
                                                                (t, self._state[product.entry_id])]
                    else:
                        self.data["state"][product.entry_id].append((t, self._state[product.entry_id]))

            else:
                for reactant in reaction_mu.reactants:
                    self.data["state"][reactant.entry_id].append((t, self._state[reactant.entry_id]))
                for product in reaction_mu.products:
                    if product.entry_id not in self.data["state"]:
                        self.data["state"][product.entry_id] = [(0.0, 0),
                                                                (t, self._state[product.entry_id])]
                    else:
                        self.data["state"][product.entry_id].append((t, self._state[product.entry_id]))

        for mol_id in self.data["state"]:
            self.data["state"][mol_id].append((t, self._state[mol_id]))

        return self.data

    def plot_trajectory(self, data=None, name=None, filename=None, num_label=10):
        """
        Plot KMC simulation data

        Args:
            data (dict): Dictionary containing output from a KMC simulation run.
                Default is None, meaning that the data stored in this
                ReactionPropagator object will be used.
            name(str): Title for the plot. Default is None.
            filename (str): Path for file to be saved. Default is None, meaning
                pyplot.show() will be used.
            num_label (int): Number of most prominent molecules to be labeled.
                Default is 10

        Returns:
            None
        """

        fig, ax = plt.subplots()

        if data is None:
            data = self.data

        # To avoid indexing errors
        if num_label > len(data["state"].keys()):
            num_label = len(data["state"].keys())

        # Sort by final concentration
        # We assume that we're interested in the most prominent products
        ids_sorted = sorted([(k, v) for k, v in data["state"].items()],
                            key=lambda x: x[1][-1][-1])
        ids_sorted = [i[0] for i in ids_sorted][::-1]

        # Only label most prominent products
        colors = plt.cm.get_cmap('hsv', num_label)
        this_id = 0

        for mol_id in data["state"]:
            ts = np.array([e[0] for e in data["state"][mol_id]])
            nums = np.array([e[1] for e in data["state"][mol_id]])
            if mol_id in ids_sorted[0:num_label]:
                for entry in self.reaction_network.entries_list:
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

        if name is None:
            title = "KMC simulation, total time {}".format(data["times"][-1])
        else:
            title = name

        ax.set(title=title,
               xlabel="Time (s)",
               ylabel="# Molecules")

        ax.legend(loc='upper right', bbox_to_anchor=(1, 1),
                  ncol=2, fontsize="small")

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
