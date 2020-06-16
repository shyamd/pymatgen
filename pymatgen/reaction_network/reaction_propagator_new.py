# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, k, R, N_A, pi
import time
from numba import jit


__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"
__credit__ = "Xiaowei Xie"



class ReactionPropagator:
    """
    Class for stochastic kinetic Monte Carlo simulation, with reactions provided
    by the Reactions present in a ReactionNetwork.
    Method is described by Gillespie (1976)

    Args:
        products: list of np.arrays, containing product ids of each forward reaction; length = # rxns
        reactants: list of np.arrays, containing reactant ids of each forward reaction; length = # rxns
        initial_state (array): Integer amounts of each species present; length = # molecule entries in ReactionNetwork
        rate_constants (array): rate constants of all forward, rev reactions [k1f k1r k2f k2r ... ]
        coord_array (array): initial coordination number of all forward, rev reactions
        species_rxn_mapping (list): list of arrays, each containing reaction indeces for each species
        volume (float): Volume in Liters (default = 1 nm^3 = 1 * 10^-24 L)

        General note:
        - Current indexing of species assumes that molecule IDs are integers, and correspond to index of a species list that Rxn Network is generated from
        - Indexing of reactions are taken from list of reactions that result from Reaction Network generation

    """
    def __init__(self, products, reactants, initial_state, rate_constants, coord_array, species_rxn_mapping, volume=1.0*10**-24):
        self.products = products
        self.reactants = reactants
        self.species_rxn_mapping = species_rxn_mapping
        self.num_species = len(self.species_rxn_mapping)
        self.num_rxns = len(self.products)
        self.rxn_ind = np.arange(2 * self.num_rxns)
        self.rate_constants = rate_constants
        self.coord_array = coord_array
        self.propensity_array = np.multiply(self.rate_constants, self.coord_array)
        self.total_propensity = np.sum(self.propensity_array)
        self.initial_state = initial_state
        self.state = self.initial_state
        self.volume = volume

        # Variables to update during simulate()
        self.times = list()
        self.reaction_history = list()

        # self._state = list()
        # self.initial_state = list()

        ## State will have number of molecules, instead of concentration
        # for molecule_id, concentration in self.initial_state_conc.items():
        #     num_mols = int(concentration * self.volume * N_A  *1000) # volume in m^3
        #     self.initial_state[molecule_id] = num_mols
        #     self._state[molecule_id] = num_mols
        """Initial loop through all reactions in network: make arrays for initial propensity calculation. 
        The rate constant array [k1f k1r k2f k2r ... ], other arrays indexed in same fashion.
        Also create a "mapping" of each species to its respective reaction it is involved in, for future convenience"""
        # self.reactions = dict()
        # self.rate_constants = np.zeros(2*self.num_rxns)
        # self.coord_array = np.zeros(2*self.num_rxns)
        # self.rxn_ind = np.arange(2 * self.num_rxns)
        # self.species_rxn_mapping = dict() ## associating reaction index to each molecule
        # for id, reaction in enumerate(self.reaction_network.reactions):
        #     self.reactions[id] = reaction
        #     self.rate_constants[2*id] = reaction.rate_constant()["k_A"]
        #     self.rate_constants[2* id + 1] = reaction.rate_constant()["k_B"]
        #     num_reactants_for = list()
        #     num_reactants_rev = list()
        #     for reactant in reaction.reactants:
        #         num_reactants_for.append(self.initial_state.get(reactant.entry_id, 0))
        #         if reactant.entry_id not in self.species_rxn_mapping:
        #             self.species_rxn_mapping[reactant.entry_id] = [2*id]
        #         else:
        #             self.species_rxn_mapping[reactant.entry_id].append(2*id)
        #     for product in reaction.products:
        #         num_reactants_rev.append(self.initial_state.get(product.entry_id, 0))
        #         if product.entry_id not in self.species_rxn_mapping:
        #             self.species_rxn_mapping[product.entry_id] = [2 * id + 1]
        #         else:
        #             self.species_rxn_mapping[product.entry_id].append(2 * id + 1)
        #
        #     ## Obtain coordination value for forward reaction
        #     if len(reaction.reactants) == 1:
        #         self.coord_array[2*id] = num_reactants_for[0]
        #     elif (len(reaction.reactants) == 2) and (reaction.reactants[0] == reaction.reactants[1]):
        #         self.coord_array[2*id] = num_reactants_for[0] * (num_reactants_for[0] - 1)
        #     elif (len(reaction.reactants) == 2) and (reaction.reactants[0] != reaction.reactants[1]):
        #         self.coord_array[2 * id] = num_reactants_for[0] * num_reactants_for[1]
        #     else:
        #         raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
        #     # For reverse reaction
        #     if len(reaction.products) == 1:
        #         self.coord_array[2*id+1] = num_reactants_rev[0]
        #     elif (len(reaction.products) == 2) and (reaction.products[0] == reaction.products[1]):
        #         self.coord_array[2*id+1] = num_reactants_rev[0] * (num_reactants_rev[0] - 1)
        #     elif (len(reaction.products) == 2) and (reaction.products[0] != reaction.products[1]):
        #         self.coord_array[2 * id+1] = num_reactants_rev[0] * num_reactants_rev[1]
        #     else:
        #         raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
        # self.propensity_array = np.multiply(self.rate_constants, self.coord_array)
        # self.total_propensity = np.sum(self.propensity_array)
        # print("Initial total propensity = ", self.total_propensity)
        # self.data = {"times": list(),
        #              "reactions": list(),
        #              "state": dict()}

    # @property
    # def state(self):
    #     return self._state
    @jit(nopython = True)
    def get_coordination(self, rxn_id, reverse):
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
        #rate_constant = reaction.rate_constant()
        if reverse:
            reactant_array = self.products[rxn_id] # Numpy array of reactant molecule IDs
            num_reactants = len(reactant_array)

        else:
            reactant_array = self.reactants[rxn_id]
            num_reactants = len(reactant_array)

        num_mols_list = list()
        #entry_ids = list() # for testing
        for reactant_id in reactant_array:
            num_mols_list.append(self.state[reactant_id])
            #entry_ids.append(reactant.entry_id)
        if num_reactants == 1:
            h_prop = num_mols_list[0]
        elif (num_reactants == 2) and (reactant_array[0] == reactant_array[1]):
            h_prop = num_mols_list[0] * (num_mols_list[0] - 1) / 2
        elif (num_reactants == 2) and (reactant_array[0] != reactant_array[1]):
            h_prop = num_mols_list[0] * num_mols_list[1]
        else:
            raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
        #propensity = h_prop * self.rate_constants[reaction_ind]
        #propensity = h_prop * k
        return h_prop
        # for testing:
        #return [reaction.reaction_type, reaction.reactants, reaction.products, reaction.rate_calculator.alpha , reaction.transition_state, "propensity = " + str(propensity), "free energy from code = " + str(reaction.free_energy()["free_energy_A"]), "calculated free energy ="  + str(-sum([r.free_energy() for r in reaction.reactants]) +  sum([p.free_energy() for p in reaction.products])),
                #"calculated k = " +  str(k_b * 298.15 / h * np.exp(-1 * (-sum([r.free_energy() for r in reaction.reactants]) +  sum([p.free_energy() for p in reaction.products]) ) * 96487 / (R * 298.15))), "k from Rxn class = " +  str(k)  ]

    jit(nopython = True)
    def update_state(self, rxn_ind, reverse):
        """ Update the system based on the reaction chosen
        Args:
            reaction (Reaction)
            reverse (bool): If True, let the reverse reaction proceed.
                Otherwise, let the forwards reaction proceed.

        Returns:
            None
        """
        if reverse:
            for reactant_id in self.products[rxn_ind]:
                self.state[reactant_id] -= 1
            for product_id in self.reactants[rxn_ind]:
                self.state[product_id] += 1

        else:
            for reactant_id in self.reactants[rxn_ind]:
                self.state[reactant_id] -= 1
            for product_id in self.products[rxn_ind]:
                self.state[product_id] += 1

    # def alter_rxn_by_product(self, product_id, k_factor_change, reaction_classes = None):
    #     """Alter the rate constant of a reaction, based on the product(s) formed. For example, decreasing the rate
    #     constant for reactions that form undesired/unstable products. The change of k is directly proportional
    #     to the probability of reaction firing.
    #
    #     Args:
    #         product (molecule entry id):
    #         k_magnitude_change (float): factor of change desired to be made to rate constant
    #         reaction_classes (list): type of reactions to consider
    #
    #     Returns:
    #         altered self.rate_constants
    #     """
    #     # search for rxns that form product
    #     # change the corresponding rate constants based on the factor of change
    #     new_k = k * 298.15 / h * k_factor_change
    #     rxn_update = self.species_rxn_mapping[product_id]
    #     # Create the list of rxn ind to update k for
    #     for ind in rxn_update:
    #         this_rxn = self.reactions[math.floor(ind / 2)]
    #         if reaction_classes is None:
    #             if ind % 2:
    #                 self.rate_constants[ind - 1] = new_k
    #             else:
    #                 self.rate_constants[ind + 1] = new_k
    #         else:
    #             for rxn_class in reaction_classes:
    #                 if this_rxn.rate_type["class"] == rxn_class:
    #                     if ind % 2:
    #                         self.rate_constants[ind - 1] = new_k
    #                     else:
    #                         self.rate_constants[ind + 1] = new_k
    #     return self.rate_constants
    #
    # @staticmethod
    # @jit(nopython = True)
    # def calculate_tau(random_1, total_prop):
    #     return -np.log(random_1) / total_prop
    #
    # @staticmethod
    # @jit(nopython = True)
    # def choose_rxn(propensity, prop_array, index_array):
    #     return index_array[np.where(np.cumsum(prop_array) >= propensity)[0][0]]
    #
    # @staticmethod
    # @jit(nopython = True)
    # def calculate_total_prop(rate_constant_array, coordination_array):
    #     propensity_array = np.multiply(rate_constant_array, coordination_array)
    #     total_propensity = np.sum(propensity_array)
    #     return [propensity_array, np.array([total_propensity])]

    @jit(nopython = True)
    def simulate(self, t_end):
        """
        Main body code of the KMC simulation. Propagates time and updates species amounts.
        Store reactions, time, and time step for each iteration
        Args:
            t_end: (float) ending time of simulation

        Returns
            final state of molecules
        """
        # If any change have been made to the state, revert them
        #self._state = self.initial_state
        t = 0.0
        # self.data = {"times": list(),
        #              "reaction_ids": list(),
        #              "state": dict()}

        #
        # for mol_id in self._state.keys():
        #     self.data["state"][mol_id] = [(0.0, self._state[mol_id])]

        step_counter = 0
        self.state_history = [[0.0, self.state[mol_id]] for mol_id in range(self.num_species)]
        while t < t_end:
            step_counter += 1
            r1 = random.random()
            r2 = random.random()
            ## Obtaining a time step tau from the probability distrubution
            ## P(t) = a*exp(-at) --> probability that any reaction occurs at time t

            tau = -np.log(r1) / self.total_propensity
            #tau = self.calculate_tau(r1, self.total_propensity)

            ## Choosing a reaction mu; need a cumulative sum of rxn propensities
            ## Discrete probability distrubution of reaction choice
            # time_start = time.time()
            random_propensity = r2 * self.total_propensity

            reaction_choice_ind = self.rxn_ind[np.where(np.cumsum(self.propensity_array) >=  random_propensity)[0][0]]
            converted_rxn_ind = math.floor(reaction_choice_ind / 2)
            #reaction_choice_ind = self.choose_rxn(random_propensity, self.propensity_array, self.rxn_ind)

            #reaction_mu = self.reactions[math.floor(reaction_choice_ind / 2 )]

            if reaction_choice_ind % 2:
                reverse = True
            else:
                reverse = False
            # time_end = time.time()
            # print("Time to choose reaction = ", time_end - time_start)

            self.update_state(converted_rxn_ind, reverse)
            # time_start = time.time()
            reactions_to_change = list()
            for reactant_id in self.reactants[converted_rxn_ind]:
                reactions_to_change.extend(list(self.species_rxn_mapping[reactant_id]))
            for product_id in self.products[converted_rxn_ind]:
                reactions_to_change.extend(list(self.species_rxn_mapping[product_id]))

            reactions_to_change = set(reactions_to_change)

            for rxn_ind in reactions_to_change:
                if rxn_ind % 2:
                    this_reverse = True
                else:
                    this_reverse = False
                this_h = self.get_coordination(rxn_ind, this_reverse)
                self.coord_array[rxn_ind] = this_h

            self.propensity_array = np.multiply(self.rate_constants, self.coord_array)
            self.total_propensity = np.sum(self.propensity_array)
            #[self.propensity_array, total_prop_array] = self.calculate_total_prop(self.rate_constants, self.coord_array)
            #self.total_propensity = total_prop_array[0]

            # time_end = time.time()
            # print("Total prop = ", self.total_propensity)
            # print("Time to calculate total propensity = ", time_end - time_start)

            self.times.append(tau)
            # self.data["reaction_ids"].append({"reaction": reaction_mu, "reverse": reverse})
            self.reaction_history.append(reaction_choice_ind)

            t += tau
            #print(t)
            if reverse:
                for reactant_id in self.products[converted_rxn_ind]:
                    self.state_history[reactant_id].append((t, self.state[reactant_id]))
                for product_id in self.reactants[converted_rxn_ind]:
                    self.state_history[product_id].append((t, self.state[product_id]))

            else:
                for reactant_id in self.reactants[converted_rxn_ind]:
                    self.state_history[reactant_id].append((t, self.state[reactant_id]))

                for product_id in self.products[converted_rxn_ind]:
                    self.state_history[product_id].append((t, self.state[product_id]))

        # for mol_id in self.data["state"]:
        #     self.data["state"][mol_id].append((t, self._state[mol_id]))

        # return self.data

    # def plot_trajectory(self, state_history = None, name=None, filename=None, num_label=10):
    #     """
    #     Plot KMC simulation data
    #
    #     Args:
    #         data (dict): Dictionary containing output from a KMC simulation run.
    #             Default is None, meaning that the data stored in this
    #             ReactionPropagator object will be used.
    #         name(str): Title for the plot. Default is None.
    #         filename (str): Path for file to be saved. Default is None, meaning
    #             pyplot.show() will be used.
    #         num_label (int): Number of most prominent molecules to be labeled.
    #             Default is 10
    #
    #     Returns:
    #         None
    #     """
    #
    #     fig, ax = plt.subplots()
    #
    #     if state_history is None:
    #         data = self.state_history
    #
    #     # To avoid indexing errors
    #     # if num_label > len(data["state"].keys()):
    #     #     num_label = len(data["state"].keys())
    #     if num_label > len(data):
    #         num_label = len(data)
    #
    #     # Sort by final concentration
    #     # We assume that we're interested in the most prominent products
    #     # ids_sorted = sorted([(k, v) for k, v in data["state"].items()],
    #     #                     key=lambda x: x[1][-1][-1])
    #     ids_sorted = sorted(range(len(data)), key = lambda k: data[k], reverse = True)
    #
    #     # ids_sorted = [i[0] for i in ids_sorted][::-1]
    #     print("top 15 species ids: ", ids_sorted[0:15])
    #     # Only label most prominent products
    #     colors = plt.cm.get_cmap('hsv', num_label)
    #     id = 0
    #     for mol_id in ids_sorted[0:num_label]:
    #         ts = np.array([e[0] for e in self.state_history[mol_id]])
    #         nums = np.array([e[1] for e in self.state_history[mol_id]])
    #         if mol_id in ids_sorted[0:num_label]:
    #             for entry in self.reaction_network.entries_list:
    #                 if mol_id == entry.entry_id:
    #                     this_composition = entry.molecule.composition.alphabetical_formula
    #                     this_charge = entry.molecule.charge
    #                     this_label = this_composition + " " + str(this_charge)
    #                     this_color = colors(id)
    #                     id +=1
    #                     #this_label = entry.entry_id
    #                     break
    #
    #             ax.plot(ts, nums, label = this_label, color = this_color)
    #         else:
    #             ax.plot(ts, nums)
    #     if name is None:
    #         title = "KMC simulation, total time {}".format(data["times"][-1])
    #     else:
    #         title = name
    #
    #     ax.set(title=title,
    #            xlabel="Time (s)",
    #            ylabel="# Molecules")
    #
    #     ax.legend(loc='upper right', bbox_to_anchor=(1, 1),
    #                 ncol=2, fontsize="small")
    #     # ax.legend(loc='best', bbox_to_anchor=(0.45, -0.175),
    #     #           ncol=5, fontsize="small")
    #
    #
    #     # if filename is None:
    #     #     plt.show()
    #     # else:
    #     #     fig.savefig(filename, dpi=600)
    #     if filename == None:
    #         plt.savefig("SimulationRun")
    #     else:
    #         plt.savefig(filename)
    #
    # def reaction_analysis(self, data = None):
    #     if data == None:
    #         data = self.data
    #     reaction_analysis_results = dict()
    #     reaction_analysis_results["endo_rxns"] = dict()
    #     rxn_count = np.zeros(2*self.num_rxns)
    #     endothermic_rxns_count = 0
    #     fired_reaction_ids = set(self.data["reaction_ids"])
    #     for ind in fired_reaction_ids:
    #         this_count = data["reaction_ids"].count(ind)
    #         rxn_count[ind] = this_count
    #         this_rxn = self.reactions[math.floor(ind/2)]
    #         if ind % 2: # reverse rxn
    #             if this_rxn.free_energy()["free_energy_B"] > 0: # endothermic reaction
    #                 endothermic_rxns_count += this_count
    #         else:
    #             if this_rxn.free_energy()["free_energy_A"] > 0: # endothermic reaction
    #                 endothermic_rxns_count += this_count
    #     reaction_analysis_results["endo_rxns"]["endo_count"] = endothermic_rxns_count
    #     sorted_rxn_ids = sorted(self.rxn_ind, key = lambda k: rxn_count[k], reverse = True)
    #     bar_rxns_labels = list()
    #     bar_rxns_count = list()
    #     bar_rxns_x = list()
    #     for i in range(15): # analysis on most frequent reactions
    #         this_rxn_id = sorted_rxn_ids[i]
    #         bar_rxns_x.append(str(this_rxn_id))
    #         this_reaction = self.reactions[math.floor(this_rxn_id / 2 )]
    #         reaction_analysis_results[this_rxn_id] = dict()
    #         reaction_analysis_results[this_rxn_id]["count"] = rxn_count[this_rxn_id]
    #         reaction_analysis_results[this_rxn_id]["reactants"] = list()
    #         reaction_analysis_results[this_rxn_id]["products"] = list()
    #         this_label = str()
    #         if this_rxn_id % 2: # reverse rxn
    #             reaction_analysis_results[this_rxn_id]["reaction_type"] = this_reaction.reaction_type()["rxn_type_B"]
    #             this_label += this_reaction.reaction_type()["rxn_type_B"]
    #             for reactant in this_reaction.products:
    #                 reaction_analysis_results[this_rxn_id]["reactants"].append((reactant.molecule.composition.alphabetical_formula , reactant.entry_id))
    #                 this_label += " " + reactant.molecule.composition.alphabetical_formula
    #             for product in this_reaction.reactants:
    #                 reaction_analysis_results[this_rxn_id]["products"].append((product.molecule.composition.alphabetical_formula , product.entry_id))
    #                 this_label += " " + product.molecule.composition.alphabetical_formula
    #             reaction_analysis_results[this_rxn_id]["rate constant"] = this_reaction.rate_constant()["k_B"]
    #         else: # forward rxn
    #             reaction_analysis_results[this_rxn_id]["reaction_type"] = this_reaction.reaction_type()["rxn_type_A"]
    #             this_label += this_reaction.reaction_type()["rxn_type_A"]
    #             for reactant in this_reaction.reactants:
    #                 reaction_analysis_results[this_rxn_id]["reactants"].append((reactant.molecule.composition.alphabetical_formula, reactant.entry_id))
    #                 this_label += " " + reactant.molecule.composition.alphabetical_formula
    #             for product in this_reaction.products:
    #                 reaction_analysis_results[this_rxn_id]["products"].append((product.molecule.composition.alphabetical_formula , product.entry_id))
    #                 this_label += " " + product.molecule.composition.alphabetical_formula
    #             reaction_analysis_results[this_rxn_id]["rate constant"] = this_reaction.rate_constant()["k_A"]
    #         bar_rxns_labels.append(this_label)
    #         bar_rxns_count.append(reaction_analysis_results[this_rxn_id]["count"])
    #     plt.figure()
    #     plt.bar(bar_rxns_x[:10], bar_rxns_count[:10])
    #     plt.xlabel("Reaction Index")
    #     plt.ylabel("Reaction Occurrence")
    #     plt.title("Top Reactions, total " + str(len(self.data["times"])) + " reactions")
    #     plt.savefig("li_limited_top_rxns")
    #     return reaction_analysis_results




