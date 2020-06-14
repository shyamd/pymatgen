# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import N_A
# import time


__author__ = "Ronald Kam, Evan Spotte-Smith, Xiaowei Xie"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"


class KineticMonteCarloSimulator:
    """
    Class for stochastic kinetic Monte Carlo simulation, with reactions provided
    by the Reactions present in a ReactionNetwork.

    Method is described by Gillespie (1976).

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

        # 2x because forwards and reverse reactions considered separately
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
