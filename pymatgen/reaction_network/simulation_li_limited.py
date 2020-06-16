import numpy as np
from pymatgen.reaction_network.reaction_propagator_new import ReactionPropagator
from pymatgen.reaction_network.reaction_network import ReactionNetwork
import time
import matplotlib.pyplot as plt
import pickle
from scipy.constants import N_A

__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

class Simulation_Li_Limited:
    def __init__(self, file_name, li_conc = 1.0, ec_conc = 3.5706, emc_conc = 7.0555, volume = 10**-24, t_end = 1):
        """ Create an initial state and reaction network, in a Li system of ~ 3200 molecules.
        Typical electrolyte composition is 1M LiPF6, 3:7 wt% EC:EMC

        Args:
        li_conc (float): Li concentration
        ec_conc (float): Ethylene carbonate concentration
        emc_conc (float): Ethyl methyl carbonate
        volume (float): Volume in Liters (default = 1 nm^3 = 1 * 10^-24 L)
        t_end (float): end time of simulation
        """

        # Set up initial conditions, use baseline Li-ion electrolyte solution
        self.volume = volume ## m^3
        self.li_conc = li_conc # mol/L
        self.file_name = file_name
        self.ec_conc = ec_conc
        self.emc_conc = emc_conc
        self.t_end = t_end
        # Impurities
        self.h2o_conc = 1.665*10**-4 # 1-5 ppm
        #self.hf_conc = 2.70*10**-3 # 30-60 ppm

        # ref_ec = Molecule.from_file("ref_ec.xyz")
        # ref_ec = MoleculeGraph.with_local_env_strategy(ref_ec, OpenBabelNN())
        # ref_emc = Molecule.from_file("ref_emc.xyz")
        # ref_emc = MoleculeGraph.with_local_env_strategy(ref_emc, OpenBabelNN())
        # ref_h2o = Molecule.from_file("ref_h2o.xyz")
        # ref_h2o = MoleculeGraph.with_local_env_strategy(ref_h2o, OpenBabelNN())

        # Put entries in a list to make ReactionNetwork
        # self.entries = loadfn("mol_entries_limited_two.json")
        #
        # for ii, entry in enumerate(self.entries):
        #     entry.entry_id = ii
        # pickle_out = open("pickle_mol_entries_limited_two_IDs", "wb")
        # pickle.dump(self.entries, pickle_out)
        # pickle_out.close()

        # pickle_in = open("pickle_mol_entries_limited_two_IDs", "rb")
        # self.entries = pickle.load(pickle_in)
        #
        # self.reaction_network = ReactionNetwork.from_input_entries(self.entries, electron_free_energy = -2.15)
        # self.reaction_network.build()
        #
        # pickle_out = open("pickle_rxnnetwork_Li-limited", "wb")
        # pickle.dump(self.reaction_network, pickle_out)
        # pickle_out.close()

        pickle_in = open("pickle_rxnnetwork_Li-limited", "rb")
        self.reaction_network = pickle.load(pickle_in)
        li_id = 2335
        ec_id = 2606
        emc_id = 1877
        h2o_id = 3306

        conc_to_amt = lambda c:  int(c * self.volume * N_A * 1000)
        self.initial_state_dict = {li_id: conc_to_amt(self.li_conc), ec_id: conc_to_amt(self.ec_conc),
                                   emc_id: conc_to_amt(self.emc_conc), h2o_id: conc_to_amt(self.h2o_conc)}
        self.num_entries = 5732 # number of entries used to create reaction network, not equivalent to the number of entries in the network
        self.initial_state = np.zeros(self.num_entries)
        for initial_molecule_id in self.initial_state_dict:
            self.initial_state[initial_molecule_id] = self.initial_state_dict[initial_molecule_id]

        self.reactants = list()
        self.products = list()
        self.num_rxns = len(self.reaction_network.reactions)

        self.rate_constants = np.zeros(2 * self.num_rxns)
        self.coord_array = np.zeros(2 * self.num_rxns)
        self.rxn_ind = np.arange(2 * self.num_rxns) # [r1_f, r1_r, r2_f, r2_r, ... ]
        self.species_rxn_mapping = [np.array([]) for entry in range(self.num_entries)] # a list of arrays of reaction inds which molecule is reactant of
        for id, reaction in enumerate(self.reaction_network.reactions):
            this_reactant_list = list()
            this_product_list = list()
            num_reactants_for = list()
            num_reactants_rev = list()
            self.rate_constants[2 * id] = reaction.rate_constant()["k_A"]
            self.rate_constants[2 * id + 1] = reaction.rate_constant()["k_B"]
            for react in reaction.reactants:
                this_reactant_list.append(react.entry_id)
                self.species_rxn_mapping[react.entry_id] = np.append(self.species_rxn_mapping[react.entry_id], 2 * id)
                num_reactants_for.append(self.initial_state_dict.get(react.entry_id, 0))
            for prod in reaction.products:
                this_product_list.append(prod.entry_id)
                self.species_rxn_mapping[prod.entry_id] = np.append(self.species_rxn_mapping[prod.entry_id], 2*id + 1)
                num_reactants_rev.append(self.initial_state_dict.get(prod.entry_id, 0))

            self.reactants.append(np.array(this_reactant_list))
            self.products.append(np.array(this_product_list))

            ## Obtain coordination value for forward reaction
            if len(reaction.reactants) == 1:
                self.coord_array[2 * id] = num_reactants_for[0]
            elif (len(reaction.reactants) == 2) and (reaction.reactants[0] == reaction.reactants[1]):
                self.coord_array[2 * id] = num_reactants_for[0] * (num_reactants_for[0] - 1)
            elif (len(reaction.reactants) == 2) and (reaction.reactants[0] != reaction.reactants[1]):
                self.coord_array[2 * id] = num_reactants_for[0] * num_reactants_for[1]
            else:
                raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
            # For reverse reaction
            if len(reaction.products) == 1:
                self.coord_array[2 * id + 1] = num_reactants_rev[0]
            elif (len(reaction.products) == 2) and (reaction.products[0] == reaction.products[1]):
                self.coord_array[2 * id + 1] = num_reactants_rev[0] * (num_reactants_rev[0] - 1)
            elif (len(reaction.products) == 2) and (reaction.products[0] != reaction.products[1]):
                self.coord_array[2 * id + 1] = num_reactants_rev[0] * num_reactants_rev[1]
            else:
                raise RuntimeError("Only single and bimolecular reactions supported by this simulation")
        # print("rxn mapping", self.species_rxn_mapping)
        self.propensity_array = np.multiply(self.rate_constants, self.coord_array)
        self.total_propensity = np.sum(self.propensity_array)
        print("Initial total propensity = ", self.total_propensity)

        self.propagator = ReactionPropagator(self.products, self.reactants, self.initial_state, self.rate_constants,
                                             self.coord_array, self.species_rxn_mapping,  self.volume)

        time_start = time.time()
        #self.simulation_data = self.propagator.simulate(self.t_end)
        self.propagator.simulate(self.t_end)
        time_end = time.time()
        self.runtime = time_end - time_start
        print("Total simulation time is: ", self.runtime)


        #self.propagator.plot_trajectory(self.simulation_data,"Simulation Results", self.file_name)
        print("Final state is: ", self.propagator._state)
        # self.rxn_analysis = self.propagator.reaction_analysis()
        # print("Reaction Analysis")
        # for analysis_key in self.rxn_analysis:
        #     print(self.rxn_analysis[analysis_key])

        #pickle_out = open("pickle_simdata_" + self.file_name, "wb")
        #pickle.dump(self.simulation_data, pickle_out)
        #pickle_out.close()

    def time_analysis(self):
        time_dict = dict()
        time_dict["t_avg"] = np.average(self.simulation_data["times"])
        time_dict["t_std"] = np.std(self.simulation_data["times"])
        time_dict["steps"] = len(self.simulation_data["times"])
        time_dict["total_t"] = self.simulation_data["times"][-1]
        return time_dict

    def plot_trajectory(self, name=None, filename=None, num_label=10):
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

        #if state_history is None:
        data = self.state_history

        # To avoid indexing errors
        # if num_label > len(data["state"].keys()):
        #     num_label = len(data["state"].keys())
        if num_label > len(data):
            num_label = len(data)

        # Sort by final concentration
        # We assume that we're interested in the most prominent products
        # ids_sorted = sorted([(k, v) for k, v in data["state"].items()],
        #                     key=lambda x: x[1][-1][-1])
        ids_sorted = sorted(range(len(self.propagator.state)), key = lambda k: data[k], reverse = True)

        # ids_sorted = [i[0] for i in ids_sorted][::-1]
        print("top 15 species ids: ", ids_sorted[0:15])
        # Only label most prominent products
        colors = plt.cm.get_cmap('hsv', num_label)
        id = 0
        for mol_id in ids_sorted[0:num_label]:
            ts = np.array([e[0] for e in self.propagator.state_history[mol_id]])
            nums = np.array([e[1] for e in self.propagator.state_history[mol_id]])
            # if mol_id in ids_sorted[0:num_label]:
            #     for entry in self.reaction_network.entries_list:
            #         if mol_id == entry.entry_id:
            #             this_composition = entry.molecule.composition.alphabetical_formula
            #             this_charge = entry.molecule.charge
            #             this_label = this_composition + " " + str(this_charge)
            #             this_color = colors(id)
            #             id +=1
            #             #this_label = entry.entry_id
            #             break
            for entry in self.reaction_network.entries_list:
                if mol_id == entry.entry_id:
                    this_composition = entry.molecule.composition.alphabetical_formula
                    this_charge = entry.molecule.charge
                    this_label = this_composition + " " + str(this_charge)
                    this_color = colors(id)
                    id +=1
                    #this_label = entry.entry_id
                    break

                ax.plot(ts, nums, label = this_label, color = this_color)
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
        # ax.legend(loc='best', bbox_to_anchor=(0.45, -0.175),
        #           ncol=5, fontsize="small")


        # if filename is None:
        #     plt.show()
        # else:
        #     fig.savefig(filename, dpi=600)
        if filename == None:
            plt.savefig("SimulationRun")
        else:
            plt.savefig(filename)

    def reaction_analysis(self, data = None):
        if data == None:
            data = self.data
        reaction_analysis_results = dict()
        reaction_analysis_results["endo_rxns"] = dict()
        rxn_count = np.zeros(2*self.num_rxns)
        endothermic_rxns_count = 0
        fired_reaction_ids = set(self.data["reaction_ids"])
        for ind in fired_reaction_ids:
            this_count = data["reaction_ids"].count(ind)
            rxn_count[ind] = this_count
            this_rxn = self.reactions[math.floor(ind/2)]
            if ind % 2: # reverse rxn
                if this_rxn.free_energy()["free_energy_B"] > 0: # endothermic reaction
                    endothermic_rxns_count += this_count
            else:
                if this_rxn.free_energy()["free_energy_A"] > 0: # endothermic reaction
                    endothermic_rxns_count += this_count
        reaction_analysis_results["endo_rxns"]["endo_count"] = endothermic_rxns_count
        sorted_rxn_ids = sorted(self.rxn_ind, key = lambda k: rxn_count[k], reverse = True)
        bar_rxns_labels = list()
        bar_rxns_count = list()
        bar_rxns_x = list()
        for i in range(15): # analysis on most frequent reactions
            this_rxn_id = sorted_rxn_ids[i]
            bar_rxns_x.append(str(this_rxn_id))
            this_reaction = self.reactions[math.floor(this_rxn_id / 2 )]
            reaction_analysis_results[this_rxn_id] = dict()
            reaction_analysis_results[this_rxn_id]["count"] = rxn_count[this_rxn_id]
            reaction_analysis_results[this_rxn_id]["reactants"] = list()
            reaction_analysis_results[this_rxn_id]["products"] = list()
            this_label = str()
            if this_rxn_id % 2: # reverse rxn
                reaction_analysis_results[this_rxn_id]["reaction_type"] = this_reaction.reaction_type()["rxn_type_B"]
                this_label += this_reaction.reaction_type()["rxn_type_B"]
                for reactant in this_reaction.products:
                    reaction_analysis_results[this_rxn_id]["reactants"].append((reactant.molecule.composition.alphabetical_formula , reactant.entry_id))
                    this_label += " " + reactant.molecule.composition.alphabetical_formula
                for product in this_reaction.reactants:
                    reaction_analysis_results[this_rxn_id]["products"].append((product.molecule.composition.alphabetical_formula , product.entry_id))
                    this_label += " " + product.molecule.composition.alphabetical_formula
                reaction_analysis_results[this_rxn_id]["rate constant"] = this_reaction.rate_constant()["k_B"]
            else: # forward rxn
                reaction_analysis_results[this_rxn_id]["reaction_type"] = this_reaction.reaction_type()["rxn_type_A"]
                this_label += this_reaction.reaction_type()["rxn_type_A"]
                for reactant in this_reaction.reactants:
                    reaction_analysis_results[this_rxn_id]["reactants"].append((reactant.molecule.composition.alphabetical_formula, reactant.entry_id))
                    this_label += " " + reactant.molecule.composition.alphabetical_formula
                for product in this_reaction.products:
                    reaction_analysis_results[this_rxn_id]["products"].append((product.molecule.composition.alphabetical_formula , product.entry_id))
                    this_label += " " + product.molecule.composition.alphabetical_formula
                reaction_analysis_results[this_rxn_id]["rate constant"] = this_reaction.rate_constant()["k_A"]
            bar_rxns_labels.append(this_label)
            bar_rxns_count.append(reaction_analysis_results[this_rxn_id]["count"])
        plt.figure()
        plt.bar(bar_rxns_x[:10], bar_rxns_count[:10])
        plt.xlabel("Reaction Index")
        plt.ylabel("Reaction Occurrence")
        plt.title("Top Reactions, total " + str(len(self.data["times"])) + " reactions")
        plt.savefig("li_limited_top_rxns")
        return reaction_analysis_results

runtime_data = dict()
runtime_data["runtime"] = list()
runtime_data["label"] = list()
runtime_data["t_avg"] = list()
runtime_data["t_std"] = list()
runtime_data["steps"] = list()

li_conc = 1.0
# 3:7 EC:EMC
ec_conc = 3.57
emc_conc = 7.0555

# Testing parameters
volumes = [10**-24]
times = [10**-12]

for v in volumes:
    for t_end in times:
        file_n = "li_limited_t_" + str(t_end) +  "_V_" + str(v) + "_ea_10000_Numba"
        this_simulation = Simulation_Li_Limited(file_n, li_conc, ec_conc, emc_conc, v,
                                                t_end)
        this_simulation.plot_trajectory("Simulation Results", file_n, num_label = 10)
        time_data = this_simulation.time_analysis()
        runtime_data["t_avg"].append(time_data["t_avg"])
        runtime_data["t_std"].append(time_data["t_std"])
        runtime_data["steps"].append(time_data["steps"])
        runtime_data["runtime"].append(this_simulation.runtime)
        runtime_data["label"].append("V_" + str(v) + "_t_" + str(t_end))


print(runtime_data["label"])
print("runtimes: ", runtime_data["runtime"])
print("Avg t steps: ", runtime_data["t_avg"])
print("Std dev t steps: ", runtime_data["t_std"])
print("Number of rxns: ", runtime_data["steps"])

# for t_end in times:
#     this_filename = "Simulation_Run_" + str(t_end)
#     this_simulation = Simulation_Li_Limited(this_filename, li_conc, ec_conc, emc_conc, volume, t_end)
#     time_data = this_simulation.time_analysis()
#     runtime_data["t_avg"].append(time_data["t_avg"])
#     runtime_data["t_std"].append(time_data["t_std"])
#     runtime_data["steps"].append(time_data["steps"])
#     runtime_data["runtime"].append(this_simulation.runtime)
#
# plt.figure()
# plt.subplot(411)
# plt.plot(times, runtime_data["t_avg"])
# plt.title("Average Time Steps")
# plt.ylabel("Time (s)")
# plt.xlabel("t_end (s)")
#
# plt.subplot(412)
# plt.plot(times, runtime_data["t_std"])
# plt.title("Std Dev Time Steps")
# plt.ylabel("Time (s)")
# plt.xlabel("t_end (s)")
#
# plt.subplot(413)
# plt.plot(times, runtime_data["steps"])
# plt.title("Number of Time Steps")
# plt.ylabel("Time steps")
# plt.xlabel("t_end (s)")
#
# plt.subplot(414)
# plt.plot(times, runtime_data["runtime"])
# plt.title("Simulation Runtime Analysis")
# plt.ylabel("Runtime (s)")
# plt.xlabel("t_end (s)")
#
# plt.savefig("Simulation_Time_Analysis_t-9")
