import pickle
import copy
from typing import Optional, List, Dict, Tuple
import itertools

import numpy as np
import networkx as nx
from scipy.constants import h, k, R

from monty.serialization import dumpfn

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.io.qchem.outputs import QCOutput
from pymatgen.reaction_network.reaction_network import (Reaction,
                                                        RedoxReaction,
                                                        IntramolSingleBondChangeReaction,
                                                        IntermolecularReaction,
                                                        CoordinationBondChangeReaction,
                                                        ReactionNetwork,
                                                        ConcertedReaction,
                                                        graph_rep_2_2,
                                                        MappingDict,
                                                        Mapping_Family_Dict,
                                                        Mapping_ReactionType_Dict,
                                                        Mapping_Energy_Dict)
from pymatgen.reaction_network.reaction_rates import (ReactionRateCalculator,
                                                      ExpandedBEPRateCalculator)

from atomate.qchem.database import QChemCalcDb


class MetalHopReaction(Reaction):
    """
    A class to define metal "hopping" bond change as follows:
        Breaking one coordination bond (AM -> A + M) while simultaneously
        forming another (B + M -> BM), with overall stoichiometry
        AM + B <-> BM + A
        Four entries with:
            M = Li or Mg
            comp(AM) = comp(A) + comp(M)
            comp(BM) + comp(B) + comp(M)
            charge(AM) = charge(A) + charge(M)
            charge(BM) = charge(B) + charge(M)
            removing all edges containing M in AM yields two disconnected
            subgraphs that are isomorphic to A and M, and likewise for BM

    NOTE: This class assumes that the reactants and products are in the order:
        reactants: AM, B
        products: BM, A

    Args:
        reactant([MoleculeEntry]): list of single molecular entry
        product([MoleculeEntry]): list of two molecular entries
        transition_state (MoleculeEntry or None): A MoleculeEntry representing a
            transition state for the reaction.
        parameters (dict): Any additional data about this reaction
    """

    def __init__(self, reactants: MoleculeEntry, products: List[MoleculeEntry],
                 metal: MoleculeEntry,
                 transition_state: Optional[MoleculeEntry] = None,
                 parameters: Optional[Dict] = None,
                 neutral_hop_barrier: Optional[float] = 0.130,
                 anion_hop_barrier: Optional[float] = 0.239):
        """
        Initializes MetalHopReaction.reactant to be in the form of a
            [MoleculeEntry], MetalHopReaction.product to be in the form of
            [MoleculeEntry],

        Args:
            reactants: list of MoleculeEntry objects of length 2
            products: list of MoleculeEntry objects of length 2
            transition_state (MoleculeEntry or None): A MoleculeEntry
                representing a transition state for the reaction.
            parameters (dict): Any additional data about this reaction
            neutral_hop_barrier (float): Energy barrier (in eV) for a metal ion
                to de-coordinate from a neutral species
            anion_hop_barrier (float): Energy barrier (in eV) for a metal ion
                to de-coordinate from an anionic species

        """

        self.metal = metal
        self.neutral_hop_barrier = neutral_hop_barrier
        self.anion_hop_barrier = anion_hop_barrier

        super().__init__(reactants, products, transition_state=transition_state,
                         parameters=parameters)

    def graph_representation(self) -> nx.DiGraph:
        """
            A method to convert a CoordinationBondChangeReaction class object
                into graph representation (nx.Digraph object).
            CoordinationBondChangeReaction must be of type 1 reactant -> 2 products

            :return nx.Digraph object of a single CoordinationBondChangeReaction object
        """

        return graph_rep_2_2(self)

    @classmethod
    def generate(cls, entries: MappingDict) -> Tuple[List[Reaction], Mapping_Family_Dict]:
        reactions = list()
        M_entries = dict()
        pairs = list()
        for formula in entries:
            if formula in ["Li1", "Mg1", "Ca1", "Zn1"]:
                if formula not in M_entries:
                    M_entries[formula] = dict()
                for charge in entries[formula][0]:
                    # Only allow cations - neutral/anionic metals probably won't be re-coordinating
                    if charge > 0:
                        assert (len(entries[formula][0][charge]) == 1)
                        M_entries[formula][charge] = entries[formula][0][charge][0]
        if M_entries != dict():
            for formula in entries:
                if "Li" in formula or "Mg" in formula or "Ca" in formula or "Zn" in formula:
                    for Nbonds in entries[formula]:
                        if Nbonds > 2:
                            for charge in entries[formula][Nbonds]:
                                for entry in entries[formula][Nbonds][charge]:
                                    for aa, atom in enumerate(entry.molecule):
                                        if str(atom.specie) in ["Li", "Mg", "Zn", "Ca"]:
                                            edge_list = list()
                                            for edge in entry.mol_graph.graph.edges():
                                                if aa in edge:
                                                    edge_list.append(edge)

                                            try:
                                                frags = entry.mol_graph.split_molecule_subgraphs(edge_list,
                                                                                                 allow_reverse=True)
                                                M_ind = None
                                                M_formula = None
                                                for ii, frag in enumerate(frags):
                                                    frag_formula = frag.molecule.composition.alphabetical_formula
                                                    if frag_formula in M_entries:
                                                        M_ind = ii
                                                        M_formula = frag_formula
                                                        break
                                                if M_ind is not None:
                                                    for ii, frag in enumerate(frags):
                                                        if ii != M_ind:
                                                            nonM_formula = frag.molecule.composition.alphabetical_formula
                                                            nonM_Nbonds = len(frag.graph.edges())
                                                            if nonM_formula in entries:
                                                                if nonM_Nbonds in entries[nonM_formula]:
                                                                    for nonM_charge in entries[nonM_formula][nonM_Nbonds]:
                                                                        M_charge = entry.charge - nonM_charge
                                                                        if M_charge in M_entries[M_formula] and M_charge > 0:
                                                                            for nonM_entry in \
                                                                                    entries[nonM_formula][nonM_Nbonds][
                                                                                        nonM_charge]:
                                                                                if frag.isomorphic_to(nonM_entry.mol_graph):
                                                                                    pairs.append((entry, nonM_entry, M_entries[M_formula][M_charge]))
                                                                                    break
                                            except MolGraphSplitError:
                                                pass

        if len(pairs) > 1:
            for combo in itertools.combinations(pairs, 2):
                m_one = combo[0][2]
                m_two = combo[1][2]
                # Only allow if metal ion is the same on both sides
                if m_one.charge == m_two.charge and m_one.formula == m_two.formula:
                    reactions.append(cls([combo[0][0], combo[1][1]],
                                         [combo[1][0], combo[0][1]],
                                         m_one))

        return reactions, dict()

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
           A method to identify type of coordination bond change reaction
           (bond breaking from one to two or forming from two to one molecules)

           Args:
              :return dictionary of the form {"class": "CoordinationBondChangeReaction",
                                              "rxn_type_A": rxn_type_A,
                                              "rxn_type_B": rxn_type_B}
              where rnx_type_A is the primary type of the reaction based on the
              reactant and product of the CoordinationBondChangeReaction
              object, and the backwards of this reaction would be rnx_type_B
        """

        rxn_string = "Metal hopping AM + B <-> BM + A"

        reaction_type = {"class": "MetalHopReaction",
                         "rxn_type_A": rxn_string,
                         "rxn_type_B": rxn_string}
        return reaction_type

    def free_energy(self, temperature=298.15) -> Mapping_Energy_Dict:
        """
              A method to determine the free energy of the coordination bond
                change reaction

              Args:
                 :return dictionary of the form {"free_energy_A": energy_A,
                                                 "free_energy_B": energy_B}
                 where free_energy_A is the primary type of the reaction based
                 on the reactant and product of the CoordinationBondChangeReaction
                 object, and the backwards of this reaction would be free_energy_B.
         """

        try:
            g_product = sum([x.free_energy(temp=temperature) for x in self.products])
            g_reactant = sum([x.free_energy(temp=temperature) for x in self.reactants])
            free_energy_A = g_product - g_reactant
            free_energy_B = g_reactant - g_product
        except TypeError:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
              A method to determine the energy of the coordination bond change
              reaction

              Args:
                 :return dictionary of the form {"energy_A": energy_A,
                                                 "energy_B": energy_B}
                 where energy_A is the primary type of the reaction based on the
                 reactant and product of the CoordinationBondChangeReaction
                 object, and the backwards of this reaction would be energy_B.
        """

        try:
            e_product = sum([x.energy for x in self.products])
            e_reactant = sum([x.energy for x in self.reactants])
            energy_A = e_product - e_reactant
            energy_B = e_reactant - e_product
        except TypeError:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate_constant(self, temperature=298.15) -> Mapping_Energy_Dict:
        rate_constant = dict()
        g = self.free_energy(temperature=temperature)

        ga = g["free_energy_A"]
        gb = g["free_energy_B"]

        q_no_m_a = self.reactants[0].charge - self.metal.charge
        q_no_m_b = self.products[0].charge - self.metal.charge

        if q_no_m_a == 0:
            barrier_a = self.neutral_hop_barrier
        else:
            barrier_a = self.anion_hop_barrier

        if q_no_m_b == 0:
            barrier_b = self.neutral_hop_barrier
        else:
            barrier_b = self.anion_hop_barrier

        if ga < barrier_a:
            rate_constant["k_A"] = k * temperature / h * np.exp(-1 * barrier_a * 96487 / (R * temperature))
        else:
            rate_constant["k_A"] = k * temperature / h * np.exp(-1 * ga * 96487 / (R * temperature))

        if gb < barrier_b:
            rate_constant["k_B"] = k * temperature / h * np.exp(-1 * barrier_b * 96487 / (R * temperature))
        else:
            rate_constant["k_B"] = k * temperature / h * np.exp(-1 * gb * 96487 / (R * temperature))

        return rate_constant

    def as_dict(self) -> dict:

        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "reactants": [r.as_dict() for r in self.reactants],
             "products": [p.as_dict() for p in self.products],
             "metal": self.metal.as_dict(),
             "transition_state": None,
             "rate_calculator": None,
             "parameters": self.parameters,
             "neutral_hop_barrier": self.neutral_hop_barrier,
             "anion_hop_barrier": self.anion_hop_barrier}

        return d

    @classmethod
    def from_dict(cls, d):
        reactants = [MoleculeEntry.from_dict(m) for m in d["reactants"]]
        products = [MoleculeEntry.from_dict(m) for m in d["products"]]
        metal = MoleculeEntry.from_dict(d["metal"])

        reaction = cls(reactants, products, metal,
                       neutral_hop_barrier=d["neutral_hop_barrier"],
                       anion_hop_barrier=d["anion_hop_barrier"])

        return reaction


def inner_reorganization_energy(a_a, b_b, a_b, b_a):
    return ((a_b - a_a) + (b_a - b_b)) / 2


db = QChemCalcDb.from_db_file("sam_db.json")

electron_free_energy = -2.15

# "environment": "smd_18.5,1.415,0.00,0.735,20.2,0.00,0.00"

# Base molecules
easy_molecules = {
    "EC": -9317.492754189294,
    "ECminus": -9318.359185217801,
    # "LiEC_plus_mono": -9519.667234808205,
    "LiEC_plus_bi": -9519.538722407473,
    # "LiEC_mono": -9521.564251504658,
    "LiEC_bi": -9521.708410009893,
    "LiEC_RO1": -9522.915866743,
    "LiEC_RO2": -9522.793398675867,
    "LiCO3": -7384.740771154649,
    "LiCO3_minus": -7389.618831945432,
    "C2H4": -2137.8046383583164,
    "LiEC_minus_RO1": -9525.812603646005,
    "LiEC_minus_RO2": -9525.70508421495,
    "LEDC": -16910.7035955349,
    "LEDC_minus_Li": -16707.348720748865,
    "LEMC_minus": -11589.2628521163,
    # "LEC": -9540.895397325152,
    # LMC?
}

annoying_molecules = {
    "Li2CO3_plus": -278.799089123002,
    "Li2CO3": -279.043401129729,
    "H2O": -76.444743506228,
    "H2O_minus": -76.463413859741,
    "OH_minus": -75.9095418284053,
    "OH": -75.7470597069103,
    "LiOH": -83.3990229264322,
    "LiHCO3": -272.053210322813,
    "LEMC": -425.902362278509,
    "LEDC_minus": -621.529073351548
}

really_annoying_molecules = {
    "Li_plus": ("Li1", 1),
    "H": ("H1", 0),
    "H_plus": ("H1", 1)
}

additional_molecules = {
    "LEDC_Li_plus": (-628.9517046286, 70.923, 119.295),
    "LEDC_minus_Li_plus": (-629.0034834217, 69.484, 117.581),
    "LEMC_Li_plus": (-433.3276381245, 65.492, 100.896),
    "LEMC_minus_Li_plus": (-433.4022379853, 65.375, 101.264),
    "not_LEDC": (-621.5075399838, 68.581, 112.190),
    "Li2EC_RO": (-357.4395468439, 52.186, 94.066),
    "LiH2O_plus": (-83.8684621095031, 17.885, 56.902),
    "LiH2O": (-83.9436750513148, 17.734, 58.872),
    "DLEMC": (-432.8595189354, 57.216, 97.752),
}

mol_entries = list()
for name, free_energy in easy_molecules.items():
    try:
        db_entry = db.db["molecules"].find_one({"free_energy": free_energy})

        molecule = Molecule.from_dict(db_entry["molecule"])
        energy = db_entry["energy_Ha"]
        enthalpy = db_entry["rot_enthalpy_kcal/mol"] + db_entry["trans_enthalpy_kcal/mol"] + db_entry["vib_enthalpy_kcal/mol"]
        entropy = db_entry["rot_entropy_cal/molK"] + db_entry["trans_entropy_cal/molK"] + db_entry["vib_entropy_cal/molK"]

        mol_entry = MoleculeEntry(molecule=molecule,
                                  energy=energy,
                                  enthalpy=enthalpy,
                                  entropy=entropy,
                                  entry_id=name)
        mol_entries.append(mol_entry)
    except:
        print("PROBLEM (free energy)", name)

for name, energy in annoying_molecules.items():
    try:
        db_entry = db.db["molecules"].find_one({"energy_Ha": energy})

        molecule = Molecule.from_dict(db_entry["molecule"])
        energy = db_entry["energy_Ha"]
        enthalpy = db_entry["rot_enthalpy_kcal/mol"] + db_entry["trans_enthalpy_kcal/mol"] + db_entry["vib_enthalpy_kcal/mol"]
        entropy = db_entry["rot_entropy_cal/molK"] + db_entry["trans_entropy_cal/molK"] + db_entry["vib_entropy_cal/molK"]

        mol_entry = MoleculeEntry(molecule=molecule,
                                  energy=energy,
                                  enthalpy=enthalpy,
                                  entropy=entropy,
                                  entry_id=name)
        mol_entries.append(mol_entry)
    except:
        print("PROBLEM (energy Ha)", name)


for name, data in really_annoying_molecules.items():
    try:
        db_entry = db.db["molecules"].find_one({"formula_alphabetical": data[0], "molecule.charge": data[1],
                                                "environment": "smd_18.5,1.415,0.00,0.735,20.2,0.00,0.00"})
        molecule = Molecule.from_dict(db_entry["molecule"])
        energy = db_entry["energy_Ha"]
        enthalpy = db_entry["enthalpy_kcal/mol"]
        entropy = db_entry["entropy_cal/molK"]
        mol_entry = MoleculeEntry(molecule=molecule,
                                  energy=energy,
                                  enthalpy=enthalpy,
                                  entropy=entropy,
                                  entry_id=name)
        mol_entries.append(mol_entry)
    except:
        print("PROBLEM (formula)", name)


for name, energies in additional_molecules.items():
    mol = Molecule.from_file("mols/{}.xyz".format(name))
    if "plus" in name and "minus" not in name:
        mol.set_charge_and_spin(1)
    mol_entry = MoleculeEntry(molecule=mol,
                              energy=energies[0],
                              enthalpy=energies[1],
                              entropy=energies[2],
                              entry_id=name)
    mol_entries.append(mol_entry)


dumpfn(mol_entries, "20200818_kmc_network_entries.json")

# Generate reaction network
# TODO: Decide on electron free energy
network = ReactionNetwork.from_input_entries(mol_entries,
                                             electron_free_energy=electron_free_energy)
network.build()

reactions, _ = MetalHopReaction.generate(network.entries)
for reaction in reactions:
    network.reactions.append(reaction)
    network.add_reaction(reaction.graph_representation())

reaction_sets = list()
to_delete = list()
for rr, reaction in enumerate(network.reactions):
    reaction_set = (sorted(reaction.reactant_ids), sorted(reaction.product_ids))
    if reaction_set in reaction_sets:
        to_delete.append(rr)
    else:
        reaction_sets.append(reaction_set)

for rr in sorted(to_delete)[::-1]:
    del network.reactions[rr]

# Define transition states
# NOTE: ring openings modified to have artificially lower barriers
# - 0.0132297
transition_states = {
    "LiEC_bi___LiEC_RO1": (-349.932441548429 - 0.0132297, 49.758, 79.589),
    "LiEC_bi___LiEC_RO2": (-349.932146021146 - 0.0132297, 49.628, 79.024),
    "LiEC_RO1___ethylene__LiCO3": (-349.955881242111, 48.640, 89.721),
    "LiEC_RO2___ethylene__LiCO3": (-349.949594721925, 48.503, 89.749),
    "LiEC_RO1___ethylene__LiCO3_minus": (-350.099875862606, 48.560, 83.607),
    "LiEC_RO2___ethylene__LiCO3_minus": (-350.099400939398, 48.367, 88.095),
    "LiCO3__LiEC___LEDC": (-621.443534450198, 67.541, 109.685),
    "LiCO3__LiEC___not_LEDC": (-621.4411141046, 67.453, 109.762),
    "LiCO3__EC___LEDC_minus_Li": (-613.989321259289, 65.226, 102.352),
    "LiEC_RO1__Li2CO3_plus___LEDC_Li_plus": (-628.8685738750, 68.623, 122.276),
    "LiEC_RO1__Li2CO3___LEDC_minus_Li_plus": (-628.9781207643, 67.608, 119.064),
    "LiEC_RO1__OH_minus___LEMC_minus": (-425.8844480361, 58.743, 94.966),
    "LiEC_RO1__H2O___LEMC__H": (-426.3629645891, 64.222, 95.803),
    # "LiEC_RO1__H2O___LMC__OH": (-426.4186524362, 64.547, 96.180),
    "LEMC__Li___DLEMC__H": (-433.3605250211, 58.456, 103.705),
    "LiEC_plus_bi__OH_minus___LEMC": (-425.8069003510, 61.101, 91.370)
}

ts = dict()
for name, thermo in transition_states.items():
    ts[name] = MoleculeEntry(molecule=QCOutput("ts/{}.out".format(name)).data["initial_molecule"],
                             energy=thermo[0],
                             enthalpy=thermo[1],
                             entropy=thermo[2],
                             entry_id=name)

# Associate transition states
print("=====\nADDING TS\n=====")
for reaction in network.reactions:
    if ("LiEC_bi" in reaction.reactant_ids and "LiEC_RO1" in reaction.product_ids) or ("LiEC_bi" in reaction.product_ids and "LiEC_RO1" in reaction.reactant_ids):
        print("LiEC___LiEC_RO1")
        reaction.update_calculator(transition_state=ts["LiEC_bi___LiEC_RO1"])
    if ("LiEC_bi" in reaction.reactant_ids and "LiEC_RO2" in reaction.product_ids) or ("LiEC_bi" in reaction.product_ids and "LiEC_RO2" in reaction.reactant_ids):
        print("LiEC___LiEC_RO2")
        reaction.update_calculator(transition_state=ts["LiEC_bi___LiEC_RO2"])
    if "LiEC_RO1" in reaction.reactant_ids and "C2H4" in reaction.product_ids and "LiCO3" in reaction.product_ids:
        print("LiEC_RO1___ethylene__LiCO3")
        reaction.update_calculator(transition_state=ts["LiEC_RO1___ethylene__LiCO3"])
    if "LiEC_RO2" in reaction.reactant_ids and "C2H4" in reaction.product_ids and "LiCO3" in reaction.product_ids:
        print("LiEC_RO2___ethylene__LiCO3")
        reaction.update_calculator(transition_state=ts["LiEC_RO2___ethylene__LiCO3"])
    if "LiEC_minus_RO1" in reaction.reactant_ids and "C2H4" in reaction.product_ids and "LiCO3_minus" in reaction.product_ids:
        print("LiEC_RO1___ethylene__LiCO3_minus")
        reaction.update_calculator(transition_state=ts["LiEC_RO1___ethylene__LiCO3_minus"])
    if "LiEC_minus_RO2" in reaction.reactant_ids and "C2H4" in reaction.product_ids and "LiCO3_minus" in reaction.product_ids:
        print("LiEC_RO2___ethylene__LiCO3_minus")
        reaction.update_calculator(transition_state=ts["LiEC_RO2___ethylene__LiCO3_minus"])
    if "LEDC_Li_plus" in reaction.reactant_ids and "LiEC_RO1" in reaction.product_ids and "Li2CO3_plus" in reaction.product_ids:
        print("LiEC_RO1__Li2CO3_plus___LEDC_Li_plus")
        reaction.update_calculator(transition_state=ts["LiEC_RO1__Li2CO3_plus___LEDC_Li_plus"])
    if "LEDC_minus_Li_plus" in reaction.reactant_ids and "LiEC_RO1" in reaction.product_ids and "Li2CO3" in reaction.product_ids:
        print("LiEC_RO1__Li2CO3_plus___LEDC_Li_plus")
        reaction.update_calculator(transition_state=ts["LiEC_RO1__Li2CO3___LEDC_minus_Li_plus"])
    if "LEMC_minus" in reaction.reactant_ids and "LiEC_RO1" in reaction.product_ids and "OH_minus" in reaction.product_ids:
        print("LiEC_RO1__OH_minus___LEMC_minus")
        reaction.update_calculator(transition_state=ts["LiEC_RO1__OH_minus___LEMC_minus"])
    if "LEMC_minus" in reaction.reactant_ids and "LiEC_minus_RO1" in reaction.product_ids and "OH" in reaction.product_ids:
        print("LiEC_minus_RO1__OH___LEMC_minus")
        reaction.update_calculator(transition_state=ts["LiEC_RO1__OH_minus___LEMC_minus"])
    if "LEMC_minus_Li_plus" in reaction.reactant_ids and "DLEMC" in reaction.product_ids and "H" in reaction.product_ids:
        print("LEMC_minus__Li_plus___DLEMC__H")
        reaction.update_calculator(transition_state=ts["LEMC__Li___DLEMC__H"])

# Add concerted reactions
print("=====\nADDING CONCERTEDS\n=====")
print("2 LiEC_RO1 -> LEDC + C2H4")
network.reactions.append(ConcertedReaction([network.entries_list[22],
                                            network.entries_list[5]],
                                           [network.entries_list[8],
                                            network.entries_list[8]],
                                           electron_free_energy=electron_free_energy))
print("LiEC_RO1 + LiEC_RO2 -> LEDC + C2H4")
network.reactions.append(ConcertedReaction([network.entries_list[22],
                                            network.entries_list[5]],
                                           [network.entries_list[8],
                                            network.entries_list[9]],
                                           electron_free_energy=electron_free_energy))
print("LiCO3 + LiEC_bi -> LEDC")
network.reactions.append(ConcertedReaction([network.entries_list[22]],
                                           [network.entries_list[1],
                                            network.entries_list[11]],
                                           transition_state=ts["LiCO3__LiEC___LEDC"],
                                           electron_free_energy=electron_free_energy))
print("LiCO3 + LiEC_bi -> not_LEDC")
network.reactions.append(ConcertedReaction([network.entries_list[23]],
                                           [network.entries_list[1],
                                            network.entries_list[11]],
                                           transition_state=ts["LiCO3__LiEC___not_LEDC"],
                                           electron_free_energy=electron_free_energy))
print("LiCO3 + EC -> LEDC_minus_Li")
network.reactions.append(ConcertedReaction([network.entries_list[20]],
                                           [network.entries_list[1],
                                            network.entries_list[15]],
                                           transition_state=ts["LiCO3__EC___LEDC_minus_Li"],
                                           electron_free_energy=electron_free_energy))
print("LiEC_RO1 + H2O -> LEMC + H")
network.reactions.append(ConcertedReaction([network.entries_list[8],
                                            network.entries_list[34]],
                                           [network.entries_list[17],
                                            network.entries_list[26]],
                                           transition_state=ts["LiEC_RO1__H2O___LEMC__H"],
                                           electron_free_energy=electron_free_energy))
print("LiEC_plus_bi + OH_minus -> LEMC")
network.reactions.append(ConcertedReaction([network.entries_list[17]],
                                           [network.entries_list[11],
                                            network.entries_list[29]],
                                           transition_state=ts["LiEC_plus_bi__OH_minus___LEMC"],
                                           electron_free_energy=electron_free_energy))
print("LEMC + Li_plus -> DLEMC + H")
# If you count based on the reactants, and not based on the complex, it's actually barrierless
network.reactions.append(ConcertedReaction([network.entries_list[35],
                                            network.entries_list[16]],
                                           [network.entries_list[13],
                                            network.entries_list[26]],
                                           # transition_state=ts["LEMC__Li___DLEMC__H"],
                                           electron_free_energy=electron_free_energy))
# print(ts["LEMC__Li___DLEMC__H"].free_energy() - network.entries_list[35].free_energy() - network.entries_list[16].free_energy())
# network.reactions.append(IntermolecularReaction([network.entries_list[14]],
#                                       [network.entries_list[0],
#                                        network.entries_list[8]],
#                                       transition_state=ts["ledc_formation"]))
# network.reactions.append(IntermolecularReaction([network.entries_list[15]],
#                                       [network.entries_list[0],
#                                        network.entries_list[8]],
#                                       transition_state=ts["attack_ro_2"]))
# TODO: add?
# # "LiEC_RO1__H2O___LMC__OH": (-426.4186524362, 64.547, 96.180),

network.reactions.append(RedoxReaction(network.entries_list[4],
                                       network.entries_list[3],
                                       electron_free_energy=electron_free_energy))

base_reference = {"dielectric": 18.5,
                  "refractive": 1.415,
                  "electron_free_energy": electron_free_energy,
                  "electrode_dist": 10.0}

print("=====\nADDING REDOX BARRIERS\n=====")

u_LiEC_plus_bi = -9520.80544639673
u_LiEC_bi = -9522.93975687255
liec_bi_0_1 = -349.8078163508 * 27.21139
liec_bi_1_0 = -349.9536654838 * 27.21139
u_LiEC_RO1 = -9523.96698199867
u_LiEC_minus_RO1 = -9526.87086244333
liec_ro1_0_minus = -350.0800263774 * 27.21139
liec_ro1_minus_0 = -349.9806026181 * 27.21139
u_LiEC_RO2 = -9523.81540716707
u_LiEC_minus_RO2 = -9526.71475518606
liec_ro2_0_minus = -350.0739243984 * 27.21139
liec_ro2_minus_0 = -349.9772943437 * 27.21139
u_LiCO3 = -7384.41022413945
u_LiCO3_minus = -7389.35190917139
lico3_minus_0 = -271.3578385712 * 27.21139
lico3_0_minus = -271.5376783148 * 27.21139
u_Li2CO3 = -7593.1588150675
u_Li2CO3_plus = -7586.51074577077
li2co3_1_0 = -278.9845805999 * 27.21139
li2co3_0_1 = -278.7684416693 * 27.21139
u_LEMC = -11589.3952818818
u_LEMC_minus = -11590.7932436233
lemc_0_minus = -425.9527912320 * 27.21139
lemc_minus_0 = -425.9014515250 * 27.21139
u_LEMC_Li_plus = -433.3276381245
u_LEMC_minus_Li_plus = -433.4022379853
lilemc_1_0 = -433.4016153425 * 27.21139
liliemc_0_1 = -433.3273702879 * 27.21139
u_EC = -9318.79298119549
u_EC_minus = -9319.53145751394
ec_0_minus = -342.4641092033 * 27.21139
ec_minus_0 = -342.3929701699 * 27.21139
u_OH = -2061.18278303802
u_OH_minus = -2065.60414741405
oh_0_minus = -75.9093119994 * 27.21139
oh_minus_0 = -75.7468409703 * 27.21139
u_LEDC_Li_plus = -628.9517046286
u_LEDC_minus_Li_plus = -629.0034834217
ledc_li_1_0 = -629.0132417768 * 27.21139
ledc_li_0_1 = -628.8786100691 * 27.21139
u_LEDC = -621.513385102203 * 27.21139
u_LEDC_minus = -621.529073351548 * 27.21139
ledc_0_minus = -621.5288353163 * 27.21139
ledc_minus_0 = -621.5131805964 * 27.21139
u_LiH2O_plus = -83.8684621095031 * 27.21139
u_LiH2O = -83.9436750513148 * 27.21139
lih2o_1_0 = -83.9435230279 * 27.21139
lih2o_0_1 = -83.8682922131 * 27.21139
u_H2O = -76.444743506228
u_H2O_minus = -76.463413859741
h2o_0_minus = -76.4627671281
h2o_minus_0 = -76.4440380125

lambda_inner = {"LiEC_bi": inner_reorganization_energy(-9520.80544639673, -9522.93975687255, -349.8078163508 * 27.21139, -349.9536654838 * 27.21139),
                "LiEC_RO1": inner_reorganization_energy(-9523.96698199867, -9526.87086244333, -350.0800263774 * 27.21139, -349.9806026181 * 27.21139),
                "LiEC_RO2": inner_reorganization_energy(-9523.81540716707, -9526.71475518606, -350.0739243984 * 27.21139, -349.9772943437 * 27.21139),
                "LiCO3": inner_reorganization_energy(-7384.41022413945, -7389.35190917139, -271.3578385712 * 27.21139, -271.5376783148 * 27.21139),
                "Li2CO3": inner_reorganization_energy(-7593.1588150675, -7586.51074577077, -278.9845805999 * 27.21139, -278.7684416693 * 27.21139),
                "LEMC": inner_reorganization_energy(-11589.3952818818, -11590.7932436233, -425.9527912320 * 27.21139, -425.9014515250 * 27.21139),
                "LEMC_Li_plus": inner_reorganization_energy(-433.3276381245 * 27.21139, -433.4022379853 * 27.21139, -433.4016153425 * 27.21139, -433.3273702879 * 27.21139),
                "LEDC_Li_plus": inner_reorganization_energy(-628.9517046286 * 27.21139, -629.0034834217 * 27.21139, -629.0132417768 * 27.21139, -628.8786100691 * 27.21139),
                "EC": inner_reorganization_energy(-9318.79298119549, -9319.53145751394, -342.4641092033 * 27.21139, -342.3929701699 * 27.21139),
                "OH": inner_reorganization_energy(-2061.18278303802, -2065.60414741405, -75.9093119994 * 27.21139, -75.7468409703 * 27.21139),
                "LEDC": inner_reorganization_energy(-621.513385102203 * 27.21139, -621.529073351548 * 27.21139, -621.5288353163 * 27.21139, -621.5131805964 * 27.21139),
                "LiH2O": inner_reorganization_energy(-83.8684621095031 * 27.21139, -83.9436750513148 * 27.21139, -83.9435230279 * 27.21139, -83.8682922131 * 27.21139),
                "H2O": inner_reorganization_energy(-76.444743506228 * 27.21139, -76.463413859741 * 27.21139, -76.4627671281 * 27.21139, -76.4440380125 * 27.21139),
                "H": 0.0}

# for k, v in lambda_inner.items():
#     print(k, v)

radii = dict()
for entry in network.entries_list:
    if entry.entry_id in lambda_inner:
        radius = np.max(entry.molecule.distance_matrix) / 2.0 + 4.0
        radii[entry.entry_id] = radius
# for x in lambda_inner:
#     if x not in radii:
#         print("PROBLEM", x)
# for k, v in radii.items():
#     print(k, v)

for reaction in network.reactions:
    if reaction.reaction_type()["class"] == "RedoxReaction":
        try:
            reference = copy.deepcopy(base_reference)
            match = False
            for key in lambda_inner:
                if key in reaction.reactant_ids or key in reaction.product_ids:
                    reference["lambda_inner"] = lambda_inner[key]
                    reference["radius"] = radii[key]
                    reaction.update_calculator(reference=reference)
                    match = True
                    break
            if not match:
                print("PROBLEM", reaction.reactant_ids, reaction.product_ids)
        except KeyError:
            print("PROBLEM", reaction.reaction_type()["class"], reaction.reactant_ids, reaction.product_ids)

print("=====\nENTRIES\n=====")
for ii, entry in enumerate(network.entries_list):
    print(ii, entry.entry_id)
print("=====\nREACTIONS\n=====")
for reaction in network.reactions:
    print(reaction.reaction_type()["class"], reaction.reactant_ids, reaction.product_ids,
          reaction.free_energy()["free_energy_A"], reaction.rate_constant()["k_A"])
print("TOTAL REACTIONS:", len(network.reactions))

try:
    dumpfn(network, "ledc_lemc_network_nomono.json")
    pickle.dump(network, open("ledc_network_pickle_nomono_rightbarrier", 'wb'))
except:
    pickle.dump(network, open("ledc_network_pickle_nomono_rightbarrier", 'wb'))