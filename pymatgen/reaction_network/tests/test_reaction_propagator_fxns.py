import numpy as np
import random
import math
from scipy.constants import N_A
from pymatgen.util.testing import PymatgenTest
from pymatgen.reaction_network.reaction_network import ReactionNetwork
from pymatgen.core import Molecule
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.reaction_network.reaction_propagator_fxns import *
import unittest
import copy
try:
    from openbabel import openbabel as ob
except ImportError:
    ob = None

__author__ = "Ronald Kam, Evan Spotte-Smith"
__email__ = "kamronald@berkeley.edu"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

class TestKMCReactionPropagatorFxns(PymatgenTest):
    def setUp(self):
        """ Create an initial state and reaction network, based on H2O molecule.
        Species include H2, H2O, H, O, O2, OH, H3O
        """
        self.volume = 10**-24 # m^3

        # 100 molecules each of H2O, H2, O2, OH-, H+
        self.num_mols = int(100)
        self.concentration = self.num_mols / N_A / self.volume / 1000

        # Make molecule objects
        H2O_mol = Molecule.from_file("H2O.xyz")
        H2O_mol1 = copy.deepcopy(H2O_mol)
        H2O_mol_1 = copy.deepcopy(H2O_mol)
        H2O_mol1.set_charge_and_spin(charge=1)
        H2O_mol_1.set_charge_and_spin(charge=-1)

        H2_mol = Molecule.from_file("H2.xyz")
        H2_mol1 = copy.deepcopy(H2_mol)
        H2_mol_1 = copy.deepcopy(H2_mol)
        H2_mol1.set_charge_and_spin(charge=1)
        H2_mol_1.set_charge_and_spin(charge=-1)

        O2_mol = Molecule.from_file("O2.xyz")
        O2_mol1 = copy.deepcopy(O2_mol)
        O2_mol_1 = copy.deepcopy(O2_mol)
        O2_mol1.set_charge_and_spin(charge=1)
        O2_mol_1.set_charge_and_spin(charge=-1)

        OH_mol = Molecule.from_file("OH.xyz")
        OH_mol1 = copy.deepcopy(OH_mol)
        OH_mol_1 = copy.deepcopy(OH_mol)
        OH_mol1.set_charge_and_spin(charge=1)
        OH_mol_1.set_charge_and_spin(charge=-1)

        H_mol = Molecule.from_file("H.xyz")
        H_mol1 = copy.deepcopy(H_mol)
        H_mol_1 = copy.deepcopy(H_mol)
        H_mol1.set_charge_and_spin(charge=1)
        H_mol_1.set_charge_and_spin(charge=-1)

        O_mol = Molecule.from_file("O.xyz")
        O_mol1 = copy.deepcopy(O_mol)
        O_mol_1 = copy.deepcopy(O_mol)
        O_mol1.set_charge_and_spin(charge=1)
        O_mol_1.set_charge_and_spin(charge=-1)

        # Make molecule entries
        # H2O 1-3
        if ob:
            H2O = MoleculeEntry(H2O_mol, energy=-76.4447861695239, correction=0, enthalpy=15.702, entropy=46.474,
                                parameters=None, entry_id = 'h2o', attribute=None)
            H2O_1 = MoleculeEntry(H2O_mol_1, energy=-76.4634569330715, correction=0, enthalpy=13.298, entropy=46.601,
                                  parameters=None, entry_id='h2o-', attribute=None)
            H2O_1p = MoleculeEntry(H2O_mol1, energy=-76.0924662469782, correction=0, enthalpy=13.697, entropy=46.765,
                                   parameters=None, entry_id='h2o+', attribute=None)
            # H2 4-6
            H2 = MoleculeEntry(H2_mol, energy=-1.17275734244991, correction=0, enthalpy=8.685, entropy=31.141,
                               parameters=None, entry_id='h2', attribute=None)
            H2_1 = MoleculeEntry(H2_mol_1, energy=-1.16232420718418, correction=0, enthalpy=3.56, entropy=33.346,
                                 parameters=None, entry_id='h2-', attribute=None)
            H2_1p = MoleculeEntry(H2_mol1, energy=-0.781383960574136, correction=0, enthalpy=5.773, entropy=32.507,
                                  parameters=None, entry_id='h2+', attribute=None)

            # OH 7-9
            OH = MoleculeEntry(OH_mol, energy=-75.7471080255785, correction=0, enthalpy=7.659, entropy=41.21,
                               parameters=None, entry_id='oh', attribute=None)
            OH_1 = MoleculeEntry(OH_mol_1, energy=-75.909589774742, correction=0, enthalpy=7.877, entropy=41.145,
                                 parameters=None, entry_id='oh-', attribute=None)
            OH_1p = MoleculeEntry(OH_mol1, energy=-75.2707068199185, correction=0, enthalpy=6.469, entropy=41.518,
                                  parameters=None, entry_id='oh+', attribute=None)
            # O2 10-12
            O2 = MoleculeEntry(O2_mol, energy=-150.291045922131, correction=0, enthalpy=4.821, entropy=46.76,
                               parameters=None, entry_id='o2', attribute=None)
            O2_1p = MoleculeEntry(O2_mol1, energy=-149.995474036502, correction=0, enthalpy=5.435, entropy=46.428,
                                  parameters=None, entry_id='o2+', attribute=None)
            O2_1 = MoleculeEntry(O2_mol_1, energy=-150.454499528454, correction=0, enthalpy=4.198, entropy=47.192,
                                 parameters=None, entry_id='o2-', attribute=None)

            # O 13-15
            O = MoleculeEntry(O_mol, energy=-74.9760564004, correction=0, enthalpy=1.481, entropy=34.254,
                              parameters=None, entry_id='o', attribute=None)
            O_1 = MoleculeEntry(O_mol_1, energy=-75.2301047938, correction=0, enthalpy=1.481, entropy=34.254,
                                parameters=None, entry_id='o-', attribute=None)
            O_1p = MoleculeEntry(O_mol1, energy=-74.5266804995, correction=0, enthalpy=1.481, entropy=34.254,
                                 parameters=None, entry_id='o+', attribute=None)
            # H 15-18
            H = MoleculeEntry(H_mol, energy=-0.5004488848, correction=0, enthalpy=1.481, entropy=26.014,
                              parameters=None, entry_id='h', attribute=None)
            H_1p = MoleculeEntry(H_mol1, energy=-0.2027210483, correction=0, enthalpy=1.481, entropy=26.066,
                                 parameters=None, entry_id='h+', attribute=None)
            H_1 = MoleculeEntry(H_mol_1, energy=-0.6430639079, correction=0, enthalpy=1.481, entropy=26.014,
                                parameters=None, entry_id='h-', attribute=None)

            self.mol_entries = [H2O, H2O_1, H2O_1p, H2, H2_1, H2_1p,
                                OH, OH_1, OH_1p, O2, O2_1p, O2_1,
                                O, O_1, O_1p, H, H_1p, H_1]

            self.reaction_network = ReactionNetwork.from_input_entries(self.mol_entries, electron_free_energy=-2.15)
            self.reaction_network.build()
            # print('number of reactions: ', len(self.reaction_network.reactions))
            # Only H2O, H2, O2 present initially
            self.initial_conditions = {'h2o': self.concentration, 'h2': self.concentration, 'o2': self.concentration, 'oh-': self.concentration, 'h+': self.concentration}
            # for ind, spec in enumerate(self.reaction_network.entries_list):
            #     print('mol_ind: ', ind, ', species_id: ', spec.entry_id)
            # for ind, reaction in enumerate(self.reaction_network.reactions):
            #     rxn_type = reaction.reaction_type.__self__
            #     print('reaction ', ind, ' is of type ', rxn_type.reaction_type["class"])
            #     print('reactants: ', [mol.entry_id for mol in reaction.reactants])
            #     print('products: ', [mol.entry_id for mol in reaction.products])

    def tearDown(self) -> None:
        if ob:
            del self.volume
            del self.num_mols
            del self.concentration
            del self.mol_entries
            del self.reaction_network
            del self.initial_conditions

    @unittest.skipIf(not ob, "OpenBabel not present. Skipping...")
    def test_initialize_simulation(self):
        # initialize the simulation

        [initial_state, species_rxn_mapping, reactant_array, product_array, coord_array, rate_constants,
         propensities, molid_index_mapping] = initialize_simulation(self.reaction_network, self.initial_conditions, self.volume)

        # Verify initial state
        num_species = len(self.reaction_network.entries_list)
        exp_initial_state = [0 for i in range(num_species)]
        exp_initial_state[2] = self.num_mols # h+
        exp_initial_state[3] = self.num_mols # oh-
        exp_initial_state[7] = self.num_mols # h2
        exp_initial_state[10] = self.num_mols #h2o
        exp_initial_state[16] = self.num_mols #o2

        self.assertEqual(exp_initial_state, initial_state)

if __name__ == "__main__":
    unittest.main()