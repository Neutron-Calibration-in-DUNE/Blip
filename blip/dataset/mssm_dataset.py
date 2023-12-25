"""
MSSM Dataset associated to the paper:
'Visualization and Efficient Generation of Constrained High-dimensional Theoretical Parameter Spaces'
Jason Baretz, Nicholas Carrara, Jacob Hollingsworth, Daniel Whiteson
J. High Energ. Phys. 2023, 62 (2023). https://doi.org/10.1007/JHEP11(2023)062
"""
import os
import getpass
import socket
import numpy as np
from datetime import datetime
import pandas as pd
import kmapper as km
from tqdm import tqdm

from blip.utils.logger import Logger
from blip.dataset.common import common_columns
from blip.dataset.common import cmssm_columns, pmssm_columns
from blip.dataset.common import base_constraints
from blip.dataset.common import cmssm_constraints, pmssm_constraints

mssm_dataset_config = {
    "name":               "default",
}


class MSSMDataset:
    """
    """
    def __init__(
        self,
        name:   str = "",
        config: dict = mssm_dataset_config,
        meta:   dict = {}
    ):
        self.name = name + '_dataset'
        self.config = config
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(name, output="both", file_mode="w")
        else:
            self.logger = Logger(name, level='warning', file_mode="w")
        self.logger.info("constructing mssm dataset.")

        # create constraint and dataset directories
        if not os.path.isdir("constraints/"):
            os.makedirs("constraints/")
        if not os.path.isdir("datasets/"):
            os.makedirs("datasets/")

        self.parse_config()

    def parse_config(self):
        if "simulation_folder" not in self.config.keys():
            self.logger.warn('simulation_folder not specified in config! Setting to "./".')
            self.config['simulation_folder'] = './'
        self.simulation_folder = self.config['simulation_folder']
        if "simulation_files" not in self.config.keys():
            self.logger.warn('simulation_files not specified in config!')
            self.config['simulation_files'] = []
        if len(self.config['simulation_files']) == 0:
            self.config['simulation_files'] = [
                file
                for file in os.listdir(self.simulation_folder)
                if '.txt' in file
            ]
        self.simulation_files = self.config['simulation_files']

        if "process_type" not in self.config.keys():
            self.logger.warn('process_type not specified in config! Setting to "cmssm".')
            self.config["process_type"] = "cmssm"
        self.process_type = self.config["process_type"][0]

        self.output_folder = self.process_type
        if not os.path.isdir(f"data/{self.output_folder}"):
            os.makedirs(f"data/{self.output_folder}")

        if self.process_type == 'cmssm':
            self.feature_labels = {
                'gut_m0': 0, 'gut_m12': 1,
                'gut_A0': 2, 'gut_tanb': 3,
                'sign_mu': 4
            }
            self.subspace_columns = cmssm_columns
            self.search_columns = self.subspace_columns + common_columns
            self.constraints = base_constraints
            self.constraints.update(cmssm_constraints)
        elif self.process_type == 'pmssm':
            self.feature_labels = {
                'gut_m1': 0, 'gut_m2': 1,
                'gut_m3': 2, 'gut_mmu': 3,
                'gut_mA': 4, 'gut_At': 5,
                'gut_Ab': 6, 'gut_Atau': 7,
                'gut_mL1': 8, 'gut_mL3': 9,
                'gut_me1': 10, 'gut_mtau1': 11,
                'gut_mQ1': 12, 'gut_mQ3': 13,
                'gut_mu1': 14, 'gut_mu3': 15,
                'gut_md1': 16, 'gut_md3': 17,
                'gut_tanb': 18
            }
            self.subspace_columns = pmssm_columns
            self.search_columns = self.subspace_columns + common_columns
            self.constraints = base_constraints
            self.constraints.update(pmssm_constraints)
        if "process_simulation" in self.config.keys():
            if self.config["process_simulation"]:
                self.generate_training_data(self.process_type)

        # create mapper object
        self.logger.info("creating Keppler Mapper.")
        self.mapper = km.KeplerMapper(verbose=0)
        self.mapper_nodes = None

    def generate_training_data(
        self,
        process_type:  list = ['all'],
    ):
        self.meta = {
            "who_created":      getpass.getuser(),
            "when_created":     datetime.now().strftime("%m-%d-%Y-%H:%M:%S"),
            "where_created":    socket.gethostname(),
            "features":         self.feature_labels,
            "classes": {
                'higgs': 0, 'dark_matter': 1, 'lightest_lsp': 2, 'muon': 3
            },
            "higgs_labels": {
                0:  'invalid', 1:    'valid'
            },
            "dark_matter_labels": {
                0:  'invalid', 1:    'valid'
            },
            "lightest_lsp_labels": {
                0:  'invalid', 1:    'valid'
            },
            "muon_labels": {
                0:  'invalid', 1:    'valid'
            },
        }
        self.features = []
        self.classes = []
        file_loop = tqdm(
            enumerate(self.simulation_files, 0),
            total=len(self.simulation_files),
            leave=True,
            position=0,
            colour='green'
        )
        for ii, input_file in file_loop:
            file_dataset = []
            try:
                file_dataset = pd.read_csv(
                    self.simulation_folder + input_file,
                    sep=',',
                    header=None,
                    names=self.search_columns,
                    usecols=self.search_columns
                )
            except Exception as e:
                self.logger.warning(f"file {input_file} threw an error: {e}")
            file_features = file_dataset[self.feature_labels.keys()]
            self.features.append(file_features.to_numpy())
            self.classes.append(self.apply_constraints(file_dataset))

            file_loop.set_description(f"MSSM File: [{ii+1}/{len(self.simulation_files)}]")

        self.features = np.concatenate(self.features)
        self.classes = np.concatenate(self.classes)

        np.savez(
            f"data/{self.output_folder}/{self.process_type}.npz",
            features=self.features,
            classes=self.classes,
            meta=self.meta
        )

    def apply_constraints(
        self,
        features
    ):
        classes = np.zeros((len(features), 4), dtype=int)
        # apply higgs constraint
        higgs = (
            (round(features['weakm_mh'], 2) - self.constraints["higgs_mass"]).abs() < self.constraints["higgs_mass_sigma"]
        )
        # apply dm relic density constraint
        relic_density_saturated = (
            (round(features['omegah2'], 2) - self.constraints["dm_relic_density"]).abs() < self.constraints["dm_relic_density_sigma"]
        )
        # apply lightest lsp constraint
        neutralino_lsp = (
            round(features['lspmass']) == round(features['weakm_mneut1'])
        )
        # apply muon g-2 constraint
        muon_moment = (
            (round(features['g-2'],8) - self.constraints["muon_magnetic_moment"]).abs() < self.constraints["muon_magnetic_moment_sigma"]
        )

        classes[higgs][:, 0] = 1
        classes[relic_density_saturated][:, 1] = 1
        classes[neutralino_lsp][:, 2] = 1
        classes[muon_moment][:, 3] = 1

        return classes

    def diff(
        self,
        X,
        Y,
    ):
        """
        Computes |X-Y|/min(X,Y)
        """
        return abs(X-Y)/min(X, Y)

    def dm_channel_label(
        self,
        dm_mass,    # D+45 (49)
        stop_mass,  # D+56 (60)
        a0_mass,    # D+42 (46)
        sigma_ann,  # D+87 (91)
        stau_mass,  # D+61 (65)
        bino,       # D+72 (76)
    ):
        """
        Assigns one of five labels to the sample according to its DM
        annihilation channel.
            0) light chi            D+45
            1) stop coannihilation
            2) A0-pole annihilation
            3) stau coannihilation
            4) well-tempered
            5) other
        """
        if (dm_mass < 70):
            return 0
        elif (self.diff(dm_mass, stop_mass) < 0.2):
            return 1
        elif (
            (self.diff(2*dm_mass, a0_mass) < 0.4) and
            (sigma_ann > 2e-27)
        ):
            return 2
        elif (self.diff(dm_mass, stau_mass) < 0.2):
            return 3
        elif (bino*bino < 0.9):
            return 4
        else:
            return 5
