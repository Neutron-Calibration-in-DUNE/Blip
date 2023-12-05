"""
Arrakis dataset class.  This class processes LArSoft Arrakis and ndlar-flow Arrakis output
and constructs BlipDatasets from them.  It also generates specific datasets for tasks 
such as the singles needed to train BlipGraph, etc.
"""
import uproot
import os
import getpass
import socket
import numpy as np
from datetime import datetime
import h5py
import imageio
from matplotlib import pyplot as plt

from blip.utils.logger import Logger
from blip.utils.utils import get_files_with_extension
from blip.dataset.common import classification_labels


class Arrakis:
    def __init__(
        self,
        name:   str = "arrakis",
        config: dict = {},
        meta:   dict = {}
    ):
        self.name = name + '_arrakis'
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
        self.logger.info("constructing arrakis dataset.")

        self.wire_experiments = ['protodune', 'microboone', 'icarus', 'sbnd', 'protodune_vd']
        self.larpix_experiments = ['2x2']

        self.wire_process_types = []
        self.larpix_process_types = []

        """
        ProtoDUNE channel mappings for different
        TPCs.  Only some of the view2 TPCs are
        part of the active volume, the rest are cryostat
        wall facing.
        """
        self.protodune_active_tpcs_view2 = [
            "tpc1", "tpc2", "tpc5", "tpc6", "tpc9", "tpc10"
        ]
        self.protodune_tpc_wire_channels = {
            "tpc0": [[0, 799], [800, 1599], [1600, 2079]],
            "tpc1": [[0, 799], [800, 1599], [2080, 2559]],
            "tpc2": [[2560, 3359], [3360, 4159], [4160, 4639]],
            "tpc3": [[2560, 3359], [3360, 4159], [4640, 5119]],
            "tpc4": [[5120, 5919], [5920, 6719], [6720, 7199]],
            "tpc5": [[5120, 5919], [5920, 6719], [7200, 7679]],
            "tpc6": [[7680, 8479], [8480, 9279], [9280, 9759]],
            "tpc7": [[7680, 8479], [8480, 9279], [9760, 10239]],
            "tpc8": [[10240, 11039], [11040, 11839], [11840, 12319]],
            "tpc9": [[10240, 11039], [11040, 11839], [12320, 12799]],
            "tpc10": [[12800, 13599], [13600, 14399], [14400, 14879]],
            "tpc11": [[12800, 13599], [13600, 14399], [14880, 15359]],
        }
        self.protodune_tpc_positions = {
            "tpc0": [[-376.8501, -366.8851], [0., 607.49875], [-0.49375, 231.16625]],
            "tpc1": [[-359.2651,   -0.1651], [0., 607.49875], [-0.49375, 231.16625]],
            "tpc2": [[0.1651, 359.2651],    [0., 607.49875], [-0.49375, 231.16625]],
            "tpc3": [[366.8851, 376.8501],  [0., 607.49875], [-0.49375, 231.16625]],
            "tpc4": [[-376.8501, -366.8851], [0., 607.49875], [231.56625, 463.22625]],
            "tpc5": [[-359.2651, -0.1651],  [0., 607.49875], [231.56625, 463.22625]],
            "tpc6": [[0.1651, 359.2651],    [0., 607.49875], [231.56625, 463.22625]],
            "tpc7": [[366.8851, 376.8501],  [0., 607.49875], [231.56625, 463.22625]],
            "tpc8": [[-376.8501, -366.8851], [0., 607.49875], [463.62625, 695.28625]],
            "tpc9": [[-359.2651, -0.1651],  [0., 607.49875], [463.62625, 695.28625]],
            "tpc10": [[0.1651, 359.2651],   [0., 607.49875], [463.62625, 695.28625]],
            "tpc11": [[366.8851, 376.8501], [0., 607.49875], [463.62625, 695.28625]],
        }
        self.protodune_active_tpc_views = {}

        """
        MicroBooNE channel mappings for different
        TPCs.  Only some of the view2 TPCs are
        part of the active volume, the rest are cryostat
        wall facing.
        """
        self.microboone_active_tpcs_view2 = {

        }
        self.microboone_tpc_wire_channels = {

        }
        self.microboone_tpc_positions = {

        }

        """
        ICARUS channel mappings for different
        TPCs.  Only some of the view2 TPCs are
        part of the active volume, the rest are cryostat
        wall facing.
        """
        self.icarus_active_tpcs_view2 = {

        }
        self.icarus_tpc_wire_channels = {

        }
        self.icarus_tpc_positions = {

        }

        """
        SBND channel mappings for different
        TPCs.  Only some of the view2 TPCs are
        part of the active volume, the rest are cryostat
        wall facing.
        """
        self.sbnd_active_tpcs_view2 = {

        }
        self.sbnd_tpc_wire_channels = {

        }
        self.sbnd_tpc_positions = {

        }

        """
        ProtoDUNE Vertical Drift channel mappings for different
        TPCs.  Only some of the view2 TPCs are
        part of the active volume, the rest are cryostat
        wall facing.
        """
        self.protodune_vd_active_tpcs_view2 = {

        }
        self.protodune_vd_tpc_wire_channels = {

        }
        self.protodune_vd_tpc_positions = {

        }

        self.active_tpcs_view2 = {}
        self.tpc_wire_channels = {}
        self.tpc_positions = {}

        self.parse_config()

    def parse_config(self):
        if "experiment" not in self.config.keys():
            self.logger.warn('no experiment specified in config! Setting to ProtoDUNE.')
            self.config['experiment'] = 'protodune'
        if self.config['experiment'] == 'protodune':
            self.logger.info('setting Arrakis parameters for ProtoDUNE')
            self.active_tpcs_view2 = self.protodune_active_tpcs_view2
            self.tpc_wire_channels = self.protodune_tpc_wire_channels
            self.tpc_positions = self.protodune_tpc_positions
        elif self.config['experiment'] == 'microboone':
            self.logger.info('setting Arrakis parameters for MicroBooNE')
            self.active_tpcs_view2 = self.microboone_active_tpcs_view2
            self.tpc_wire_channels = self.microboone_tpc_wire_channels
            self.tpc_positions = self.microboone_tpc_positions
        elif self.config['experiment'] == 'icarus':
            self.logger.info('setting Arrakis parameters for ICARUS')
            self.active_tpcs_view2 = self.icarus_active_tpcs_view2
            self.tpc_wire_channels = self.icarus_tpc_wire_channels
            self.tpc_positions = self.icarus_tpc_positions
        elif self.config['experiment'] == 'sbnd':
            self.logger.info('setting Arrakis parameters for SBND')
            self.active_tpcs_view2 = self.sbnd_active_tpcs_view2
            self.tpc_wire_channels = self.sbnd_tpc_wire_channels
            self.tpc_positions = self.sbnd_tpc_positions
        elif self.config['experiment'] == 'protodune_vd':
            self.logger.info('setting Arrakis parameters for ProtoDUNE-VD')
            self.active_tpcs_view2 = self.protodune_vd_active_tpcs_view2
            self.tpc_wire_channels = self.protodune_vd_tpc_wire_channels
            self.tpc_positions = self.protodune_vd_tpc_positions
        elif self.config['experiment'] == '2x2':
            self.logger.info('setting Arrakis parameters for the 2x2')
        else:
            self.logger.error(f'specified experiment "{self.config["experiment"]}" not an allowed type!')

        if "simulation_folder" not in self.config.keys():
            self.logger.warn('simulation_folder not specified in config! Setting to "/local_data/".')
            self.config['simulation_folder'] = '/local_data/'
        self.simulation_folder = self.config['simulation_folder']
        if "simulation_files" not in self.config.keys():
            self.logger.warn('simulation_files not specified in config! setting to "[]"')
            self.config['simulation_files'] = []

        # if simulation_files == [], grab all .root or .h5 files in the simulation_folder
        if self.config['experiment'] in self.wire_experiments:
            if self.config['simulation_files'] == []:
                self.logger.info(
                    f'no simulation_files specified, grabbing all .root files in directory {self.simulation_folder}'
                )
                self.config['simulation_files'] = get_files_with_extension(self.simulation_folder, '.root')
            self.simulation_files = self.config['simulation_files']
            # create output folders for processed simulation
            self.output_folders = {
                simulation_file: simulation_file.replace('.root', '')
                for simulation_file in self.simulation_files
            }
        else:
            if self.config['simulation_files'] == []:
                self.logger.info(
                    f'no simulation_files specified, grabbing all .h5 files in directory {self.simulation_folder}'
                )
                self.config['simulation_files'] = get_files_with_extension(self.simulation_folder, '.h5')
            self.simulation_files = self.config['simulation_files']
            # create output folders for processed simulation
            self.output_folders = {
                simulation_file: simulation_file.replace('.h5', '')
                for simulation_file in self.simulation_files
            }
        for output_folder in self.output_folders.values():
            if not os.path.isdir(f"/local_data/{output_folder}"):
                os.makedirs(f"/local_data/{output_folder}")

        if "process_type" not in self.config.keys():
            self.logger.warn('process_type not specified in config! Setting to "[all]".')
            self.config["process_type"] = ["all"]
        self.process_type = self.config["process_type"]

    def load_root_arrays(
        self,
        input_file:    str = ''
    ):
        try:
            self.uproot_file = uproot.open(self.simulation_folder + input_file)
        except:
            self.logger.error(
                f'error while atttempting to load input file {self.simulation_folder + input_file}'
            )

        self.mc_map = None
        self.energy_deposit_point_cloud = None
        self.wire_plane_point_cloud = None
        self.op_det_point_cloud = None
        for key in self.uproot_file.keys():
            if 'mc_edep_point_cloud' in key:
                self.energy_deposit_point_cloud = self.uproot_file[key].arrays(library="np")
            elif 'mc_wire_plane_point_cloud' in key:
                self.wire_plane_point_cloud = self.uproot_file[key].arrays(library="np")
            elif 'mc_op_det_point_cloud' in key:
                self.op_det_point_cloud = self.uproot_file[key].arrays(library="np")
            elif 'mc_maps' in key:
                self.mc_map = self.uproot_file[key].arrays(library="np")

    def load_flow_arrays(
        self,
        input_file:    str = ''
    ):
        try:
            self.h5_file = h5py.File(self.simulation_folder + input_file, 'r')
        except:
            self.logger.error(
                f'error while atttempting to load input file {self.simulation_folder + input_file}'
            )

    def prep_larsoft_training_data(self):
        self.meta = {}
        self.mc_maps = {}
        self.energy_deposit_point_clouds = {}
        self.wire_plane_point_clouds = {}
        self.op_det_point_clouds = {}

        for tpc, tpc_ranges in self.tpc_positions.items():
            self.meta[tpc] = {
                "who_created":      getpass.getuser(),
                "when_created":     datetime.now().strftime("%m-%d-%Y-%H:%M:%S"),
                "where_created":    socket.gethostname(),
                "view_features": {
                    "channel": 0, "tdc": 1, "adc": 2
                },
                "edep_features": {
                    "t": 0, "x": 1, "y": 2, "z": 3, "energy": 4, "num_photons": 5, "num_electrons": 6
                },
                "classes": {
                    "source": 0, "topology": 1, "particle": 2, "physics": 3, "hit": 4
                },
                "clusters": {
                    "topology":  0, "particle": 1, "physics": 2
                },
                "hits": {
                    "mean": 0, "rms": 1, "amplitude": 2, "charge": 3
                },
                "source_labels": {
                    key: value
                    for key, value in classification_labels["source"].items()
                },
                "topology_labels": {
                    key: value
                    for key, value in classification_labels["topology"].items()
                },
                "particle_labels": {
                    key: value
                    for key, value in classification_labels["particle"].items()
                },
                "physics_labels": {
                    key: value
                    for key, value in classification_labels["physics"].items()
                },
                "hit_labels": {
                    key: value
                    for key, value in classification_labels["hit"].items()
                },
            }
            self.mc_maps[tpc] = {
                'pdg_code': [],
                'parent_track_id': [],
                'ancestor_track_id': [],
                'ancestor_level': []
            }
            self.energy_deposit_point_clouds[tpc] = {
                'edep_features': [],
                'edep_classes':  [],
                'edep_clusters': [],
            }
            self.wire_plane_point_clouds[tpc] = {
                'view_0_features':  [],
                'view_0_classes':   [],
                'view_0_clusters':  [],
                'view_0_hits':      [],
                'view_1_features':  [],
                'view_1_classes':   [],
                'view_1_clusters':  [],
                'view_1_hits':      [],
                'view_2_features':  [],
                'view_2_classes':   [],
                'view_2_clusters':  [],
                'view_2_hits':      []
            }

    def generate_larsoft_training_data(
        self,
        input_file:   str = '',
        limit_tpcs:   list = []
    ):
        self.prep_larsoft_training_data()
        for process in self.process_type:
            if process == 'energy_deposit_point_cloud':
                self.generate_larsoft_energy_deposit_point_cloud(
                    self.simulation_folder + input_file,
                    limit_tpcs=limit_tpcs
                )
            elif process == 'wire_plane_point_cloud':
                self.generate_larsoft_wire_plane_point_cloud(
                    self.simulation_folder + input_file,
                    limit_tpcs=limit_tpcs
                )
            elif process == 'op_det_point_cloud':
                self.generate_larsoft_op_det_point_cloud(
                    self.simulation_folder + input_file,
                    limit_tpcs=limit_tpcs
                )
            elif process == 'mc_maps':
                self.generate_larsoft_mc_maps(
                    self.simulation_folder + input_file,
                    limit_tpcs=limit_tpcs
                )
            elif process == 'all':
                self.generate_larsoft_energy_deposit_point_cloud(
                    self.simulation_folder + input_file,
                    limit_tpcs=limit_tpcs
                )
                self.generate_larsoft_wire_plane_point_cloud(
                    self.simulation_folder + input_file,
                    limit_tpcs=limit_tpcs
                )
                self.generate_larsoft_op_det_point_cloud(
                    self.simulation_folder + input_file,
                    limit_tpcs=limit_tpcs
                )
                self.generate_larsoft_mc_maps(
                    self.simulation_folder + input_file,
                    limit_tpcs=limit_tpcs
                )
            else:
                self.logger.error(f'specified process type {process} not allowed!')

        for tpc, tpc_ranges in self.tpc_positions.items():
            if len(limit_tpcs) != 0 and tpc not in limit_tpcs:
                continue
            np.savez(
                f"/local_data/{self.output_folders[input_file]}/{tpc}.npz",
                edep_features=self.energy_deposit_point_clouds[tpc]['edep_features'],
                edep_classes=self.energy_deposit_point_clouds[tpc]['edep_classes'],
                edep_clusters=self.energy_deposit_point_clouds[tpc]['edep_clusters'],
                view_0_features=self.wire_plane_point_clouds[tpc]['view_0_features'],
                view_0_classes=self.wire_plane_point_clouds[tpc]['view_0_classes'],
                view_0_clusters=self.wire_plane_point_clouds[tpc]['view_0_clusters'],
                view_0_hits=self.wire_plane_point_clouds[tpc]['view_0_hits'],
                view_1_features=self.wire_plane_point_clouds[tpc]['view_1_features'],
                view_1_classes=self.wire_plane_point_clouds[tpc]['view_1_classes'],
                view_1_clusters=self.wire_plane_point_clouds[tpc]['view_1_clusters'],
                view_1_hits=self.wire_plane_point_clouds[tpc]['view_1_hits'],
                view_2_features=self.wire_plane_point_clouds[tpc]['view_2_features'],
                view_2_classes=self.wire_plane_point_clouds[tpc]['view_2_classes'],
                view_2_clusters=self.wire_plane_point_clouds[tpc]['view_2_clusters'],
                view_2_hits=self.wire_plane_point_clouds[tpc]['view_2_hits'],
                mc_maps=self.mc_maps[tpc],
                meta=self.meta[tpc]
            )

    def generate_larpix_training_data(
        self,
        input_file:   str = '',
        limit_tpcs:   list = [], 
    ):
        pass

    def generate_larsoft_mc_maps(
        self,
        input_file:         str = '',
        limit_tpcs:         list = [],
    ):
        if self.mc_maps is None:
            self.logger.warn(f'no mc_maps data in file {input_file}!')
            return
        self.logger.info(
            f"generating 'mc_maps' training data from file: {input_file}"
        )
        for tpc, tpc_ranges in self.tpc_wire_channels.items():
            if len(limit_tpcs) != 0 and tpc not in limit_tpcs:
                continue
            for event in range(len(self.mc_map['pdg_code_map.first'])):
                self.mc_maps[tpc]['pdg_code'].append({
                    self.mc_map['pdg_code_map.first'][event][ii]: self.mc_map['pdg_code_map.second'][event][ii]
                    for ii in range(len(self.mc_map['pdg_code_map.first'][event]))
                })
                self.mc_maps[tpc]['parent_track_id'].append({
                    self.mc_map['parent_track_id_map.first'][event][ii]: self.mc_map['parent_track_id_map.second'][event][ii]
                    for ii in range(len(self.mc_map['parent_track_id_map.first'][event]))
                })
                self.mc_maps[tpc]['ancestor_track_id'].append({
                    self.mc_map['ancestor_track_id_map.first'][event][ii]: self.mc_map['ancestor_track_id_map.second'][event][ii]
                    for ii in range(len(self.mc_map['ancestor_track_id_map.first'][event]))
                })
                self.mc_maps[tpc]['ancestor_level'].append({
                    self.mc_map['ancestor_level_map.first'][event][ii]: self.mc_map['ancestor_level_map.second'][event][ii]
                    for ii in range(len(self.mc_map['ancestor_level_map.first'][event]))
                })

    def generate_larsoft_energy_deposit_point_cloud(
        self,
        input_file:         str = '',
        separate_unique:    bool = False,
        unique_label:       str = 'topology',
        replace_topology_label:  int = -1,
        replace_particle_label:  int = -1,
        replace_physics_label:  int = -1,
        max_events:     int = 5000,
        limit_tpcs:     list = [],
        make_gifs:      bool = False
    ):
        """
        We iterate over each tpc and collect all (x,y,z) points for each
        point cloud into a features array, together with (source, topology, particle) as
        the categorical information and (topology, particle) as clustering
        information.
        """
        if self.energy_deposit_point_cloud is None:
            self.logger.warn(f'no energy_deposit_point_cloud data in file {input_file}!')
            return
        self.logger.info(
            f"generating 'energy_deposit_point_cloud' training data from file: {input_file}"
        )
        edep_t = self.energy_deposit_point_cloud['edep_t']
        edep_x = self.energy_deposit_point_cloud['edep_x']
        edep_y = self.energy_deposit_point_cloud['edep_y']
        edep_z = self.energy_deposit_point_cloud['edep_z']
        edep_energy = self.energy_deposit_point_cloud['edep_energy']
        edep_num_photons = self.energy_deposit_point_cloud['edep_num_photons']
        edep_num_electrons = self.energy_deposit_point_cloud['edep_num_electrons']

        # construct ids and names for source, topology and particle labels
        source_label = self.energy_deposit_point_cloud['source_label']
        topology_label = self.energy_deposit_point_cloud['topology_label']
        particle_label = self.energy_deposit_point_cloud['particle_label']
        physics_label = self.energy_deposit_point_cloud['physics_label']
        unique_topology_label = self.energy_deposit_point_cloud['unique_topology']
        unique_particle_label = self.energy_deposit_point_cloud['unique_particle']
        unique_physics_label = self.energy_deposit_point_cloud['unique_physics']

        for tpc, tpc_ranges in self.tpc_positions.items():
            if len(limit_tpcs) != 0 and tpc not in limit_tpcs:
                continue
            edep_t_tpc = []
            edep_x_tpc = []
            edep_y_tpc = []
            edep_z_tpc = []
            edep_energy_tpc = []
            edep_num_photons_tpc = []
            edep_num_electrons_tpc = []
            source_label_tpc = []
            topology_label_tpc = []
            particle_label_tpc = []
            physics_label_tpc = []
            unique_topology_label_tpc = []
            unique_particle_label_tpc = []
            unique_physics_label_tpc = []

            for event in range(len(edep_t)):
                view_mask = (
                    (edep_x[event] >= tpc_ranges[0][0]) &
                    (edep_x[event] < tpc_ranges[0][1]) &
                    (edep_y[event] >= tpc_ranges[1][0]) &
                    (edep_y[event] < tpc_ranges[1][1]) &
                    (edep_z[event] >= tpc_ranges[2][0]) &
                    (edep_z[event] < tpc_ranges[2][1]) &
                    # (source_label[event] >= 0) &        # we don't want 'undefined' points in our dataset.
                    (topology_label[event] >= 0) &         # i.e., things with a label == -1
                    (particle_label[event] >= 0)
                )
                if np.sum(view_mask) > 0:
                    if separate_unique:
                        if unique_label == 'topology':
                            unique_labels = unique_topology_label[event]
                        elif unique_label == 'particle':
                            unique_labels = unique_particle_label[event]
                        elif unique_label == 'physics':
                            unique_labels = unique_physics_label[event]
                        else:
                            self.logger.error(f'specified unique_label type {unique_label} not allowed!')
                        for label in np.unique(unique_labels):
                            unique_mask = view_mask & (unique_labels == label)
                            edep_t_tpc.append(edep_t[event][unique_mask])
                            edep_x_tpc.append(edep_x[event][unique_mask])
                            edep_y_tpc.append(edep_y[event][unique_mask])
                            edep_z_tpc.append(edep_z[event][unique_mask])
                            edep_energy_tpc.append(edep_energy[event][unique_mask])
                            edep_num_photons_tpc.append(edep_num_photons[event][unique_mask])
                            edep_num_electrons_tpc.append(edep_num_electrons[event][unique_mask])
                            source_label_tpc.append(source_label[event][unique_mask])

                            # if we want to replace the topology label (temp solution for single gammas)
                            if replace_topology_label != -1:
                                topology_label[event][unique_mask] = replace_topology_label

                            topology_label_tpc.append(topology_label[event][unique_mask])

                            # if we want to replace the particle label (temp solution for single gammas)
                            if replace_particle_label != -1:
                                particle_label[event][unique_mask] = replace_particle_label

                            particle_label_tpc.append(particle_label[event][unique_mask])

                            # if we want to replace the physics label (temp solution for single gammas)
                            if replace_physics_label != -1:
                                physics_label[event][unique_mask] = replace_physics_label

                            physics_label_tpc.append(physics_label[event][unique_mask])
                            unique_topology_label_tpc.append(unique_topology_label[event][unique_mask])
                            unique_particle_label_tpc.append(unique_particle_label[event][unique_mask])
                            unique_physics_label_tpc.append(unique_physics_label[event][unique_mask])
                    else:
                        edep_t_tpc.append(edep_t[event][view_mask])
                        edep_x_tpc.append(edep_x[event][view_mask])
                        edep_y_tpc.append(edep_y[event][view_mask])
                        edep_z_tpc.append(edep_z[event][view_mask])
                        edep_energy_tpc.append(edep_energy[event][view_mask])
                        edep_num_photons_tpc.append(edep_num_photons[event][view_mask])
                        edep_num_electrons_tpc.append(edep_num_electrons[event][view_mask])
                        source_label_tpc.append(source_label[event][view_mask])
                        topology_label_tpc.append(topology_label[event][view_mask])
                        particle_label_tpc.append(particle_label[event][view_mask])
                        physics_label_tpc.append(physics_label[event][view_mask])
                        unique_topology_label_tpc.append(unique_topology_label[event][view_mask])
                        unique_particle_label_tpc.append(unique_particle_label[event][view_mask])
                        unique_physics_label_tpc.append(unique_physics_label[event][view_mask])

                if len(edep_t_tpc) >= max_events:
                    self.logger.info(f'reached max_events: {max_events} for input file {input_file}; returning.')
                    break

            edep_t_tpc = np.array(edep_t_tpc, dtype=object)
            edep_x_tpc = np.array(edep_x_tpc, dtype=object)
            edep_y_tpc = np.array(edep_y_tpc, dtype=object)
            edep_z_tpc = np.array(edep_z_tpc, dtype=object)
            edep_energy_tpc = np.array(edep_energy_tpc, dtype=object)
            edep_num_photons_tpc = np.array(edep_num_photons_tpc, dtype=object)
            edep_num_electrons_tpc = np.array(edep_num_electrons_tpc, dtype=object)
            source_label_tpc = np.array(source_label_tpc, dtype=object)
            topology_label_tpc = np.array(topology_label_tpc, dtype=object)
            particle_label_tpc = np.array(particle_label_tpc, dtype=object)
            physics_label_tpc = np.array(physics_label_tpc, dtype=object)
            unique_topology_label_tpc = np.array(unique_topology_label_tpc, dtype=object)
            unique_particle_label_tpc = np.array(unique_particle_label_tpc, dtype=object)
            unique_physics_label_tpc = np.array(unique_physics_label_tpc, dtype=object)

            if len(edep_t_tpc.flatten()) == 0:
                continue
            features = np.array([
                np.vstack((
                    edep_t_tpc[ii],
                    edep_x_tpc[ii],
                    edep_y_tpc[ii],
                    edep_z_tpc[ii],
                    edep_energy_tpc[ii],
                    edep_num_photons_tpc[ii],
                    edep_num_electrons_tpc[ii])).T
                for ii in range(len(edep_t_tpc))],
                dtype=object
            )
            classes = np.array([
                np.vstack((
                    source_label_tpc[ii],
                    topology_label_tpc[ii],
                    particle_label_tpc[ii],
                    physics_label_tpc[ii])).T
                for ii in range(len(edep_t_tpc))],
                dtype=object
            )
            clusters = np.array([
                np.vstack((
                    unique_topology_label_tpc[ii],
                    unique_particle_label_tpc[ii],
                    unique_physics_label_tpc[ii])).T
                for ii in range(len(edep_t_tpc))],
                dtype=object
            )

            self.meta[tpc]["num_events"] = len(features)
            self.meta[tpc]["edep_source_points"] = {
                key: np.count_nonzero(np.concatenate(source_label_tpc) == key)
                for key, value in classification_labels["source"].items()
            }
            self.meta[tpc]["edep_topology_points"] = {
                key: np.count_nonzero(np.concatenate(topology_label_tpc) == key)
                for key, value in classification_labels["topology"].items()
            }
            self.meta[tpc]["edep_particle_points"] = {
                key: np.count_nonzero(np.concatenate(particle_label_tpc) == key)
                for key, value in classification_labels["particle"].items()
            }
            self.meta[tpc]["edep_physics_points"] = {
                key: np.count_nonzero(np.concatenate(physics_label_tpc) == key)
                for key, value in classification_labels["physics"].items()
            }
            self.meta[tpc]["edep_total_points"] = len(np.concatenate(features))

            self.energy_deposit_point_clouds[tpc]['edep_features'] = features
            self.energy_deposit_point_clouds[tpc]['edep_classes'] = classes
            self.energy_deposit_point_clouds[tpc]['edep_clusters'] = clusters

    def generate_larsoft_wire_plane_point_cloud(
        self,
        input_file:         str = '',
        separate_unique:    bool = False,
        unique_label:       str = 'topology',
        replace_topology_label:  int = -1,
        replace_particle_label:  int = -1,
        replace_physics_label:  int = -1,
        limit_tpcs:     list = [],
        max_events:     int = 5000,
        make_gifs:      bool = False,
    ):
        """
        We iterate over each view (wire plane) and collect all
        (channel, tdc, adc) points for each point cloud into a features
        array, together with (source, topology, particle) as
        the categorical information and (topology, particle) as clustering
        information.
        """
        if self.wire_plane_point_cloud is None:
            self.logger.warn(f'no wire_plane_point_cloud data in file {input_file}!')
            return
        self.logger.info(
            f"generating 'wire_plane_point_cloud' training data from file: {input_file}"
        )

        channel = self.wire_plane_point_cloud['channel']
        tdc = self.wire_plane_point_cloud['tdc']
        energy = self.wire_plane_point_cloud['energy'] * 10e5
        adc = self.wire_plane_point_cloud['adc']

        # construct ids and names for source, topology and particle labels
        source_label = self.wire_plane_point_cloud['source_label']
        topology_label = self.wire_plane_point_cloud['topology_label']
        particle_label = self.wire_plane_point_cloud['particle_label']
        physics_label = self.wire_plane_point_cloud['physics_label']
        unique_topology_label = self.wire_plane_point_cloud['unique_topology']
        unique_particle_label = self.wire_plane_point_cloud['unique_particle']
        unique_physics_label = self.wire_plane_point_cloud['unique_physics']
        hit_mean = self.wire_plane_point_cloud['hit_mean']
        hit_rms = self.wire_plane_point_cloud['hit_rms']
        hit_amplitude = self.wire_plane_point_cloud['hit_amplitude']
        hit_charge = self.wire_plane_point_cloud['hit_charge']

        for tpc, tpc_ranges in self.tpc_wire_channels.items():
            self.wire_plane_point_cloud[tpc] = {}
            if len(limit_tpcs) != 0 and tpc not in limit_tpcs:
                continue
            for v, tpc_view in enumerate(tpc_ranges):
                """
                For each point cloud, we want to normalize adc against
                all point clouds in the data set, so that it is independent
                of the specific detector readout.
                """
                channel_view = []
                tdc_view = []
                adc_view = []
                energy_view = []
                source_label_view = []
                topology_label_view = []
                particle_label_view = []
                physics_label_view = []
                unique_topology_label_view = []
                unique_particle_label_view = []
                unique_physics_label_view = []
                hit_class_view = []
                hit_mean_view = []
                hit_rms_view = []
                hit_amplitude_view = []
                hit_charge_view = []

                gif_frames = []

                for event in range(len(channel)):
                    view_mask = (
                        (channel[event] >= tpc_view[0]) &
                        (channel[event] < tpc_view[1]) &
                        # (source_label[event] >= 0) &        # we don't want 'undefined' points in our dataset.
                        (topology_label[event] > 0) &         # i.e., things with a label == -1
                        (particle_label[event] > 0)
                    )
                    if np.sum(view_mask) > 0:
                        # if we want to separate out unique instances of event types
                        if separate_unique:
                            if unique_label == 'topology':
                                unique_labels = unique_topology_label[event]
                            elif unique_label == 'particle':
                                unique_labels = unique_particle_label[event]
                            elif unique_label == 'physics':
                                unique_labels = unique_physics_label[event]
                            else:
                                self.logger.error(f'specified unique_label type {unique_label} not allowed!')
                            for label in np.unique(unique_labels):
                                unique_mask = view_mask & (unique_labels == label)
                                if label == -1:
                                    continue
                                if sum(unique_mask) < 3:
                                    continue
                                if np.sum(adc[event][unique_mask]) == 0:
                                    continue
                                channel_view.append(channel[event][unique_mask])
                                tdc_view.append(tdc[event][unique_mask])
                                adc_view.append(adc[event][unique_mask])
                                energy_view.append(energy[event][unique_mask])
                                source_label_view.append(source_label[event][unique_mask])

                                # if we want to replace the topology label (temp solution for single gammas)
                                if replace_topology_label != -1:
                                    topology_label[event][unique_mask] = replace_topology_label

                                topology_label_view.append(topology_label[event][unique_mask])

                                # if we want to replace the particle label (temp solution for single gammas)
                                if replace_particle_label != -1:
                                    particle_label[event][unique_mask] = replace_particle_label

                                particle_label_view.append(particle_label[event][unique_mask])

                                # if we want to replace the physics label (temp solution for single gammas)
                                if replace_physics_label != -1:
                                    physics_label[event][unique_mask] = replace_physics_label

                                physics_label_view.append(physics_label[event][unique_mask])
                                unique_topology_label_view.append(unique_topology_label[event][unique_mask])
                                unique_particle_label_view.append(unique_particle_label[event][unique_mask])
                                unique_physics_label_view.append(unique_physics_label[event][unique_mask])

                                hit_class = np.zeros_like(hit_mean[event][unique_mask])
                                hit_class[(hit_mean[event][unique_mask] != -1)] = 1
                                hit_class_view.append(hit_class)
                                hit_mean_view.append(hit_mean[event][unique_mask])
                                hit_rms_view.append(hit_rms[event][unique_mask])
                                hit_amplitude_view.append(hit_amplitude[event][unique_mask])
                                hit_charge_view.append(hit_charge[event][unique_mask])

                                print(channel[event][unique_mask])
                                print(tdc[event][unique_mask])
                                print(adc[event][unique_mask])
                                print(unique_particle_label[event][unique_mask])
                                print(unique_topology_label[event][unique_mask])
                                print(particle_label[event][unique_mask])

                                # create animated GIF of events
                                if make_gifs:
                                    pos = np.vstack((
                                        channel[event][unique_mask],
                                        tdc[event][unique_mask],
                                        np.abs(adc[event][unique_mask])
                                    )).astype(float)
                                    summed_adc = np.sum(adc[event][unique_mask])
                                    mins = np.min(pos, axis=1)
                                    maxs = np.max(pos, axis=1)
                                    for kk in range(len(pos)-1):
                                        denom = (maxs[kk] - mins[kk])
                                        if denom == 0:
                                            pos[kk] = 0 * pos[kk]
                                        else:
                                            pos[kk] = 2 * (pos[kk] - mins[kk])/(denom) - 1

                                    self.create_class_gif_frame(
                                        pos,
                                        'test',
                                        summed_adc,
                                        event,
                                        [mins[1], maxs[1]],
                                        [mins[0], maxs[0]]
                                    )
                                    gif_frames.append(
                                        imageio.v2.imread(
                                            f"/local_data/blip_plots/.img/img_{event}.png"
                                        )
                                    )

                        # otherwise save the entire view
                        else:
                            channel_view.append(channel[event][view_mask])
                            tdc_view.append(tdc[event][view_mask])
                            adc_view.append(adc[event][view_mask])
                            energy_view.append(energy[event][view_mask])
                            source_label_view.append(source_label[event][view_mask])
                            topology_label_view.append(topology_label[event][view_mask])
                            particle_label_view.append(particle_label[event][view_mask])
                            physics_label_view.append(physics_label[event][view_mask])
                            unique_topology_label_view.append(unique_topology_label[event][view_mask])
                            unique_particle_label_view.append(unique_particle_label[event][view_mask])
                            unique_physics_label_view.append(unique_physics_label[event][view_mask])

                            hit_class = np.zeros_like(hit_mean[event][view_mask])
                            hit_class[(hit_mean[event][view_mask] != -1)] = 1
                            hit_class_view.append(hit_class)
                            hit_mean_view.append(hit_mean[event][view_mask])
                            hit_rms_view.append(hit_rms[event][view_mask])
                            hit_amplitude_view.append(hit_amplitude[event][view_mask])
                            hit_charge_view.append(hit_charge[event][view_mask])

                    if len(channel_view) >= max_events:
                        self.logger.info(f'reached max_events: {max_events} for input file {input_file}; returning.')
                        break

                if make_gifs:
                    if len(gif_frames) == 0:
                        continue
                    imageio.mimsave(
                        "/local_data/blip_plots/singles_test.gif",
                        gif_frames,
                        duration=2000
                    )

                channel_view = np.array(channel_view, dtype=object)
                tdc_view = np.array(tdc_view, dtype=object)
                adc_view = np.array(adc_view, dtype=object)
                energy_view = np.array(energy_view, dtype=object)
                source_label_view = np.array(source_label_view, dtype=object)
                topology_label_view = np.array(topology_label_view, dtype=object)
                particle_label_view = np.array(particle_label_view, dtype=object)
                physics_label_view = np.array(physics_label_view, dtype=object)
                unique_topology_label_view = np.array(unique_topology_label_view, dtype=object)
                unique_particle_label_view = np.array(unique_particle_label_view, dtype=object)
                unique_physics_label_view = np.array(unique_physics_label_view, dtype=object)
                hit_class_view = np.array(hit_class_view, dtype=object)
                hit_mean_view = np.array(hit_mean_view, dtype=object)
                hit_rms_view = np.array(hit_rms_view, dtype=object)
                hit_amplitude_view = np.array(hit_amplitude_view, dtype=object)
                hit_charge_view = np.array(hit_charge_view, dtype=object)

                adc_view_sum = np.array([sum(a) for a in adc_view])
                adc_view_normalized = adc_view / adc_view_sum

                if len(channel_view.flatten()) == 0:
                    continue
                features = np.array([
                    np.vstack((channel_view[ii], tdc_view[ii], adc_view_normalized[ii])).T
                    for ii in range(len(channel_view))],
                    dtype=object
                )
                classes = np.array([
                    np.vstack((
                        source_label_view[ii],
                        topology_label_view[ii],
                        particle_label_view[ii],
                        physics_label_view[ii],
                        hit_class_view[ii])).T
                    for ii in range(len(channel_view))],
                    dtype=object
                )
                clusters = np.array([
                    np.vstack((
                        unique_topology_label_view[ii],
                        unique_particle_label_view[ii],
                        unique_physics_label_view[ii])).T
                    for ii in range(len(channel_view))],
                    dtype=object
                )
                hits = np.array([
                    np.vstack((
                        hit_mean_view[ii],
                        hit_rms_view[ii],
                        hit_amplitude_view[ii],
                        hit_charge_view[ii])).T
                    for ii in range(len(channel_view))],
                    dtype=object
                )

                self.meta[tpc]["num_events"] = len(features)
                self.meta[tpc][f"view_{v}_source_points"] = {
                    key: np.count_nonzero(np.concatenate(source_label_view) == key)
                    for key, value in classification_labels["source"].items()
                }
                self.meta[tpc][f"view_{v}_topology_points"] = {
                    key: np.count_nonzero(np.concatenate(topology_label_view) == key)
                    for key, value in classification_labels["topology"].items()
                }
                self.meta[tpc][f"view_{v}_particle_points"] = {
                    key: np.count_nonzero(np.concatenate(particle_label_view) == key)
                    for key, value in classification_labels["particle"].items()
                }
                self.meta[tpc][f"view_{v}_physics_points"] = {
                    key: np.count_nonzero(np.concatenate(physics_label_view) == key)
                    for key, value in classification_labels["physics"].items()
                }
                self.meta[tpc][f"view_{v}_total_points"] = len(np.concatenate(features))
                self.meta[tpc][f"view_{v}_adc_sum"] = adc_view_sum
                self.wire_plane_point_clouds[tpc][f'view_{v}_features'] = features
                self.wire_plane_point_clouds[tpc][f'view_{v}_classes'] = classes
                self.wire_plane_point_clouds[tpc][f'view_{v}_clusters'] = clusters
                self.wire_plane_point_clouds[tpc][f'view_{v}_hits'] = hits

    def generate_larsoft_op_det_point_cloud(
        self,
        input_file:         str = '',
        separate_unique:    bool = False,
        unique_label:       str = 'topology',
        replace_physics_label:  int = -1,
        max_events:     int = 5000,
        limit_tpcs:     list = [],
        make_gifs:      bool = False
    ):
        """
        """
        if self.op_det_point_cloud is None:
            self.logger.warn(f'no op_det_point_cloud data in file {input_file}!')
            return
        self.logger.info(
            f"generating 'op_det_point_cloud' training data from file: {input_file}"
        )

    def generate_larsoft_singles_training_data(
        self,
        input_file:     str = '',
        unique_label:   str = 'topology',
        replace_topology_label:  int = -1,
        replace_particle_label:  int = -1,
        replace_physics_label:  int = -1,
        max_events:     int = 5000,
        limit_tpcs:     list = [],
        make_gifs:      bool = False
    ):
        """
        """
        if make_gifs:
            if not os.path.isdir("/local_data/blip_plots/"):
                os.makedirs("/local_data/blip_plots/")
            if not os.path.isdir("/local_data/blip_plots/.img"):
                os.makedirs("/local_data/blip_plots/.img")

        self.prep_larsoft_training_data()
        for process in self.process_type:
            if process == 'energy_deposit_point_cloud':
                self.generate_larsoft_energy_deposit_point_cloud(
                    self.simulation_folder + input_file,
                    separate_unique=True,
                    unique_label=unique_label,
                    replace_topology_label=replace_topology_label,
                    replace_particle_label=replace_particle_label,
                    replace_physics_label=replace_physics_label,
                    max_events=max_events,
                    limit_tpcs=limit_tpcs,
                    make_gifs=make_gifs
                )
            elif process == 'wire_plane_point_cloud':
                self.generate_larsoft_wire_plane_point_cloud(
                    self.simulation_folder + input_file,
                    separate_unique=True,
                    unique_label=unique_label,
                    replace_topology_label=replace_topology_label,
                    replace_particle_label=replace_particle_label,
                    replace_physics_label=replace_physics_label,
                    max_events=max_events,
                    limit_tpcs=limit_tpcs,
                    make_gifs=make_gifs
                )
            elif process == 'op_det_point_cloud':
                self.generate_larsoft_op_det_point_cloud(
                    self.simulation_folder + input_file,
                    separate_unique=True,
                    unique_label=unique_label,
                    replace_topology_label=replace_topology_label,
                    replace_particle_label=replace_particle_label,
                    replace_physics_label=replace_physics_label,
                    max_events=max_events,
                    limit_tpcs=limit_tpcs,
                    make_gifs=make_gifs
                )
            elif process == 'mc_maps':
                self.generate_larsoft_mc_maps(
                    self.simulation_folder + input_file
                )
            elif process == 'all':
                self.generate_larsoft_energy_deposit_point_cloud(
                    self.simulation_folder + input_file,
                    separate_unique=True,
                    unique_label=unique_label,
                    replace_topology_label=replace_topology_label,
                    replace_particle_label=replace_particle_label,
                    replace_physics_label=replace_physics_label,
                    max_events=max_events,
                    limit_tpcs=limit_tpcs,
                    make_gifs=make_gifs
                )
                self.generate_larsoft_wire_plane_point_cloud(
                    self.simulation_folder + input_file,
                    separate_unique=True,
                    unique_label=unique_label,
                    replace_topology_label=replace_topology_label,
                    replace_particle_label=replace_particle_label,
                    replace_physics_label=replace_physics_label,
                    max_events=max_events,
                    limit_tpcs=limit_tpcs,
                    make_gifs=make_gifs
                )
                self.generate_larsoft_op_det_point_cloud(
                    self.simulation_folder + input_file,
                    separate_unique=True,
                    unique_label=unique_label,
                    replace_topology_label=replace_topology_label,
                    replace_particle_label=replace_particle_label,
                    replace_physics_label=replace_physics_label,
                    max_events=max_events,
                    limit_tpcs=limit_tpcs,
                    make_gifs=make_gifs
                )
                self.generate_larsoft_mc_maps(
                    self.simulation_folder + input_file
                )
            else:
                self.logger.error(f'specified process type {process} not allowed!')

        for tpc, tpc_ranges in self.tpc_positions.items():
            if len(limit_tpcs) != 0 and tpc not in limit_tpcs:
                continue
            np.savez(
                f"/local_data/{self.output_folders[input_file]}_singles/{tpc}.npz",
                edep_features=self.energy_deposit_point_clouds[tpc]['edep_features'],
                edep_classes=self.energy_deposit_point_clouds[tpc]['edep_classes'],
                edep_clusters=self.energy_deposit_point_clouds[tpc]['edep_clusters'],
                view_0_features=self.wire_plane_point_clouds[tpc]['view_0_features'],
                view_0_classes=self.wire_plane_point_clouds[tpc]['view_0_classes'],
                view_0_clusters=self.wire_plane_point_clouds[tpc]['view_0_clusters'],
                view_0_hits=self.wire_plane_point_clouds[tpc]['view_0_hits'],
                view_1_features=self.wire_plane_point_clouds[tpc]['view_1_features'],
                view_1_classes=self.wire_plane_point_clouds[tpc]['view_1_classes'],
                view_1_clusters=self.wire_plane_point_clouds[tpc]['view_1_clusters'],
                view_1_hits=self.wire_plane_point_clouds[tpc]['view_1_hits'],
                view_2_features=self.wire_plane_point_clouds[tpc]['view_2_features'],
                view_2_classes=self.wire_plane_point_clouds[tpc]['view_2_classes'],
                view_2_clusters=self.wire_plane_point_clouds[tpc]['view_2_clusters'],
                view_2_hits=self.wire_plane_point_clouds[tpc]['view_2_hits'],
                mc_maps=self.mc_maps[tpc],
                meta=self.meta[tpc]
            )

    def create_class_gif_frame(
        self,
        pos,
        class_label,
        summed_adc,
        image_number,
        xlim:   list = [-1.0, 1.0],
        ylim:   list = [-1.0, 1.0],
    ):
        # fix this later
        pass
        fig, axs = plt.subplots(figsize=(8, 8))
        axs.scatter(
            pos[0],   # channel
            pos[1],   # tdc
            marker='o',
            s=pos[2],
            c=pos[2],
            label=r"$\Sigma$"+f" ADC: {summed_adc:.2f}"
        )
        axs.set_xlim(-1.2, 1.2)
        axs.set_ylim(-1.2, 1.2)
        axs.set_xlabel("Channel [id normalized]")
        axs.set_ylabel("TDC (ns normalized)")
        plt.title(f"Point cloud {image_number} for class {class_label}")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(
            f"/local_data/blip_plots/.img/img_{image_number}.png",
            transparent=False,
            facecolor='white'
        )
        plt.close()
