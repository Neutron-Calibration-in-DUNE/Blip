from ctypes import sizeof
import uproot
import os
import getpass
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy import stats as st
from datetime import datetime

from blip.utils.logger import Logger
from blip.dataset.common import *

class Arrakis:
    def __init__(self,
        name:   str="arrakis",
        config: dict={},
        meta:   dict={}
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
            self.logger = Logger(name, file_mode="w")
        self.logger.info(f"constructing arrakis dataset.")

        self.simulation_files = []
        self.output_folders = {}
        
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
            "tpc0": [[0,799],[800,1599],[1600,2079]],
            "tpc1": [[0,799],[800,1599],[2080,2559]],
            "tpc2": [[2560,3359],[3360,4159],[4160,4639]],
            "tpc3": [[2560,3359],[3360,4159],[4640,5119]],
            "tpc4": [[5120,5919],[5920,6719],[6720,7199]],
            "tpc5": [[5120,5919],[5920,6719],[7200,7679]],
            "tpc6": [[7680,8479],[8480,9279],[9280,9759]],
            "tpc7": [[7680,8479],[8480,9279],[9760,10239]],
            "tpc8": [[10240,11039],[11040,11839],[11840,12319]],
            "tpc9": [[10240,11039],[11040,11839],[12320,12799]],
            "tpc10": [[12800,13599],[13600,14399],[14400,14879]],
            "tpc11": [[12800,13599],[13600,14399],[14880,15359]],
        }

        self.parse_config()

    def parse_config(self):
        if "simulation_folder" not in self.config.keys():
            self.logger.warn(f'simulation_folder not specified in config! Setting to "./".')
            self.config['simulation_folder'] = './'
        self.simulation_folder = self.config['simulation_folder']
        if "simulation_files" not in self.config.keys():
            self.logger.warn(f'simulation_files not specified in config!')
            self.config['simulation_files'] = []
        self.simulation_files = self.config['simulation_files']
        self.output_folders = {
            self.simulation_folder + simulation_file: simulation_file.replace('.root','') 
            for simulation_file in self.simulation_files
        }
        for output_folder in self.output_folders.values():
            if not os.path.isdir(f"data/{output_folder}"):
                os.makedirs(f"data/{output_folder}")
        if "process_type" not in self.config.keys():
            self.logger.warn(f'process_type not specified in config! Setting to "all".')
            self.config["process_type"] = "all"
        self.process_type = self.config["process_type"]
        if "process_simulation" in self.config.keys():
            if self.config["process_simulation"]:
                for ii, input_file in enumerate(self.simulation_files):
                    self.load_arrays(self.simulation_folder, input_file)
                    self.generate_training_data(self.process_type, self.simulation_folder + input_file)

    def load_arrays(self,
        input_folder:   str='',
        input_file:    str=''
    ):
        try:
            self.uproot_file = uproot.open(input_folder + input_file)
        except:
            self.logger.error(
                f'error while atttempting to load input file {input_folder + input_file}'
            )
        
        self.energy_deposit_point_cloud = None
        self.wire_plane_point_cloud = None
        self.op_det_point_cloud = None
        for key in self.uproot_file.keys():
            if 'energy_deposit_point_cloud' in key:
                self.energy_deposit_point_cloud = self.uproot_file[key].arrays(library="np")
            elif 'mc_wire_plane_point_cloud' in key:
                self.wire_plane_point_cloud = self.uproot_file[key].arrays(library="np")
            elif 'mc_op_det_point_cloud' in key:
                self.op_det_point_cloud = self.uproot_file[key].arrays(library="np")

    def generate_training_data(self,
        process_type:   str='all',
        input_file:    str=''
    ):
        if process_type == 'energy_deposit_point_cloud':
            self.generate_energy_deposit_point_cloud(input_file)
        elif process_type == 'view_tpc_point_cloud':
            self.generate_view_tpc_point_cloud(input_file)
        elif process_type == 'op_det_point_cloud':
            self.generate_op_det_point_cloud(input_file)
        elif process_type == 'all':
            self.generate_energy_deposit_point_cloud(input_file)
            self.generate_view_tpc_point_cloud(input_file)
            self.generate_op_det_point_cloud(input_file)
        
    def generate_energy_deposit_point_cloud(self,
        input_file: str=''
    ):
        """
        """
        if self.energy_deposit_point_cloud == None:
            self.logger.warn(f'no energy_deposit_point_cloud data in file {input_file}!')
            return
        self.logger.info(
            f"generating 'energy_deposit_point_cloud' training data from file: {input_file}"
        )

    def generate_view_tpc_point_cloud(self,
        input_file: str=''
    ):
        """
        We iterate over each view (wire plane) and collect all
        (channel, tdc, adc) points for each point cloud into a features
        array, together with (source, shape, particle) as
        the categorical information and (shape, particle) as clustering
        information.
        """
        if self.wire_plane_point_cloud == None:
            self.logger.warn(f'no wire_plane_point_cloud data in file {input_file}!')
            return
        self.logger.info(
            f"generating 'view_tpc_point_cloud' training data from file: {input_file}"
        )
        
        channel = self.wire_plane_point_cloud['channel']
        tdc = self.wire_plane_point_cloud['tdc']
        energy = self.wire_plane_point_cloud['energy'] * 10e5
        adc = self.wire_plane_point_cloud['adc']

        # construct ids and names for source, shape and particle labels
        source_label = self.wire_plane_point_cloud['source_label']
        shape_label = self.wire_plane_point_cloud['shape_label']
        particle_label = self.wire_plane_point_cloud['particle_label']
        unique_shape_label = self.wire_plane_point_cloud['unique_shape']
        unique_particle_label = self.wire_plane_point_cloud['unique_particle']

        for tpc, tpc_ranges in self.protodune_tpc_wire_channels.items():
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
                shape_label_view = []
                particle_label_view = []
                unique_shape_label_view = []
                unique_particle_label_view = []

                for event in range(len(channel)):
                    view_mask = (
                        (channel[event] >= tpc_view[0]) & 
                        (channel[event] < tpc_view[1]) & 
                        #(source_label[event] >= 0) &        # we don't want 'undefined' points in our dataset.
                        (shape_label[event] >= 0) &         # i.e., things with a label == -1
                        (particle_label[event] >= 0)
                    )
                    if np.sum(view_mask) > 0:
                        channel_view.append(channel[event][view_mask])
                        tdc_view.append(tdc[event][view_mask])
                        adc_view.append(adc[event][view_mask])
                        energy_view.append(energy[event][view_mask])
                        source_label_view.append(source_label[event][view_mask])
                        shape_label_view.append(shape_label[event][view_mask])
                        particle_label_view.append(particle_label[event][view_mask])
                        unique_shape_label_view.append(unique_shape_label[event][view_mask])
                        unique_particle_label_view.append(unique_particle_label[event][view_mask])

                channel_view = np.array(channel_view, dtype=object)
                tdc_view = np.array(tdc_view, dtype=object)
                adc_view = np.array(adc_view, dtype=object)
                energy_view = np.array(energy_view, dtype=object)
                source_label_view = np.array(source_label_view, dtype=object)
                shape_label_view = np.array(shape_label_view, dtype=object)
                particle_label_view = np.array(particle_label_view, dtype=object)
                unique_shape_label_view = np.array(unique_shape_label_view, dtype=object)
                unique_particle_label_view = np.array(unique_particle_label_view, dtype=object)

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
                    np.vstack((source_label_view[ii], shape_label_view[ii], particle_label_view[ii])).T
                    for ii in range(len(channel_view))],
                    dtype=object
                )          
                clusters = np.array([
                    np.vstack((unique_shape_label_view[ii], unique_particle_label_view[ii])).T
                    for ii in range(len(channel_view))],
                    dtype=object
                )
                hits = np.array([])
                merge_tree = np.array([])

                meta = {
                    "who_created":      getpass.getuser(),
                    "when_created":     datetime.now().strftime("%m-%d-%Y-%H:%M:%S"),
                    "where_created":    socket.gethostname(),
                    "num_events":       len(features),
                    "view":             v,
                    "mc_maps":          {},
                    "features": {
                        "channel": 0, "tdc": 1, "adc": 2
                    },
                    "classes": {
                        "source": 0, "shape": 1, "particle": 2
                    },
                    "clusters": {
                        "shape":  0, "particle": 1
                    },
                    "source_labels": {
                        key: value
                        for key, value in classification_labels["source"].items()
                    },
                    "source_points": {
                        key: np.count_nonzero(np.concatenate(source_label_view) == key)
                        for key, value in classification_labels["source"].items()
                    },
                    "shape_labels": {
                        key: value
                        for key, value in classification_labels["shape"].items()
                    },
                    "shape_points": {
                        key: np.count_nonzero(np.concatenate(shape_label_view) == key)
                        for key, value in classification_labels["shape"].items()
                    },
                    "particle_labels": {
                        key: value
                        for key, value in classification_labels["particle"].items()
                    },      
                    "particle_points": {
                        key: np.count_nonzero(np.concatenate(particle_label_view) == key)
                        for key, value in classification_labels["particle"].items()
                    },
                    "total_points":     len(np.concatenate(features)),
                    "adc_view_sum":     adc_view_sum,    
                }
                    
                np.savez(
                    f"data/{self.output_folders[input_file]}/view{v}_{tpc}.npz",
                    features=features,
                    classes=classes,
                    clusters=clusters,
                    hists=hits,
                    merge_tree=merge_tree,
                    meta=meta
                )
    
    def generate_op_det_point_cloud(self,
        input_file: str=''
    ):
        """
        """
        if self.op_det_point_cloud == None:
            self.logger.warn(f'no op_det_point_cloud data in file {input_file}!')
            return
        self.logger.info(
            f"generating 'op_det_point_cloud' training data from file: {input_file}"
        )