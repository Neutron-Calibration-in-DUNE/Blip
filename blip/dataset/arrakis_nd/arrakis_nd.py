from ctypes import sizeof
import uproot
import os
import getpass
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy import stats as st
from datetime import datetime
import h5py
import numpy.lib.recfunctions as rfn
from collections import defaultdict
import json
from tqdm import tqdm

from h5flow.core import H5FlowStage, resources

from blip.utils.logger import Logger
from blip.dataset.common import *
from blip.dataset.arrakis_nd.simulation_wrangler import SimulationWrangler
from blip.dataset.arrakis_nd.simulation_labeling_logic import SimulationLabelingLogic

# class ArrakisND(H5FlowStage):
class ArrakisND:
    class_version = '0.0.0' # keep track of a version number for each class

    default_custom_param = None
    default_obj_name = 'obj0'
    """
    The module class helps to organize meta data and objects related to different tasks
    and execute those tasks based on a configuration file.  The spirit of the 'Module' class
    is to mimic some of the functionality of LArSoft, e.g. where you can specify a chain
    of tasks to be completed, the ability to have nested config files where default parameters
    can be overwritten.

    We'd like to put this into H5FlowStage format so that it can be used in larnd-sim.

    larnd-sim files have the following sets of arrays

    trajectories (mc truth):
        These are the true particle trajectories (or paths) through the detector for all particles, 
        both neutral and charged, excluding the incident neutrino. Each true particle may have multiple 
        trajectories if the trajectory was split/broken by edep-sim with each having their own unique track ID.

        event_id:       unique ID for an interesting window of time; for beam events this corresponds to a spill
        vertex_id:      the vertex ID number, corresponds to an individual generator interaction
        traj_id:        the monotonic trajectory (track) ID, guaranteed to be unique within a file
        local_traj_id:  the original edep-sim trajectory (track) ID, may not be unique
        parent_id:      the trajectory (track) ID of the parent trajectory, if the trajectory is a primary particle the ID is -1
        E_start:        the total energy in [MeV] at the start of the trajectory
        pxyz_start:     the momentum 3-vector (px, py, pz) in [MeV] at the start of the trajectory
        xyz_start:      the start position 3-vector (x, y, z) in [cm] of the trajectory (specifically the position of the first trajectory point)
        t_start:        the start time of the trajectory in [us]
        E_end:          the total energy in [MeV] at the end of the trajectory
        pxyz_end:       the momentum 3-vector (px, py, pz) in [MeV] at the end of the trajectory
        xyz_end:        the end position 3-vector (x, y, z) in [cm] of the trajectory (specifically the position of the last trajectory point)
        t_end:          the end time of the trajectory in [us]
        pdg_id:         the PDG code of the particle
        start_process:  physics process for the start of the trajectory as defined by GEANT4
        start_subprocess: physics subprocess for the start of the trajectory as defined by GEANT4
        end_process:    physics process for the end of the trajectory as defined by GEANT4
        end_subprocess: physics subprocess for the end of the trajectory as defined by GEANT4

    segments (energy depositions):
        These are the true energy deposits (or energy segments) for active parts of the detector from edep-sim. 
        Each segment corresponds to some amount of energy deposited over some distance. Some variables are filled 
        during the larndsim stage of processing.

        event_id:       unique ID for an interesting window of time; for beam events this corresponds to a spill
        vertex_id:      the vertex ID number, corresponds to an individual generator interaction
        segment_id:     the segment ID number
        traj_id:        the trajectory (track) ID of the edep-sim trajectory that created this energy deposit
        x_start:        the x start position [cm]
        y_start:        the y start position [cm]
        z_start:        the z start position [cm]
        t0_start:       the start time [us]
        x_end:          the x end position [cm]
        y_end:          the y end position [cm]
        z_end:          the z end position [cm]
        t0_end:         the start time [us]
        x:              the x mid-point of the segment [cm] -> (x_start + x_end) / 2
        y:              the y mid-point of the segment [cm] -> (y_start + y_end) / 2
        z:              the z mid-point of the segment [cm] -> (z_start + z_end) / 2
        t0:             the time mid-point [us] -> (t0_start + t0_end) / 2
        pdg_id:         PDG code of the particle that created this energy deposit
        dE:             the energy deposited in this segment [MeV]
        dx:             the length of this segment [cm]
        dEdx:           the calculated energy per length [MeV/cm]
        tran_diff:      (ADD INFO)
        long_diff:      (ADD INFO)
        n_electrons:    (ADD INFO)
        n_photons:      (ADD INFO)
        pixel_plane:    (ADD INFO)
        t/t_start/t_end: (ADD INFO)
    
    flow files have the following sets of arrays

    charge:

    mc_truth:
        calib_final_hit_backtrack:
            fraction:   fraction of the segment associated to the hit
            segment_id: segment id associated to the hit
        interactions:
        light:
            segment_id: segment id associated to the hit
            n_photons_det:
            t0_det:
        packet_fraction:
        segments:
        stack:
        trajectories:

    Associations between calib_final_hits and particles/edeps can be made with the 'calib_final_hit_backtrack' 
    array inside of the mc_truth dataset in the flow files.  Each calib_final_hit has an associated segment id and a
    fraction of the edep that corresponds to the hit.

    First, we'll have to collect information using an event mask, and then arange the mc_truth info for particles
    and edeps.  Then, we will construct the 3d charge points and apply the labeling logic.
    
    H5FlowStage
    '''
        Base class for loop stage. Provides the following attributes:

         - ``name``: instance name of stage (declared in configuration file)
         - ``classname``: stage class
         - ``class_version``: a ``str`` version number (``'major.minor.fix'``, default = ``'0.0.0'``)
         - ``data_manager``: an ``H5FlowDataManager`` instance used to access the output file
         - ``requires``: a list of dataset names to load when calling ``H5FlowStage.load()``
         - ``comm``: MPI world communicator (if needed, else ``None``)
         - ``rank``: MPI group rank
         - ``size``: MPI group size

         To build a custom stage, inherit from this base class and implement
         the ``init()`` and the ``run()`` methods.

         Example::

            class ExampleStage(H5FlowStage):
                class_version = '0.0.0' # keep track of a version number for each class

                default_custom_param = None
                default_obj_name = 'obj0'

                def __init__(**params):
                    super(ExampleStage,self).__init__(**params)

                    # grab parameters from configuration file here, e.g.
                    self.custom_param = params.get('custom_param', self.default_custom_param)
                    self.obj_name = self.name + '/' + params.get('obj_name', self.default_obj_name)

                def init(self, source_name):
                    # declare any new datasets and set dataset metadata, e.g.

                    self.data_manager.set_attrs(self.obj_name,
                        classname=self.classname,
                        class_version=self.class_version,
                        custom_param=self.custom_param,
                        )
                    self.data_manager.create_dset(self.obj_name)

                def run(self, source_name, source_slice):
                    # load, process, and save new data objects

                    data = self.load(source_name, source_slice)
    """
    def __init__(self,
        name:   str="arrakis_nd",
        config: dict={},
        meta:   dict={},
        **params
    ):
        # super(ArrakisND, self).__init__(**params)
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
        self.logger.info(f"constructing arrakis_nd dataset.")

        self.simulation_files = []
        self.output_folders = {}

        """
        2x2 channel mappings for different
        TPCs.
        """
        self.nd_2x2_tpc_positions = {
            "tpc0": [[-376.8501, -366.8851],[0., 607.49875],[-0.49375, 231.16625]],
            "tpc1": [[-359.2651,   -0.1651],[0., 607.49875],[-0.49375, 231.16625]],
            "tpc2": [[0.1651, 359.2651],    [0., 607.49875],[-0.49375, 231.16625]],
            "tpc3": [[366.8851, 376.8501],  [0., 607.49875],[-0.49375, 231.16625]],
        }
        self.simulation_wrangler = SimulationWrangler()
        self.simulation_labeling_logic = SimulationLabelingLogic(self.simulation_wrangler)

        self.parse_config()
    
    def init(self, source_name):
        # declare any new datasets and set dataset metadata, e.g.
        self.data_manager.set_attrs(
            self.obj_name,
            classname=self.classname,
            class_version=self.class_version,
            custom_param=self.custom_param
        )
        self.data_manager.create_dset(self.obj_name)

    def set_config(self,
        config_file:    str
    ):
        self.logger.info(f"parsing config file: {config_file}.")
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
                    self.run_arrakis_nd(input_file)
                    self.generate_training_data(self.process_type, input_file)

    def run(self, source_name, source_slice):
        # load, process, and save new data objects
        pass

    def run_arrakis_nd(self,
        simulation_file
    ):
        flow_file = h5py.File(self.simulation_folder + '/' + simulation_file, 'r')
        try:
            charge = flow_file['charge']
            combined = flow_file['combined']
            geometry_info = flow_file['geometry_info']
            lar_info = flow_file['lar_info']
            light = flow_file['light']
            mc_truth = flow_file['mc_truth']
            run_info = flow_file['run_info']
        except:
            self.logger.error(f'there was a problem processing flow file {simulation_file}')
        
        trajectories = mc_truth['trajectories']['data']
        segments = mc_truth['segments']['data']
        stacks = mc_truth['stack']['data']
        hits_back_track = mc_truth['calib_final_hit_backtrack']['data']
        hits = charge['calib_final_hits']['data']

        trajectory_events = trajectories['event_id']
        segment_events = segments['event_id']
        stack_events = stacks['event_id']

        unique_events = np.unique(segment_events)

        event_loop = tqdm(
            enumerate(unique_events, 0), 
            total=len(unique_events), 
            leave=True,
            position=0,
            colour='green'
        )
        for ii, event in event_loop:
            trajectory_event_mask = (trajectory_events == event)
            segment_event_mask = (segment_events == event)
            stack_event_mask = (stack_events == event)
            
            hits_back_track_mask = np.any(
                np.isin(hits_back_track['segment_id'], segments[segment_event_mask]['segment_id']), 
                axis=1
            )
            self.simulation_wrangler.process_event(
                event,
                event_trajectories=trajectories[trajectory_event_mask],
                event_segments=segments[segment_event_mask],
                event_stacks=stacks[stack_event_mask],
                hits_back_track=hits_back_track[hits_back_track_mask],
                hits=hits[hits_back_track_mask]
            )
            self.simulation_labeling_logic.process_event()
            self.simulation_wrangler.save_event()
            event_loop.set_description(f"Running ArrakisND - Event: [{ii+1}/{len(unique_events)}]")
    
    def generate_training_data(self,
        process_type:  list=['all'],
        simulation_file:    str=''
    ):
        self.meta = {}
        self.mc_maps = {}
        self.segment_point_clouds = {}
        self.hits_point_clouds = {}
        self.op_det_point_clouds = {}

        for tpc, tpc_ranges in self.nd_2x2_tpc_positions.items():
            self.meta[tpc] = {
                "who_created":      getpass.getuser(),
                "when_created":     datetime.now().strftime("%m-%d-%Y-%H:%M:%S"),
                "where_created":    socket.gethostname(),
                "simulation_file":  simulation_file,
                # "segment_features": {
                #     "channel": 0, "tdc": 1, "adc": 2
                # },
                "hits_features": {
                    "x": 0, "y": 1, "z": 2, "t_drift": 3, "ts_pps": 4, "Q": 5, "E": 6
                },
                "classes": {
                    "source": 0, "topology": 1, "particle": 2, "physics": 3
                },
                "clusters": {
                    "topology":  0, "particle": 1, "physics": 2
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
            }
            self.mc_maps[tpc] = {
                'pdg_code': [],
                'parent_track_id': [],
                'ancestor_track_id': [],
                'ancestor_level': []
            }
            # self.segment_point_clouds[tpc] = {
            #     'segment_features': [],
            #     'segment_classes':  [],
            #     'segment_clusters': [],                
            # }
            self.hits_point_clouds[tpc] = {
                'hits_features':  [],
                'hits_classes':   [],
                'hits_clusters':  [],
            }   

        for process in process_type:
            if process == 'segment_point_cloud':
                self.generate_segment_point_cloud(simulation_file)
            elif process == 'hits_point_cloud':
                self.generate_hits_point_cloud(simulation_file)
            elif process == 'mc_maps':
                self.generate_mc_maps(simulation_file)
            elif process == 'all':
                self.generate_segment_point_cloud(simulation_file)
                self.generate_hits_point_cloud(simulation_file)
                self.generate_mc_maps(simulation_file)
            else:
                self.logger.error(f'specified process type {process} not allowed!')
        
        for tpc, tpc_ranges in self.nd_2x2_tpc_positions.items():
            print(self.hits_point_clouds[tpc].items())
            np.savez(
                f"data/{self.output_folders[self.simulation_folder + simulation_file]}/{tpc}.npz",
                hits_features=self.hits_point_clouds[tpc]['hits_features'],
                hits_classes=self.hits_point_clouds[tpc]['hits_classes'],
                hits_clusters=self.hits_point_clouds[tpc]['hits_clusters'],
                mc_maps=self.mc_maps[tpc],
                meta=self.meta[tpc]
            )
    

    def generate_mc_maps(self,
        input_file: str=''
    ):
        pass

    def generate_segment_point_cloud(self,
        input_file: str=''
    ):
        pass
    
    def generate_hits_point_cloud(self,
        input_file: str=''
    ):
        """
        We iterate over each tpc and collect all
        points for each point cloud into a features
        array, together with (source, topology, particle) as
        the categorical information and (topology, particle) as clustering
        information.
        """
        self.logger.info(
            f"generating 'simulation_wrangler.det_point_clouds' training data from file: {input_file}"
        )
        det_point_cloud = self.simulation_wrangler.det_point_clouds

        for tpc, tpc_ranges in self.nd_2x2_tpc_positions.items():
            """
            For each point cloud, we want to normalize adc against
            all point clouds in the data set, so that it is independent 
            of the specific detector readout.
            """

            x_view = []
            y_view = []
            z_view = []
            t_drift_view = []
            ts_pps_view = []
            Q_view = []
            E_view = []
            source_label_view = []
            topology_label_view = []
            particle_label_view = []
            physics_label_view = []
            unique_topology_label_view = []
            unique_particle_label_view = []
            unique_physics_label_view = []

            for event, point_cloud in det_point_cloud.items():
                view_mask = (
                    (point_cloud.x >= tpc_ranges[0][0]) & 
                    (point_cloud.x <  tpc_ranges[0][1]) & 
                    (point_cloud.y >= tpc_ranges[1][0]) & 
                    (point_cloud.y <  tpc_ranges[1][1]) &
                    (point_cloud.z >= tpc_ranges[2][0]) & 
                    (point_cloud.z <  tpc_ranges[2][1])
                    # (point_cloud.topology_label >= 0) &
                    # (point_cloud.particle_label >= 0)
                )
                if np.sum(view_mask) > 0:
                    x_view.append(point_cloud.x[view_mask])
                    y_view.append(point_cloud.y[view_mask])
                    z_view.append(point_cloud.z[view_mask])
                    t_drift_view.append(point_cloud.t_drift[view_mask])
                    ts_pps_view.append(point_cloud.ts_pps[view_mask])
                    Q_view.append(point_cloud.Q[view_mask])
                    E_view.append(point_cloud.E[view_mask])
                    source_label_view.append(point_cloud.source_label[view_mask])
                    topology_label_view.append(point_cloud.topology_label[view_mask])
                    particle_label_view.append(point_cloud.particle_label[view_mask])
                    physics_label_view.append(point_cloud.physics_label[view_mask])
                    unique_topology_label_view.append(point_cloud.unique_topologies[view_mask])
                    unique_particle_label_view.append(point_cloud.unique_particles[view_mask])
                    unique_physics_label_view.append(point_cloud.unique_physicses[view_mask])

            x_view = np.array(x_view, dtype=object)
            y_view = np.array(y_view, dtype=object)
            z_view = np.array(z_view, dtype=object)
            t_drift_view = np.array(t_drift_view, dtype=object)
            ts_pps_view = np.array(ts_pps_view, dtype=object)
            Q_view = np.array(Q_view, dtype=object)
            E_view = np.array(E_view, dtype=object)
            source_label_view = np.array(source_label_view, dtype=object)
            topology_label_view = np.array(topology_label_view, dtype=object)
            particle_label_view = np.array(particle_label_view, dtype=object)
            physics_label_view = np.array(physics_label_view, dtype=object)
            unique_topology_label_view = np.array(unique_topology_label_view, dtype=object)
            unique_particle_label_view = np.array(unique_particle_label_view, dtype=object)
            unique_physics_label_view = np.array(unique_physics_label_view, dtype=object)

            if len(x_view.flatten()) == 0:
                continue
            features = np.array([
                np.vstack((
                    x_view[ii], y_view[ii], z_view[ii], 
                    t_drift_view[ii], ts_pps_view[ii], 
                    Q_view[ii], E_view[ii]
                )).T
                for ii in range(len(x_view))],
                dtype=object
            )
            classes = np.array([
                np.vstack((
                    source_label_view[ii], 
                    topology_label_view[ii], 
                    particle_label_view[ii], 
                    physics_label_view[ii])).T
                for ii in range(len(x_view))],
                dtype=object
            )          
            clusters = np.array([
                np.vstack((
                    unique_topology_label_view[ii], 
                    unique_particle_label_view[ii],
                    unique_physics_label_view[ii])).T
                for ii in range(len(x_view))],
                dtype=object
            )

            self.meta[tpc]["num_events"] = len(features)
            self.meta[tpc][f"tpc_source_points"] = {
                key: np.count_nonzero(np.concatenate(source_label_view) == key)
                for key, value in classification_labels["source"].items()
            }
            self.meta[tpc][f"tpc_topology_points"] = {
                key: np.count_nonzero(np.concatenate(topology_label_view) == key)
                for key, value in classification_labels["topology"].items()
            }
            self.meta[tpc][f"tpc_particle_points"] = {
                key: np.count_nonzero(np.concatenate(particle_label_view) == key)
                for key, value in classification_labels["particle"].items()
            }
            self.meta[tpc][f"tpc_physics_points"] = {
                key: np.count_nonzero(np.concatenate(physics_label_view) == key)
                for key, value in classification_labels["physics"].items()
            }
            self.meta[tpc][f"tpc_total_points"] = len(np.concatenate(features))   
            self.hits_point_clouds[tpc][f'hits_features'] = features
            self.hits_point_clouds[tpc][f'hits_classes'] = classes
            self.hits_point_clouds[tpc][f'hits_clusters'] = clusters
    
    def generate_op_det_point_cloud(self,
        input_file: str=''
    ):
        """
        """
        pass



