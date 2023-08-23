"""

"""
import numpy as np
from matplotlib import pyplot as plt
import copy
import h5py

from blip.dataset.arrakis_nd.det_point_cloud import DetectorPointCloud

class SimulationWrangler:
    """
    """
    def __init__(self):

        self.det_point_cloud = DetectorPointCloud()
        self.det_point_clouds = {}

        self.trackid_parentid = {}
        self.trackid_pdgcode = {}
        self.trackid_process = {}
        self.trackid_subprocess = {}
        self.trackid_endprocess = {}
        self.trackid_endsubprocess = {}
        self.trackid_energy = {}
        self.trackid_daughters = {}
        self.trackid_progeny = {}
        self.trackid_descendants = {}
        self.trackid_ancestorlevel = {}
        self.trackid_ancestry = {} 

        self.trackid_segmentid = {}
        self.segmentid_trackid = {}

        self.trackid_hit = {}
        self.segmentid_hit = {}


    def clear_event(self):

        self.det_point_cloud.clear()

        self.trackid_parentid = {}
        self.trackid_pdgcode = {}
        self.trackid_process = {}
        self.trackid_subprocess = {}
        self.trackid_endprocess = {}
        self.trackid_endsubprocess = {}
        self.trackid_energy = {}
        self.trackid_daughters = {}
        self.trackid_progeny = {}
        self.trackid_descendants = {}
        self.trackid_ancestorlevel = {}
        self.trackid_ancestry = {}

        self.trackid_segmentid = {}
        self.segmentid_trackid = {}

        self.trackid_hit = {}
        self.segmentid_hit = {}
    
    def clear_point_clouds(self):
        self.det_point_clouds = {}
    
    def set_hit_labels(self,
        hit, trackid, topology, 
        particle, physics, unique_topology
    ):
        track_index = self.get_index_trackid(hit, trackid)
        if track_index != -1:
            self.det_point_cloud.topology_labels[hit][track_index] = topology
            self.det_point_cloud.particle_labels[hit][track_index] = particle
            self.det_point_cloud.physics_labels[hit][track_index] = physics
            self.det_point_cloud.unique_topologies[hit][track_index] = unique_topology
            self.det_point_cloud.unique_particles[hit][track_index] = trackid
        self.det_point_cloud.topology_label[hit] = topology
        self.det_point_cloud.particle_label[hit] = particle
        self.det_point_cloud.physics_label[hit] = physics
        self.det_point_cloud.unique_topology[hit] = unique_topology
        self.det_point_cloud.unique_particle[hit] = trackid

    def process_event(self,
        event_id,
        event_trajectories,
        event_segments,
        event_stacks,
        hits_back_track,
        hits
    ):
        self.clear_event()
        self.det_point_cloud.event = event_id
        self.process_event_trajectories(event_trajectories)
        self.process_event_stacks(event_stacks)
        self.process_event_segments(event_segments)
        self.process_event_hits(hits, hits_back_track)
    
    def save_event(self):
        self.det_point_cloud.x = np.array(self.det_point_cloud.x)
        self.det_point_cloud.y = np.array(self.det_point_cloud.y)
        self.det_point_cloud.z = np.array(self.det_point_cloud.z)
        self.det_point_cloud.t_drift = np.array(self.det_point_cloud.t_drift)
        self.det_point_cloud.ts_pps = np.array(self.det_point_cloud.ts_pps)
        self.det_point_cloud.Q = np.array(self.det_point_cloud.Q)
        self.det_point_cloud.E = np.array(self.det_point_cloud.E)
        self.det_point_cloud.source_label = np.array(self.det_point_cloud.source_label)
        self.det_point_cloud.topology_label = np.array(self.det_point_cloud.topology_label)
        self.det_point_cloud.particle_label = np.array(self.det_point_cloud.particle_label)
        self.det_point_cloud.physics_label = np.array(self.det_point_cloud.physics_label)
        self.det_point_cloud.unique_topology = np.array(self.det_point_cloud.unique_topology)
        self.det_point_cloud.unique_particle = np.array(self.det_point_cloud.unique_particle)
        self.det_point_cloud.unique_physics = np.array(self.det_point_cloud.unique_physics)
        self.det_point_clouds[self.det_point_cloud.event] = copy.deepcopy(self.det_point_cloud)

    def process_event_trajectories(self,
        event_trajectories
    ):
        for ii, particle in enumerate(event_trajectories):
            track_id = particle[2]                          
            self.trackid_parentid[track_id] = particle[4]
            self.trackid_pdgcode[track_id] = particle[13]
            self.trackid_process[track_id] = particle[14]
            self.trackid_subprocess[track_id] = particle[15]
            self.trackid_endprocess[track_id] = particle[16]
            self.trackid_endsubprocess[track_id] = particle[17]
            self.trackid_energy[track_id] = particle[5]
            
            # iterate over daughters
            self.trackid_daughters[track_id] = []
            self.trackid_descendants[track_id] = []
            if particle[4] != -1:
                self.trackid_daughters[particle[4]].append(track_id)
                self.trackid_descendants[particle[4]].append(track_id)
            self.trackid_progeny[track_id] = []
            # iterate over ancestry
            level = 0
            mother = particle[4]
            temp_track_id = particle[2]
            ancestry = []
            while mother != -1:
                level += 1
                temp_track_id = mother
                ancestry.append(mother)
                mother = self.trackid_parentid[temp_track_id]
                
                if level > 1 and mother != -1:
                    self.trackid_progeny[mother].append(temp_track_id)
                    self.trackid_descendants[mother].append(temp_track_id)

            self.trackid_ancestorlevel[track_id] = level
            self.trackid_ancestry[track_id] = ancestry
            self.trackid_hit[track_id] = []

    def process_event_stacks(self,
        event_stacks
    ):
        pass
        
    def process_event_segments(self,
        event_segments
    ):
        for ii, segment in enumerate(event_segments):
            self.trackid_segmentid[segment['traj_id']] = segment['segment_id']
            self.segmentid_trackid[segment['segment_id']] = segment['traj_id']
            self.segmentid_hit[segment['segment_id']] = []
    
    def process_event_hits(self,
        event_hits,
        event_hits_back_track
    ):
        for ii, hit in enumerate(event_hits):
            segment_ids = event_hits_back_track['segment_id'][ii]
            segment_fractions = event_hits_back_track['fraction'][ii]
            self.det_point_cloud.add_point(
                hit['x'], 
                hit['y'],
                hit['z'],
                hit['t_drift'], 
                hit['ts_pps'], 
                hit['Q'], 
                hit['E'], 
                segment_ids[(segment_ids != 0)],
                segment_fractions[(segment_ids != 0)]
            )
            for segmentid in segment_ids[(segment_ids != 0)]:
                if segmentid in self.segmentid_hit.keys():
                    self.segmentid_hit[segmentid].append(ii)
                    self.trackid_hit[self.segmentid_trackid[segmentid]].append(ii)
    
    def get_primaries_generator_label(self,
        label
    ):
        pass
    
    def get_primaries_pdg_code(self,
        pdg
    ):
        primaries = []
        for track_id, parent_id in self.trackid_parentid.items():
            if parent_id == 0 and self.trackid_pdgcode[track_id] == pdg:
                primaries.append(track_id)
        return primaries

    def get_trackid_pdg_code(self,
        pdg
    ):
        trackid = []
        for track_id, pdg_code in self.trackid_pdgcode.items():
            if pdg_code == pdg:
                trackid.append(track_id)
        return trackid   

    def filter_trackid_abs_pdg_codese(self,
        trackids, pdg
    ):
        trackid = []
        for track_id in trackids:
            if self.trackid_pdgcode[track_id] == pdg:
                trackid.append(track_id)
        return trackid
    
    def get_index_trackid(self,
        hit, trackid
    ):
        for ii, particle in enumerate(self.det_point_cloud.particle_labels[hit]):
            if particle == trackid:
                return ii
        return -1