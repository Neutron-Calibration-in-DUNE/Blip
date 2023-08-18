"""

"""
import numpy as np
from matplotlib import pyplot as plt
import h5py

from blip.dataset.arrakis_nd.det_point_cloud import DetectorPointCloud

class SimulationWrangler:
    """
    """
    def __init__(self):

        self.det_point_cloud = DetectorPointCloud()

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

    def process_event(self,
        event_trajectories,
        event_segments,
        event_stacks,
        hits_back_track,
        hits
    ):
        self.clear_event()
        self.process_event_trajectories(event_trajectories)
        self.process_event_stacks(event_stacks)
        self.process_event_segments(event_segments)
        self.process_event_hits(hits, hits_back_track)
        print(self.det_point_cloud)

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
            daughters = []

            self.trackid_daughters[track_id] = daughters
            self.trackid_progeny[track_id] = []
            self.trackid_descendants[track_id] = daughters

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

    def process_event_stacks(self,
        event_stacks
    ):
        pass
        
    def process_event_segments(self,
        event_segments
    ):
        for i, segment in enumerate(event_segments):
            self.trackid_segmentid[segment['track_id']] = segment['segment_id']
            self.segmentid_trackid[segment['segment_id']] = segment['track_id']
    
    def process_event_hits(self,
        event_hits,
        event_hits_back_track
    ):
        for ii, hit in enumerate(event_hits):
            segment_ids = event_hits_back_track['segment_id'][ii]
            self.det_point_cloud.add_point(
                hit['x'], 
                hit['y'],
                hit['z'],
                hit['t_drift'], 
                hit['ts_pps'], 
                hit['Q'], 
                hit['E'], 
                segment_ids[segment_ids != 0]
            )
    