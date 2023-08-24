"""
"""
from blip.dataset.common import *
from blip.dataset.arrakis_nd.simulation_wrangler import SimulationWrangler

class SimulationLabelingLogic:
    """
    """
    def __init__(self,
        simulation_wrangler
    ):
        self.simulation_wrangler = simulation_wrangler

        self.topology_label = 0
    
    def iterate_topology_label(self):
        self.topology_label += 1
        return self.topology_label
    
    def set_labels(self,
        hits, segments, trackid,
        topology, physics, unique_topology
    ):
        for hit in hits:
            self.simulation_wrangler.set_hit_labels(
                hit, trackid, topology, 
                trackid, physics, unique_topology
            )

    def process_event(self):
        
        self.topology_label = 0
        self.process_muons()

    def process_muons(self):
        muons = self.simulation_wrangler.get_trackid_pdg_code(13)
        for muon in muons:
            muon_daughters = self.simulation_wrangler.trackid_daughters[muon]
            muon_progeny = self.simulation_wrangler.trackid_progeny[muon]
            muon_hits = self.simulation_wrangler.trackid_hit[muon]
            # muon_segments = self.simulation_wrangler.trackid_segmentid[muon]
            muon_segments = []

            cluster_label = self.iterate_topology_label()
            self.set_labels(
                muon_hits, muon_segments, muon,
                TopologyLabel.Track, PhysicsLabel.MIPIonization,
                cluster_label
            )
            
            # muon_elec_daughters = self.simulation_wrangler.filter_trackid_abs_pdg_code(muon_daughters, 11)