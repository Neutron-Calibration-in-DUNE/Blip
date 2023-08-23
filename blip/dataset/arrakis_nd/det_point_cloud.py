"""
"""
import numpy as np
import h5py

class DetectorPointCloud:
    """
    """
    def __init__(self):
        self.clear()
    
    def clear(self):
        
        self.event = -1
        self.x = []
        self.y = []
        self.z = []
        self.t_drift = []
        self.ts_pps = []
        self.Q = []
        self.E = []
        self.segment_ids = []
        self.segment_fractions = []

        self.source_label = []
        self.topology_label = []
        self.particle_label = []
        self.physics_label = []

        self.unique_topology = []
        self.unique_particle = []
        self.unique_physics = []

        self.source_labels = []
        self.topology_labels = []
        self.particle_labels = []
        self.physics_labels = []

        self.unique_topologies = []
        self.unique_particles = []
        self.unique_physicses = []


    def add_point(self,
        x, y, z, 
        t_drift, ts_pps, Q, E, 
        segment_ids, segment_fractions

    ):
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
        self.t_drift.append(t_drift)
        self.ts_pps.append(ts_pps)
        self.Q.append(Q)
        self.E.append(E)
        self.segment_ids.append(segment_ids)
        self.segment_fractions.append(segment_fractions)

        self.source_label.append(-1)
        self.topology_label.append(-1)
        self.particle_label.append(-1)
        self.physics_label.append(-1)

        self.unique_topology.append(-1)
        self.unique_particle.append(-1)
        self.unique_physics.append(-1)

        self.source_labels.append([-1 for ii in range(len(segment_ids))])
        self.topology_labels.append([-1 for ii in range(len(segment_ids))])
        self.particle_labels.append([-1 for ii in range(len(segment_ids))])
        self.physics_labels.append([-1 for ii in range(len(segment_ids))])

        self.unique_topologies.append([-1 for ii in range(len(segment_ids))])
        self.unique_particles.append([-1 for ii in range(len(segment_ids))])
        self.unique_physicses.append([-1 for ii in range(len(segment_ids))])
    