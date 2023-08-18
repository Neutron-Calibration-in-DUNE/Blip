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
        self.x = []
        self.y = []
        self.z = []
        self.t_drift = []
        self.ts_pps = []
        self.Q = []
        self.E = []
        self.segment_id = []

        self.source_label = []
        self.topology_label = []
        self.particle_label = []
        self.physics_label = []

        self.unique_topology = []
        self.unique_particle = []
        self.unique_physics = []


    def add_point(self,
        x, y, z, t_drift, ts_pps, Q, E, segment_id
    ):
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
        self.t_drift.append(t_drift)
        self.ts_pps.append(ts_pps)
        self.Q.append(Q)
        self.E.append(E)
        self.segment_id.append(segment_id)

        self.source_label.append(-1)
        self.topology_label.append(-1)
        self.particle_label.append(-1)
        self.physics_label.append(-1)

        self.unique_topology.append(-1)
        self.unique_particle.append(-1)
        self.unique_physics.append(-1)
    