"""
Implementation of the blip model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch_geometric.transforms as T
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool


from blip.models.common import activations, normalizations
from blip.utils.sampling import *
from blip.utils.grouping import *
from blip.utils.utils import *
from blip.models import GenericModel, PointNet

set_abstraction_config = {
    'sampling': {
        'method':   farthest_point_sampling,
        'number_of_centroids':  1024,
    },
    'grouping': {
        'method':       query_ball_point,
        'radii':                0.05,
        'number_of_samples':    32,
    },
    'pointnet': {
        'method':   PointNet,
    },
}

class SetAbstraction(GenericModel):
    """
    """
    def __init__(self,
        name:   str='set_abstraction',
        cfg:    dict=set_abstraction_config
    ):
        super(SetAbstraction, self).__init__(name, cfg)
        self.cfg = cfg

        self.number_of_centroids = self.cfg['sampling']['number_of_centroids']
        self.radii = self.cfg['grouping']['radii']
        self.number_of_samples = self.cfg['grouping']['number_of_samples']

        self.sampling_method = self.cfg['sampling']['method']
        self.grouping_method = self.cfg['grouping']['method']
        self.pointnet_method = self.cfg['pointnet']['method']

        # construct the model
        self.forward_views      = {}
        self.forward_view_map   = {}
        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        self.logger.info(f"Attempting to build {self.name} architecture using cfg: {self.cfg}")
        
        # record the info
        self.logger.info(
            f"Constructed SetAbstraction"
        )
    
    def forward(self,
        position, 
        embedding = None,
    ):
        """
        Iterate over the sampling + grouping stage
        Iterate over the model dictionary
        """
        position = position.to(self.device)
        if embedding is not None:
            embedding = embedding.to(self.device)

        position_dimension = position.shape

        # Grab centroids using farthest point sampling.
        sampled_positions = self.sampling_method(
            position, self.number_of_centroids
        )

        # Iterate over each grouping radius
        group_indices = self.grouping_method(
            sampled_positions,
            position,
            self.radii,
            self.number_of_samples
        )
        group_positions = index_positions(position, group_indices)
        
        # Shift grouped points relative to centroid
        if self.number_of_centroids is not None:
            group_positions -= sampled_positions.view(
                self.number_of_centroids, 1, position_dimension
            )
        
        # Grab local embedding points if embedding is not empty
        if embedding is not None:
            group_embedding = index_positions(embedding, group_indices)
            group_embedding = torch.cat([group_embedding, group_positions], dim=-1)
        else:
            group_embedding = group_positions

        # Pass the local samples through the PointNet layer
        pointnet_embedding = self.pointnet_method(group_embedding)
        sampled_embedding = pointnet_embedding['reductions']
        
        return {
            'sampled_positions': sampled_positions,
            'sampled_embedding': sampled_embedding,
        }



        