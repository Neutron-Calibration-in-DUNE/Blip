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
from blip.models import GenericModel, PointNet, SamplingAndGrouping

set_abstraction_config = {
    "sampling_method":  "farthest_point",
    "sampling_num_samples": 512,
    "grouping_method":  "query_ball_point",
    "grouping_type":    "multi-scale", 
    "grouping_radii":   [0.1, 0.2, 0.4],
    "grouping_samples": [16, 32, 128],
    "pointnet_num_embeddings":  2,
    "pointnet_embedding_mlp_layers":  [2, 2],
    "pointnet_embedding_type":    "dynamic_edge_conv",
    "pointnet_number_of_neighbors": 20,
    "pointnet_aggregation": ["max", "max"],
    "pointnet_input_dimension": 3,
}

class SetAbstraction(GenericModel):
    """
    """
    def __init__(self,
        name:   str='set_abstraction',
        config: dict=set_abstraction_config
    ):
        super(SetAbstraction, self).__init__(name, config)
        self.config = config

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
        self.logger.info(f"Attempting to build {self.name} architecture using config: {self.config}")
        self.sampling_and_grouping = SamplingAndGrouping(
            self.name + "_sampling_and_grouping",
            self.config
        )
        self.pointnet = PointNet(
            self.name + "_pointnet",
            self.config
        )
        # record the info
        self.logger.info(
            f"Constructed SetAbstraction"
        )
    
    def forward(self,
        positions,
        batches
    ):
        """
        Iterate over the sampling + grouping stage
        and then PointNet.
        """
        positions = positions.to(self.device)
        batches = batches.to(self.device)

        sampling_and_grouping = self.sampling_and_grouping(positions, batches)
        pointnet_embedding = self.pointnet(positions, batches, sampling_and_grouping)
        
        return {
            'sampling_and_grouping': sampling_and_grouping,
            #'pointnet_embedding':    pointnet_embedding,
        }



        