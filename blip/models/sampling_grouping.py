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
import torch_cluster


from blip.models.common import activations, normalizations
from blip.models import GenericModel

sampling_grouping_config = {
    "sampling_method":      "farthest_point",
    "sampling_num_samples": 512,
    "grouping_method":      "query_ball_point",
    "grouping_type":        "multi-scale", 
    "grouping_radii":       [0.1, 0.2, 0.4],
    "grouping_samples":     [16, 32, 128],
}

class SamplingAndGrouping(GenericModel):
    """
    """
    def __init__(self,
        name:   str='sampling_grouping',
        cfg:    dict=sampling_grouping_config
    ):
        super(SamplingAndGrouping, self).__init__(name, cfg)
        self.cfg = cfg

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
        if self.cfg["sampling_method"] == "farthest_point":
            self.sampling_method = self.furthest_point_sampling
        if self.cfg["grouping_method"] == "query_ball_point":
            self.grouping_method = self.query_ball_point

        # pairwise distance calculator
        self.pairwise_distance = nn.PairwiseDistance(p=2, keepdim=True)

        # record the info
        self.logger.info(
            f"Constructed sampling_grouping with methods:"
        )

    def furthest_point_sampling(self,
        positions,
    ):
        ratio = float(self.cfg['sampling_num_samples'])/len(positions)
        # only sampling according to the (tdc,channel) dimensions, not adc.
        sampled_indices = torch_cluster.fps(positions[:,:2], None, ratio)
        sampled_positions = positions[sampled_indices]
        return sampled_positions, sampled_indices

    def query_ball_point(self,
        positions,
        centroids,
    ):
        grouped_positions = []
        grouped_indices = []
        pairwise_distances = torch.cdist(centroids, positions, p=2)
        for ii, radii in enumerate(self.cfg['grouping_radii']):
            if self.cfg['grouping_type'] == 'multi-scale':
                # Shift grouped points relative to centroid
                for jj, distances in enumerate(pairwise_distances):
                    grouped_index = (distances <= (radii * radii)).nonzero(as_tuple=True)
                    grouped_position = positions[grouped_index]
                    grouped_position -= centroids[jj]
                    grouped_positions.append(grouped_position)
                    grouped_indices.append(grouped_index)
            else:
                pass
        return grouped_positions, grouped_indices
    
    def forward(self,
        positions,
    ):
        """
        Iterate over the model dictionary
        """
        positions = positions.to(self.device)
        """
        Iterate over the sampling + grouping stage
        """
        # Grab centroids using farthest point sampling.
        sampled_positions, sampled_indices = self.sampling_method(
            positions
        )

        # Iterate over each grouping radius
        grouped_positions, grouped_indices = self.grouping_method(
            positions,
            sampled_positions
        )

        return {
            'sampled_positions':    sampled_positions,
            'sampled_indices':      sampled_indices,
            'grouped_positions':    grouped_positions,
            'grouped_indices':      grouped_indices
        }