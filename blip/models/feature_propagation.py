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
from blip.models import GenericModel

feature_propagation_config = {
    'in_channels':  3,
    'mlp':          [256, 256]
}

class FeaturePropagation(GenericModel):
    """
    """
    def __init__(self,
        name:   str='feature_propagation',
        cfg:    dict=feature_propagation_config
    ):
        super(FeaturePropagation, self).__init__(name, cfg)
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
        
        """
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        self.logger.info(f"Attempting to build {self.name} architecture using cfg: {self.cfg}")
        _convolution_dict = OrderedDict()

        self.convolution_dict = nn.ModuleDict(_convolution_dict)

        # record the info
        self.logger.info(
            f"Constructed FeaturePropagation with dictionary:"
            + f"\n{self.convolution_dict}"
        )
    
    def forward(self,
        positions,
        batches,
        indices,
        embedding,
        prev_indices,
        prev_embedding
    ):     
        """
        Iterate over the model dictionary
        """
        current_positions = positions[indices]
        prev_positions = positions[prev_indices]
        if len(current_positions) == 1:
            interpolated_points = embedding.repeat(len(current_positions), 1)
        else:
            pairwise_distances = torch.cdist(prev_positions, current_positions, p=2)
            pairwise_distances, indices = pairwise_distances.sort(dim=-1)

            pairwise_distances = 1.0 / (pairwise_distances + 1e-8)
            weights = torch.sum(pairwise_distances, dim=1, keepdim=True)
            weights = pairwise_distances / weights
            
            interpolated_points = torch.matmul(weights, embedding)

        if prev_embedding is not None:
            interpolated_points = torch.cat([prev_embedding, interpolated_points], dim=1)
        
        return interpolated_points
        