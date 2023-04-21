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
        
        # for ii in range(len(self.cfg['mlp'])):
        #     _convolution_dict[f'conv_{ii}'] = nn.Conv2d(
        #         last_channel, self.cfg['mlp'][ii], 1
        #     )
        #     _convolution_dict[f'batch_norm_{ii}'] = nn.BatchNorm2d(
        #         self.cfg['mlp'][ii]
        #     )
        #     last_channel = self.cfg['mlp'][ii]

        self.convolution_dict = nn.ModuleDict(_convolution_dict)

        # record the info
        self.logger.info(
            f"Constructed FeaturePropagation with dictionary:"
            + f"\n{self.convolution_dict}"
        )
    
    def forward(self,
        positions,
        embedding,
        prev_positions,
        prev_embedding
    ):
        print(positions.shape)
        print(embedding.shape)
        print(prev_positions.shape)
        
        """
        Iterate over the model dictionary
        """
        if len(positions) == 1:
            interpolated_points = embedding.repeat(len(positions), 1)
        else:
            pairwise_distances = torch.cdist(prev_positions, positions, p=2)
            pairwise_distances = 1.0 / (pairwise_distances + 1e-8)
            print(pairwise_distances.shape)
            weights = torch.sum(pairwise_distances, dim=1, keepdim=True)
            weights = pairwise_distances / weights
            print(weights.shape)
            print(embedding.shape)
            interpolated_points = torch.sum(embedding * weights.view(len(prev_positions), 1), dim=1)

        if prev_embedding is not None:
            new_points = torch.cat([prev_embedding, interpolated_points])
        else:
            new_points = interpolated_points
        
        return new_points
        