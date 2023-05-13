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

pointnet_config = {
    "sampling_method":  "farthest_point",
    "sampling_num_samples": 512,
    "grouping_method":  "query_ball_point",
    "grouping_type":    "multi-scale", 
    "grouping_radii":   [0.1, 0.2, 0.4],
    "grouping_samples": [16, 32, 128],
    "pointnet_num_embeddings":          2,
    "pointnet_embedding_mlp_layers":    [2, 2],
    "pointnet_embedding_type":          "dynamic_edge_conv",
    "pointnet_number_of_neighbors":     20,
    "pointnet_aggregation":             ["max", "max"],
    "pointnet_input_dimension":         3,
}

class PointNet(GenericModel):
    """
    """
    def __init__(self,
        name:   str='pointnet',
        config: dict=pointnet_config,
        device: str='cpu'
    ):
        super(PointNet, self).__init__(name, config, device)
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

        self.input_dimension = self.config['pointnet_input_dimension']
        
        """
        Feature extraction layers
        Each radii (for multi-scale grouping) will have a set
        of pointnet embedding layers applied.
        """
        self.embedding_dicts = []
        for jj, radii in enumerate(self.config['grouping_radii']):
            _embedding_dict = OrderedDict()
            _reduction_dict = OrderedDict()
            _input_dimension = self.config['pointnet_input_dimension']
            for ii in range(self.config['pointnet_num_embeddings']):
                if self.config['pointnet_embedding_type'] == 'dynamic_edge_conv':
                    _embedding_dict[f'embedding_{ii}'] = DynamicEdgeConv(
                        MLP([2 * _input_dimension] + self.config['pointnet_embedding_mlp_layers'][ii]), 
                        self.config['pointnet_number_of_neighbors'],
                        self.config['pointnet_aggregation'][ii]
                    )
                _input_dimension = self.config['pointnet_embedding_mlp_layers'][ii][-1]
            self.embedding_dicts.append(nn.ModuleDict(_embedding_dict))
        # create the dictionaries
        # record the info
        self.logger.info(
            f"Constructed PointNet with dictionaries:"
            #+ f"\n{self.embedding_dict}\n{self.reduction_dict}"
        )

    
    def forward(self,
        positions,
        batches,
        sampling_and_grouping=None,
    ):
        """
        Iterate over the model dictionary
        """
        positions = positions.to(self.device)
        batches = batches.to(self.device)
        print(self.device)
        if sampling_and_grouping is not None:
            sampled_positions = sampling_and_grouping['sampled_positions']
            sampled_batches = sampling_and_grouping['sampled_batches']
            grouped_positions = sampling_and_grouping['grouped_positions']

        grouped_embedding = [[] for jj in range(len(self.config['grouping_radii']))]

        for jj, radii in enumerate(self.config['grouping_radii']):
            positions = grouped_positions[jj]
            print(jj)
            for ii, position in enumerate(positions):
                for kk, layer in enumerate(self.embedding_dicts[jj].keys()):
                    position = self.embedding_dicts[jj][layer](position)
                grouped_embedding[jj].append(position)

        return {
            'grouped_embedding':    grouped_embedding, 
        }