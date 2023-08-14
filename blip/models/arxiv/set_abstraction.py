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
    "pointnet_embedding_mlp_layers":2,
    "pointnet_embedding_type":      "dynamic_edge_conv",
    "pointnet_number_of_neighbors": 20,
    "pointnet_aggregation":         "max",
    "pointnet_input_dimension":     3,
}

class SetAbstraction(GenericModel):
    """
    """
    def __init__(self,
        name:   str='set_abstraction',
        config: dict=set_abstraction_config,
        meta:   dict={}
    ):
        super(SetAbstraction, self).__init__(name, config, meta)
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

        _embedding_dict = OrderedDict()
        for jj, radii in enumerate(self.config['grouping_radii']): 
            _input_dimension = self.config['pointnet_input_dimension']
            if self.config['pointnet_embedding_type'] == 'dynamic_edge_conv':
                _embedding_dict[f'embedding_{jj}'] = DynamicEdgeConv(
                    MLP([2 * _input_dimension] + self.config['pointnet_embedding_mlp_layers']), 
                    self.config['pointnet_number_of_neighbors'],
                    self.config['pointnet_aggregation']
                )
        self.embedding_dict = nn.ModuleDict(_embedding_dict)
        # self.sampling_and_grouping = SamplingAndGrouping(
        #     self.name + "_sampling_and_grouping",
        #     self.config
        # )
        # self.pointnet = PointNet(
        #     self.name + "_pointnet",
        #     self.config
        # )
        # record the info
        self.logger.info(
            f"Constructed SetAbstraction"
        )
    
    # def furthest_point_sampling(self,
    #     positions,
    #     batches
    # ):
    #     ratio = len(torch.unique(batches))*float(self.config['sampling_num_samples'])/len(positions)
    #     # only sampling according to the (tdc,channel) dimensions, not adc.
    #     sampled_indices = torch_cluster.fps(positions, batches, ratio)
    #     sampled_positions = positions[sampled_indices]
    #     sampled_batches = batches[sampled_indices]
    #     return sampled_positions, sampled_batches, sampled_indices

    # def query_ball_point(self,
    #     positions,
    #     batches,
    #     centroids,
    #     centroid_batches
    # ):
    #     grouped_positions = [[] for ii in range(len(self.config['grouping_radii']))]
    #     #grouped_indices = [[] for ii in range(len(self.config['grouping_radii']))]
    #     pairwise_distances = torch.cdist(centroids, positions, p=2)
    #     for ii, radii in enumerate(self.config['grouping_radii']):
    #         if self.config['grouping_type'] == 'multi-scale':
    #             # Shift grouped points relative to centroid
    #             for jj, distances in enumerate(pairwise_distances):
    #                 grouped_index = ((distances <= (radii * radii)) & (batches == centroid_batches[jj])).nonzero(as_tuple=True)
    #                 grouped_position = positions[grouped_index]
    #                 grouped_position -= centroids[jj]
    #                 grouped_positions[ii].append(grouped_position)
    #                 #grouped_indices[ii].append(grouped_index)
    #         else:
    #             pass
    #     return grouped_positions, grouped_indices
    
    def forward(self,
        positions,
        batches,
        embedding
    ):
        """
        Iterate over the sampling + grouping stage
        and then PointNet.
        """
        positions = positions.to(self.device)
        batches = batches.to(self.device)

        ratio = len(torch.unique(batches))*float(self.config['sampling_num_samples'])/len(positions)
        
        sampled_indices = torch_cluster.fps(positions, batches, ratio)
        sampled_positions = positions[sampled_indices]
        sampled_batches = batches[sampled_indices]

        sampled_embedding = [[] for ii in range(len(sampled_positions))]

        pairwise_distances = torch.cdist(sampled_positions, positions, p=2)

        for ii, radii in enumerate(self.config['grouping_radii']):
            # if self.config['grouping_type'] == 'multi-scale':
            # Shift grouped points relative to centroid
            for jj, distances in enumerate(pairwise_distances):
                grouped_index = ((distances <= (radii * radii)) & (batches == sampled_batches[jj])).nonzero(as_tuple=True)
                grouped_position = positions[grouped_index]
                grouped_position -= sampled_positions[jj]
                if embedding is not None:
                    grouped_embedding = embedding[grouped_index]
                    grouped_embedding = torch.cat([grouped_embedding, grouped_position], dim=-1)
                else:
                    grouped_embedding = grouped_position
                if len(grouped_embedding) == 1:
                    grouped_embedding = torch.cat([grouped_embedding, grouped_embedding])
                grouped_embedding = self.embedding_dict[f'embedding_{ii}'](grouped_embedding)
                max_values = torch.max(grouped_embedding, dim=0)
                sampled_embedding[jj].append(max_values.values)
        
        for ii in range(len(sampled_positions)):
            sampled_embedding[ii] = torch.cat(sampled_embedding[ii])
        sampled_embedding = torch.stack(sampled_embedding)

        return sampled_positions, sampled_batches, sampled_embedding

        # sampling_and_grouping = self.sampling_and_grouping(positions, batches)
        # pointnet_embedding = self.pointnet(positions, batches, sampling_and_grouping)
        
        # return {
        #     'sampling_and_grouping': sampling_and_grouping,
        #     #'pointnet_embedding':    pointnet_embedding,
        # }



        