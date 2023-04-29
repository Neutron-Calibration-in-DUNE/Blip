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
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from sklearn.cluster import DBSCAN


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
    "coarse_graining_eps":  0.1,
    "eps_vals":         [0.1, 0.2, 0.4],
    "pointnet_embedding_mlp_layers":2,
    "pointnet_embedding_type":      "dynamic_edge_conv",
    "pointnet_number_of_neighbors": 20,
    "pointnet_aggregation":         "max",
    "pointnet_input_dimension":     3,
}

class SetAbstractionDBSCAN(GenericModel):
    """
    """
    def __init__(self,
        name:   str='set_abstraction',
        config: dict=set_abstraction_config
    ):
        super(SetAbstractionDBSCAN, self).__init__(name, config)
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
        for jj, eps in enumerate(self.config['eps_embedding_vals']): 
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
            f"Constructed SetAbstractionDBSCAN"
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
        indices,
        embedding,
    ):
        """
        Iterate over the sampling + grouping stage
        and then PointNet.
        """
        if indices is not None:
            positions = positions.to(self.device)[indices]
            batches = batches.to(self.device)[indices]
        else:
            positions = positions.to(self.device)
            batches = batches.to(self.device)
        unique_batchs = torch.unique(batches)

        if embedding is not None:
            positions = torch.cat((positions, embedding), dim=1)

        # sampling stage uses the coarse graining eps value
        # to generate clusters.  points with -1 get added to the 
        # list, as well as the closest points to the barycenters
        # of each cluster.
        coarse_grained_indices = []
        for batch in torch.unique(batches):
            pos = positions[(batches == batch)]
            clustering = DBSCAN(eps=self.config["coarse_graining_eps"], min_samples=2).fit(pos.detach().cpu())
            labels = torch.tensor(clustering.labels_)
            unique_labels = torch.unique(labels)
            minus_ones = torch.where(labels == -1)[0]
            coarse_grained_indices.append(minus_ones)
            for label in unique_labels:
                if label == -1:
                    continue
                group = pos[(labels == label)]   
                indices = torch.argwhere(labels == label)
                mean = torch.mean(group, axis=0)
                closest_point = torch.cdist(mean.unsqueeze(0), group).argmin()
                coarse_grained_indices.append(indices[closest_point])
        coarse_grained_indices = torch.cat(coarse_grained_indices)

        # now we go over all the eps values, run embeddings on 
        # each cluster, and append the embeddings to the associated
        # coarse grained indices.
        embeddings = [[] for ii in range(len(coarse_grained_indices))]

        for ii, eps in enumerate(self.config["eps_embedding_vals"]):
            for batch in torch.unique(batches):
                pos = positions[(batches == batch)]
                point_net = self.embedding_dict[f'embedding_{ii}'](pos)

                clustering = DBSCAN(eps=eps, min_samples=2).fit(pos.detach().cpu())
                labels = torch.tensor(clustering.labels_)
                unique_labels = torch.unique(labels)
                for label in unique_labels:   
                    indices = torch.argwhere(labels == label)
                    coarse_indices = torch.where(
                        (coarse_grained_indices.view(1, -1) == indices.view(-1, 1)).any(dim=0)
                    )[0]
                    # print(coarse_indices)
                    # coarse_indices = coarse_grained_indices[
                    #     (coarse_grained_indices.view(1, -1) == indices.view(-1, 1)).any(dim=0)
                    # ]
                    if label == -1:
                        for jj, index in enumerate(coarse_indices):
                            embeddings[index].append(point_net[coarse_grained_indices[index]])
                    else:
                        max_values = torch.max(point_net[coarse_grained_indices[coarse_indices]], dim=0).values.to(self.device)
                        for index in coarse_indices:
                            embeddings[index].append(max_values)
        
        for ii in range(len(coarse_grained_indices)):
            embeddings[ii] = torch.cat(embeddings[ii])
        embeddings = torch.stack(embeddings)
        return coarse_grained_indices, embeddings
    
        # ratio = len(torch.unique(batches))*float(self.config['sampling_num_samples'])/len(positions)
        
        # sampled_indices = torch_cluster.fps(positions, batches, ratio)
        # sampled_positions = positions[sampled_indices]
        # sampled_batches = batches[sampled_indices]

        # sampled_embedding = [[] for ii in range(len(sampled_positions))]

        # pairwise_distances = torch.cdist(sampled_positions, positions, p=2)

        # for ii, radii in enumerate(self.config['grouping_radii']):
        #     # if self.config['grouping_type'] == 'multi-scale':
        #     # Shift grouped points relative to centroid
        #     for jj, distances in enumerate(pairwise_distances):
        #         grouped_index = ((distances <= (radii * radii)) & (batches == sampled_batches[jj])).nonzero(as_tuple=True)
        #         grouped_position = positions[grouped_index]
        #         grouped_position -= sampled_positions[jj]
        #         if embedding is not None:
        #             grouped_embedding = embedding[grouped_index]
        #             grouped_embedding = torch.cat([grouped_embedding, grouped_position], dim=-1)
        #         else:
        #             grouped_embedding = grouped_position
        #         if len(grouped_embedding) == 1:
        #             grouped_embedding = torch.cat([grouped_embedding, grouped_embedding])
        #         grouped_embedding = self.embedding_dict[f'embedding_{ii}'](grouped_embedding)
        #         max_values = torch.max(grouped_embedding, dim=0)
        #         sampled_embedding[jj].append(max_values.values)
        
        # for ii in range(len(sampled_positions)):
        #     sampled_embedding[ii] = torch.cat(sampled_embedding[ii])
        # sampled_embedding = torch.stack(sampled_embedding)

        # return sampled_positions, sampled_batches, sampled_embedding

        # sampling_and_grouping = self.sampling_and_grouping(positions, batches)
        # pointnet_embedding = self.pointnet(positions, batches, sampling_and_grouping)
        
        # return {
        #     'sampling_and_grouping': sampling_and_grouping,
        #     #'pointnet_embedding':    pointnet_embedding,
        # }



        