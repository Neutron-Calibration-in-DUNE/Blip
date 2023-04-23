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
from blip.models import GenericModel, SetAbstraction, SetAbstractionMultiScaleGrouping
from blip.models import FeaturePropagation
from blip.models import Segmentation
from blip.utils.sampling import *
from blip.utils.grouping import *
from blip.models import PointNet

pointnet_plusplus_config = {
    "model_type":   "PointNet++",
    "set_abstraction_layers": {
        "sampling_methods":     ["farthest_point", "farthest_point", "farthest_point"],
        "sampling_num_samples": [1024, 512, 256],
        "grouping_methods":     ["query_ball_point", "query_ball_point", "query_ball_point"],
        "grouping_type":        ["multi-scale", "multi-scale", "all"],
        "grouping_radii":       [[0.1, 0.2, 0.4], [0.2, 0.4, 0.8], []],
        "grouping_samples":     [[16, 32, 128], [32, 64, 128], []],
        "pointnet_num_embeddings":      [2, 2, 2],
        "pointnet_embedding_mlp_layers":[[2, 2], [2, 2], [2, 2]],
        "pointnet_embedding_type":      ["dynamic_edge_conv", "dynamic_edge_conv", "dynamic_edge_conv"],
        "pointnet_number_of_neighbors": [20, 20, 20],
        "pointnet_aggregation":         [["max", "max"], ["max", "max"], ["max", "max"]],
        "pointnet_input_dimension":     [3, 5, 7]
    },
    "segmentation_layers": {
        "num_inputs":   15
    },
    "classification_layers":    {},
}

class PointNetPlusPlus(GenericModel):
    """
    """
    def __init__(self,
        name:       str='pointnet_plusplus',
        config:    dict=pointnet_plusplus_config
    ):
        super(PointNetPlusPlus, self).__init__(name, config)
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

        _set_abstraction_dict = OrderedDict()
        _feature_propagation_dict = OrderedDict()
        _segmentation_dict = OrderedDict()
        _classification_dict = OrderedDict()
        
        for ii, layer in enumerate(self.config['set_abstraction_layers']["sampling_methods"]):
            _set_abstraction_dict[f'set_abstraction_layers_{ii}'] = SetAbstraction(
                self.name + f"_set_abstraction_layer_{ii}",
                {
                    "sampling_method":      self.config['set_abstraction_layers']['sampling_methods'][ii],
                    "sampling_num_samples": self.config['set_abstraction_layers']['sampling_num_samples'][ii],
                    "grouping_method":      self.config['set_abstraction_layers']['grouping_methods'][ii],
                    "grouping_type":        self.config['set_abstraction_layers']['grouping_type'][ii],
                    "grouping_radii":       self.config['set_abstraction_layers']['grouping_radii'][ii],
                    "grouping_samples":     self.config['set_abstraction_layers']['grouping_samples'][ii],
                    "pointnet_embedding_mlp_layers":self.config['set_abstraction_layers']['pointnet_embedding_mlp_layers'][ii],
                    "pointnet_embedding_type":      self.config['set_abstraction_layers']['pointnet_embedding_type'][ii],
                    "pointnet_number_of_neighbors": self.config['set_abstraction_layers']['pointnet_number_of_neighbors'][ii],
                    "pointnet_aggregation":         self.config['set_abstraction_layers']['pointnet_aggregation'][ii],
                    "pointnet_input_dimension":     self.config['set_abstraction_layers']['pointnet_input_dimension'][ii]
                }
            )
        for ii, layer in enumerate(self.config['set_abstraction_layers']["sampling_methods"]):
            _feature_propagation_dict[f'feature_propagaion_layers_{ii}'] = FeaturePropagation(
                self.name + f"_feature_propagation_layer_{ii}",
                {
                }
            )
        _segmentation_dict['segmentation_layer'] = Segmentation(
            self.name + f"_segmentation_layer",
            self.config['segmentation_layer']
        )
        # for ii in range(len(self.config['classification']['mlp'])-1):
        #     _classification_dict[f'mlp_{ii}'] = nn.Linear(
        #         self.config['classification']['mlp'][ii],
        #         self.config['classification']['mlp'][ii+1]
        #     )
        #     _classification_dict[f'batch_norm_{ii}'] = nn.BatchNorm1d(self.config['classification']['mlp'][ii+1])
        #     _classification_dict[f'relu_{ii}'] = F.relu
        #     _classification_dict[f'dropout_{ii}'] = nn.Dropout(self.config['classification']['dropout'][ii])
        # _classification_dict['output'] = nn.Linear(self.config['classification']['mlp'][-1], self.number_of_classes)
        # _classification_dict['softmax'] = F.log_softmax
        
        
        # create the dictionaries
        self.set_abstraction_dict = nn.ModuleDict(_set_abstraction_dict)
        self.feature_propagation_dict = nn.ModuleDict(_feature_propagation_dict)
        self.segmentation_dict = nn.ModuleDict(_segmentation_dict)
        # self.classification_dict = nn.ModuleDict(_classification_dict)

        # record the info
        self.logger.info(
            f"Constructed PointNetClassification with dictionaries:"
            # + f"\n{self.set_abstraction_dict}\n{self.classification_dict}."
        )

    def forward(self,
        data
    ):
        """
        Iterate over the model dictionary
        """
        positions = data.to(self.device).pos
        batches = data.to(self.device).batch
        sampling_and_grouping = {}

        samples = positions
        embedding = None
        samples_list = [positions]
        embedding_list = [None]

        for ii, layer in enumerate(self.set_abstraction_dict.keys()):
            samples, batches, embedding = self.set_abstraction_dict[layer](samples, batches, embedding)
            samples_list.append(samples)
            embedding_list.append(embedding)

        for ii, layer in enumerate(self.feature_propagation_dict.keys()):
            embedding_list[-(ii+2)] = self.feature_propagation_dict[layer](
                samples_list[-(ii+1)], embedding_list[-(ii+1)],
                samples_list[-(ii+2)], embedding_list[-(ii+2)]
            )
        
        segmentation = self.segmentation_dict["segmentation_layer"](
            positions, embedding_list[0]
        )
        classification = None

        return segmentation, classification


        