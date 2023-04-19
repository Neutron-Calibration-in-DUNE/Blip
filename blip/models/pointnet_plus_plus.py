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
from blip.models import GenericModel, SetAbstraction, SetAbstractionMultiScaleGrouping
from blip.utils.sampling import *
from blip.utils.grouping import *
from blip.models import PointNet

pointnet_plusplus_cfg = {
    "model_type":   "PointNet++",
    "set_abstraction_layers": {
        "sampling_methods": ["farthest_point", "farthest_point", "farthest_point"],
        "sampling_num_samples": [1024, 512, 256],
        "grouping_methods": ["query_ball_point", "query_ball_point", "query_ball_point"],
        "grouping_type":    ["multi-scale", "multi-scale", "all"],
        "grouping_radii":   [[0.1, 0.2, 0.4], [0.2, 0.4, 0.8], []],
        "grouping_samples": [[16, 32, 128], [32, 64, 128], []],
        "pointnet_embeddings":  [2, 2, 2],
        "pointnet_type":    ["dynamic_edge_conv", "dynamic_edge_conv", "dynamic_edge_conv"],
        "pointnet_number_of_neighbors": [20, 20, 20],
        "pointnet_aggregation": [["max", "max"], ["max", "max"], ["max", "max"]],
    },
    "segmentation_layers":      {},
    "classification_layers":    {},
}

class PointNetPlusPlus(GenericModel):
    """
    """
    def __init__(self,
        name:   str='pointnet_plusplus',
        cfg:    dict=pointnet_plusplus_cfg
    ):
        super(PointNetPlusPlus, self).__init__(name, cfg)
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

        _set_abstraction_dict = OrderedDict()
        _classification_dict = OrderedDict()
        for ii, layer in enumerate(self.cfg['set_abstraction_layers']["sampling_methods"]):
            _set_abstraction_dict[f'set_abstraction_layers_{ii}'] = SetAbstraction(
                self.name + f"_set_abstraction_layer_{ii}",
                {
                    "sampling_method":  self.cfg['set_abstraction_layers']['sampling_methods'][ii],
                    "sampling_num_samples":  self.cfg['set_abstraction_layers']['sampling_num_samples'][ii],
                    "grouping_method":  self.cfg['set_abstraction_layers']['grouping_methods'][ii],
                    "grouping_type":  self.cfg['set_abstraction_layers']['grouping_type'][ii],
                    "grouping_radii":  self.cfg['set_abstraction_layers']['grouping_radii'][ii],
                    "grouping_samples":  self.cfg['set_abstraction_layers']['grouping_samples'][ii],
                }
            )
        # for ii in range(len(self.cfg['classification']['mlp'])-1):
        #     _classification_dict[f'mlp_{ii}'] = nn.Linear(
        #         self.cfg['classification']['mlp'][ii],
        #         self.cfg['classification']['mlp'][ii+1]
        #     )
        #     _classification_dict[f'batch_norm_{ii}'] = nn.BatchNorm1d(self.cfg['classification']['mlp'][ii+1])
        #     _classification_dict[f'relu_{ii}'] = F.relu
        #     _classification_dict[f'dropout_{ii}'] = nn.Dropout(self.cfg['classification']['dropout'][ii])
        # _classification_dict['output'] = nn.Linear(self.cfg['classification']['mlp'][-1], self.number_of_classes)
        # _classification_dict['softmax'] = F.log_softmax
        
        
        # create the dictionaries
        self.set_abstraction_dict = nn.ModuleDict(_set_abstraction_dict)
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
        batch_size, = positions.shape
        embedding = None

        for ii, layer in enumerate(self.set_abstraction_dict.keys()):
            positions, embedding = self.set_abstraction_dict[layer](positions, embedding)
        output = embedding.view(batch_size, self.cfg['classification']['mlp'][0])
        # for ii, layer in enumerate(self.classification_dict.keys()):
        #     output = self.classification_dict[layer](output)
        
        return {
            'output':   output,
            'embedding':embedding,
        }


        