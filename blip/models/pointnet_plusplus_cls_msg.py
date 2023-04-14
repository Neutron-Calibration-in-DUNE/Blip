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

pointnet_plusplus_cls_msg_config = {
    # input dimension
    'input_dimension':      3,
    # number of classes
    'number_of_classes':    10,
    # set abstraction layers
    'set_abstraction_msg': {
        'layer1':   {
            'sampling': {
                'method':   farthest_point_sampling,
                'number_of_centroids':  512,
            },
            'grouping': {
                'method':       query_ball_point,
                'radii_list':           [0.1, 0.2, 0.4],
                'number_of_samples':    [16, 32, 128],
            },
            'pointnet': {
                'method':   PointNet,
            },
        },
        'layer2':   {
            'sampling': {
                'method':   farthest_point_sampling,
                'number_of_centroids':  128,
            },
            'grouping': {
                'method':       query_ball_point,
                'radii_list':           [0.2, 0.4, 0.8],
                'number_of_samples':    [32, 64, 128],
            },
            'pointnet': {
                'method':   PointNet,
            },
        },
    },
    'set_abstraction': {
        'sampling': {
            'method':   farthest_point_sampling,
            'number_of_centroids':  None,
        },
        'grouping': {
            'method':       query_ball_point,
            'radii_list':           None,
            'number_of_samples':    None,
        },
        'pointnet': {
            'method':   PointNet,
        },
    },
    'classification':   {
        'mlp':      [1024, 512, 256],
        'dropout':  [0.4, 0.5, 0.0],
    },
}

class PointNetPlusPlusClassificationMSG(GenericModel):
    """
    """
    def __init__(self,
        name:   str='pointnet_plusplus_cls_msg',
        cfg:    dict=pointnet_plusplus_cls_msg_config
    ):
        super(PointNetPlusPlusClassificationMSG, self).__init__(name, cfg)
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
        self.number_of_classes = self.cfg['number_of_classes']

        _set_abstraction_dict = OrderedDict()
        _classification_dict = OrderedDict()
        for layer in range(len(self.cfg['set_abstraction_msg']).keys()):
            _set_abstraction_dict[f'set_abstraction_msg_{layer}'] = SetAbstractionMultiScaleGrouping(
                self.cfg['set_abstraction_msg'][layer]
            )
        _set_abstraction_dict['set_abstraction'] = SetAbstraction(
            self.cfg['set_abstraction']
        )
        for ii in range(len(self.cfg['classification']['mlp'])-1):
            _classification_dict[f'mlp_{ii}'] = nn.Linear(
                self.cfg['classification']['mlp'][ii],
                self.cfg['classification']['mlp'][ii+1]
            )
            _classification_dict[f'batch_norm_{ii}'] = nn.BatchNorm1d(self.cfg['classification']['mlp'][ii+1])
            _classification_dict[f'relu_{ii}'] = F.relu
            _classification_dict[f'dropout_{ii}'] = nn.Dropout(self.cfg['classification']['dropout'][ii])
        _classification_dict['output'] = nn.Linear(self.cfg['classification']['mlp'][-1], self.number_of_classes)
        _classification_dict['softmax'] = F.log_softmax
        
        
        # create the dictionaries
        self.set_abstraction_dict = nn.ModuleDict(_set_abstraction_dict)
        self.classification_dict = nn.ModuleDict(_classification_dict)

        # record the info
        self.logger.info(
            f"Constructed PointNetClassification with dictionaries:"
            + f"\n{self.set_abstraction_dict}\n{self.classification_dict}."
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
        for ii, layer in enumerate(self.classification_dict.keys()):
            output = self.classification_dict[layer](output)
        
        return {
            'output':   output,
            'embedding':embedding,
        }


        