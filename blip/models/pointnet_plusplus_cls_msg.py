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

pointnet_plusplus_cls_msg_config = {
    # input dimension
    'input_dimension':      3,
    # number of classes
    'number_of_classes':    10,
    # set abstraction layers
    'set_abstraction_msg': {
        'layer1':   {
            'num_points':   512,
            'radius_list':  [0.1, 0.2, 0.4],
            'num_samples':  [16, 32, 128],
            'in_channels':  3,
            'mlp_list':     [
                [32, 32, 64],
                [64, 64, 128],
                [64, 96, 128]
            ],
        },
        'layer2':   {
            'num_points':   128,
            'radius_list':  [0.2, 0.4, 0.8],
            'num_samples':  [32, 64, 128],
            'in_channels':  320,
            'mlp_list':     [
                [64, 64, 128],
                [128, 128, 256],
                [128, 128, 256]
            ],
        },
    },
    'set_abstraction': {
        'num_points':   None,
        'radius':       None,
        'num_samples':  None,
        'in_channels':  643,
        'mlp':          [256, 512, 1024],
        'group_all':    True,
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
        
        """
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
        x
    ):
        """
        Iterate over the model dictionary
        """
        x = x.to(self.device)
        B, _, _ = x[0].shape

        for ii, layer in enumerate(self.embedding_dict.keys()):

        