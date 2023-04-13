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

set_abstraction_msg_config = {
    'num_points':   1024,
    'radius_list':  [0.05, 0.1],
    'num_samples':  [16, 32],
    'in_channels':  3,
    'mlp_list':     [[16, 16, 32], [32, 32, 64]]
}

class SetAbstractionMultiScaleGrouping(GenericModel):
    """
    """
    def __init__(self,
        name:   str='set_abstraction_msg',
        cfg:    dict=set_abstraction_msg_config
    ):
        super(SetAbstractionMultiScaleGrouping, self).__init__(name, cfg)
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
        
        self.input_dimension = self.cfg['in_channels']
        self.group_all = self.cfg['group_all']
        for ii in range(len(self.cfg['mlp_list'])):
            last_channel = self.cfg['in_channels']
            for jj in range(len(self.cfg['mlp'][ii])):
                _convolution_dict[f'conv_{ii}_{jj}'] = nn.Conv2d(
                    last_channel, self.cfg['mlp'][ii][jj], 1
                )
                _convolution_dict[f'batch_norm_{ii}_{jj}'] = nn.BatchNorm2d(
                    self.cfg['mlp'][ii][jj]
                )
                last_channel = self.cfg['mlp'][ii][jj]

        self.convolution_dict = nn.ModuleDict(_convolution_dict)

        # record the info
        self.logger.info(
            f"Constructed SetAbstractionMultiScaleGrouping with dictionary:"
            + f"\n{self.convolution_dict}"
        )
    
    def forward(self,
        x
    ):
        """
        Iterate over the model dictionary
        """
        x = x.to(self.device)
        