"""
Implementation of the WaveNet model using pytorch
"""
import numpy as np
import torch
import copy
import torch.nn as nn
from collections import OrderedDict
import torch_geometric.transforms as T
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, PointNetConv, PointTransformerConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import MinkowskiEngine as ME

from blip.models.common import activations, normalizations
from blip.models import GenericModel
from blip.models.causal_dilated_conv1d_model import CausalDilatedConv1D

residual_block_config = {
    "res_channels": 0,
    "skip_channels":0,
    "dimension":    1,
    "kernel_size":  0,
    "dilation":     0,
    "bias":         False,
}

class ResidualBlock(ME.MinkowskiNetwork, GenericModel):
    """
    """
    def __init__(self,
        name:   str='residual_block',
        config: dict=residual_block_config,
    ):
        super(ResidualBlock, self).__init__(self.config["dimension"])
        self.name = name
        self.config = config

        # construct the model
        self.forward_views      = {}
        self.forward_view_map   = {}

        # construct the model
        self.construct_model()

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.

        
        """
        self.logger.info(f"Attempting to build ResidualBlock architecture using config: {self.config}")

        _model_dict = OrderedDict()
        dilated_dict = {
            'in_channels':  self.config['res_channels'],
            'out_channels': self.config['res_channels'],
            'dimension':    self.config['dimension'],
            'kernel_size':  self.config['kernel_size'],
            'dilation':     self.config['dilation'],
            'bias':         self.config['bias']
        }
        _model_dict[f'{self.name}_dilated'] = CausalDilatedConv1D(
            f'{self.name}_dilated',
            dilated_dict,
        )
        _model_dict[f'{self.name}_tanh'] = nn.Tanh()
        _model_dict[f'{self.name}_sigmoid'] = nn.Sigmoid()

        # make residual and skip connection layers
        residual_dict = copy.deepcopy(dilated_dict)
        residual_dict['kernel_size'] = 1
        residual_dict['dilation'] = 1
        _model_dict[f'{self.name}_residual'] = CausalDilatedConv1D(
            f'{self.name}_residual',
            residual_dict,
        )
        skip_dict = copy.deepcopy(dilated_dict)
        skip_dict['out_channels'] = self.config['skip_channels']
        _model_dict[f'{self.name}_skip'] = CausalDilatedConv1D(
            f'{self.name}_skip',
            skip_dict,
        )

        self.model_dict = nn.ModuleDict(_model_dict)

        # record the info
        self.logger.info(
            f"Constructed ResidualBlock with dictionaries:"
        )

    
    def forward(self,
        x
    ):
        """
        Iterate over the model dictionary
        """
        conv1d = self.model_dict[f'{self.name}_dilated'](x)
        tanh = self.model_dict[f'{self.name}_tanh'](conv1d)
        sigm = self.model_dict[f'{self.name}_sigmoid'](conv1d)
        product = tanh * sigm
        residual = self.model_dict[f'{self.name}_residual'](product) + x
        skip = self.model_dict[f'{self.name}_skip'](x)
        
        return residual, skip