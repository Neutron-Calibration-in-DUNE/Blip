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

blip_config = {
    # input dimension
    'input_dimension':  3,
    # number of dynamic edge convs
    'num_dynamic_edge_conv':    2,
    # edge conv layer values
    'edge_conv_mlp_layers': [
        [64, 64],
        [64, 64]
    ],
    'number_of_neighbors':  20,
    'aggregation_operators': [
        'sum', 'sum'
    ],
    # linear layer
    'linear_output':    128,
    'mlp_output_layers': [128, 256, 32],
    'augmentations':    [
        T.RandomJitter(0.03), 
        T.RandomFlip(1), 
        T.RandomShear(0.2)
    ],
    # number of augmentations per batch
    'number_of_augmentations': 2
}

class BLIP(GenericModel):
    """
    """
    def __init__(self,
        name:   str='blip',
        cfg:    dict=blip_config
    ):
        super(BLIP, self).__init__(name, cfg)
        self.cfg = cfg
        self.augmentation = T.Compose(self.cfg['augmentations'])

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
        self.logger.info(f"Attempting to build chunc architecture using cfg: {self.cfg}")
        _edge_conv_dict = OrderedDict()
        _linear_dict = OrderedDict()
        _mlp_dict = OrderedDict()

        self.input_dimension = self.cfg['input_dimension']
        _input_dimension = self.cfg['input_dimension']
        _num_edge_conv_outputs = 0
        # Feature extraction
        # an example would be
        # self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64]), k, aggr)
        # self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        for ii in range(self.cfg['num_dynamic_edge_conv']):
            _edge_conv_dict[f'edge_conv_{ii}'] = DynamicEdgeConv(
                MLP([2 * _input_dimension] + self.cfg['edge_conv_mlp_layers'][ii]), 
                self.cfg['number_of_neighbors'],
                self.cfg['aggregation_operators'][ii]
            )
            _input_dimension = self.cfg['edge_conv_mlp_layers'][ii][-1]
            _num_edge_conv_outputs += _input_dimension
        
        # add linear layer Encoder head
        _linear_dict[f'linear_layer'] = Linear(
            _num_edge_conv_outputs, 
            self.cfg['linear_output']
        )

        # add output mlp Projection head (See explanation in SimCLRv2)
        _mlp_dict[f'mlp_output'] = MLP(
            self.cfg['mlp_output_layers'],
            norm=None
        )
        
        # create the dictionaries
        self.edge_conv_dict = nn.ModuleDict(_edge_conv_dict)
        self.linear_dict = nn.ModuleDict(_linear_dict)
        self.mlp_dict = nn.ModuleDict(_mlp_dict)

        # record the info
        self.logger.info(f"Constructed BLIP with dictionaries:\n{self.edge_conv_dict}\n{self.linear_dict}\n{self.mlp_dict}.")

    
    def forward(self,
        x
    ):
        """
        Iterate over the model dictionary
        """
        x = x.to(self.device)
        # if self.training:
        # Get augmentations of the batch
        pools, compacts = [], []
        for ii in range(self.cfg['number_of_augmentations']):
            augmentations = self.augmentation(x)
            pos, batch = augmentations.pos, augmentations.batch
            for ii, layer in enumerate(self.edge_conv_dict.keys()):
                pos = self.edge_conv_dict[layer](pos, batch)
                if ii == 0:
                    linear_input = pos
                else:
                    linear_input = torch.cat([linear_input, pos], dim=1)

            linear_output = self.linear_dict['linear_layer'](linear_input)
            linear_pool = global_max_pool(linear_output, batch)
            linear_compact = self.mlp_dict['mlp_output'](linear_pool)
            pools.append(linear_pool)
            compacts.append(linear_compact)

        indices = torch.arange(0, compacts[0].size(0), device=compacts[0].device)
        labels = torch.cat([indices for ii in range(len(compacts))])
        compacts = torch.cat(compacts)
        pools = torch.cat(pools)

        return pools, compacts, labels
    
    def forward_eval(self,
        x
    ):
        """
        Iterate over the model dictionary
        """
        x = x.to(self.device)
        # if self.training:
        pos, batch = x.pos, x.batch
        for ii, layer in enumerate(self.edge_conv_dict.keys()):
            pos = self.edge_conv_dict[layer](pos, batch)
            if ii == 0:
                linear_input = pos
            else:
                linear_input = torch.cat([linear_input, pos], dim=1)

        linear_output = self.linear_dict['linear_layer'](linear_input)
        linear_pool = global_max_pool(linear_output, batch)
        linear_compact = self.mlp_dict['mlp_output'](linear_pool)

        labels = x.category

        return linear_pool, linear_compact, labels