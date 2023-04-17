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
    # input dimension
    'input_dimension':  3,
    # number of dynamic edge convs
    'num_embedding':    2,
    # edge conv layer values
    'embedding_mlp_layers': [
        [64, 64],
        [64, 64]
    ],
    'number_of_neighbors':  20,
    'aggregation_operators': [
        'max', 'max'
    ],
    # linear layer
    'linear_output':    128,
    'mlp_output_layers': [128, 256, 32],
}

class PointNet(GenericModel):
    """
    """
    def __init__(self,
        name:   str='pointnet',
        cfg:    dict=pointnet_config
    ):
        super(PointNet, self).__init__(name, cfg)
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
        _embedding_dict = OrderedDict()
        _reduction_dict = OrderedDict()

        self.input_dimension = self.cfg['input_dimension']
        _input_dimension = self.cfg['input_dimension']
        _num_embedding_outputs = 0
        # Feature extraction
        # an example would be
        for ii in range(self.cfg['num_embedding']):
            _embedding_dict[f'embedding_{ii}'] = DynamicEdgeConv(
                MLP([2 * _input_dimension] + self.cfg['embedding_mlp_layers'][ii]), 
                self.cfg['number_of_neighbors'],
                self.cfg['aggregation_operators'][ii]
            )
            _input_dimension = self.cfg['embedding_mlp_layers'][ii][-1]
            _num_embedding_outputs += _input_dimension
        
        # add linear layer Encoder head
        _reduction_dict[f'linear_layer'] = Linear(
            _num_embedding_outputs, 
            self.cfg['linear_output']
        )
        # add output mlp Projection head (See explanation in SimCLRv2)
        _reduction_dict[f'mlp_output'] = MLP(
            self.cfg['mlp_output_layers']
        )
        
        # create the dictionaries
        self.embedding_dict = nn.ModuleDict(_embedding_dict)
        self.reduction_dict = nn.ModuleDict(_reduction_dict)

        # record the info
        self.logger.info(
            f"Constructed PointNet with dictionaries:"
            + f"\n{self.embedding_dict}\n{self.reduction_dict}"
        )

    
    def forward(self,
        positions,
        batch
    ):
        """
        Iterate over the model dictionary
        """

        pos = positions
        for ii, layer in enumerate(self.embedding_dict.keys()):
            pos = self.embedding_dict[layer](pos, batch)
            if ii == 0:
                linear_input = pos
            else:
                linear_input = torch.cat([linear_input, pos], dim=1)

        embeddings = self.reduction_dict['linear_layer'](linear_input)
        pools = global_max_pool(embeddings, batch)
        reductions = self.reduction_dict['mlp_output'](pools)

        return {
            'embeddings': embeddings,
            'pools': pools, 
            'reductions': reductions, 
        }