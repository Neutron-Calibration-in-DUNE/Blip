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

pointnet_classification_config = {
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
        'sum', 'sum'
    ],
    # linear layer
    'linear_output':    128,
    'mlp_output_layers': [128, 256, 32],
    'classification_layers': [32, 64, 32, 10],
    'augmentations':    [
        T.RandomJitter(0.03), 
        T.RandomFlip(1), 
        T.RandomShear(0.2)
    ],
    # number of augmentations per batch
    'number_of_augmentations': 2
}

class PointNetClassification(GenericModel):
    """
    """
    def __init__(self,
        name:   str='pointnet_classification',
        cfg:    dict=pointnet_classification_config
    ):
        super(PointNetClassification, self).__init__(name, cfg)
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
        self.logger.info(f"Attempting to build {self.name} architecture using cfg: {self.cfg}")
        _embedding_dict = OrderedDict()
        _reduction_dict = OrderedDict()
        _classification_dict = OrderedDict()

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
            self.cfg['mlp_output_layers'],
            norm=None
        )

        _classification_dict[f'classification_output'] = MLP(
            self.cfg['classification_layers']
        )
        
        # create the dictionaries
        self.embedding_dict = nn.ModuleDict(_embedding_dict)
        self.reduction_dict = nn.ModuleDict(_reduction_dict)
        self.classification_dict = nn.ModuleDict(_classification_dict)

        # record the info
        self.logger.info(f"Constructed PointNetClassification with dictionaries:\n{self.embedding_dict}\n{self.reduction_dict}\n{self.reduction_dict}\n{self.classification_dict}.")

    
    def forward(self,
        x
    ):
        """
        Iterate over the model dictionary
        """
        x = x.to(self.device)
        # if self.training:
        # Get augmentations of the batch
        pools, compacts, classifications = [], [], []
        for ii in range(self.cfg['number_of_augmentations']):
            augmentations = self.augmentation(x)
            pos, batch = augmentations.pos, augmentations.batch
            for ii, layer in enumerate(self.embedding_dict.keys()):
                pos = self.embedding_dict[layer](pos, batch)
                if ii == 0:
                    linear_input = pos
                else:
                    linear_input = torch.cat([linear_input, pos], dim=1)

            linear_output = self.reduction_dict['linear_layer'](linear_input)
            linear_pool = global_max_pool(linear_output, batch)
            linear_compact = self.reduction_dict['mlp_output'](linear_pool)

            classification = self.classification_dict['classification_output'](linear_compact)
            pools.append(linear_pool)
            compacts.append(linear_compact)
            classifications.append(classification)

        compacts = torch.cat(compacts)
        pools = torch.cat(pools)
        classifications = torch.cat(classifications)

        return pools, compacts, classifications
    
    def forward_eval(self,
        x
    ):
        """
        Iterate over the model dictionary
        """
        x = x.to(self.device)
        # if self.training:
        pos, batch = x.pos, x.batch
        for ii, layer in enumerate(self.embedding_dict.keys()):
            pos = self.embedding_dict[layer](pos, batch)
            if ii == 0:
                linear_input = pos
            else:
                linear_input = torch.cat([linear_input, pos], dim=1)

        linear_output = self.reduction_dict['linear_layer'](linear_input)
        linear_pool = global_max_pool(linear_output, batch)
        linear_compact = self.reduction_dict['mlp_output'](linear_pool)

        labels = x.category

        return linear_pool, linear_compact, labels