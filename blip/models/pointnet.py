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
    "input_dimension":      3,
    "classifications":      ["source", "shape", "particle"],
    'augmentations':    {
        'jitter':   0.03,
        'flip':     1.0,
        'shear':    0.2
    },
    'number_of_augmentations':  2,
    "embedding_type":       "dynamic_edge_conv",
    "number_of_embeddings": 4,
    "number_of_neighbors":  [5, 10, 20, 30],
    "aggregation":          ["max", "max", "max", "max"],    
    "embedding_mlp_layers": [
        [5, 10, 25, 10],
        [10, 25, 50, 25],
        [20, 30, 40, 30],
        [30, 50, 75, 50]
    ],
    'linear_output':        128,
    'mlp_output_layers':    [128, 256, 32],
    'out_channels':         [8, 7, 32],
}

class PointNet(GenericModel):
    """
    """
    def __init__(self,
        name:   str='pointnet',
        config: dict=pointnet_config,
        device: str='cpu'
    ):
        super(PointNet, self).__init__(
            name, config, device
        )
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
        augmentations = []
        for augmentation in self.config["augmentations"]:
            if augmentation == 'jitter':
                augmentations.append(T.RandomJitter(self.config['augmentations'][augmentation]))
            elif augmentation == 'flip':
                for ii in range(len(self.config['augmentations'][augmentation]['positions'])):
                    augmentations.append(T.RandomFlip(
                        self.config['augmentations'][augmentation]['positions'][ii],
                        self.config['augmentations'][augmentation]['probabilities'][ii]
                    ))
            elif augmentation == 'shear':
                augmentations.append(T.RandomShear(self.config['augmentations'][augmentation]))
        self.augmentations = T.Compose(augmentations)
        self.number_of_augmentations = self.config['number_of_augmentations']
        self.logger.info(f"Attempting to build {self.name} architecture using config: {self.config}")
        
        """
        Feature extraction layers
        Each radii (for multi-scale grouping) will have a set
        of pointnet embedding layers applied.
        """
        self.embedding_dicts = []
        _embedding_dict = OrderedDict()
        _reduction_dict = OrderedDict()
        _classification_dict = OrderedDict()

        _input_dimension = self.config['input_dimension']
        _num_embedding_outputs = 0
        
        for ii in range(self.config['number_of_embeddings']):
            if self.config['embedding_type'] == 'dynamic_edge_conv':
                _embedding_dict[f'embedding_{ii}'] = DynamicEdgeConv(
                    MLP([2 * _input_dimension] + self.config['embedding_mlp_layers'][ii]), 
                    self.config['number_of_neighbors'][ii],
                    self.config['aggregation'][ii]
                )
            _input_dimension = self.config['embedding_mlp_layers'][ii][-1]
            _num_embedding_outputs += _input_dimension

        # add linear layer Encoder head
        _reduction_dict[f'linear_layer'] = Linear(
            _num_embedding_outputs, 
            self.config['linear_output']
        )
        # add output mlp Projection head (See explanation in SimCLRv2)
        for ii, classification in enumerate(self.config["classifications"]):
            _classification_dict[f'{classification}'] = MLP(
                self.config['mlp_output_layers'] + [self.config['out_channels'][ii]]
            )

        self.embedding_dict = nn.ModuleDict(_embedding_dict)
        self.reduction_dict = nn.ModuleDict(_reduction_dict)
        self.classification_dict = nn.ModuleDict(_classification_dict)
        self.softmax = nn.Softmax(dim=1)

        # record the info
        self.logger.info(
            f"Constructed PointNet with dictionaries:"
        )

    
    def forward(self,
        data
    ):
        """
        Iterate over the model dictionary
        """
        if self.training:
            reductions, classifications = [], [[]]
            for kk in range(self.number_of_augmentations):
                augmentations = self.augmentations(data).to(self.device)
                pos, batch = augmentations.pos, augmentations.batch
                for ii, embedding in enumerate(self.embedding_dict.keys()):
                    pos = self.embedding_dict[embedding](pos, batch)
                    if ii == 0:
                        linear_input = pos
                    else:
                        linear_input = torch.cat([linear_input, pos], dim=1)
                linear_output = self.reduction_dict['linear_layer'](linear_input)
                linear_pool = global_max_pool(linear_output, batch)
                for jj, classification in enumerate(self.classification_dict.keys()):
                    classifications[jj].append(self.classification_dict[classification](linear_pool))
                reductions.append(linear_pool)
            outputs = {
                classification: torch.cat(classifications[jj])
                for jj, classification in enumerate(self.classification_dict.keys())
            }
            outputs['reductions'] = torch.cat(reductions)
        else:
            pos = data.pos.to(self.device)
            batch = data.batch.to(self.device)
            for ii, embedding in enumerate(self.embedding_dict.keys()):
                pos = self.embedding_dict[embedding](pos, batch)
                if ii == 0:
                    linear_input = pos
                else:
                    linear_input = torch.cat([linear_input, pos], dim=1)
            linear_output = self.reduction_dict['linear_layer'](linear_input)
            linear_pool = global_max_pool(linear_output, batch)
            outputs = {
                classifications: self.classification_dict[classifications](linear_pool)
                for classifications in self.classification_dict.keys()
            }
            outputs['reductions'] = linear_pool
        return outputs