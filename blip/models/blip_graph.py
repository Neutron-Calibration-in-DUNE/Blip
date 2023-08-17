"""
Implementation of the blip graph model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch_geometric.transforms as T
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, PointNetConv, PointTransformerConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


from blip.models.common import activations, normalizations
from blip.models import GenericModel

blip_graph_config = {
    "input_dimension":      3,
    "classifications":      ["source", "shape", "particle"],
    'augmentations':    {
        'jitter':   0.03,
        'flip':     1.0,
        'shear':    0.2,
        'number_of_augmentations':  2
    },
    'embedding': {
        'embedding_1': {
            "embedding_type":       "dynamic_edge_conv",
            "number_of_neighbors":  5,
            "aggregation":          "max",    
            "embedding_mlp_layers": [5, 10, 25, 10],
        },
        'embedding_2': {
            "embedding_type":   "point_net_conv",
            "local_mlp_layers": [5, 10, 25, 10],
            "add_self_loops":   True,
        },
    },
    'reduction': {
        'linear_output':        128,
        'reduction_type':       'max_pool',
    },
    'classification': {
        'mlp_output_layers':    [128, 256, 32],
        'out_channels':         [8, 7, 32],
    },
}

class BlipGraph(GenericModel):
    """
    """
    def __init__(self,
        name:   str='blip_graph',
        config: dict=blip_graph_config,
        meta:   dict={}
    ):
        super(BlipGraph, self).__init__(
            name, config, meta
        )
        self.config = config

        # construct the model
        self.forward_views      = {}
        self.forward_view_map   = {}

        # construct augmentations
        self.construct_augmentations()
        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()

    def construct_augmentations(self):
        """
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
            elif augmentation == "rotate":
                for ii in range(len(self.config['augmentations'][augmentation]['degrees'])):
                    augmentations.append(T.RandomRotate(
                        self.config['augmentations'][augmentation]['degrees'][ii],
                        self.config['augmentations'][augmentation]['axis'][ii]
                    ))
            elif augmentation == 'number_of_augmentations':
                self.number_of_augmentations = self.config['augmentations'][augmentation]

        self.augmentations = T.Compose(augmentations)

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.

        This network consists of three sections:
            (1) embedding layers - graph layers for embedding the point clouds.
            (2) reduction layers - layer for reducing the embedding to a fixed dimension.
            (3) classification layers - layers for classifying the reduced embeddings.

        each section should have its own config subsection like the following:

        BlipGraph:
            augmentations:
                ...
            embedding:
                ...
            reduction:
                ...
            classification:
                ...
        """
        self.logger.info(f"Attempting to build BlipGraph architecture using config: {self.config}")

        self.embedding_dicts = []
        _embedding_dict = OrderedDict()
        _reduction_dict = OrderedDict()
        _classification_dict = OrderedDict()

        _input_dimension = self.config['input_dimension']
        _num_embedding_outputs = 0
        
        # iterate over embeddings
        embedding_config = self.config['embedding']
        for ii, embedding in enumerate(embedding_config.keys()):
            if embedding_config[embedding]['embedding_type'] == 'dynamic_edge_conv':
                _embedding_dict[embedding] = DynamicEdgeConv(
                    nn=MLP([2 * _input_dimension] + embedding_config[embedding]['embedding_mlp_layers']), 
                    k=embedding_config[embedding]['number_of_neighbors'],
                    aggr=embedding_config[embedding]['aggregation']
                )
                _input_dimension = embedding_config['embedding_mlp_layers'][-1]
            elif embedding_config[embedding]['embedding_type'] == 'point_net_conv':
                _embedding_dict[embedding] = PointNetConv(
                    local_nn=MLP([_input_dimension + self.config['input_dimension']] + embedding_config[embedding]['local_mlp_layers']), 
                    add_self_loops=embedding_config[embedding]['add_self_loops']
                )
                _input_dimension = embedding_config[embedding]['local_mlp_layers'][-1]
            else:
                self.logger.error(f'specified embedding type {embedding_config[embedding]["embedding_type"]} not allowed!')
            _num_embedding_outputs += _input_dimension

        if self.config["add_summed_adc"]:
            self.config['classification']['mlp_output_layers'][0] += 1

        # reduction layer
        reduction_config = self.config['reduction']
        # add linear layer Encoder head
        _reduction_dict['linear_layer'] = Linear(
            _num_embedding_outputs, 
            reduction_config['linear_output']
        )
        if reduction_config['reduction_type'] == 'add_pool':
            self.pooling_layer = global_add_pool
        elif reduction_config['reduction_type'] == 'mean_pool':
            self.pooling_layer = global_mean_pool
        else:
            self.pooling_layer = global_max_pool

        # classification layer
        classifcation_config = self.config['classification']
        # add output mlp Projection head (See explanation in SimCLRv2)
        for ii, classification in enumerate(self.config["classifications"]):
            _classification_dict[f'{classification}'] = MLP(
                classifcation_config['mlp_output_layers'] + [classifcation_config['out_channels'][ii]]
            )

        self.embedding_dict = nn.ModuleDict(_embedding_dict)
        self.reduction_dict = nn.ModuleDict(_reduction_dict)
        self.classification_dict = nn.ModuleDict(_classification_dict)
        self.softmax = nn.Softmax(dim=1)

        # record the info
        self.logger.info(
            f"Constructed BlipGraph with dictionaries:"
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
                # Create augmentations
                augmentations = self.augmentations(data).to(self.device)
                pos, batch = augmentations.pos, augmentations.batch
                if self.config["add_summed_adc"]:
                    summed_adc = augmentations.summed_adc

                # Pass through embedding dictionary
                for ii, embedding in enumerate(self.embedding_dict.keys()):
                    pos = self.embedding_dict[embedding](pos, batch)
                    if ii == 0:
                        linear_input = pos
                    else:
                        linear_input = torch.cat([linear_input, pos], dim=1)

                # Pass through reduction dictionary
                linear_output = self.reduction_dict['linear_layer'](linear_input)
                
                # Apply Pooling
                linear_pool = self.pooling_layer(linear_output, batch)

                if self.config["add_summed_adc"]:
                    linear_pool = torch.cat([linear_pool, summed_adc.unsqueeze(1)], dim=1)

                # Pass through classification dictionary
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
            if self.config["add_summed_adc"]:
                summed_adc = data.summed_adc.to(self.device)
            for ii, embedding in enumerate(self.embedding_dict.keys()):
                pos = self.embedding_dict[embedding](pos, batch)
                if ii == 0:
                    linear_input = pos
                else:
                    linear_input = torch.cat([linear_input, pos], dim=1)

            # Pass through reduction dictionary
            linear_output = self.reduction_dict['linear_layer'](linear_input)
            linear_pool = self.pooling_layer(linear_output, batch)
            
            if self.config["add_summed_adc"]:
                linear_pool = torch.cat([linear_pool, summed_adc.unsqueeze(1)], dim=1)
            outputs = {
                classifications: self.classification_dict[classifications](linear_pool)
                for classifications in self.classification_dict.keys()
            }
            outputs['reductions'] = linear_pool
        return outputs