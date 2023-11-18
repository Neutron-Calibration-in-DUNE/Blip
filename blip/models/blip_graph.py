"""
Implementation of the BlipGraph model using pytorch.

BlipGraph consists of the following architecture:


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
from torch_geometric.nn import fps, radius


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
            "embedding_type":       "point_net_conv",
            "local_mlp_layers":     [5, 10, 25, 10],
            "fps_ratio":            0.5,
            "cluster_radius":       0.25,
            "max_number_neighbors": 5,
            "add_self_loops":       True,
        },
        'embedding_3': {
            "embedding_type":       'point_transformer_conv',
            "pos_nn_layers":        [10,25,10],
            "attn_nn_layers":       [10,25,10],
            "fps_ratio":            0.25,
            "cluster_radius":       0.25,
            "max_number_neighbors": 5,
            "add_self_loops":       True,
        }
    },
    'reduction': {
        'linear_output':        128,
        'reduction_type':       'max_pool',
        'projection_head':      [128, 256, 128]
    },
    'classification': {
        'mlp_output_layers':    [128, 256, 32],
        'out_channels':         [8, 7, 32],
    },
}

# PointNetConv Modules
class PointNetModule(torch.nn.Module):
    """
    Methodology for using PointNetConv is to not reduce
    positions into cluster centers, but rather return the
    full set of positions after each layer.
    """
    def __init__(self, 
        local_mlp:      MLP,
        fps_ratio:      float=0.5, 
        cluster_radius: float=0.25, 
        max_number_neighbors:   int=10,
        add_self_loops:     bool=False
    ):
        super().__init__()
        self.local_mlp = local_mlp
        self.fps_ratio = fps_ratio
        self.cluster_radius = cluster_radius
        self.max_number_neighbors = max_number_neighbors
        self.add_self_loops = add_self_loops
        self.point_net_conv = PointNetConv(
            local_nn=self.local_mlp, 
            add_self_loops=self.add_self_loops
        )

    def forward(self, pos, batch):
        # find indices with furthest point sampling 
        idx = fps(
            pos, batch, 
            ratio=self.fps_ratio
        )
        # gather clusters
        row, col = radius(
            pos, pos[idx], 
            self.cluster_radius, 
            batch, batch[idx],
            max_num_neighbors=self.max_number_neighbors
        )
        # determine edges for convolution
        edge_index = torch.stack([col, row], dim=0)
        output = self.point_net_conv(
            None, 
            pos, 
            edge_index
        )
        return output

# PointTransformerConv Modules
class PointTransformerModule(torch.nn.Module):
    """
    Methodology for using PointTransformerConv is to not reduce
    positions into cluster centers, but rather return the
    full set of positions after each layer.
    """
    def __init__(self, 
        in_channels:    int,
        out_channels:   int,
        pos_nn:         MLP,
        attn_nn:        MLP,
        fps_ratio:      float=0.5, 
        cluster_radius: float=0.25, 
        max_number_neighbors:   int=10,
        add_self_loops:     bool=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos_nn = pos_nn
        self.attn_nn = attn_nn
        self.fps_ratio = fps_ratio
        self.cluster_radius = cluster_radius
        self.max_number_neighbors = max_number_neighbors
        self.add_self_loops = add_self_loops
        self.point_transformer_conv = PointTransformerConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            pos_nn=self.pos_nn,
            attn_nn=self.attn_nn,
            add_self_loops=self.add_self_loops
        )

    def forward(self, pos, batch):
        # find indices with furthest point sampling 
        idx = fps(
            pos, batch, 
            ratio=self.fps_ratio
        )
        # gather clusters
        row, col = radius(
            pos, pos[idx], 
            self.cluster_radius, 
            batch, batch[idx],
            max_num_neighbors=self.max_number_neighbors
        )
        # determine edges for convolution
        edge_index = torch.stack([col, row], dim=0)
        output = self.point_transformer_conv(
            pos, 
            pos, 
            edge_index
        )
        return output


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

        # construct augmentations
        self.construct_augmentations()
        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()

    def construct_augmentations(self):
        """
        Currently there are four augmentations that can be implemented
        as part of the contrastive learning in BlipGraph, 
        'jitter', 'flip', 'shear' and 'rotate'.

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
                _input_dimension = embedding_config[embedding]['embedding_mlp_layers'][-1]
            elif embedding_config[embedding]['embedding_type'] == 'point_net_conv':
                _embedding_dict[embedding] = PointNetModule(
                    local_mlp=MLP(
                        [_input_dimension] + 
                        embedding_config[embedding]['local_mlp_layers']
                    ), 
                    fps_ratio=embedding_config[embedding]['fps_ratio'],
                    cluster_radius=embedding_config[embedding]['cluster_radius'],
                    max_number_neighbors=embedding_config[embedding]['max_number_neighbors'],
                    add_self_loops=embedding_config[embedding]['add_self_loops']
                )
                _input_dimension = embedding_config[embedding]['local_mlp_layers'][-1]
            elif embedding_config[embedding]['embedding_type'] == 'point_transformer_conv':
                _embedding_dict[embedding] = PointTransformerModule(
                    in_channels=_input_dimension,
                    out_channels=embedding_config[embedding]['pos_nn_layers'][-1],
                    pos_nn=MLP(
                        [_input_dimension] + 
                        embedding_config[embedding]['pos_nn_layers']
                    ),
                    attn_nn=MLP(
                        [embedding_config[embedding]['pos_nn_layers'][-1]] + 
                        embedding_config[embedding]['attn_nn_layers']
                    ), 
                    fps_ratio=embedding_config[embedding]['fps_ratio'],
                    cluster_radius=embedding_config[embedding]['cluster_radius'],
                    max_number_neighbors=embedding_config[embedding]['max_number_neighbors'],
                    add_self_loops=embedding_config[embedding]['add_self_loops']
                )
                _input_dimension = embedding_config[embedding]['pos_nn_layers'][-1]
            else:
                self.logger.error(f'specified embedding type {embedding_config[embedding]["embedding_type"]} not allowed!')
            _num_embedding_outputs += _input_dimension


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

        if self.config["add_summed_adc"]:
            reduction_config['linear_output'] += 1

        # projection head
        if "projection_head" in reduction_config.keys():
            self.projection_head = True
            _reduction_dict['projection_head'] = MLP(
                [reduction_config['linear_output']] + reduction_config['projection_head']
            )
            reduction_output = reduction_config['projection_head'][-1]
        else:
            self.projection_head = False
            reduction_output = reduction_config['linear_output']

        # classification layer
        classifcation_config = self.config['classification']
        # add output mlp Projection head (See explanation in SimCLRv2)
        for ii, classification in enumerate(self.config["classifications"]):
            _classification_dict[f'{classification}'] = MLP(
                [reduction_output] + classifcation_config['mlp_output_layers'] + [classifcation_config['out_channels'][ii]]
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
            reductions, projection_head, classifications = [], [], [[] for kk in range(self.number_of_augmentations)]
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
                
                reductions.append(linear_pool)

                if self.projection_head:
                    linear_pool = self.reduction_dict['projection_head'](linear_pool)
                projection_head.append(linear_pool)

                # Pass through classification dictionary
                for jj, classification in enumerate(self.classification_dict.keys()):
                    classifications[jj].append(self.classification_dict[classification](linear_pool))

            outputs = {
                classification: torch.cat(classifications[jj])
                for jj, classification in enumerate(self.classification_dict.keys())
            }
            outputs['reductions'] = torch.cat(reductions)
            outputs['projection_head'] = torch.cat(projection_head)
            outputs['augmentations'] = self.number_of_augmentations
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
            
            reductions = linear_pool

            if self.projection_head:
                linear_pool = self.reduction_dict['projection_head'](linear_pool)

            outputs = {
                classifications: self.classification_dict[classifications](linear_pool)
                for classifications in self.classification_dict.keys()
            }
            outputs['reductions'] = reductions
            outputs['projection_head'] = linear_pool
            outputs['augmentations'] = 1
        return outputs