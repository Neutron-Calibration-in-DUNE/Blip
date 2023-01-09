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
        []
    ],
    'number_of_neighbors':  20,
    'aggregation_operators': [
        'max', 'max'
    ],
    # linear layer
    'linear_output':    128,
    'mlp_output_layers': [128, 256, 32],
    'augmentations':    [
        T.RandomJitter(0.03), T.RandomFlip(1), T.RandomShear(0.2)
    ],
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
        _model_dict = OrderedDict()

        self.input_dimension = self.cfg['input_dimension']
        # Feature extraction
        # an example would be
        # self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64]), k, aggr)
        # self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        for ii in range(self.cfg['num_dynamic_edge_conv']):
            _model_dict[f'edge_conv_{ii}'] = DynamicEdgeConv(
                MLP([]), 
                self.cfg['number_of_neighbors'],
                self.cfg['aggregation_operators'][ii]
            )
        _num_edge_conv_outputs = sum(self.cfg['edge_conv_mlp_layers'][:,-1])

        # add linear layer Encoder head
        _model_dict[f'linear_layer'] = Linear(
            _num_edge_conv_outputs, 
            self.cfg['linear_output']
        )

        # add output mlp Projection head (See explanation in SimCLRv2)
        _model_dict[f'mlp_output'] = MLP(
            self.cfg['mlp_output_layers'],
            norm=None
        )
        
        # create the dictionaries
        self.module_dict = nn.ModuleDict(_model_dict)
        # record the info
        self.logger.info(f"Constructed BLIP with dictionary: {self.module_dict}.")

    
    def forward(self,
        x
    ):
        """
        Iterate over the model dictionary
        """
        x = x[0].to(self.device)

        if train:
            # Get 2 augmentations of the batch
            print(x.pos.size())
            augm_1 = self.augmentation(x)
            augm_2 = self.augmentation(x)

            # Extract properties
            pos_1, batch_1 = augm_1.pos, augm_1.batch
            pos_2, batch_2 = augm_2.pos, augm_2.batch

            # Get representations for first augmented view
            x1 = self.conv1(pos_1, batch_1)
            x2 = self.conv2(x1, batch_1)
            h_points_1 = self.lin1(torch.cat([x1, x2], dim=1))

            # Get representations for second augmented view
            x1 = self.conv1(pos_2, batch_2)
            x2 = self.conv2(x1, batch_2)
            h_points_2 = self.lin1(torch.cat([x1, x2], dim=1))
            
            # Global representation
            h_1 = global_max_pool(h_points_1, batch_1)
            h_2 = global_max_pool(h_points_2, batch_2)
        else:
            x1 = self.conv1(x.pos, x.batch)
            x2 = self.conv2(x1, x.batch)
            h_points = self.lin1(torch.cat([x1, x2], dim=1))
            return global_max_pool(h_points, x.batch)

        # Transformation for loss function
        compact_h_1 = self.mlp(h_1)
        compact_h_2 = self.mlp(h_2)

        return h_1, h_2, compact_h_1, compact_h_2
    