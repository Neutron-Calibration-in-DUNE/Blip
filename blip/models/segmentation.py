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

segmentation_config = {
    
}

class Segmentation(GenericModel):
    """
    """
    def __init__(self,
        name:   str='segmentation',
        config: dict=segmentation_config
    ):
        super(Segmentation, self).__init__(name, config)
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
        self.logger.info(f"Attempting to build {self.name} architecture using config: {self.config}")
        """
        """
        _segmentation_dict = OrderedDict()
        input_dimension = self.config['num_inputs']
        _segmentation_dict[f"mlp_layer"] = MLP(
            [input_dimension] + self.config["mlp_layers"] + [self.config["classifications"]]
        )

        self.segmentation_dict = nn.ModuleDict(_segmentation_dict)
        
        # create the dictionaries
        # record the info
        self.logger.info(
            f"Constructed Segmentation layer with dictionaries:"
            #+ f"\n{self.embedding_dict}\n{self.reduction_dict}"
        )

    
    def forward(self,
        positions,
        embedding,
    ):
        """
        Iterate over the model dictionary
        """
        inputs = torch.cat([positions, embedding], dim=1)
        return self.segmentation_dict["mlp_layer"](inputs)
        