"""
Implementation of the LinearEvaluation model using pytorch
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

linear_evaluation_config = {

}

class LinearEvaluation(GenericModel):
    """
    """
    def __init__(self,
        name:   str='linear_evaluation',
        config: dict=linear_evaluation_config,
        meta:   dict={}
    ):
        super(LinearEvaluation, self).__init__(
            name, config, meta
        )
        self.config = config
         
        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()


    def construct_model(self):
        """
        This model takes in a trained contrastive learning model and appends
        to the embedding layer a single linear layer which is trained to separate
        the classes linearly.
        """
        self.logger.info(f"Attempting to build LinearEvaluation architecture using config: {self.config}")


        # record the info
        self.logger.info(
            f"Constructed LinearEvaluation with dictionaries:"
        )

    
    def forward(self,
        data
    ):
        outputs = []
        return outputs