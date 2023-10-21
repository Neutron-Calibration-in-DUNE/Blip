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
from blip.models.blip_graph import BlipGraph

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
        _classification_dict = OrderedDict()
        self.logger.info(f"Attempting to build LinearEvaluation architecture using config: {self.config}")

        self.blip_graph_model = self.config['model']
        checkpoint = torch.load(self.blip_graph_model)
        self.blip_graph_config = checkpoint['model_config']
        
        self.logger.info(f"Loading BlipGraph from {self.config['model']}.")
        if self.blip_graph_config["add_summed_adc"]:
            self.blip_graph_config['reduction']['linear_output'] -= 1
        self.blip_graph = BlipGraph(
            'blip_graph',
            self.blip_graph_config,
            self.meta
        )
        self.blip_graph.load_state_dict(checkpoint['model_state_dict'])
        
        reduction_config = self.blip_graph_config['reduction']
        classifcation_config = self.blip_graph_config['classification']
        # now attach the linear layer to the reductions layer of the BlipGraph.
        for ii, classification in enumerate(self.config["classifications"]):
            _classification_dict[f'{classification}'] = MLP(
                [reduction_config['linear_output']] + [classifcation_config['out_channels'][ii]]
            )
        self.classification_dict = nn.ModuleDict(_classification_dict)
        self.softmax = nn.Softmax(dim=1)

        # record the info
        self.logger.info(
            f"Constructed LinearEvaluation with dictionaries:"
        )
    
    def forward(self,
        data
    ):
        self.blip_graph.eval()
        blip_graph_outputs = self.blip_graph(data)
        for classifications in self.classification_dict.keys():
            blip_graph_outputs['classifications'] = self.classification_dict[classifications](blip_graph_outputs['reductions'])
        return blip_graph_outputs