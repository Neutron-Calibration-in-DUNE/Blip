"""
Implementation of the LinearEvaluation model using pytorch
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from torch_geometric.nn import MLP

from blip.models import GenericModel
from blip.models.blip_graph import BlipGraph

linear_evaluation_config = {

}


class LinearEvaluation(GenericModel):
    """
    """
    def __init__(
        self,
        name:   str = 'linear_evaluation',
        config: dict = linear_evaluation_config,
        meta:   dict = {}
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
        if isinstance(self.blip_graph_model, str):
            try:
                checkpoint = torch.load(self.blip_graph_model)
            except Exception as exception:
                self.logger.error(
                    f"failed to load model from ckpt: {self.blip_graph_model}! Does this file exist?" +
                    f" exception: {exception}"
                )
            try:
                self.blip_graph_config = checkpoint['model_config']
            except Exception as exception:
                self.logger.error(
                    f"could not load 'model_config' from checkpoint of {self.blip_graph_model}!" +
                    f" exception: {exception}"
                )

            self.logger.info(f"loading BlipGraph from {self.config['model']}.")
            if self.blip_graph_config["add_summed_adc"]:
                self.blip_graph_config['reduction']['linear_output'] -= 1

            self.blip_graph = BlipGraph(
                'blip_graph',
                self.blip_graph_config,
                self.meta
            )
            self.blip_graph.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.logger.info("setting BlipGraph from config.")
            self.blip_graph = self.config["model"]
            self.config["model"] = ''
            self.blip_graph_config = self.blip_graph.config

        reduction_config = self.blip_graph_config['reduction']
        classification_config = self.blip_graph_config['classification']
        # now attach the linear layer to the reductions layer of the BlipGraph.
        for ii, classification in enumerate(self.blip_graph_config["classifications"]):
            _classification_dict[f'{classification}'] = MLP(
                [reduction_config['linear_output']] + [classification_config['out_channels'][ii]],
                act='LeakyReLU',
            )
        self.classification_dict = nn.ModuleDict(_classification_dict)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        data
    ):
        self.blip_graph.eval()
        blip_graph_outputs = self.blip_graph(data)
        for classifications in self.classification_dict.keys():
            blip_graph_outputs['classifications'] = self.classification_dict[classifications](blip_graph_outputs['reductions'])
        return blip_graph_outputs
