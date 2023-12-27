"""
Implementation of the CausalDilatedConv1D model using pytorch
"""
import torch.nn as nn
from collections import OrderedDict
import MinkowskiEngine as ME

from blip.models import GenericModel

causal_dilated_conv1d_config = {
    "in_channels":  0,
    "out_channels": 0,
    "dimension":    1,
    "kernel_size":  0,
    "dilation":     0,
    "bias":         False
}


class CausalDilatedConv1D(ME.MinkowskiNetwork, GenericModel):
    """
    """
    def __init__(
        self,
        name:   str = 'causal_dilated_conv1d',
        config: dict = causal_dilated_conv1d_config,
    ):
        super(CausalDilatedConv1D, self).__init__(config['dimension'])
        self.name = name
        self.config = config

        # construct the model
        self.forward_views = {}
        self.forward_view_map = {}

        # construct the model
        self.construct_model()

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        self.logger.info(f"Attempting to build CausalDilatedConv1D architecture using config: {self.config}")

        self.ignore_out_index = (self.config['kernel_size'] - 1) * self.config['dilation']

        _model_dict = OrderedDict()
        _model_dict[f'{self.name}_conv1d'] = nn.Conv1d(
            in_channels=self.config['in_channels'],
            out_channels=self.config['out_channels'],
            kernel_size=self.config['kernel_size'],
            dilation=self.config['dilation'],
            bias=self.config['bias']
        )
        self.model_dict = nn.ModuleDict(_model_dict)

    def forward(
        self,
        x
    ):
        """
        Iterate over the model dictionary
        """
        return self.model_dict[f'{self.name}_conv1d'](x)[..., :-self.ignore_out_index]
