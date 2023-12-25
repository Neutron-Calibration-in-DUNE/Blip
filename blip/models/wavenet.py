"""
Implementation of the WaveNet model using pytorch
"""
import torch
import copy
import torch.nn as nn
from collections import OrderedDict

from blip.models import GenericModel
from blip.models.causal_dilated_conv1d_model import CausalDilatedConv1D
from blip.models.residual_block_stack import ResidualBlockStack

wavenet_config = {
    "in_channels":  0,
    "out_channels": 0,
    "dimension":    1,
    "kernel_size":  0,
    "dilation":     0,
    "bias":         False,
    "stack_size":   0,
    "layer_size":   0
}


class Wavenet(GenericModel):
    """
    """
    def __init__(
        self,
        name:   str = 'wavenet',
        config: dict = wavenet_config,
        meta:   dict = {}
    ):
        super(Wavenet, self).__init__(
            name, config, meta
        )
        self.config = config

        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        self.logger.info(f"attempting to build Wavenet architecture using config: {self.config}")

        _model_dict = OrderedDict()
        causal_conv1d_dict = copy.deepcopy(self.config)
        causal_conv1d_dict['res_channels'] = self.config['in_channels']
        causal_conv1d_dict['skip_channels'] = self.config['out_channels']
        causal_conv1d_dict['dilation'] = 1
        _model_dict[f'{self.name}_causal'] = CausalDilatedConv1D(
            f'{self.name}_causal',
            causal_conv1d_dict,
        )
        residual_block_stack_dict = copy.deepcopy(self.config)
        residual_block_stack_dict['res_channels'] = self.config['in_channels']
        residual_block_stack_dict['skip_channels'] = self.config['out_channels']
        _model_dict[f'{self.name}_residual_block_stack'] = ResidualBlockStack(
            f'{self.name}_residual_block_stack',
            residual_block_stack_dict
        )
        _model_dict[f'{self.name}_dense_relu'] = nn.Relu()
        _model_dict[f'{self.name}_conv1d'] = nn.Conv1d(
            self.config['in_channels'],
            self.config['in_channels'],
            kernel_size=(1, 1),
            bias=False
        )
        _model_dict[f'{self.name}_dense_softmax'] = nn.SoftMax(dim=1)
        self.model_dict = nn.ModuleDict(_model_dict)

    def forward(
        self,
        data
    ):
        """
        Iterate over the model dictionary
        """
        outputs = {}
        x = self.model_dict[f'{self.name}_causal'](x)
        residual_output, skip_connections = self.model_dict[f'{self.name}_residual_block_stack'](x)
        dense_output = torch.sum(skip_connections, dim=2)
        dense_output = self.model_dict[f'{self.name}_dense_relu'](dense_output)
        dense_output = self.model_dict[f'{self.name}_conv1d'](dense_output)
        dense_output = self.model_dict[f'{self.name}_dense_relu'](dense_output)
        dense_output = self.model_dict[f'{self.name}_conv1d'](dense_output)
        softmax_output = self.model_dict[f'{self.name}_dense_softmax'](dense_output)
        outputs['output'] = softmax_output
        return outputs
