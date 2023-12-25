"""
Implementation of the WaveNet model using pytorch
"""
import torch
import copy
import torch.nn as nn
from collections import OrderedDict
import MinkowskiEngine as ME

from blip.models import GenericModel
from blip.models.residual_block import ResidualBlock

residual_block_stack_config = {
    "res_channels": 0,
    "skip_channels": 0,
    "dimension":    1,
    "kernel_size":  0,
    "dilation":     0,
    "bias":         False,
    "stack_size":   0,
    "layer_size":   0
}


class ResidualBlockStack(ME.MinkowskiNetwork, GenericModel):
    """
    """
    def __init__(
        self,
        name:   str = 'residual_block_stack',
        config: dict = residual_block_stack_config,
    ):
        super(ResidualBlockStack, self).__init__(self.config["dimension"])
        self.name = name
        self.config = config

        # construct the model
        self.construct_model()

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        self.logger.info(f"Attempting to build ResidualBlockStack architecture using config: {self.config}")

        _model_dict = OrderedDict()

        # generate the collection of dilations
        self.dilations_for_all_stacks = []
        for stack in range(self.config['stack_size']):
            dilations = []
            for layer in range(self.config['layer_size']):
                dilations.append(2**layer)
            self.dilations_for_all_stacks.append(dilations)
        for ii, dilation_per_stack in enumerate(self.dilations_for_all_stacks):
            for jj, dilation in enumerate(dilation_per_stack):
                residual_dict = copy.deepcopy(self.config)
                residual_dict['dilation'] = dilation
                _model_dict[f'{self.name}_stack_{ii}_{dilation}'] = ResidualBlock(
                    f'{self.name}_stack_{ii}_{dilation}',
                    residual_dict
                )

        self.model_dict = nn.ModuleDict(_model_dict)

    def forward(
        self,
        x
    ):
        """
        Iterate over the model dictionary
        """
        skip_outputs = []
        residual_output = x
        for ii, dilation_per_stack in enumerate(self.dilations_for_all_stacks):
            for jj, dilation in enumerate(dilation_per_stack):
                residual_output, skip_output = self.model_dict[f'{self.name}_stack_{ii}_{dilation}'](residual_output)
                skip_outputs.append(skip_output)
        return residual_output, torch.stack(skip_outputs)
