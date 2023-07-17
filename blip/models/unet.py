
 

"""
Implementation of UNet
"""
import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F
from time import time
from collections import OrderedDict

from blip.utils.logger import Logger
from blip.models.common import activations
from blip.models import GenericModel

class DoubleConv2DLayer(GenericModel):
    def __init__(self,
        name,           # a unique name identifier for this layer
        in_channels,    # number of input channels
        out_channels,   # number of output channels
        kernel_size:    int=3,    # size of the kernel
        stride:         int=1,    # stride
        padding:        int=1,    # padding
        activation:     str='relu', # choice of activation function
        batch_norm:     bool=True, # whether to use batch_norm
    ):
        super(DoubleConv2DLayer, self).__init__(
            name,
            {}
        )
        self.logger = Logger(name, file_mode="w")
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bias = False
        else:
            self.bias = True
        self.activation = activation
        if activation not in activations.keys():
            self.logger.error(f"specified activation '{self.config['activation']} not in allowed list ({activations.keys()})!")
        self.logger.info(f"Creating DoubleConv2DLayer {self.name} with in_channels: {self.in_channels} and out_channels: {self.out_channels}, activation: {activations[self.activation]} and batch_norm: {self.batch_norm}")
        self.logger.info(f"DoubleConvLayer {self.name} has kernel_size: {self.kernel_size}, stride: {self.stride} and padding: {self.padding}")
        # check that parameters are consistent
        self.construct_model()

    def construct_model(self):
        """
        Create the model dictionary
        """
        _dict = OrderedDict()
        # first conv layer
        _dict[f'{self.name}_conv1'] = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=self.bias
        )
        if self.batch_norm:
            _dict[f'{self.name}_batch_norm1'] = nn.BatchNorm2d(self.out_channels)
        _dict[f'{self.name}_{self.activation}1'] = activations[self.activation]
        # second conv layer
        _dict[f'{self.name}_conv2'] = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            bias=self.bias
        )
        if self.batch_norm:
            _dict[f'{self.name}_batch_norm2'] = nn.BatchNorm2d(self.out_channels)
        _dict[f'{self.name}_{self.activation}2'] = activations[self.activation]
        # construct the model
        self.module_dict = nn.ModuleDict(_dict)
        self.logger.info(f"Constructed DoubleConv2DLayer: {self.module_dict}.")

    def forward(self, 
        x
    ):
        """
        Iterate over the module dictionary.
        """
        for layer in self.module_dict.keys():
            x = self.module_dict[layer](x)
        return x

# Here are a set of standard UNet parameters, which must be 
# adjusted by the user for each application
unet_config = {
    'in_channels':  3,
    'out_channels': 1,  # this is the number of classes for the SS
    'filtrations':  [64, 128, 256, 512],    # the number of filters in each downsample
    # standard double_conv parameters
    'double_conv_kernel':   3,
    'double_conv_stride':   1,
    'double_conv_padding':  1,
    'double_conv_activation':   'relu_inplace',
    'double_conv_batch_norm':   True,
    # conv transpose layers
    'conv_transpose_kernel':    2,
    'conv_transpose_stride':    2,
    'conv_transpose_padding':   0,
    # max pooling layer
    'max_pooling_kernel':   2,
    'max_pooling_stride':   2,
}

class UNet(GenericModel):
    """
    """
    def __init__(self,
        name:   str,
        config: dict=unet_config,
        meta:   dict={}    # configuration parameters
    ):
        super(UNet, self).__init__(name, config, meta)
        self.name = name
        self.logger = Logger(self.name, file_mode="w")
        self.config = config
        # check config
        self.logger.info(f"checking UNet architecture using config: {self.config}")
        for item in unet_config.keys():
            if item not in self.config:
                self.logger.error(f"parameter {item} was not specified in config file {self.config}")
                raise AttributeError
        
        # construct the model
        self.construct_model()

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        self.logger.info(f"Attempting to build UNet architecture using config: {self.config}")
        _down_dict = OrderedDict()
        _up_dict = OrderedDict()
        # iterate over the down part
        in_channels = self.config['in_channels']
        for filter in self.config['filtrations']:
            _down_dict[f'down_filter_double_conv{filter}'] = DoubleConv2DLayer(
                name=f'down_{filter}',
                in_channels=in_channels,
                out_channels=filter,
                kernel_size=self.config['double_conv_kernel'],
                stride=self.config['double_conv_stride'],
                padding=self.config['double_conv_padding'],
                activation=self.config['double_conv_activation'],
                batch_norm=self.config['double_conv_batch_norm'],
            )
            # set new in channel to current filter size
            in_channels = filter
        # iterate over the up part
        for filter in reversed(self.config['filtrations']):
            _up_dict[f'up_filter_transpose{filter}'] = nn.ConvTranspose2d(
                in_channels=2*filter,   # adding the skip connection, so the input doubles
                out_channels=filter,
                kernel_size=self.config['conv_transpose_kernel'],
                stride=self.config['conv_transpose_stride'],
                padding=self.config['conv_transpose_padding']    
            )
            _up_dict[f'up_filter_double_conv{filter}'] = DoubleConv2DLayer(
                name=f'up_{filter}',
                in_channels=2*filter,
                out_channels=filter,
                kernel_size=self.config['double_conv_kernel'],
                stride=self.config['double_conv_stride'],
                padding=self.config['double_conv_padding'],
                activation=self.config['double_conv_activation'],
                batch_norm=self.config['double_conv_batch_norm'],
            )
        # create bottleneck layer
        self.bottleneck = DoubleConv2DLayer(
            name=f"bottleneck_{self.config['filtrations'][-1]}",
            in_channels=self.config['filtrations'][-1],
            out_channels=2*self.config['filtrations'][-1],
            kernel_size=self.config['double_conv_kernel'],
            stride=self.config['double_conv_stride'],
            padding=self.config['double_conv_padding'],
            activation=self.config['double_conv_activation'],
            batch_norm=self.config['double_conv_batch_norm'],
        )
        # create output layer
        self.output = nn.Conv2d(
            self.config['filtrations'][0],    # to match first filtration
            self.config['out_channels'],      # to the number of classes
            kernel_size=1                  # a one-one convolution
        )
        # create the max pooling layer
        self.max_pooling = nn.MaxPool2d(
            kernel_size=self.config['max_pooling_kernel'],
            stride=self.config['max_pooling_stride']
        )
        # create the dictionaries
        self.module_down_dict = nn.ModuleDict(_down_dict)
        self.module_up_dict = nn.ModuleDict(_up_dict)
        # record the info
        self.logger.info(f"Constructed UNet with down: {self.module_down_dict} and up: {self.module_up_dict}.")
        self.logger.info(f"Bottleneck layer: {self.bottleneck}, output layer: {self.output} and max pooling: {self.max_pooling}.")

    def forward(self, 
        x
    ):
        """
        Iterate over the module dictionary.
        """
        x = x[0].to(self.device)
        # record the skip connections
        skip_connections = {}
        # iterate over the down part
        for filter in self.config['filtrations']:
            x = self.module_down_dict[f'down_filter_double_conv{filter}'](x)
            skip_connections[f'{filter}'] = x
            x = self.max_pooling(x)
        # through the bottleneck layer
        x = self.bottleneck(x)
        for filter in reversed(self.config['filtrations']):
            x = self.module_up_dict[f'up_filter_transpose{filter}'](x)
            # concatenate the skip connections
            skip_connection = skip_connections[f'{filter}']
            # check for compatibility
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.module_up_dict[f'up_filter_double_conv{filter}'](concat_skip)
        
        return self.output(x)