"""
SparseUNet implementation using MinkowskiEngine
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import MinkowskiEngine as ME
from collections import OrderedDict
from datetime import datetime
import getpass
import sys
import os
import csv

from blip.models import GenericModel
from blip.models.common import Identity, sparse_activations

def get_activation(
    activation: str,
):
    if activation in sparse_activations.keys():
        return sparse_activations[activation]

class VariableConv(ME.MinkowskiNetwork):
    """
    """
    def __init__(self,
        name, 
        in_channels, 
        out_channels,
        kernel_size:    int=3,
        stride:         int=1,
        dilation:       int=1,
        activation:     str='relu',
        batch_norm:     bool=True,
        dimension:      int=3, 
        num_of_conv: int=2
    ):
        """
        """
        super(VariableConv, self).__init__(dimension)
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.dimension = dimension
        self.batch_norm = batch_norm
        self.num_of_conv = num_of_conv
        if self.batch_norm:
            self.bias = False
        else:
            self.bias = True
        self.activation = activation
        self.activation_fn = get_activation(self.activation)
        self.construct_model()

    def construct_model(self):
        """
        Create model dictionary
        """
        _dict = OrderedDict()
        
        # create conv layer
    
        for i in range(1, self.num_of_conv):
            _dict[f'{self.name}_conv[i]'] = ME.MinkowskiConvolution(
                in_channels  = self.in_channels,
                out_channels = self.out_channels,
                kernel_size  = self.kernel_size,
                stride       = self.stride,
                dilation     = self.dilation,
                bias         = self.bias,
                dimension    = self.dimension
            )
            if self.batch_norm:
                _dict[f'{self.name}_batch_norm_{i}'] = ME.MinkowskiBatchNorm(self.out_channels)
            _dict[f'{self.name}_{self.activation}_{i}'] = self.activation_fn
            
        # second conv layer
        #_dict[f'{self.name}_conv2'] = ME.MinkowskiConvolution(
        #    in_channels  = self.out_channels,
        #    out_channels = self.out_channels,
        #   kernel_size  = self.kernel_size,
        #    stride       = self.stride,
        #    dilation     = self.dilation,
        #    bias         = self.bias,
        #    dimension    = self.dimension
        #)
        #if self.batch_norm:
        #    _dict[f'{self.name}_batch_norm2'] = ME.MinkowskiBatchNorm(self.out_channels)
        #_dict[f'{self.name}_{self.activation}2'] = self.activation_fn
        #self.module_dict = nn.ModuleDict(_dict)

    def forward(self, 
        x
    ):
        """
        Iterate over the module dictionary.
        """
        for layer in self.module_dict.keys():
            x = self.module_dict[layer](x)
        return x

""" 
    Here are a set of standard UNet parameters, which must be 
    adjusted by the user for each application
"""
sparse_unet_params = {
    'in_channels':  1,
    'out_channels': 1,  # this is the number of classes for the semantic segmentation
    'filtrations':  [64, 128, 256, 512],    # the number of filters in each downsample
    'variable_conv_params': {
        'kernel':       3,
        'stride':       1,
        'dilation':     1,
        'activation':   'relu',
        'dimension':    3,
        'batch_norm':   True,
    },
    'conv_transpose_params': {
        'kernel':    2,
        'stride':    2,
        'dilation':  1,
        'dimension': 3,
    },
    'max_pooling_params': {
        'kernel':   2,
        'stride':   2,
        'dilation': 1,
        'dimension':3,
    }
}

class SparseUNet(GenericModel):
    """
    """
    def __init__(self,
        name:   str='my_unet',      # name of the model
        config: dict=sparse_unet_params,
        meta:   dict={}
    ):
        super(SparseUNet, self).__init__(name, config, meta)
        self.name = name
        self.config = config
        # check config
        self.logger.info(f"checking SparseUNet architecture using config: {self.config}")
        for item in sparse_unet_params.keys():
            if item not in self.config:
                self.logger.error(f"parameter {item} was not specified in config file {self.config}")
                raise AttributeError
        if ((self.config["variable_conv_params"]["dimension"] != 
             self.config["conv_transpose_params"]["dimension"]) or 
            (self.config["variable_conv_params"]["dimension"] !=
             self.config["max_pooling_params"]["dimension"])):
            self.logger.error(
                "dimensions for 'variable_conv_params', 'conv_transpose_params' and" +  
                f"'max_pooling_params' (with values {self.config['variable_conv_params']['dimension']}" +
                f", {self.config['conv_transpose_params']['dimension']} and " + 
                f"{self.config['max_pooling_params']['dimension']}) do not match!"
            )
            raise AttributeError

        # construct the model
        self.construct_model()
        self.save_model(flag='init')

    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        The Convolution Transpose in ME has the following constructor arguments:
            MinkowskiConvolutionTranspose(
            in_channels, 
            out_channels, 
            kernel_size=-1, 
            stride=1, 
            dilation=1, 
            bias=False, 
            kernel_generator=None, 
            expand_coordinates=False, 
            convolution_mode=<ConvolutionMode.DEFAULT: 0>, 
            dimension=None)
        The Convolution layer in ME has the following constructor arguments:
            MinkowskiConvolution(
            in_channels, 
            out_channels, 
            kernel_size=-1, 
            stride=1, 
            dilation=1, 
            bias=False, 
            kernel_generator=None, 
            expand_coordinates=False, 
            convolution_mode=<ConvolutionMode.DEFAULT: 0>, 
            dimension=None)
        The Max Pooling layer from ME has the following constructor arguments:
            MinkowskiMaxPooling(
            kernel_size, 
            stride=1, 
            dilation=1, 
            kernel_generator=None, 
            dimension=None)
        """
        self.logger.info(f"Attempting to build UNet architecture using config: {self.config}")
        _down_dict = OrderedDict()
        _up_dict = OrderedDict()
        _classification_dict = OrderedDict()

        # iterate over the down part
        in_channels = self.config['in_channels']
        for filter in self.config['filtrations']:
            _down_dict[f'down_filter_double_conv{filter}'] = VariableConv(
                name=f'down_{filter}',
                in_channels=in_channels,
                out_channels=filter,
                kernel_size=self.config['variable_conv_params']['kernel_size'],
                stride=self.config['variable_conv_params']['stride'],
                dilation=self.config['variable_conv_params']['dilation'],
                dimension=self.config['variable_conv_params']['dimension'],
                activation=self.config['variable_conv_params']['activation'],
                batch_norm=self.config['variable_conv_params']['batch_norm'],
            )
            # set new in channel to current filter size
            in_channels = filter

        # iterate over the up part
        for filter in reversed(self.config['filtrations']):
            _up_dict[f'up_filter_transpose{filter}'] = ME.MinkowskiConvolutionTranspose(
                in_channels=2*filter,   # adding the skip connection, so the input doubles
                out_channels=filter,
                kernel_size=self.config['conv_transpose_params']['kernel_size'],
                stride=self.config['conv_transpose_params']['stride'],
                dilation=self.config['conv_transpose_params']['dilation'],
                dimension=self.config['conv_transpose_params']['dimension']    
            )
            _up_dict[f'up_filter_double_conv{filter}'] = VariableConv(
                name=f'up_{filter}',
                in_channels=2*filter,
                out_channels=filter,
                kernel_size=self.config['variable_conv_params']['kernel_size'],
                stride=self.config['variable_conv_params']['stride'],
                dilation=self.config['variable_conv_params']['dilation'],
                dimension=self.config['variable_conv_params']['dimension'],
                activation=self.config['variable_conv_params']['activation'],
                batch_norm=self.config['variable_conv_params']['batch_norm'],
            )

        # create bottleneck layer
        self.bottleneck = VariableConv(
            name=f"bottleneck_{self.config['filtrations'][-1]}",
            in_channels=self.config['filtrations'][-1],
            out_channels=2*self.config['filtrations'][-1],
            kernel_size=self.config['variable_conv_params']['kernel_size'],
            stride=self.config['variable_conv_params']['stride'],
            dilation=self.config['variable_conv_params']['dilation'],
            dimension=self.config['variable_conv_params']['dimension'],
            activation=self.config['variable_conv_params']['activation'],
            batch_norm=self.config['variable_conv_params']['batch_norm'],
        )

        # create output layer
        for ii, classification in enumerate(self.config['classifications']):
            _classification_dict[f"{classification}"] = ME.MinkowskiConvolution(
                in_channels=self.config['filtrations'][0],      # to match first filtration
                out_channels=self.config['out_channels'][ii],   # to the number of classes
                kernel_size=1,                                  # a one-one convolution
                dimension=self.config['variable_conv_params']['dimension'],
            )

        # create the max pooling layer
        self.max_pooling = ME.MinkowskiMaxPooling(
            kernel_size=self.config['max_pooling_params']['kernel_size'],
            stride=self.config['max_pooling_params']['stride'],
            dilation=self.config['max_pooling_params']['dilation'],
            dimension=self.config['max_pooling_params']['dimension']
        )

        # create the dictionaries
        self.module_down_dict = nn.ModuleDict(_down_dict)
        self.module_up_dict = nn.ModuleDict(_up_dict)
        self.classification_dict = nn.ModuleDict(_classification_dict)
        # record the info
        self.logger.info(f"Constructed UNet with down: {self.module_down_dict} and up: {self.module_up_dict}.")
        self.logger.info(f"Bottleneck layer: {self.bottleneck}, output layer: {self.classification_dict} and max pooling: {self.max_pooling}.")

    def forward(self, 
        data
    ):
        """
        Convert input, which should be a tuple a Data 
        object to a ME.SparseTensor(feats, coords).
        Iterate over the module dictionary.
        """ 
        x = ME.SparseTensor(
            features=data.x, 
            coordinates=torch.cat(
                (data.batch.unsqueeze(1), data.pos),
                dim=1
            ).int(), 
            device=self.device
        )
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

            concat_skip = ME.cat(skip_connection, x)
            x = self.module_up_dict[f'up_filter_double_conv{filter}'](concat_skip)

        return {
            classifications: self.classification_dict[classifications](x).features
            for classifications in self.classification_dict.keys()
        }