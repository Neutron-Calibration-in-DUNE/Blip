
"""
Generic model code.
"""
import torch
import os
import csv
import getpass
from torch import nn
from time import time
from datetime import datetime
from collections import OrderedDict

import MinkowskiEngine as ME

from blip.models import GenericModel
from blip.models.common import Identity, sparse_activations

def get_activation(
    activation: str,
):
    if activation in sparse_activations.keys():
        return sparse_activations[activation]
    
point_proposal_config = {
    'proposal_type':                'fcos', # fixed, centernet, fcos
    'apply_non_max_suppression':    True,
    'dimension':            3,
    # shared convolution section
    'ppn_in_channels':      128, 
    'shared_convolutions':  [256, 512],
    'shared_kernel_size':   3,
    'shared_stride':        1,
    'shared_dilation':      1,
    'shared_activation':     'relu',
    'shared_batch_norm':    True,
    # classification section
    'classification_anchors':       10,
    'classification_size':          1,
    'classification_kernel_size':   1,
    # regression section
    'regression_anchors':       10,
    'regression_size':          7,
    'regression_kernel_size':   1,
}

class PointProposalNetwork(ME.MinkowskiNetwork):
    """
    """
    def __init__(self,
        name, 
        config: dict=point_proposal_config,
        meta:   dict={} 
    ):
        """
        """
        self.name = name
        self.config = config
        super(PointProposalNetwork, self).__init__(self.config['dimension'])
        self.proposal_type = self.config['proposal_type']
        self.apply_non_max_suppression = self.config['apply_non_max_suppression']
        self.in_channels = self.config['ppn_in_channels']
        self.out_channels = self.config['shared_convolutions'][-1]
        self.dimension = self.config['dimension']
        self.batch_norm = self.config['shared_batch_norm']
        if self.batch_norm:
            self.bias = False
        else:
            self.bias = True
        self.activation = self.config['shared_activation']
        self.activation_fn = get_activation(self.activation)
        self.construct_model()

    def construct_model(self):
        """
        Create model dictionary
        """
        if self.in_channels != self.out_channels:
            self.residual = ME.MinkowskiLinear(
                self.in_channels, self.out_channels, bias=self.bias
            )
        else:
            self.residual = Identity()
        
        _convolution_layers = OrderedDict()
        _regression_layers = OrderedDict()
        _classification_layers = OrderedDict()

        in_channels = self.config['ppn_in_channels']
        for ii, layer in enumerate(self.config['shared_convolutions']):
            _convolution_layers[f'ppn_convolution_{ii}'] = ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=self.config['shared_convolutions'][ii],
                kernel_size=self.config['shared_kernel_size'],
                stride=self.config['shared_stride'],
                dilation=self.config['shared_dilation'],
                bias=self.bias,
                dimension=self.dimension
            )
            if self.batch_norm:
                _convolution_layers[f'ppn_convolution_{ii}_batch_norm'] = ME.MinkowskiBatchNorm(
                    self.config['shared_convolutions'][ii]
                )
            in_channels = self.config['shared_convolutions'][ii]
        if self.proposal_type == 'fixed':
            self.regression_out_channels = self.config['regression_anchors'] * self.config['regression_size']
            self.classification_out_channels = self.config['classification_anchors'] * self.config['classification_size']
        else:
            self.regression_out_channels = 4
            self.classification_out_channels = self.config['classification_size']

        _regression_layers['ppn_regression'] = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=self.regression_out_channels,
            kernel_size=self.config['regression_kernel_size'],
            stride=1,
            dimension=self.dimension
        )
        _classification_layers['ppn_classification'] = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=self.classification_out_channels,
            kernel_size=self.config['classification_kernel_size'],
            stride=1,
            dimension=self.dimension
        )

        self.convolution_dict = nn.ModuleDict(_convolution_layers)
        self.regression_dict = nn.ModuleDict(_regression_layers)
        self.classification_dict = nn.ModuleDict(_classification_layers)

    def forward(self, 
        x
    ):
        """
        Iterate over the module dictionary.
        """
        identity = self.residual(x)
        for layer in self.convolution_dict.keys():
            x = self.convolution_dict[layer](x)
            x = self.activation_fn(x)
        x = x + identity
        regression = self.regression_dict['ppn_regression'](x).features
        classification = self.classification_dict['ppn_classification'](x).features
        return {
            'ppn_regression':       regression,
            'ppn_classification':   classification
        }
