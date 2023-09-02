"""
Implementation of the chunc model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from blip.models.common import activations, normalizations
from blip.models import GenericModel

chunc_config = {
    # dimension of the input variables
    'input_dimension':      5,
    # encoder parameters
    'encoder_dimensions':   [10, 25, 50, 25, 10],
    'encoder_activation':   'leaky_relu',
    'encoder_activation_params':    {
        'negative_slope': 0.02
    },
    'encoder_normalization':'bias',
    # desired dimension of the latent space
    'latent_dimension':     5,
    # decoder parameters
    'decoder_dimensions':   [10, 25, 50, 25, 10],
    'decoder_activation':   'leaky_relu',
    'decoder_activation_params':    {
        'negative_slope': 0.02
    },
    'decoder_normalization':'bias',
    # output activation
    'output_activation':    'linear',
    'output_activation_params':     {},
}

class CHUNCNet(GenericModel):
    """
    
    """
    def __init__(self,
        name:   str='chuncnet',
        config: dict=chunc_config,
        meta:   dict={}
    ):
        super(CHUNCNet, self).__init__(
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
        self.logger.info(f"Attempting to build chunc architecture using cfg: {self.config}")
        _encoder_dict = OrderedDict()
        _latent_dict = OrderedDict()
        _decoder_dict = OrderedDict()
        _output_dict = OrderedDict()

        self.input_dimension = self.config['input_dimension']
        input_dimension = self.input_dimension
        # iterate over the encoder
        for ii, dimension in enumerate(self.config['encoder_dimensions']):
            if self.config['encoder_normalization'] == 'bias':
                _encoder_dict[f'encoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=True
                )
            else:
                _encoder_dict[f'encoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=False
                )
                _encoder_dict[f'encoder_{ii}_batchnorm'] = nn.BatchNorm1d(
                    num_features=dimension
                )
            _encoder_dict[f'encoder_{ii}_activation'] = activations[self.config['encoder_activation']](**self.config['encoder_activation_params'])
            input_dimension=dimension
            
        # create the latent space
        _latent_dict['latent_layer'] = nn.Linear(
            in_features=dimension,
            out_features=self.config['latent_dimension'],
            bias=False
        )
                
        input_dimension = self.config['latent_dimension']
        # iterate over the decoder
        for ii, dimension in enumerate(self.config['decoder_dimensions']):
            if self.config['decoder_normalization'] == 'bias':
                _decoder_dict[f'decoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=True
                )
            else:
                _decoder_dict[f'decoder_{ii}'] = nn.Linear(
                    in_features=input_dimension,
                    out_features=dimension,
                    bias=False
                )
                _decoder_dict[f'decoder_{ii}_batchnorm'] = nn.BatchNorm1d(
                    num_features=dimension
                )
            _decoder_dict[f'decoder_{ii}_activation'] = activations[self.config['decoder_activation']](**self.config['decoder_activation_params'])
            input_dimension=dimension
        # create the output
        _output_dict['output'] = nn.Linear(
            in_features=dimension,
            out_features=self.input_dimension,
            bias=False
        )
        if self.config['output_activation'] != 'linear':
            _output_dict['output_activation'] = activations[self.config['output_activation']](**self.config['output_activation_params'])
        # create the dictionaries
        self.encoder_dict = nn.ModuleDict(_encoder_dict)
        self.latent_dict = nn.ModuleDict(_latent_dict)
        self.decoder_dict = nn.ModuleDict(_decoder_dict)
        self.output_dict = nn.ModuleDict(_output_dict)
        # record the info
        self.logger.info(f"constructed chunc with dictionaries:\n{self.encoder_dict}\n{self.latent_dict}\n{self.decoder_dict}\n{self.output_dict}.")

    def forward(self,
        data
    ):
        """
        Iterate over the model dictionary
        """
        x = data['x'].to(self.device)
        # first the encoder
        for layer in self.encoder_dict.keys():
            x = self.encoder_dict[layer](x)
        x = self.latent_dict['latent_layer'](x)
        for layer in self.decoder_dict.keys():
            x = self.decoder_dict[layer](x)
        for layer in self.output_dict.keys():
            x = self.output_dict[layer](x)
        return {
            'decoder':  x, 
            'latent':   self.forward_views['latent_layer']
        }

    def sample(self,
        x
    ):
        """
        Returns an output given a input from the latent space
        """
        x = x.to(self.device)
        for layer in self.decoder_dict.keys():
            x = self.decoder_dict[layer](x)
        for layer in self.output_dict.keys():
            x = self.output_dict[layer](x)
        return {
            'decoder': x
        }

    def latent(self,
        data,
    ):
        """
        Get the latent representation of an input
        """
        x = data['x'].to(self.device)
        # first the encoder
        for layer in self.encoder_dict.keys():
            x = self.encoder_dict[layer](x)
        x = self.latent_dict['latent_layer'](x)
        return {
            'latent': x
        }