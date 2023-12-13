
"""
Generic model code.
"""
import torch
import os
import csv
import getpass
from torch import nn
import numpy as np
from time import time
from datetime import datetime
from collections import OrderedDict

from blip.models.generic_model import GenericModel

blip_segmentation_config = {
    "depth":    5,
    
}


class BlipSegmentation(GenericModel):
    """
    Wrapper of torch nn.Module that generates a BlipSegmentation
    """
    def __init__(
        self,
        name:   str = 'blip_segmentation',
        config: dict = blip_segmentation_config,
        meta:   dict = {}
    ):
        super(BlipSegmentation, self).__init__(
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
        self.logger.info(f"Attempting to build BlipSegmentation architecture using config: {self.config}")

        _encoder_dict = OrderedDict()
        _decoder_dict = OrderedDict()
        _bottleneck_dict = OrderedDict()






        self.encoder_dict = nn.ModuleDict(_encoder_dict)
        self.decoder_dict = nn.ModuleDict(_decoder_dict)
        self.bottleneck_dict = nn.ModuleDict(_bottleneck_dict)

        # record the info
        self.logger.info(
            "Constructed BlipSegmentation with dictionaries:"
        )

    def forward(
        self,
        data
    ):
        x = self.encoder_dict['layer'](x)
        
                
        
        
        