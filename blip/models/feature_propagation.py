
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

from blip.utils.logger import Logger
from blip.models import GenericModel

generic_config = {
    "no_params":    "no_values"
}

class FeaturePropagation(GenericModel):
    """
    Wrapper of torch nn.Module that generates a FeaturePropagation
    """
    def __init__(self,
        name:   str,
        config: dict=generic_config,
        meta:   dict={}
    ):
        super(FeaturePropagation, self).__init__(name, config, meta)
                    
    def construct_model(self):
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.

        """
        self.logger.info(f"Attempting to build FeaturePropagation architecture using config: {self.config}")

        _model_dict = OrderedDict()
        self.model_dict = nn.ModuleDict(_model_dict)

        # record the info
        self.logger.info(
            f"Constructed FeaturePropagation with dictionaries:"
        )

    def forward(self, x):
        pass

