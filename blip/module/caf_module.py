
"""
Module for constructing CAFs (Common Analysis Files) from
Blip Datasets.
"""
import torch
import os
import csv
import getpass
from torch import nn
from time import time
from datetime import datetime

from blip.utils.logger import Logger

generic_config = {
    "no_params":    "no_values"
}

class CAFModule(nn.Module):
    """
    """
    def __init__(self,
        name:   str,
        config: dict={},
        mode:   str='',
        meta:   dict={}
    ):
        self.name = name + "_ml_module"
        super(CAFModule, self).__init__(
            self.name, config, mode, meta
        )

    def set_device(self,
        device
    ):
        self.device = device
    
    def set_config(self,
        config_file:    str
    ):
        self.config_file = config_file
        self.parse_config()
    
    def parse_config(self):
        pass
    
    def run_module(self):
        pass
        