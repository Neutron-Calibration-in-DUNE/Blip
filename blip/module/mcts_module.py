
"""
Monte Carlo Tree Search Module code.
"""
import torch
import os
import csv
import getpass
from torch import nn
from time import time
from datetime import datetime

from blip.module import GenericModule

generic_config = {
    "no_params":    "no_values"
}


class MCTSModule(GenericModule):
    """
    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        mode:   str = '',
        meta:   dict = {}
    ):
        self.name = name + "_ml_module"
        super(MCTSModule, self).__init__(
            self.name, config, mode, meta
        )

    def set_device(
        self,
        device
    ):
        self.device = device

    def set_config(
        self,
        config_file:    str
    ):
        self.config_file = config_file
        self.parse_config()

    def parse_config(self):
        self.logger.error('"parse_config" not implemented in Module!')

    def run_module(self):
        self.logger.error('"run_module" not implemented in Module!')
