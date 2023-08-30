
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

from blip.utils.logger import Logger

generic_config = {
    "no_params":    "no_values"
}

class BlipNetModule(nn.Module):
    """
    """
    def __init__(self,
        name:   str,
        config: dict={},
        mode:   str='',
        meta:   dict={}
    ):
        self.name = name + "_ml_module"
        super(BlipNetModule, self).__init__(
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
        self.logger.error(f'"parse_config" not implemented in Module!')
    
    def run_module(self):
        self.logger.info(f'running blip_net module')
        self.construct_merge_tree()
        self.evaluate_merge_tree_scores()
        self.run_mcts()

    def construct_merge_tree(self):
        pass

    def evaluate_merge_tree_scores(self):
        pass

    def run_mcts(self):
        pass
        