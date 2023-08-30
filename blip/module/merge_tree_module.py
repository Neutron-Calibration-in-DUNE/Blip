
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
from blip.topology.merge_tree import MergeTree

generic_config = {
    "no_params":    "no_values"
}

class MergeTreeModule(nn.Module):
    """
    """
    def __init__(self,
        name:   str,
        config: dict={},
        mode:   str='',
        meta:   dict={}
    ):
        self.name = name + "_ml_module"
        super(MergeTreeModule, self).__init__(
            self.name, config, mode, meta
        )

        self.merge_tree = None

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
        self.logger.info(f'setting up merge_tree')
        self.merge_tree = MergeTree()
    
    def run_module(self):
        self.logger.info(f'running merge_tree module.')
        """
        Set up progress bar.
        """
        if (progress_bar == True):
            inference_loop = tqdm(
                enumerate(inference_loader, 0), 
                total=len(list(inference_indices)), 
                leave=rewrite_bar,
                colour='magenta'
            )
        else:
            inference_loop = enumerate(inference_loader, 0)
        for ii, data in inference_loop:
            vietoris_rips, tree = self.merge_tree.create_merge_tree(
                data
            )        