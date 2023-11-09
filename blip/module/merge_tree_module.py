
"""
Generic model code.
"""
import torch
import getpass
from torch    import nn
from time     import time
from datetime import datetime

import os,csv,random,copy,ot,persim,warnings
import numpy             as np
import networkx          as nx
import gudhi             as gd
import matplotlib.pyplot as plt
from ripser                   import ripser
from scipy.cluster.hierarchy  import dendrogram, linkage, cut_tree
from scipy.optimize           import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances
from bisect                   import bisect_left
from hopcroftkarp             import HopcroftKarp

from blip.utils.logger          import Logger
from blip.utils.utils           import print_colored
from blip.module.generic_module import GenericModule
warnings.filterwarnings("ignore")


generic_config = { "no_params":    "no_values" }



class MergeTreeModule(GenericModule):
    """
    Creates a merge tree from a pointcloud input:
    - pointCloud: a Euclidean point cloud of shape (num points) x (dimension). The merge tree
                    is generated from the Vietoris-Rips filtration of the point cloud
    Merge trees can be 'decorated' with higher-dimensional homological data by the 'fit_barcode' method.
    The result is a 'decorated merge tree'.
    """

    def set_device(self, device):
        self.device = device
    
    def set_config(self, config_file: str):
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
                data)