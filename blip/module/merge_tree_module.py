
"""
Generic model code.
"""
import torch
import getpass
from torch    import nn
from time     import time
from datetime import datetime

import os,csv,random,copy,ot
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

from blip.utils.logger        import Logger
from blip.topology.merge_tree import MergeTree

generic_config = {
    "no_params":    "no_values"
}

class MergeTreeModule(nn.Module):
    """
    Creates a merge tree from a pointcloud input:
    - pointCloud: a Euclidean point cloud of shape (num points) x (dimension). The merge tree
                    is generated from the Vietoris-Rips filtration of the point cloud
    Merge trees can be 'decorated' with higher-dimensional homological data by the 'fit_barcode' method.
    The result is a 'decorated merge tree'.
    """
    def __init__(self,
                name:   str,
                config: dict={},
                mode:   str='',
                meta:   dict={},
                tree          = None,
                height        = None,
                pointCloud    = None,
                node_distance = 0,
                simplify      = False
                ):
       
        print("START: Merge Tree Class")
        self.name = name + "_ml_module"
        self.T              = tree
        self.pointCloud     = pointCloud
        self.node_distance  = node_distance
        self.leaf_barcode   = None
        self.ultramatrix    = None
        self.label          = None
        self.inverted_label = None

        L = linkage(pointCloud)
        print("HERE",L)
        T, height = linkage_to_merge_tree(L,pointCloud)
        print("T",T)
        print("height",height)

        self.tree   = T
        self.height = height

        ### THIS IS NOT WORKING NOW simplify=False by default ###
        if simplify:
            print("entering here")
            TNew, heightNew = simplify_merge_tree(self.tree,self.height)
            self.tree   = TNew
            self.height = heightNew


        super(MergeTreeModule, self).__init__(
            self.name, config, mode, meta, self.tree, self.height, self.pointCloud, self.node_distance, simplify
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

    """
    Creating a Decorated Merge Tree
    """
    def fit_barcode(self,
                    degree = 1,
                    leaf_barcode = None):

        if leaf_barcode is not None: self.leaf_barcode = leaf_barcode

        else:
            if self.T          is not None: raise Exception('fit_barcode for directly defined merge tree requires leaf_barcode input')
            if self.pointCloud is not None:
                dgm = ripser(self.pointCloud,maxdim = degree)['dgms'][-1]
                leaf_barcode_init = decorate_merge_tree(self.tree, self.height, self.pointCloud, dgm)
                leaf_barcode = {key: [bar for bar in leaf_barcode_init[key] if bar[1]-bar[0] > 0] for key in leaf_barcode_init.keys()}
                self.barcode = dgm
                self.leaf_barcode = leaf_barcode

    """
    Getting the ultramatrix from a labeling of the merge tree
    """
    def fit_ultramatrix(self,label = None):

        if label is None: label = {n:j for (j,n) in enumerate(self.tree.nodes())}
        ultramatrix, inverted_label = get_ultramatrix(self.tree,self.height,label)

        self.ultramatrix = ultramatrix
        self.label = label
        self.inverted_label = inverted_label

    """
    Merge tree manipulation
    """
    def threshold(self,threshold):

        if self.leaf_barcode is None:
            T_thresh, height_thresh = threshold_merge_tree(self.tree,self.height,threshold)
            self.tree           = T_thresh
            self.height         = height_thresh
            self.ultramatrix    = None
            self.label          = None
            self.inverted_label = None

        else:
            T_thresh, height_thresh, leaf_barcode_thresh = simplify_decorated_merge_tree(self.tree,self.height,self.leaf_barcode,threshold)
            self.tree           = T_thresh
            self.height         = height_thresh
            self.leaf_barcode   = leaf_barcode_thresh
            self.ultramatrix    = None
            self.label          = None
            self.inverted_label = None

    def subdivide(self,subdiv_heights):

        T_sub      = self.tree
        height_sub = self.height

        for h in subdiv_heights: T_sub, height_sub = subdivide_edges_single_height(T_sub,height_sub,h)

        self.tree   = T_sub
        self.height = height_sub

    def copy(self): return copy.copy(self)

    
    """
    Visualization Tools
    """
    # For general merge trees
    def draw(self, axes = False): draw_merge_tree(self.tree,self.height,axes = axes)

    def draw_with_labels(self,label): draw_labeled_merge_tree(self.tree,self.height,label)


    def draw_decorated(self,tree_thresh,barcode_thresh):
        if self.pointCloud is not None:
            _, _, _, _, _ = visualize_DMT_pointcloud(self.tree,
                                                     self.height,
                                                     self.barcode,
                                                     self.pointCloud,
                                                     tree_thresh,
                                                     barcode_thresh)
                