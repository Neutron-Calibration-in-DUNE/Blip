"""

"""
import numpy    as np
import networkx as nx
from torch  import nn
from ripser import ripser
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

import sys; sys.path.insert(0, '..')
from blip.module.merge_tree_module import *


class MergeTree:
    """
    """
    def __init__(self,
                # name:   str,
                config: dict={},
                mode:   str ='',
                meta:   dict={},
                tree          = None,
                height        = None,
                pointCloud    = None,
                node_distance = 0,
                simplify      = False
                ):
       
        print("START: Merge Tree Class")
        # self.name = name + "_ml_module"
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


    def create_merge_tree(self, data):
        pass