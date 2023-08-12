"""

"""
import numpy as np
from torch import nn
from ripser import ripser
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

class MergeTree:
    """
    """
    def __init__(self):
        pass

    def create_merge_tree(self,
        data
    ):
        pass

    def linkage_to_merge_tree(L,X):

        nodeIDs = np.unique(L[:,:2]).astype(int)
        num_leaves = X.shape[0]

        edge_list = []
        height = dict()

        for j in range(num_leaves):
            height[j] = 0

        for j in range(L.shape[0]):
            edge_list.append((int(L[j,0]),num_leaves+j))
            edge_list.append((int(L[j,1]),num_leaves+j))
            height[num_leaves+ j] = L[j,2]

        T = nx.Graph()
        T.add_nodes_from(nodeIDs)
        T.add_edges_from(edge_list)

        return T, height