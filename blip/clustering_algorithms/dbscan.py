"""
Wrapper for dbscan
"""
import torch
import torch.nn as nn
import sklearn.cluster

from blip.clustering_algorithms import GenericClusteringAlgorithm

class DBSCAN(GenericClusteringAlgorithm):
    """
    """
    def __init__(self,
        name:   str='dbscan',
        eps:    float=1.0,
        min_samples:    int=10,
        device:         str='cpu'
    ):
        super(DBSCAN, self).__init__(name, device)
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = sklearn.cluster.DBSCAN(
            eps=self.eps, 
            min_samples=self.min_samples
        )

    def cluster(self,
        data,
    ):
        labels = self.dbscan.fit(data.pos.to(self.device)).labels_
        return labels