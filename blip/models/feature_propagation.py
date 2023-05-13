"""
Implementation of the blip model using pytorch
"""
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch_geometric.transforms as T
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
from sklearn.cluster import DBSCAN
import time


from blip.models.common import activations, normalizations
from blip.models import GenericModel

feature_propagation_config = {
    'in_channels':  3,
    'mlp':          [256, 256]
}

class FeaturePropagation(GenericModel):
    """
    """
    def __init__(self,
        name:   str='feature_propagation',
        config: dict=feature_propagation_config,
        device: str='cpu'
    ):
        super(FeaturePropagation, self).__init__(name, config, device)
        self.config = config

        # construct the model
        self.forward_views      = {}
        self.forward_view_map   = {}
        # construct the model
        self.construct_model()
        # register hooks
        self.register_forward_hooks()

    def construct_model(self):
        """
        
        """
        """
        The current methodology is to create an ordered
        dictionary and fill it with individual modules.
        """
        self.logger.info(f"Attempting to build {self.name} architecture using config: {self.config}")
        _convolution_dict = OrderedDict()
        _mlp_dict = OrderedDict()
        _mlp_dict['mlp'] = MLP(
            self.config['embedding_mlp_layers']
        )
        self.convolution_dict = nn.ModuleDict(_convolution_dict)
        self.mlp_dict = nn.ModuleDict(_mlp_dict)
        # record the info
        self.logger.info(
            f"Constructed FeaturePropagation with dictionary:"
            + f"\n{self.convolution_dict}"
        )
    
    def forward(self,
        positions,
        batches,
        indices,
        embedding,
        prev_indices,
        prev_embedding
    ):     
        """
        Iterate over the model dictionary
        """
        start = time.time()
        embeddings = [[] for ii in range(len(prev_indices))]
        for batch in torch.unique(batches):
            s = time.time()
            current_positions = positions[(batches == batch)][indices]
            if embedding is not None:
                current_positions = torch.cat((current_positions, embedding), dim=1)
            prev_positions = positions[(batches == batch)][prev_indices]
            if prev_embedding is not None:
                prev_positions = torch.cat((prev_positions, prev_embedding), dim=1)
            e = time.time()
            print(f"positions: {e - s}")
            s = time.time()
            clustering = DBSCAN(eps=self.config["coarse_graining_eps"], min_samples=2).fit(prev_positions.detach().cpu())
            e = time.time()
            print(f"clustering: {e - s}")
            labels = torch.tensor(clustering.labels_)
            unique_labels = torch.unique(labels)
            s = time.time()
            for label in unique_labels:   
                new_indices = torch.argwhere(labels == label)
                if label == -1:
                    for jj, index in enumerate(new_indices):
                        old_index = torch.argwhere(prev_indices[index] == indices).squeeze(0)
                        embeddings[index].append(self.mlp_dict['mlp'](
                            torch.cat((prev_positions[index], current_positions[old_index]), dim=1)
                        ))
                else:
                    old_index = torch.argwhere(
                        (indices.view(1, -1) == prev_indices[new_indices].view(-1, 1)).any(dim=0)
                    ).squeeze(0)
                    for jj, index in enumerate(new_indices):
                        embeddings[index].append(self.mlp_dict[f'mlp'](
                            torch.cat((prev_positions[index], current_positions[old_index]), dim=1)
                        ))
            e = time.time()
            print(f"mlp: {e - s}")
        for ii in range(len(prev_indices)):
            embeddings[ii] = torch.cat(embeddings[ii]).squeeze(0)
        embeddings = torch.stack(embeddings)
        return embeddings
        