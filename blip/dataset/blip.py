

import json
import os
import os.path as osp
import shutil
from typing import Callable, List, Optional, Union
import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import read_txt_array


class GammaDataset(InMemoryDataset):
    def __init__(self, root, num_samples, transform=None, pre_transform=None, pre_filter=None):
        self.num_samples = num_samples
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['../data/gamma_pointcloud.npz']

    @property
    def processed_file_names(self):
        return ['data.pt']
        ...

    def process(self):
        # Read data into huge `Data` list.
        data = np.load('../data/gamma_pointcloud.npz', allow_pickle=True)
        pos = data['pos'][:self.num_samples]
        y = data['labels'][:self.num_samples]
        data_list = [
            Data(
                pos=torch.tensor(pos[ii]).type(torch.float),
                x=torch.zeros(pos[ii].shape).type(torch.float),
                y=torch.full((len(pos[ii]),1),y[ii]).type(torch.long), 
                category=torch.tensor([y[ii]]).type(torch.long)
            )
            for ii in range(self.num_samples)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])