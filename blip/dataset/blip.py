

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

from blip.utils.logger import Logger

class BlipDataset(InMemoryDataset):
    def __init__(self, 
        name:   str,
        input_file: str,
        features:   list=None,
        classes:    list=None,
        sample_weights: str=None,
        class_weights:  str=None,
        normalized:  bool=True,
        root:   str=".", 
        transform=None, 
        pre_transform=None, 
        pre_filter=None
    ):
        self.name = name
        self.input_file = input_file 
        self.logger = Logger(self.name, file_mode='w')
        self.logger.info(f"constructing dataset.")
        self.features = features
        self.classes = classes
        self.sample_weights = sample_weights
        self.class_weights = class_weights
        if sample_weights != None:
            self.use_sample_weights = True
        else:
            self.use_sample_weights = False
        if class_weights != None:
            self.use_class_weights = True
        else:
            self.use_class_weights = False
        self.normalized = normalized

        data = np.load(self.input_file, allow_pickle=True)
        self.meta = data['meta'].item()
        self.number_classes = self.meta['num_group_classes']
        self.labels = self.meta['group_classes']

        self.logger.info(f"setting 'features': {self.features}.")
        self.logger.info(f"setting 'classes': {self.classes}.")
        self.logger.info(f"setting 'sample_weights': {self.sample_weights}.")
        self.logger.info(f"setting 'class_weights': {self.class_weights}.")
        self.logger.info(f"setting 'use_sample_weights': {self.use_sample_weights}.")
        self.logger.info(f"setting 'use_class_weights': {self.use_class_weights}.")
        self.logger.info(f"setting 'normalize': {self.normalized}.")

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.input_file]

    @property
    def processed_file_names(self):
        return ['data.pt']
        ...

    def process(self):
        # Read data into huge `Data` list.
        data = np.load(self.input_file, allow_pickle=True)
        pos = data['positions']
        summed_adc = data['summed_adc']
        y = data['group_labels']

        data_list = [
            Data(
                pos=torch.tensor(pos[ii]).type(torch.float),
                x=torch.zeros(pos[ii].shape).type(torch.float),
                y=torch.full((len(pos[ii]),1),y[ii]).type(torch.long), 
                category=torch.tensor([y[ii]]).type(torch.long),
                summed_adc=torch.tensor(summed_adc[ii]).type(torch.float)
            )
            for ii in range(len(pos))
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])