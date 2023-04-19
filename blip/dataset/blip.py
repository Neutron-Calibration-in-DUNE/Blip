

import json
import os
import os.path as osp
import shutil
from typing import Callable, List, Optional, Union
import numpy as np
import torch

from torch_geometric.data import (
    Data,
    Dataset,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import read_txt_array

from blip.utils.logger import Logger

class BlipDataset(InMemoryDataset):
    def __init__(self, 
        name:   str,
        input_files:    list=None,
        features:       list=None,
        classes:        list=None,
        sample_weights: str=None,
        class_weights:  str=None,
        normalized:     bool=True,
        root:   str=".", 
        transform=None, 
        pre_transform=None, 
        pre_filter=None
    ):
        self.name = name
        self.input_files = input_files
        self.logger = Logger(self.name, file_mode='w')
        self.logger.info(f"constructing dataset.")
        self.features = features
        self.classes = classes
        self.sample_weights = sample_weights
        self.class_weights = class_weights
        self.number_of_events = 0
        if sample_weights != None:
            self.use_sample_weights = True
        else:
            self.use_sample_weights = False
        if class_weights != None:
            self.use_class_weights = True
        else:
            self.use_class_weights = False
        self.normalized = normalized

        self.meta = []
        for input_file in self.input_files:
            data = np.load(input_file, allow_pickle=True)
            self.meta.append(data['meta'].item())

        self.logger.info(f"setting 'features': {self.features}.")
        self.logger.info(f"setting 'classes': {self.classes}.")
        self.logger.info(f"setting 'sample_weights': {self.sample_weights}.")
        self.logger.info(f"setting 'class_weights': {self.class_weights}.")
        self.logger.info(f"setting 'use_sample_weights': {self.use_sample_weights}.")
        self.logger.info(f"setting 'use_class_weights': {self.use_class_weights}.")
        self.logger.info(f"setting 'normalize': {self.normalized}.")

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [input_file for input_file in self.input_files]

    @property
    def processed_file_names(self):
        return [f'data_{ii}.pt' for ii in range(self.number_of_events)]
        ...

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def process(self):
        # Read data into huge `Data` list.
        index = 0
        for jj, raw_path in enumerate(self.input_files):
            data = np.load(raw_path, allow_pickle=True)
            pos = data['point_cloud']
            adc = data['adc']
            y = data['labels']

            for ii in range(len(pos)):
                
                event = Data(
                    pos=torch.tensor(pos[ii]).type(torch.float),
                    x=torch.zeros(pos[ii].shape).type(torch.float),
                    #y=torch.full((len(pos[ii]),1),y[ii]).type(torch.long), 
                    category=torch.tensor(y[ii]).type(torch.long),
                    summed_adc=torch.tensor(adc[ii]).type(torch.float)
                )
                
                if self.pre_filter is not None:
                    event = self.pre_filter(event)

                if self.pre_transform is not None:
                    event = self.pre_transform(event)

                torch.save(event, osp.join(self.processed_dir, f'data_{index}.pt'))
                index += 1
        self.number_of_events = index