

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
        dataset_type:   str='voxel',
        input_folder:   str='',
        input_files:    list=None,
        positions:      list=None,
        features:       list=None,
        classes:        list=None,
        consolidate_classes:    list=None,
        sample_weights: str=None,
        class_weights:  str=None,
        normalized:     bool=True,
        root:   str=".", 
        transform=None, 
        pre_transform=None, 
        pre_filter=None,
        device=None
    ):
        self.name = name
        self.logger = Logger(self.name, output="both", file_mode='w')
        self.logger.info(f"constructing dataset.")

        self.dataset_type = dataset_type
        if dataset_type == 'voxel':
            self.position_type = torch.int
        else:
            self.position_type = torch.float
        self.input_folder = input_folder
        self.input_files = [self.input_folder + input_file for input_file in input_files]

        self.positions = positions
        self.features = features
        self.classes = classes
        self.consolidate_classes = consolidate_classes

        self.sample_weights = sample_weights
        self.class_weight_labels = class_weights
        
        self.number_of_events = 0

        self.meta = []
        for input_file in self.input_files:
            data = np.load(input_file, allow_pickle=True)
            self.meta.append(data['meta'].item())

        # Set up weights
        if sample_weights != None:
            self.use_sample_weights = True
        else:
            self.use_sample_weights = False
        if class_weights != None:
            self.use_class_weights = True
            self.class_weights = {
                key: torch.tensor(np.sum([
                    [self.meta[ii][f"{key}_points"][jj] 
                     for jj in self.meta[ii][f"{key}_points"].keys()
                    ] for ii in range(len(self.meta))
                ], axis=0), dtype=torch.float)
                for key in class_weights
            }
            self.class_weight_totals = {
                key: float(torch.sum(value))
                for key, value in self.class_weights.items()
            }
            for key, value in self.class_weights.items():
                for ii, val in enumerate(value):
                    if val != 0:
                        self.class_weights[key][ii] = self.class_weight_totals[key] / float(len(value) * val)
        else:
            self.use_class_weights = False
            self.class_weights = {}
        
        self.normalized = normalized
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.device = device

        self.position_indices = [
            self.meta[0]["features"][position] 
            for position in self.positions
        ]
        self.feature_indices = [
            self.meta[0]["features"][feature] 
            for feature in self.features
        ]
        self.class_labels = {
            label: self.meta[0][f"{label}_labels"]
            for label in self.classes
        }
        
        # determine if the list of class labels 
        # contains everything from the dataset list.
        # first, we check if [""] is an entry in the
        # consolidation list, and if so, replace it with
        # the left over classes which are not mentioned.
        if self.consolidate_classes is not None:
            for label in self.class_labels:
                all_labels = list(self.class_labels[label].values())
                for ii, labels in enumerate(self.consolidate_classes[label]):
                    for jj, item in enumerate(labels):
                        if item in all_labels:
                            all_labels.remove(item)
                if len(all_labels) != 0:
                    if [""] in self.consolidate_classes[label]:
                        for ii, labels in enumerate(self.consolidate_classes[label]):
                            if labels == [""]:
                                self.consolidate_classes[label][ii] = all_labels
                    else:
                        self.logger.error(
                            f"consolidate_classes does not contain an exhaustive" +
                            f" list for label '{label}'!  Perhaps you forgot to include" +
                            f" the ['']? Leftover classes are {all_labels}."
                        )
            # now we create a map from old indices to new
            self.consolidation_map = {
                label: {}
                for label in self.classes
            }
            for label in self.class_labels:
                for key, val in self.class_labels[label].items():
                    for jj, labels in enumerate(self.consolidate_classes[label]):
                        if val in labels:
                            self.consolidation_map[label][key] = jj
        self.class_indices = [
            self.meta[0]["classes"][label]
            for label in self.classes
        ]
        self.number_classes = {
            label: len(self.class_labels[label])
            for label in self.classes
        }
        self.logger.info(f"setting 'dataset_type:   {self.dataset_type}.")
        self.logger.info(f"setting 'positions':     {self.positions}.")
        self.logger.info(f"setting 'features':      {self.features}.")
        self.logger.info(f"setting 'classes':       {self.classes}.")
        self.logger.info(f"setting 'sample_weights':{self.sample_weights}.")
        self.logger.info(f"setting 'class_weights': {self.class_weights}.")
        self.logger.info(f"setting 'use_sample_weights': {self.use_sample_weights}.")
        self.logger.info(f"setting 'use_class_weights':  {self.use_class_weights}.")
        self.logger.info(f"setting 'normalize':     {self.normalized}.")

        super().__init__(root, transform, pre_transform, pre_filter, log=False)

    def consolidate_class(self, 
        classes
    ):
        for ii in range(len(classes)):
            for jj, label in enumerate(self.consolidation_map):
                classes[ii][jj] = self.consolidation_map[label][classes[ii][jj]]
        return classes

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
        self.logger.info(f"processing {len(self.input_files)} files.")
        self.input_events = {
            raw_path: []
            for raw_path in self.input_files
        }
        for jj, raw_path in enumerate(self.input_files):
            data = np.load(raw_path, allow_pickle=True)
            features = data['features']
            classes = data['classes']

            for ii in range(len(features)):
                if self.consolidate_classes is not None:
                    event_classes = self.consolidate_class(classes[ii][:, self.class_indices])
                else:
                    event_classes = classes[ii][:, self.class_indices]
                if len(self.feature_indices) != 0:
                    event = Data(
                        pos=torch.tensor(features[ii][:, self.position_indices]).type(self.position_type),
                        x=torch.tensor(features[ii][:, self.feature_indices]).type(torch.float),
                        category=torch.tensor(event_classes).type(torch.long),
                    )
                else:
                    event = Data(
                        pos=torch.tensor(features[ii][:, self.position_indices]).type(self.position_type),
                        x=torch.ones((len(features[ii]),1)).type(torch.float),
                        category=torch.tensor(event_classes).type(torch.long),
                    )
                if self.pre_filter is not None:
                    event = self.pre_filter(event)

                if self.pre_transform is not None:
                    event = self.pre_transform(event)

                torch.save(event, osp.join(self.processed_dir, f'data_{index}.pt'))
                self.input_events[raw_path].append(index)
                index += 1
        self.number_of_events = index
        self.logger.info(f"processed {self.number_of_events} events.")
    
    def append_input_files(self,
        dict_name,
        input_dict,
        indices
    ):
        for jj, raw_path in enumerate(self.input_files):
            loaded_file = np.load(raw_path, allow_pickle=True)
            loaded_arrays = {
                key: loaded_file[key] 
                for key in loaded_file.files
            }
            events = [ii for ii in list(indices) if ii in self.input_events[raw_path]]
            output = {
                classes: input_dict[classes][events]
                for classes in input_dict.keys()
            }
            # otherwise add the array and save
            loaded_arrays.update(output)
            np.savez(
                raw_path,
                **loaded_arrays
            )
            