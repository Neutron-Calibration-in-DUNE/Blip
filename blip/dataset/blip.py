

import json
import os
import os.path as osp
import glob
import shutil
from typing import Callable, List, Optional, Union
import numpy as np
from sklearn.cluster import DBSCAN
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

blip_dataset_config = {
    "name":             "default",
    # type of this blip dataset
    "dataset_type":     "voxel",
    # input folder and files    
    "dataset_folder":     "data/",
    "dataset_files":      [""],
    # positions, features, classes, etc.
    "positions":        [],
    "features":         [],
    "classes":          [],
    "consolidate_classes":  [],
    "sample_weights":   [],
    "class_weights":    [],
    "normalized":       True,
    # clustering parameters
    "dbscan_min_samples":   10,
    "dbscan_eps":       10.0,
    "cluster_class":    "shape",
    "cluster_label":    "blip",
    "cluster_positions":    [],
    # transforms and filters
    "root":             ".",
    "transform":        None,
    "pre_transform":    None,
    "pre_filter":       None,
    # device
    "device":           "cpu",
}

class BlipDataset(InMemoryDataset):
    def __init__(self, 
        config: dict=blip_dataset_config,
    ):
        self.config = config
        # setup name and logger for this dataset
        self.name = config["name"]
        self.logger = Logger(self.name, output="both", file_mode='w')
        self.logger.info(f"constructing dataset.")

        self.number_of_events = 0
        self.root = self.config["root"]
        self.device = self.config["device"]
        self.skip_processing = self.config["skip_processing"]
        if self.skip_processing:
            if os.path.isdir('processed/'):
                for path in os.listdir('processed/'):
                    if os.path.isfile(os.path.join('processed/', path)):
                        self.number_of_events += 1
            self.logger.info(f'found {self.number_of_events} processed files.')
        
        self.configure_dataset()
        self.configure_variables()
        self.configure_meta()
        self.configure_clustering()
        self.configure_weights()
        self.configure_transforms()

        super().__init__(
            self.root, self.transform, 
            self.pre_transform, self.pre_filter, 
            log=False
        )

    def configure_meta(self):
        # get meta dictionaries from files
        self.meta = []
        for input_file in self.dataset_files:
            data = np.load(input_file, allow_pickle=True)
            self.meta.append(data['meta'].item())

        # set up maps for positions, features, etc.
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
            for label in self.meta[0]["classes"].keys()
        }
        self.class_indices = [
            self.meta[0]["classes"][label]
            for label in self.classes
        ]
        self.number_classes = {
            label: len(self.class_labels[label])
            for label in self.meta[0]["classes"].keys()
        }
        self.class_labels_by_name = {
            label: {val: key for key, val in self.class_labels[label].items()}
            for label in self.class_labels.keys()
        }
        self.class_label_names = [
            label for label in self.meta[0]["classes"].keys()
        ]
        self.class_label_index = {
            label: ii
            for ii, label in enumerate(self.meta[0]["classes"].keys())
        }
        self.class_label_indices = {
            label: {key: ii for ii, key in enumerate(self.class_labels[label].keys())}
            for label in self.class_labels.keys()
        }

    def configure_dataset(self):
        # set dataset type
        if "dataset_type" not in self.config.keys():
            self.logger.error(f'no dataset_type specified in config!')
        self.dataset_type = self.config["dataset_type"]
        if self.dataset_type == 'voxel':
            self.position_type = torch.int
        else:
            self.position_type = torch.float
        self.logger.info(f"setting 'dataset_type: {self.dataset_type}.")

        # default to what's in the configuration file. May decide to deprecate in the future
        if ( "dataset_folder" in self.config.keys() ) :
            self.dataset_folder = self.config["dataset_folder"]
            self.logger.info(
                    f"Set dataset path from Configuration." +
                    f" dataset_folder: {self.dataset_folder}"
                    )
        elif ( 'BLIP_DATASET_PATH' in os.environ ):
            self.logger.debug(f'Found BLIP_DATASET_PATH in environment')
            self.dataset_folder = os.environ['BLIP_DATASET_PATH']
            self.logger.info(
                    f"Setting dataset path from Enviroment." +
                    f" BLIP_DATASET_PATH = {self.dataset_folder}"
                    )
        else :
            self.logger.error(f'No dataset_folder specified in environment or configuration file!')

        if "dataset_files" not in self.config.keys():
            self.logger.error(f'no dataset_files specified in config!')
        if isinstance(self.config["dataset_files"], list):
            self.dataset_files = [
                self.dataset_folder + input_file 
                for input_file in self.config["dataset_files"]
            ]
        elif isinstance(self.config["dataset_files"], str):
            if self.config["dataset_files"] == "all":
                self.logger.info(f'searching {self.dataset_folder} recursively for all .npz files.')
                self.dataset_files = glob.glob(self.dataset_folder + '**/*.npz', recursive=True)
            else:
                try:
                    self.logger.info(f'searching {self.dataset_folder} recursively for all {self.config["dataset_files"]} files.')
                    self.dataset_files = glob.glob(self.dataset_folder + f'**/{self.config["dataset_files"]}', recursive=True)
                except:
                    self.logger.error(f'specified "dataset_files" parameter: {self.config["dataset_files"]} incompatible!')
        else:       
            self.logger.error(f'specified "dataset_files" parameter: {self.config["dataset_files"]} incompatible!')

    def configure_variables(self):
        # set positions, features and classes
        self.positions = self.config["positions"]
        self.features = self.config["features"]
        self.classes = self.config["classes"]
        self.consolidate_classes = self.config["consolidate_classes"]
        self.sample_weights = self.config["sample_weights"]
        self.class_weights = self.config["class_weights"]
        self.normalized = self.config["normalized"]

        self.logger.info(f"setting 'positions':     {self.positions}.")
        self.logger.info(f"setting 'features':      {self.features}.")
        self.logger.info(f"setting 'classes':       {self.classes}.")
        self.logger.info(f"setting 'consolidate_classes':   {self.consolidate_classes}")
        self.logger.info(f"setting 'sample_weights':{self.sample_weights}.")
        self.logger.info(f"setting 'class_weights': {self.class_weights}.")
        self.logger.info(f"setting 'normalize':     {self.normalized}.")

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

    def configure_clustering(self):
        if self.dataset_type != "cluster":
            return
        self.dbscan_min_samples = self.config["dbscan_min_samples"]
        self.dbscan_eps = self.config["dbscan_eps"]
        self.cluster_class = self.config["cluster_class"]
        self.cluster_label = self.config["cluster_label"]
        self.cluster_positions = self.config["cluster_positions"]
        self.cluster_position_indices = [
            self.meta[0]["features"][position] 
            for position in self.cluster_positions
        ]
        self.logger.info(f"setting 'dbscan_min_samples': {self.dbscan_min_samples}.")
        self.logger.info(f"setting 'dbscan_eps': {self.dbscan_eps}.")
        self.logger.info(f"setting 'cluster_class': {self.cluster_class}.")
        self.logger.info(f"setting 'cluster_label': {self.cluster_label}.")
        self.logger.info(f"setting 'cluster_positions': {self.cluster_positions}")

        self.cluster_class_index = self.class_label_index[self.cluster_class]
        self.cluster_label_value = self.class_labels_by_name[self.cluster_class][self.cluster_label]

        self.dbscan = DBSCAN(
            eps=self.dbscan_eps, 
            min_samples=self.dbscan_min_samples
        )

    def configure_weights(self):
        # set up weights
        if self.sample_weights != None:
            self.use_sample_weights = True
        else:
            self.use_sample_weights = False
        if self.class_weights != None:
            self.use_class_weights = True
            self.class_weights = {
                key: torch.tensor(np.sum([
                    [self.meta[ii][f"{key}_points"][jj] 
                     for jj in self.meta[ii][f"{key}_points"].keys()
                    ] for ii in range(len(self.meta))
                ], axis=0), dtype=torch.float)
                for key in self.class_weights
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
    
    def configure_transforms(self):
        self.transform = self.config["transform"]
        self.pre_transform = self.config["pre_transform"]
        self.pre_filter = self.config["pre_filter"]

    def consolidate_class(self, 
        classes
    ):
        for ii in range(len(classes)):
            for jj, label in enumerate(self.consolidation_map):
                classes[ii][jj] = self.consolidation_map[label][classes[ii][jj]]
        return classes

    @property
    def raw_file_names(self):
        return [input_file for input_file in self.dataset_files]

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
        self.input_events = {
            raw_path: []
            for raw_path in self.dataset_files
        }
        self.cluster_events = {
            raw_path: []
            for raw_path in self.dataset_files
        }
        self.cluster_indicies = {
            raw_path: []
            for raw_path in self.dataset_files
        }
        if self.skip_processing:
            self.logger.info(f'skipping processing of data.')
            return
        self.index = 0
        self.logger.info(f"processing {len(self.dataset_files)} files.")
        
        for jj, raw_path in enumerate(self.dataset_files):
            data = np.load(raw_path, allow_pickle=True)
            features = data['features']
            classes = data['classes']
            cluster_events = []
            for ii in range(len(features)):
                # gather event features and classes
                event_features = features[ii]
                # check if classes need to be consolidated
                if self.consolidate_classes is not None:
                    event_classes = self.consolidate_class(classes[ii])
                else:
                    event_classes = classes[ii]

                if self.dataset_type == "cluster":
                    self.process_cluster(event_features, event_classes, raw_path)
                    cluster_events.append(np.full(len(event_features), ii, dtype=int))
                elif self.dataset_type == "voxel":
                    self.process_voxel(event_features, event_classes, raw_path)
            if self.dataset_type == "cluster":
                self.cluster_events[raw_path] = np.concatenate(cluster_events)
        self.number_of_events = self.index
        self.logger.info(f"processed {self.number_of_events} events.")

    def process_cluster(self,
        event_features,
        event_classes,
        raw_path
    ):
        # create clusters using DBSCAN
        cluster_mask = (event_classes[:, self.cluster_class_index] == self.cluster_label_value)
        cluster_positions = event_features[:, self.cluster_position_indices][cluster_mask]
        cluster_classes = event_classes[:, self.class_indices][cluster_mask]
        
        cluster_labels = self.dbscan.fit(cluster_positions).labels_
        unique_labels = np.unique(cluster_labels)

        # for each unique cluster label, create a separate
        # dataset.
        for kk in unique_labels:
            if kk == -1:
                continue
            temp_mask = (cluster_labels == kk)
            temp_positions = event_features[:, self.position_indices][cluster_mask][temp_mask]
            min_positions = np.min(temp_positions, axis=0)
            max_positions = np.max(temp_positions, axis=0)
            scale = max_positions - min_positions
            scale[(scale == 0)] = max_positions[(scale == 0)]
            temp_positions = 2 * (temp_positions - min_positions) / scale - 1
            temp_classes = cluster_classes[temp_mask]
            if len(self.feature_indices) != 0:
                event = Data(
                    pos=torch.tensor(temp_positions).type(self.position_type),
                    x=torch.tensor(event_features[:, self.feature_indices][cluster_mask][temp_mask]).type(torch.float),
                    category=torch.tensor(temp_classes).type(torch.long)
                )
            else:
                event = Data(
                    pos=torch.tensor(temp_positions).type(self.position_type),
                    x=torch.ones((len(event_features[cluster_mask][temp_mask]),1)).type(torch.float),
                    category=torch.tensor(temp_classes).type(torch.long)
                )
            if self.pre_filter is not None:
                event = self.pre_filter(event)
            if self.pre_transform is not None:
                event = self.pre_transform(event)

            torch.save(event, osp.join(self.processed_dir, f'data_{self.index}.pt'))
            self.input_events[raw_path].append(self.index)
            self.cluster_indicies[raw_path].append(np.where(temp_mask))
            self.index += 1
                           
    def process_voxel(self,
        event_features,
        event_classes,
        raw_path
    ):
        if len(self.feature_indices) != 0:
            event = Data(
                pos=torch.tensor(event_features[:, self.position_indices]).type(self.position_type),
                x=torch.tensor(event_features[:, self.feature_indices]).type(torch.float),
                category=torch.tensor(event_classes[:, self.class_indices]).type(torch.long),
            )
        else:
            event = Data(
                pos=torch.tensor(event_features[:, self.position_indices]).type(self.position_type),
                x=torch.ones((len(event_features),1)).type(torch.float),
                category=torch.tensor(event_classes[:, self.class_indices]).type(torch.long),
            )
        if self.pre_filter is not None:
            event = self.pre_filter(event)

        if self.pre_transform is not None:
            event = self.pre_transform(event)

        torch.save(event, osp.join(self.processed_dir, f'data_{self.index}.pt'))
        self.input_events[raw_path].append(self.index)
        self.index += 1
    
    def append_dataset_files(self,
        dict_name,
        input_dict,
        indices
    ):
        for jj, raw_path in enumerate(self.dataset_files):
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
            output['cluster_events'] = self.cluster_events[raw_path]
            output['cluster_indices'] = self.cluster_indicies[raw_path]
            # otherwise add the array and save
            loaded_arrays.update(output)
            np.savez(
                raw_path,
                **loaded_arrays
            )

