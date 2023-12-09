from ctypes import sizeof
import torch
import torch.nn as nn
import uproot
import os
import numpy as np
import socket
import matplotlib.pyplot as plt
from scipy import stats as st
from datetime import datetime
from torch.utils.data import Dataset

from blip.utils.logger import Logger
from blip.dataset.generic_dataset import GenericDataset
from blip.dataset.common import *

vanilla_dataset_config = {
    "name":               "default",
    # input folder and files    
    "dataset_folder":   "data/",
    "dataset_files":    [""],
    # positions, features, classes, etc.
    "features":     [],
    "classes":      [],
    # normalizations
    "features_normalization":   [],
    # weights
    "class_weights":    [],
    "sample_weights":   [],
    # masks
    "class_mask":   "",
    "label_mask":   "",
    # device
    "device":           "cpu",
}

class VanillaDataset(GenericDataset):
    """
    Datasets are stored as numpy arrays with two main branches,
    'features' and 'classes'.  'meta' stores information about 
    the dataset such as when is was created and who created it,
    etc.  It should contain a "features" dictionary, whose
    entries are the names of each feature and the corresponding
    column index in the features array.  Same for classes.

        meta = {
            "who_created":      getpass.getuser(),
            "when_created":     datetime.now().strftime("%m-%d-%Y-%H:%M:%S"),
            "where_created":    socket.gethostname(),
            "features": {
                "x": 0, "y": 1, "z": 2
            },
            "classes": {
                "validity": 0
            },
        }

    There may be additional arrays in the file, such as those
    containing event weights, class weights, or other items.
    """
    def __init__(self, 
        name:   str="",
        config: dict=vanilla_dataset_config,
        meta:   dict={}
    ):
        self.name = name + '_dataset'
        self.config = config
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(name, output="both", file_mode="w")
        else:
            self.logger = Logger(name, level='warning', file_mode="w")
        self.logger.info("constructing vanilla dataset.")

        self.number_of_events = 0
        self.meta = {}
        self.configure_dataset()
        self.configure_variables()
        self.configure_meta()
        self.configure_data()
        self.configure_weights()
        self.configure_normalization()

        GenericDataset.__init__(self)

    def configure_dataset(self):
        self.meta['feature_type'] = torch.float
        self.meta['class_type'] = torch.long

        # default to what's in the configuration file. May decide to deprecate in the future
        if ("dataset_folder" in self.config.keys()) :
            self.dataset_folder = self.config["dataset_folder"]
            self.logger.info(
                    f"Set dataset path from Configuration." +
                    f" dataset_folder: {self.dataset_folder}"
                    )
        elif ('BLIP_DATASET_PATH' in os.environ):
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
        self.meta['blip_features'] = self.config["features"]
        self.meta['blip_classes'] = self.config["classes"]

        if "labels" in self.config:
            self.meta['blip_labels'] = self.config["labels"]
        else:
            self.meta['blip_labels'] = [[] for ii in range(len(self.meta['blip_classes']))]

        if "classes_mask" in self.config:
            self.meta['blip_classes_mask'] = self.config["classes_mask"]
        else:
            self.meta['blip_classes_mask'] = []
            self.logger.info(f"setting 'classes_mask':    {self.meta['blip_classes_mask']}.")

        if "labels_mask" in self.config:
            self.meta['blip_labels_mask'] = self.config["labels_mask"]
        else:
            self.meta['blip_labels_mask'] = []
            self.logger.info(f"setting 'labels_mask':    {self.meta['blip_labels_mask']}.")

        if "consolidate_classes" in self.config:
            self.meta['consolidate_classes'] = self.config["consolidate_classes"]
        else:
            self.meta['consolidate_classes'] = None

        if "sample_weights" in self.config:
            self.meta['sample_weights'] = self.config["sample_weights"]
        else:
            self.meta['sample_weights'] = False

        if "class_weights" in self.config:
            self.class_weights = self.config["class_weights"]
        else:
            self.class_weights = False
        
        self.logger.info(f"setting 'features':          {self.meta['blip_features']}.")
        self.logger.info(f"setting 'classes':           {self.meta['blip_classes']}.")
        self.logger.info(f"setting 'consolidate_classes':   {self.meta['consolidate_classes']}")
        self.logger.info(f"setting 'sample_weights':    {self.meta['sample_weights']}.")
        self.logger.info(f"setting 'class_weights':     {self.class_weights}.")

        # determine if the list of class labels 
        # contains everything from the dataset list.
        # first, we check if [""] is an entry in the
        # consolidation list, and if so, replace it with
        # the left over classes which are not mentioned.
        if self.meta['consolidate_classes'] is not None:
            for label in self.class_labels:
                all_labels = list(self.class_labels[label].values())
                for ii, labels in enumerate(self.meta['consolidate_classes'][label]):
                    for jj, item in enumerate(labels):
                        if item in all_labels:
                            all_labels.remove(item)
                if len(all_labels) != 0:
                    if [""] in self.meta['consolidate_classes'][label]:
                        for ii, labels in enumerate(self.meta['consolidate_classes'][label]):
                            if labels == [""]:
                                self.meta['consolidate_classes'][label][ii] = all_labels
                    else:
                        self.logger.error(
                            f"consolidate_classes does not contain an exhaustive" +
                            f" list for label '{label}'!  Perhaps you forgot to include" +
                            f" the ['']? Leftover classes are {all_labels}."
                        )
            # now we create a map from old indices to new
            self.consolidation_map = {
                label: {}
                for label in self.meta['blip_classes']
            }
            for label in self.class_labels:
                for key, val in self.class_labels[label].items():
                    for jj, labels in enumerate(self.meta['consolidate_classes'][label]):
                        if val in labels:
                            self.consolidation_map[label][key] = jj
    
    def configure_meta(self):
        # get meta dictionaries from files
        temp_meta = []
        for input_file in self.dataset_files:
            try:
                data = np.load(input_file, allow_pickle=True)
                temp_meta.append(data['meta'].item())
            except:
                self.logger.error(f'error reading file "{input_file}"!')
        try:
            self.meta['features'] = temp_meta[0]['features']
            self.meta['classes'] = temp_meta[0]['classes']
            for key in self.meta['classes'].keys():
                self.meta[f'{key}_labels'] = temp_meta[0][f'{key}_labels']
        except:
            self.logger.error(f'error collecting meta information from file {input_file}!')
        
        # Check that meta info is consistent over the different files
        for ii in range(len(temp_meta)-1):
            if self.meta['features'] != temp_meta[ii+1]['features']:
                self.logger.error(f'conflicting meta information found in file {self.dataset_files[0]} and {self.dataset_files[ii+1]}')
            if self.meta['classes'] != temp_meta[ii+1]['classes']:
                self.logger.error(f'conflicting meta information found in file {self.dataset_files[0]} and {self.dataset_files[ii+1]}')
            for key in self.meta['classes'].keys():
                if self.meta[f'{key}_classes'] != temp_meta[ii+1][f'{key}_classes']:
                    self.logger.error(f'conflicting meta information found in file {self.dataset_files[0]} and {self.dataset_files[ii+1]}')
        # arange dictionaries for label<->value<->index maps
        self.meta['features_names'] = list(self.meta['features'].keys())
        self.meta['features_values'] = list(self.meta['features'].values())
        self.meta['features_names_by_value'] = {val: key for key, val in self.meta['features'].items()}
        self.meta['classes_names'] = list(self.meta['classes'].keys())
        self.meta['classes_values'] = list(self.meta['classes'].values())
        self.meta['classes_names_by_value'] = {val: key for key, val in self.meta['classes'].items()}
        self.meta['classes_labels_names'] = {
            label:   list(self.meta[f'{key}_labels'].values())
            for label in self.meta['classes'].keys()
        }
        self.meta['classes_labels_values'] = {
            label:   list(self.meta[f'{key}_labels'].keys())
            for label in self.meta['classes'].keys()
        }
        self.meta['classes_labels_names_by_value'] = {
            label:   {key: val for key, val in self.meta[f'{key}_labels'].items()}
            for label in self.meta['classes'].keys()
        }
        self.meta['classes_labels_values_by_name'] = {
            label:   {val: key for key, val in self.meta[f'{key}_labels'].items()}
            for label in self.meta['classes'].keys()
        }

        # Check that config variables match meta info
        for feature in self.meta['blip_features']:
            if feature not in self.meta['features']:
                self.logger.error(f'specified feature "{feature}" variable not in arrakis meta!')
        for ii, classes in enumerate(self.meta['blip_classes']):
            if classes not in self.meta['classes']:
                self.logger.error(f'specified classes "{classes}" variable not in arrakis meta!')
            if len(self.meta['blip_labels']) != 0:
                for label in self.meta['blip_labels'][ii]:
                    if label not in self.meta['classes_labels_names'][classes]:
                        self.logger.error(f'specified label "{classes}:{label}" not in arrakis meta!')
        try:
            self.meta['blip_features_indices'] = [
                self.meta["features"][feature] 
                for feature in self.meta['blip_features']
            ]
            self.meta['blip_features_indices_by_name'] = {
                feature: ii
                for ii, feature in enumerate(self.meta['blip_features'])
            }
        except:
            self.logger.error(f'failed to get feature indices from meta!')
        try:    
            self.meta['blip_classes_indices'] = [
                self.meta["classes"][label]
                for label in self.meta['blip_classes']
            ]
            self.meta['blip_classes_indices_by_name'] = {
                classes: ii
                for ii, classes in enumerate(self.meta['blip_classes'])
            }
        except:
            self.logger.error(f'failed to get classes indices from meta!')
        try:
            self.meta['blip_labels_values'] = {}
            self.meta['blip_labels_values_map'] = {}
            self.meta['blip_labels_values_inverse_map'] = {}
            for ii, classes in enumerate(self.meta['blip_classes']):
                if len(self.meta['blip_labels']) == 0:
                    self.meta['blip_labels_values'][classes] = list(self.meta['classes_labels_values_by_name'][classes].values())
                else:
                    if len(self.meta['blip_labels'][ii]) == 0:
                        self.meta['blip_labels_values'][classes] = list(self.meta['classes_labels_values_by_name'][classes].values())
                    else:
                        self.meta['blip_labels_values'][classes] = [
                            self.meta['classes_labels_values_by_name'][classes][key]
                            for key in self.meta['blip_labels'][ii]
                        ]
                self.meta['blip_labels_values_map'][classes] = {
                    val: ii
                    for ii, val in enumerate(self.meta['blip_labels_values'][classes])
                }
                self.meta['blip_labels_values_inverse_map'][classes] = {
                    ii: val 
                    for ii, val in enumerate(self.meta['blip_labels_values'][classes])
                }
        except:
            self.logger.error(f'failed to arange classes labels from meta!')

        # Configure masks for classes and corresponding labels.  
        if "classes_mask" in self.config:
            self.meta['blip_classes_mask_indices'] = {
                classes: self.meta['classes'][classes]
                for classes in self.meta['blip_classes_mask']
            }
        else:
            self.meta['blip_classes_mask_indices'] = {}
        if "labels_mask" in self.config:
            self.meta['blip_classes_labels_mask_values'] = {
                classes: [
                    self.meta['classes_labels_values_by_name'][classes][label]
                    for label in self.meta['blip_labels_mask'][ii]
                ]
                for ii, classes in enumerate(self.meta['blip_classes_mask'])
            }
        else:
            self.meta['classes_labels_indices'] = {}
    
    def apply_event_masks(self,
        event_features, event_classes
    ):
        mask = np.array([True for ii in range(len(event_features))])
        if "classes_mask" in self.config:
            # Apply 'classes_mask' and 'labels_mask'
            for classes, class_index in self.meta['blip_classes_mask_indices'].items():
                for jj, label_value in enumerate(self.meta['blip_classes_labels_mask_values'][classes]):
                    mask &= (event_classes[:, class_index] == label_value)
            # Apply mask for 'labels'
            for classes in self.meta['blip_classes']:
                class_index = self.meta["classes"][classes]
                for jj, label_value in enumerate(self.meta['blip_labels_values'][classes]):
                    mask |= (event_classes[:, class_index] == label_value)
        
        # Apply masks
        event_features = event_features[mask].astype(np.float)
        event_classes = event_classes[mask].astype(np.int64)

        # Convert class labels to ordered list
        temp_classes = event_classes.copy()
        for classes in self.meta['blip_classes']:
            class_index = self.meta["classes"][classes]
            for key, val in self.meta['blip_labels_values_map'][classes].items():
                temp_mask = (temp_classes[:, class_index] == key)
                event_classes[temp_mask, class_index] = val

        # Grab indices of interest
        return event_features, event_classes, mask

    def process_event(self,
        event_features, event_classes, raw_path
    ):
        event_features, event_classes, mask = self.apply_event_masks(event_features, event_classes)
        self.meta['event_mask'][raw_path].append(mask)
        self.features.append(event_features)
        self.classes.append(event_classes)
        self.meta['input_events'][raw_path].append([self.index])
        self.index += 1

    def configure_data(self):
        self.meta['input_events'] = {
            raw_path: []
            for raw_path in self.dataset_files
        }
        self.meta['event_mask'] = {
            raw_path: []
            for raw_path in self.dataset_files
        }
        self.index = 0
        self.features = []
        self.classes = []
        self.logger.info(f"processing {len(self.dataset_files)} files.")
        for jj, raw_path in enumerate(self.dataset_files):
            data = np.load(raw_path, allow_pickle=True)
            features = data[f'features']
            classes = data[f'classes']
            input_events = []
            # Iterate over all events in this file
            for ii in range(len(features)):
                # gather event features and classes
                event_features = features[ii]
                event_classes = classes[ii]
                self.process_event(
                    event_features, event_classes, raw_path
                )
                input_events.append(self.index)
            self.meta['input_events'][raw_path].append(input_events)
        self.features = np.concatenate(self.features)
        self.classes = np.concatenate(self.classes)
        self.number_of_events = self.index
        self.logger.info(f"processed {self.number_of_events} events.")

    def configure_weights(self):
        # set up weights
        if self.config['sample_weights'] != None:
            if (len(self.config['sample_weights']) > 0):
                self.use_sample_weights = True
            else:
                self.use_sample_weights = False
        else:
            self.use_sample_weights = False
        if self.config['class_weights'] != None:
            if (len(self.config['class_weights']) > 0):
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
        else:
            self.use_class_weights = False
            self.class_weights = {}

    def configure_normalization(self):
        if self.normalize:
            if 'means' in self.meta.keys():
                self.means = self.meta['means']
                # check that all features are there
                for item in self.meta['features']:
                    if item not in self.means.keys():
                        self.replace_meta = True
                        self.logger.warning(f"feature {item} is not present in list of means, calculating mean value")
                        self.means[item] = np.mean(self.features[:, self.meta['features'][item]])
                        self.meta['means'][item] = self.means[item]
            else:
                self.replace_meta = True
                self.logger.info(f"means information is not stored in meta, calculating means.")
                self.means = {}
                for item in self.meta['features']:
                    self.means[item] = np.mean(self.features[:, self.meta['features'][item]])
                self.meta['means'] = self.means

            if 'stds' in self.meta.keys():
                self.stds = self.meta['stds']
                # check that all features are there
                for item in self.meta['features']:
                    if item not in self.stds.keys():
                        self.replace_meta = True
                        self.logger.warning(f"feature {item} is not present in list of stds, calculating std value")
                        self.stds[item] = np.std(self.features[:, self.meta['features'][item]])
                        self.meta['stds'][item] = self.stds[item]
            else:
                self.replace_meta = True
                self.logger.info(f"stds information is not stored in meta, calculating stds.")
                self.stds = {}
                for item in self.meta['features']:
                    self.stds[item] = np.std(self.features[:, self.meta['features'][item]])
                self.meta['stds'] = self.stds
            self.feature_means = [self.means[item] for item in self.meta['features']]
            self.feature_stds  = [self.stds[item] for item in self.meta['features']]

        
        # turn arrays into torch tensors
        self.features = torch.tensor(self.features, dtype=torch.float)
        if self.normalize:
            self.feature_means = torch.tensor(self.feature_means, dtype=torch.float)
            self.feature_stds  = torch.tensor(self.feature_stds, dtype=torch.float)
        self.classes = torch.tensor(self.classes, dtype=torch.float)
        if self.use_sample_weights:
            self.sample_weights = torch.tensor(self.sample_weights, dtype=torch.float)
        if self.use_class_weights:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)
        
        self.feature_shape = self.features[0].shape
        self.class_shape = self.classes[0].shape
        if self.use_sample_weights:
            self.sample_weights_shape = self.sample_weights[0].shape
        else:
            self.sample_weights_shape = None
        if self.use_class_weights:
            self.class_weights_shape = self.class_weights[0].shape
        else:
            self.class_weights_shape = None
            
        self.logger.info(f"processed {self.number_of_events} events.")

    def consolidate_class(self, 
        classes
    ):
        for ii in range(len(classes)):
            for jj, label in enumerate(self.consolidation_map):
                classes[ii][jj] = self.consolidation_map[label][classes[ii][jj]]
        return classes

    def __len__(self):
        return self.number_of_events

    def __getitem__(self, idx):
        x = self.features[idx, self.meta['blip_features_indices']]
        if self.normalize:
            x = (x - self.feature_means[self.meta['blip_features_indices']])/self.feature_stds[self.meta['blip_features_indices']]
        y = self.classes[idx, self.meta['blip_classes_indices']]
        if not self.use_sample_weights:
            z = []
        else:
            z = self.sample_weights[idx, self.sample_weight_idx]
        return {
            'x':                x,
            'category':         y,
            'sample_weights':   z
        }
    
    def feature(self,
        feature:    str,
    ):
        if feature not in self.meta['features']:
            self.logger.error(
                f"attempting to access feature {feature} which " + 
                "is not in 'features': {self.meta['features']}"
            )
        return self.features[:,  self.meta['features'][feature]]
    
    def normalize(self,
        x
    ):
        return (x - self.feature_means[self.meta['blip_features_indices']])/self.feature_stds[self.meta['blip_features_indices']]

    def unnormalize(self,
        x,
        detach=False
    ):
        x = self.feature_means[self.meta['blip_features_indices']] + self.feature_stds[self.meta['blip_features_indices']] * x
        if detach:
            return x.detach().numpy()
        else:
            return x       
                           
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
            events = [
                [ii for ii in list(indices) if ii in self.meta['input_events'][raw_path][jj]] 
                for jj in range(len(self.meta['input_events'][raw_path]))
            ]
            classes_prefix = ""
            if self.meta['dataset_type'] == "cluster":
                classes_prefix = f"{self.dbscan_eps}_{self.dbscan_min_samples}_"
            output = {
                f"{classes_prefix}{classes}": [input_dict[classes][event] for event in events]
                for classes in input_dict.keys()
            }
            output['event_mask'] = self.meta['event_mask'][raw_path]
            output['labels_values_map'] = self.meta['labels_values_map']
            output['labels_values_inverse_map'] = self.meta['labels_values_inverse_map']
            if self.meta['dataset_type'] == "cluster":
                output[f'{self.dbscan_eps}_{self.dbscan_min_samples}_cluster_ids'] = self.meta['cluster_ids'][raw_path]
                output[f'{self.dbscan_eps}_{self.dbscan_min_samples}_cluster_indices'] = self.meta['cluster_indices'][raw_path]
            # otherwise add the array and save
            loaded_arrays.update(output)
            np.savez(
                raw_path,
                **loaded_arrays
            )