

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
from blip.dataset.generic_dataset import GenericDataset
from blip.dataset.common import *

blip_dataset_config = {
    "name":             "default",
    # type of this blip dataset
    "dataset_type":     "voxel",
    # input folder and files    
    "dataset_folder":     "data/",
    "dataset_files":      [""],
    # positions, features, classes, etc.
    "positions":            [],
    "features":             [],
    "classes":              [],
    "normalized":           True,
    # clustering parameters
    "dbscan_min_samples":   10,
    "dbscan_eps":       10.0,
    "cluster_class":    "topology",
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

class BlipDataset(InMemoryDataset, GenericDataset):
    """
    Blip datasets are constructed with Data objects from torch geometric.
    Each example, or entry, in the dataset corresponds to a LArTPC event.
    Each of these events has a corresponding 'Data' object from pyg.
    The Data objects have the following set of attributes:

        x - a 'n x d_f' array of node features.
        pos - a 'n x d_p' array of node positions.
        y - a 'n x d_c' array of class labels.
        edge_index - a '2 x n_e' array of edge indices.
        edge_attr - a 'n_e x d_e' array of edge_features.

    We add an additional array of cluster labels which we call 'clusters', which has
    a topology of 'n x d_cl', where 'd_cl' is the dimension (number of classes)
    for clusters.

    The Arrakis datasets prepared in .npz format contain a 'features' array, which
    is a 'n x d_f' array, a classes array, which is 'n x d_c', and a clusters array
    which is 'n x d_cl'.  It is then straightforward to assign 'features' to both
    'x' and 'pos' by providing the config parameters 'positions' and 'features'
    which should list the names of the features to be assigned to either 'x' or 'pos'.

    The same goes for 'classes' and 'clusters', which the user can specify at run 
    time.  There are also options for masking each of these arrays by providing one of

        'class_mask' - a list of classes to apply masks too

    """
    def __init__(self, 
        name:   str="",
        config: dict=blip_dataset_config,
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
        self.logger.info(f"constructing blip dataset.")

        self.number_of_events = 0
        self.root = self.config["root"]
        self.skip_processing = self.config["skip_processing"]
        if self.skip_processing:
            if os.path.isdir('processed/'):
                for path in os.listdir('processed/'):
                    if os.path.isfile(os.path.join('processed/', path)):
                        self.number_of_events += 1
            self.logger.info(f'found {self.number_of_events} processed files.')
        self.meta = {}
        self.configure_dataset()
        self.configure_variables()
        self.configure_meta()
        self.configure_clustering()
        self.configure_weights()
        self.configure_transforms()

        GenericDataset.__init__(self)
        InMemoryDataset.__init__(self,
            self.root, self.transform, 
            self.pre_transform, self.pre_filter, 
            log=False
        )

    def configure_meta(self):
        # get meta dictionaries from files
        temp_arrakis_meta = []
        for input_file in self.dataset_files:
            try:
                data = np.load(input_file, allow_pickle=True)
                temp_arrakis_meta.append(data['meta'].item())
            except:
                self.logger.error(f'error reading file "{input_file}"!')
        try:
            self.meta['features'] = temp_arrakis_meta[0]['features']
            self.meta['classes'] = temp_arrakis_meta[0]['classes']
            self.meta['clusters'] = temp_arrakis_meta[0]['clusters']
            self.meta['hits'] = temp_arrakis_meta[0]['hits']
            self.meta['source_labels'] = temp_arrakis_meta[0]['source_labels']
            self.meta['topology_labels'] = temp_arrakis_meta[0]['topology_labels']
            self.meta['particle_labels'] = temp_arrakis_meta[0]['particle_labels']
            self.meta['physics_labels'] = temp_arrakis_meta[0]['physics_labels']
            self.meta['source_points'] = temp_arrakis_meta[0]['source_points']
            self.meta['topology_points'] = temp_arrakis_meta[0]['topology_points']
            self.meta['particle_points'] = temp_arrakis_meta[0]['particle_points']
            self.meta['total_points'] = temp_arrakis_meta[0]['total_points']
            self.meta['adc_view_sum'] = [temp_arrakis_meta[0]['adc_view_sum']]
        except:
            self.logger.error(f'error collecting meta information from arrakis file {input_file}!')
        
        # Check that meta info is consistent over the different files
        for ii in range(len(temp_arrakis_meta)-1):
            if self.meta['features'] != temp_arrakis_meta[ii+1]['features']:
                self.logger.error(f'conflicting meta information found in file {self.dataset_files[0]} and {self.dataset_files[ii+1]}')
            if self.meta['classes'] != temp_arrakis_meta[ii+1]['classes']:
                self.logger.error(f'conflicting meta information found in file {self.dataset_files[0]} and {self.dataset_files[ii+1]}')
            if self.meta['clusters'] != temp_arrakis_meta[ii+1]['clusters']:
                self.logger.error(f'conflicting meta information found in file {self.dataset_files[0]} and {self.dataset_files[ii+1]}')
            if self.meta['hits'] != temp_arrakis_meta[ii+1]['hits']:
                self.logger.error(f'conflicting meta information found in file {self.dataset_files[0]} and {self.dataset_files[ii+1]}')
            if self.meta['source_labels'] != temp_arrakis_meta[ii+1]['source_labels']:
                self.logger.error(f'conflicting meta information found in file {self.dataset_files[0]} and {self.dataset_files[ii+1]}')
            if self.meta['topology_labels'] != temp_arrakis_meta[ii+1]['topology_labels']:
                self.logger.error(f'conflicting meta information found in file {self.dataset_files[0]} and {self.dataset_files[ii+1]}')
            if self.meta['particle_labels'] != temp_arrakis_meta[ii+1]['particle_labels']:
                self.logger.error(f'conflicting meta information found in file {self.dataset_files[0]} and {self.dataset_files[ii+1]}')
            for key in self.meta['source_points'].keys():
                self.meta['source_points'][key] += temp_arrakis_meta[ii+1]['source_points'][key]
            for key in self.meta['topology_points'].keys():
                self.meta['topology_points'][key] += temp_arrakis_meta[ii+1]['topology_points'][key]
            for key in self.meta['particle_points'].keys():
                self.meta['particle_points'][key] += temp_arrakis_meta[ii+1]['particle_points'][key]
            self.meta['total_points'] += temp_arrakis_meta[ii+1]['total_points']
            self.meta['adc_view_sum'].append(temp_arrakis_meta[ii+1]['adc_view_sum'])

        self.meta['features_names'] = list(self.meta['features'].keys())
        self.meta['features_values'] = list(self.meta['features'].values())
        self.meta['features_names_by_value'] = {val: key for key, val in self.meta['features'].items()}
        self.meta['classes_names'] = list(self.meta['classes'].keys())
        self.meta['classes_values'] = list(self.meta['classes'].values())
        self.meta['classes_names_by_value'] = {val: key for key, val in self.meta['classes'].items()}
        self.meta['clusters_names'] = list(self.meta['clusters'].keys())
        self.meta['clusters_values'] = list(self.meta['clusters'].values())
        self.meta['clusters_names_by_value'] = {val: key for key, val in self.meta['clusters'].items()}
        self.meta['hits_names'] = list(self.meta['hits'].keys())
        self.meta['hits_values'] = list(self.meta['hits'].values())
        self.meta['hits_names_by_value'] = {val: key for key, val in self.meta['hits'].items()}
        self.meta['classes_labels_names'] = {
            'source':   list(self.meta['source_labels'].values()),
            'topology':    list(self.meta['topology_labels'].values()),
            'particle': list(self.meta['particle_labels'].values()),
            'physics': list(self.meta['physics_labels'].values())
        }
        self.meta['classes_labels_values'] = {
            'source':   list(self.meta['source_labels'].keys()),
            'topology':    list(self.meta['topology_labels'].keys()),
            'particle': list(self.meta['particle_labels'].keys()),
            'physics': list(self.meta['physics_labels'].keys())
        }
        self.meta['classes_labels_names_by_value'] = {
            'source':   {key: val for key, val in self.meta['source_labels'].items()},
            'topology':    {key: val for key, val in self.meta['topology_labels'].items()},
            'particle': {key: val for key, val in self.meta['particle_labels'].items()},
            'physics': {key: val for key, val in self.meta['physics_labels'].items()}
        }
        self.meta['classes_labels_values_by_name'] = {
            'source':   {val: key for key, val in self.meta['source_labels'].items()},
            'topology':    {val: key for key, val in self.meta['topology_labels'].items()},
            'particle': {val: key for key, val in self.meta['particle_labels'].items()},
            'physics': {val: key for key, val in self.meta['physics_labels'].items()}
        }

        # Check that config variables match meta info
        for position in self.meta['blip_positions']:
            if position not in self.meta['features']:
                self.logger.error(f'specified position "{position}" variable not in arrakis meta!')
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
        for ii, clusters in enumerate(self.meta['blip_clusters']):
            if clusters not in self.meta['clusters']:
                self.logger.error(f'specified clusters "{clusters}" variable not in arrakis meta!')

        if "hits" in self.config:
            for ii, hits in enumerate(self.meta['blip_hits']):
                if hits not in self.meta['hits']:
                    self.logger.error(f'specified hits "{hits}" variable not in arrakis meta!')

        # Set up maps for positions, features, etc.
        try:
            self.meta['blip_position_indices'] = [
                self.meta["features"][position] 
                for position in self.meta['blip_positions']
            ]
            self.meta['blip_positions_indices_by_name'] = {
                position: ii
                for ii, position in enumerate(self.meta['blip_positions'])
            }
        except:
            self.logger.error(f'failed to get position indices from meta!')
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
            self.meta['blip_hits_indices'] = [
                self.meta["hits"][label]
                for label in self.meta['blip_hits']
            ]
            self.meta['blip_hits_indices_by_name'] = {
                hits: ii
                for ii, hits in enumerate(self.meta['blip_hits'])
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
        try:
            self.meta['blip_clusters_indices'] = [
                self.meta["clusters"][label]
                for label in self.meta['blip_clusters']
            ]
            self.meta['blip_clusters_indices_by_name'] = {
                cluster: ii
                for ii, cluster in enumerate(self.meta['blip_clusters'])
            }
        except:
            self.logger.error(f'failed to get clusters indices from meta!')

        # Configure masks for classes and corresponding labels.  
        if "classes_mask" in self.config:
            self.meta['blip_classes_mask_indices'] = {
                classes: self.meta['classes'][classes]
                for classes in self.meta['blip_classes_mask']
            }
        if "labels_mask" in self.config:
            self.meta['blip_classes_labels_mask_values'] = {
                classes: [
                    self.meta['classes_labels_values_by_name'][classes][label]
                    for label in self.meta['blip_labels_mask'][ii]
                ]
                for ii, classes in enumerate(self.meta['blip_classes_mask'])
            }

    def configure_dataset(self):
        # set dataset type
        if "dataset_type" not in self.config.keys():
            self.logger.error(f'no dataset_type specified in config!')
        self.dataset_type = self.config["dataset_type"]
        if self.dataset_type == 'voxel':
            self.meta['position_type'] = torch.int
        else:
            self.meta['position_type'] = torch.float
        self.meta['feature_type'] = torch.float
        self.meta['class_type'] = torch.long
        self.meta['cluster_type'] = torch.long
        self.meta['hit_type'] = torch.float

        self.logger.info(f"setting 'dataset_type: {self.dataset_type}.")

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
        self.meta['blip_positions'] = self.config["positions"]
        self.meta['blip_features'] = self.config["features"]
        self.meta['blip_classes'] = self.config["classes"]

        if "labels" in self.config:
            self.meta['blip_labels'] = self.config["labels"]
        else:
            self.meta['blip_labels'] = [[] for ii in range(len(self.meta['blip_classes']))]

        if "clusters" in self.config:
            self.meta['blip_clusters'] = self.config["clusters"]
            self.logger.info(f"setting 'clusters':      {self.meta['blip_clusters']}")
        else:
            self.meta['blip_clusters'] = None

        if "hits" in self.config:
            self.meta['blip_hits'] = self.config["hits"]
            self.logger.info(f"setting 'hits':      {self.meta['blip_hits']}")
        else:
            self.meta['blip_hits'] = None

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
        
        self.normalized = self.config["normalized"]

        self.logger.info(f"setting 'positions':     {self.meta['blip_positions']}.")
        self.logger.info(f"setting 'features':      {self.meta['blip_features']}.")
        self.logger.info(f"setting 'classes':       {self.meta['blip_classes']}.")
        self.logger.info(f"setting 'consolidate_classes':   {self.meta['consolidate_classes']}")
        self.logger.info(f"setting 'sample_weights':{self.meta['sample_weights']}.")
        self.logger.info(f"setting 'class_weights': {self.class_weights}.")
        self.logger.info(f"setting 'normalize':     {self.normalized}.")

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

    def configure_clustering(self):
        if self.dataset_type != "cluster":
            return
        self.dbscan_min_samples = self.config["dbscan_min_samples"]
        self.dbscan_eps = self.config["dbscan_eps"]
        self.meta['clustering_positions'] = self.config["cluster_positions"]
        if 'cluster_category_type' in self.config.keys():
            self.meta['cluster_category_type'] = self.config['cluster_category_type']
        else:
            self.meta['cluster_category_type'] = 'segmentation'
        self.meta['cluster_position_indices'] = [
            self.meta['blip_positions_indices_by_name'][position] 
            for position in self.meta['clustering_positions']
        ]
        self.logger.info(f"setting 'dbscan_min_samples': {self.dbscan_min_samples}.")
        self.logger.info(f"setting 'dbscan_eps': {self.dbscan_eps}.")
        self.logger.info(f"setting 'cluster_positions': {self.meta['clustering_positions']}")

        self.dbscan = DBSCAN(
            eps=self.dbscan_eps, 
            min_samples=self.dbscan_min_samples
        )

    def configure_weights(self):
        # set up weights
        if self.meta['sample_weights'] != None:
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

    def apply_event_masks(self,
        event_features, event_classes, event_clusters, event_hits
    ):
        mask = np.array([True for ii in range(len(event_features))])
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
        event_features = event_features[mask]
        event_classes = event_classes[mask]
        event_clusters = event_clusters[mask]
        event_hits = event_hits[mask]

        # Separate positions and features
        event_positions = event_features[:, self.meta['blip_position_indices']]
        if len(self.meta['blip_features_indices']) != 0:
            event_features = event_features[:, self.meta['blip_features_indices']]
        else:
            event_features = np.ones((len(event_features),1))

        # Convert class labels to ordered list
        for classes in self.meta['blip_classes']:
            class_index = self.meta["classes"][classes]
            for key, val in self.meta['blip_labels_values_map'][classes].items():
                event_classes[:, class_index][(event_classes[:, class_index] == key)] = val
        event_classes = event_classes[:, self.meta['blip_classes_indices']]
        event_clusters = event_clusters[:, self.meta['blip_clusters_indices']]
        event_hits = event_hits[:, self.meta['blip_hits_indices']]

        # Grab indices of interest
        return event_positions, event_features, event_classes, event_clusters, event_hits, mask

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
        self.meta['input_events'] = {
            raw_path: []
            for raw_path in self.dataset_files
        }
        self.meta['event_mask'] = {
            raw_path: []
            for raw_path in self.dataset_files
        }
        self.meta['cluster_indices'] = {
            raw_path: []
            for raw_path in self.dataset_files
        }
        self.meta['cluster_ids'] = {
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
            clusters = data['clusters']
            hits = data['hits']
            # Iterate over all events in this file
            for ii in range(len(features)):
                # gather event features and classes
                event_features = features[ii]
                event_classes = classes[ii]
                event_clusters = clusters[ii]
                event_hits = hits[ii]
                if self.dataset_type == "cluster":
                    self.process_cluster(
                        event_features, event_classes, 
                        event_clusters, event_hits, raw_path
                    )
                elif self.dataset_type == "voxel":
                    self.process_voxel(
                        event_features, event_classes, 
                        event_clusters, event_hits, raw_path
                    )
        self.number_of_events = self.index
        self.logger.info(f"processed {self.number_of_events} events.")

    def process_cluster(self,
        event_features,
        event_classes,
        event_clusters,
        event_hits,
        raw_path
    ):
        event_positions, event_features, event_classes, event_clusters, event_hits, mask = self.apply_event_masks(
            event_features, event_classes, event_clusters, event_hits
        )
        # # check if classes need to be consolidated
        # if self.meta['consolidate_classes'] is not None:
        #     event_classes = self.consolidate_class(classes[ii])
        # else:
        #     event_classes = classes[ii]

        # create clusters using DBSCAN
        if np.sum(mask) == 0:
            return
        cluster_labels = self.dbscan.fit(
            event_positions[:, self.meta['cluster_position_indices']]
        ).labels_
        unique_labels = np.unique(cluster_labels)

        self.meta['event_mask'][raw_path].append(mask)
        self.meta['cluster_ids'][raw_path].append(cluster_labels)
        input_events = []
        cluster_indices = []

        # for each unique cluster label, 
        # create a separate dataset.
        for kk in unique_labels:
            if kk == -1:
                continue
            cluster_mask = (cluster_labels == kk)
            if np.sum(cluster_mask) == 0:
                continue
            cluster_positions = event_positions[cluster_mask]
            cluster_features = event_features[cluster_mask]
            cluster_classes = event_classes[cluster_mask]
            cluster_clusters = event_clusters[cluster_mask]
            if self.meta['cluster_category_type'] == 'classification':
                cluster_classes = [
                    np.bincount(cluster_classes[:, ll]).argmax()
                    for ll in range(len(self.meta['blip_classes_indices']))
                ]

            # Normalize cluster
            min_positions = np.min(cluster_positions, axis=0)
            max_positions = np.max(cluster_positions, axis=0)
            scale = max_positions - min_positions
            scale[(scale == 0)] = max_positions[(scale == 0)]
            summed_adc = np.sum(cluster_positions[:,2])
            cluster_positions = 2 * (cluster_positions - min_positions) / scale - 1

            event = Data(
                pos=torch.tensor(cluster_positions).type(self.meta['position_type']),
                x=torch.tensor(cluster_features).type(self.meta['feature_type']),
                category=torch.tensor(cluster_classes).type(self.meta['class_type']),
                clusters = torch.tensor(cluster_clusters).type(self.meta['cluster_type']),
                summed_adc = torch.tensor(summed_adc).type(torch.float),
                # Cluster ID is unique to clustering events
                cluster_id=kk
            )

            if self.pre_filter is not None:
                event = self.pre_filter(event)
            if self.pre_transform is not None:
                event = self.pre_transform(event)

            torch.save(event, osp.join(self.processed_dir, f'data_{self.index}.pt'))
            cluster_indices.append(self.index)
            input_events.append(self.index)
            self.index += 1

        self.meta['input_events'][raw_path].append(input_events)
        self.meta['cluster_indices'][raw_path].append(cluster_indices)
                           
    def process_voxel(self,
        event_features,
        event_classes,
        event_clusters,
        event_hits,
        raw_path
    ):
        event_positions, event_features, event_classes, event_clusters, event_hits, mask = self.apply_event_masks(
            event_features, event_classes, event_clusters, event_hits
        )
        self.meta['event_mask'][raw_path].append(mask)
        # # check if classes need to be consolidated
        # if self.meta['consolidate_classes'] is not None:
        #     event_classes = self.consolidate_class(classes[ii])
        # else:
        #     event_classes = classes[ii]
        event = Data(
            pos=torch.tensor(event_positions).type(self.meta['position_type']),
            x=torch.tensor(event_features).type(self.meta['feature_type']),
            category=torch.tensor(event_classes).type(self.meta['class_type']),
            clusters=torch.tensor(event_clusters).type(self.meta['cluster_type']),
            hits=torch.tensor(event_hits).type(self.meta['hit_type'])
        )
        if self.pre_filter is not None:
            event = self.pre_filter(event)

        if self.pre_transform is not None:
            event = self.pre_transform(event)

        torch.save(event, osp.join(self.processed_dir, f'data_{self.index}.pt'))
        self.meta['input_events'][raw_path].append([self.index])
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
            events = [
                [ii for ii in list(indices) if ii in self.meta['input_events'][raw_path][jj]] 
                for jj in range(len(self.meta['input_events'][raw_path]))
            ]
            classes_prefix = ""
            if self.dataset_type == "cluster":
                classes_prefix = f"{self.dbscan_eps}_{self.dbscan_min_samples}_"
            output = {
                f"{classes_prefix}{classes}": [input_dict[classes][event] for event in events]
                for classes in input_dict.keys()
            }
            output['event_mask'] = self.meta['event_mask'][raw_path]
            output['blip_labels_values_map'] = self.meta['blip_labels_values_map']
            output['blip_labels_values_inverse_map'] = self.meta['blip_labels_values_inverse_map']
            if self.dataset_type == "cluster":
                output[f'{self.dbscan_eps}_{self.dbscan_min_samples}_cluster_ids'] = self.meta['cluster_ids'][raw_path]
                output[f'{self.dbscan_eps}_{self.dbscan_min_samples}_cluster_indices'] = self.meta['cluster_indices'][raw_path]
            # otherwise add the array and save
            loaded_arrays.update(output)
            # loaded_arrays.update(self.cluster_labels[raw_path])
            np.savez(
                raw_path,
                **loaded_arrays
            )

