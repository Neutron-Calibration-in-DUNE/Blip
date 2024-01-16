import os
import glob
import torch
import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from torch_geometric.data import InMemoryDataset

from blip.utils.logger import Logger

generic_config = {
    "name":             "default",
    "dataset_params":   None,
    'processed_directory':             "/local_data/",
    "transform":        None,
    "pre_transform":    None,
    "pre_filter":       None,
    "dataset_folder":   "/local_data/",
    "dataset_files":    [""],
    "variables": {
        "positions":    [],
        "features":     [],
        "classes":      [],
        "clusters":     [],
        "hits":         [],
        "positions_normalization":  [],
        "features_normalization":   [],
        "class_mask":   [""],
        "label_mask":   [[""]],
        # default 2.6 us dt per hit.
        # ~4 mm, different pixel pitches.
        "voxelization": []
    },
    "weights": {
        "class_weights":    [],
        "sample_weights":   [],
    },
    "clustering": {
        "dbscan_min_samples": 10,
        "dbscan_eps":         10.0,
        "cluster_positions":  [""],
        "cluster_category_type": "classification",
    },
}


class GenericDataset(InMemoryDataset):
    """
    Datasets are constructed with Data objects from torch geometric.
    Each example, or entry, in the dataset corresponds to a LArTPC event.
    Each of these events has a corresponding 'Data' object from pyg.
    The Data objects have the following set of attributes:

        x   - a 'n x d_f' array of node features.
        pos - a 'n x d_p' array of node positions.
        y   - a 'n x d_c' array of class labels.
        edge_index - a '2 x n_e' array of edge indices.
        edge_attr  - a 'n_e x d_e' array of edge_features.

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
    def __init__(
        self,
        name:   str = 'generic',
        config: dict = generic_config,
        meta:   dict = {}
    ):
        self.name = name + '_dataset'
        self.config = config
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(self.name, output="both",   file_mode="w")
        else:
            self.logger = Logger(self.name, level='warning', file_mode="w")

        self.meta['number_of_events'] = 0

        if "dataset_params" in self.config.keys():
            if self.config["dataset_params"] is not None:
                if os.path.isfile(self.config['dataset_params']):
                    self.logger.info(f'loading dataset_params from file {self.config["dataset_params"]}')
                    self.load_params(self.config['dataset_params'])
                else:
                    self.logger.error(f'problem loading dataset_params from file {self.config["dataset_params"]}!')

        self.process_generic_config()
        self.process_config()
        self.save_params()

        InMemoryDataset.__init__(
            self,
            self.processed_directory,
            self.transform,
            self.pre_transform,
            self.pre_filter,
            log=False
        )

    def save_params(self):
        with open(self.processed_directory + '/dataset.params', 'wb') as file:
            pickle.dump(self.config, file)

    def load_params(
        self,
        params_file
    ):
        with open(params_file, 'rb') as file:
            old_config = pickle.load(file)
            old_config.update(self.config)
            self.config = old_config

    def process_generic_config(self):
        self.process_root()
        self.process_skip_processing()
        self.process_transform()
        self.process_pre_transform()
        self.process_pre_filter()
        self.process_dataset_folder()
        self.process_dataset_files()
        self.process_meta()
        self.process_variables()
        self.process_consolidate_classes()
        self.process_maps()
        self.process_clustering()
        self.process_weights()
        self.process_voxelization()

    def process_root(self):
        # set up 'processed_directory' directory.  this is mainly for processed data.
        if 'processed_directory' in self.config.keys():
            if os.path.isdir(self.config['processed_directory']):
                self.processed_directory = self.config['processed_directory']
            else:
                self.logger.warn(
                    f'specified root directory {self.config["processed_directory"]} doesnt exist. attempting to create directory'
                )
                try:
                    os.makedirs(self.config['processed_directory'])
                except Exception as exception:
                    self.logger.warn(
                        f'attempt at making directory {self.config["processed_directory"]} failed.  setting root to /local_data/'
                    )
                    self.processed_directory = self.meta['local_data']
        else:
            self.processed_directory = self.meta['local_data']
        self.logger.info(f'set "processed_directory" directory to {self.processed_directory}')

    def process_skip_processing(self):
        # set skip_processing
        if "skip_processing" in self.config.keys():
            if not isinstance(self.config["skip_processing"], bool):
                self.logger.error(f'skip_processing set to {self.config["skip_processing"]}, but should be a bool!')
            else:
                self.skip_processing = self.config["skip_processing"]

        if self.skip_processing:
            if os.path.isdir(self.processed_directory + '/processed/'):
                for path in os.listdir(self.processed_directory + '/processed/'):
                    if 'data' in path and '.pt' in path:
                        self.meta['number_of_events'] += 1
            self.logger.info(f"found {self.meta['number_of_events']} processed files.")

    def process_transform(self):
        # set transform
        if "transform" in self.config.keys():
            self.transform = self.config["transform"]
        else:
            self.transform = None
        self.logger.info(f'setting transform to {self.transform}')

    def process_pre_transform(self):
        # set pre_transform
        if "pre_transform" in self.config.keys():
            self.pre_transform = self.config["pre_transform"]
        else:
            self.pre_transform = None
        self.logger.info(f'setting pre_transform to {self.pre_transform}')

    def process_pre_filter(self):
        # set pre_filter
        if "pre_filter" in self.config.keys():
            self.pre_filter = self.config["pre_filter"]
        else:
            self.pre_filter = None
        self.logger.info(f'setting pre_filter to {self.pre_filter}')

    def process_dataset_folder(self):
        # default to what's in the configuration file. May decide to deprecate in the future
        if ("dataset_folder" in self.config.keys()):
            self.dataset_folder = self.config["dataset_folder"]
            self.logger.info(
                "Set dataset path from Configuration." +
                f" dataset_folder: {self.dataset_folder}"
            )
        elif ('BLIP_DATASET_PATH' in os.environ):
            self.logger.debug('Found BLIP_DATASET_PATH in environment')
            self.dataset_folder = os.environ['BLIP_DATASET_PATH']
            self.logger.info(
                "Setting dataset path from Enviroment." +
                f" BLIP_DATASET_PATH = {self.dataset_folder}"
            )
        else:
            self.logger.error('No dataset_folder specified in environment or configuration file!')
        if not os.path.isdir(self.dataset_folder):
            self.logger.error(f'Specified dataset folder "{self.dataset_folder}" does not exist!')

    def process_dataset_files(self):
        if "dataset_files" not in self.config.keys():
            self.logger.error('no dataset_files specified in config!')
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
                    self.logger.info(
                        f'searching {self.dataset_folder} recursively for all {self.config["dataset_files"]} files.')
                    self.dataset_files = glob.glob(self.dataset_folder + f'**/{self.config["dataset_files"]}', recursive=True)
                except Exception as exception:
                    self.logger.error(f'specified "dataset_files" parameter: {self.config["dataset_files"]} incompatible!')
        else:
            self.logger.error(f'specified "dataset_files" parameter: {self.config["dataset_files"]} incompatible!')

    def process_meta(self):
        # get meta dictionaries from files
        self.meta_consistency = True
        temp_meta = []
        for input_file in self.dataset_files:
            try:
                data = np.load(input_file, allow_pickle=True)
                temp_meta.append(data['meta'].item())
            except Exception as exception:
                self.logger.error(f'error reading file "{input_file}"!')
        try:
            for key, value in temp_meta[0].items():
                if "created" in key:
                    continue
                self.meta[key] = value
        except Exception as exception:
            self.logger.error(f'error collecting meta information from file {self.dataset_files[0]}!')

        # Check that meta info is consistent over the different files
        for ii in range(len(temp_meta)-1):
            for key, value in temp_meta[0].items():
                if key not in temp_meta[ii].keys():
                    self.logger.warn(f'meta item {key} in {self.dataset_files[0]} not in meta of {self.dataset_files[ii]}!')
                else:
                    if "points" in key:
                        if "total" in key:
                            self.meta[key] += temp_meta[ii][key]
                        else:
                            for label, points in temp_meta[ii][key].items():
                                self.meta[key][label] += points
                    elif "events" in key:
                        continue
                    else:
                        if isinstance(value, list):
                            for item in value:
                                if item not in temp_meta[ii][key]:
                                    self.logger.warn(
                                        f'meta item {key}:{value} in {self.dataset_files[0]} not equal to meta' +
                                        f' item {key}:{temp_meta[ii][key]} of {self.dataset_files[ii]}!'
                                    )

        # arange dictionaries for label<->value<->index maps
        for data_type in ["edep_features", "view_features", "det_features", "mc_features", "features", "clusters", "hits"]:
            if (f"{data_type}" in self.meta.keys()):
                self.meta[f'{data_type}_names'] = list(self.meta[f'{data_type}'].keys())
                self.meta[f'{data_type}_values'] = list(self.meta[f'{data_type}'].values())
                self.meta[f'{data_type}_names_by_value'] = {
                    val: key for key, val in self.meta[f'{data_type}'].items()
                }
        if "features" not in self.meta.keys():
            if "edep_features" in self.meta.keys() and "view_features" in self.meta.keys():
                self.meta['features'] = {**self.meta['edep_features'], **self.meta['view_features']}
                self.meta['features_names'] = self.meta['edep_features_names'] + self.meta['view_features_names']
                self.meta['features_values'] = self.meta['edep_features_values'] + self.meta['view_features_values']
                self.meta['features_names_by_value'] = {
                    **self.meta['edep_features_names_by_value'], **self.meta['view_features_names_by_value']
                }
            elif "edep_features" in self.meta.keys():
                self.meta['features'] = self.meta['edep_features']
                self.meta['features_names'] = self.meta['edep_features_names']
                self.meta['features_values'] = self.meta['edep_features_values']
                self.meta['features_names_by_value'] = {
                    **self.meta['edep_features_names_by_value']
                }
            elif "det_features" in self.meta.keys():
                self.meta['features'] = self.meta['det_features']
                self.meta['features_names'] = self.meta['det_features_names']
                self.meta['features_values'] = self.meta['det_features_values']
                self.meta['features_names_by_value'] = {
                    **self.meta['det_features_names_by_value']
                }
        self.meta['classes_names'] = list(self.meta['classes'].keys())
        self.meta['classes_values'] = list(self.meta['classes'].values())
        self.meta['classes_names_by_value'] = {val: key for key, val in self.meta['classes'].items()}

        # arange label maps.  if we don't want undefined and noise then
        # don't include them in these maps and they won't appear downstream.
        if "skip_undefined" not in self.config.keys():
            self.logger.warn('skip_undefined not specified in config! setting to "True"')
            self.config["skip_undefined"] = True
        self.meta['skip_undefined'] = self.config['skip_undefined']
        if self.meta["skip_undefined"]:
            for label in self.meta['classes'].keys():
                del self.meta[f'{label}_labels'][-1]
                del self.meta[f'{label}_labels'][0]

        self.meta['classes_labels_names'] = {
            label:   list(self.meta[f'{label}_labels'].values())
            for label in self.meta['classes'].keys()
        }
        self.meta['classes_labels_values'] = {
            label:   list(self.meta[f'{label}_labels'].keys())
            for label in self.meta['classes'].keys()
        }
        self.meta['classes_labels_names_by_value'] = {
            label:   {key: val for key, val in self.meta[f'{label}_labels'].items()}
            for label in self.meta['classes'].keys()
        }
        self.meta['classes_labels_values_by_name'] = {
            label:   {val: key for key, val in self.meta[f'{label}_labels'].items()}
            for label in self.meta['classes'].keys()
        }

    def process_variables(self):
        # set positions, features and classes
        if "variables" not in self.config.keys():
            self.logger.error("variables section not specified in config!")
        variables_config = self.config["variables"]

        if "positions" not in variables_config.keys():
            self.logger.error("variables:positions not specified in config!")
        self.meta['blip_positions'] = variables_config["positions"]
        self.logger.info(f"setting 'positions':     {self.meta['blip_positions']}.")
        for position in self.meta['blip_positions']:
            if position not in self.meta['features']:
                self.logger.error(f'specified position "{position}" variable not in arrakis meta!')

        if "classes" not in variables_config.keys():
            self.logger.error("variables:classes not specified in config!")
        self.meta['blip_classes'] = variables_config["classes"]
        self.logger.info(f"setting 'classes':      {self.meta['blip_classes']}.")
        for ii, classes in enumerate(self.meta['blip_classes']):
            if classes not in self.meta['classes']:
                self.logger.error(f'specified classes "{classes}" variable not in arrakis meta!')

        if "features" in variables_config.keys():
            self.meta['blip_features'] = variables_config["features"]
            for feature in self.meta['blip_features']:
                if feature not in self.meta['features']:
                    self.logger.error(f'specified feature "{feature}" variable not in arrakis meta!')
        else:
            self.meta['blip_features'] = None
        self.logger.info(f"setting 'features':      {self.meta['blip_features']}.")

        if "labels" in variables_config.keys():
            self.meta['blip_labels'] = variables_config["labels"]
            for ii, classes in enumerate(self.meta['blip_classes']):
                if len(self.meta['blip_labels']) != 0:
                    for label in self.meta['blip_labels'][ii]:
                        if label not in self.meta['classes_labels_names'][classes]:
                            self.logger.error(f'specified label "{classes}:{label}" not in arrakis meta!')
        else:
            self.meta['blip_labels'] = [[] for ii in range(len(self.meta['blip_classes']))]
        self.logger.info(f"setting 'labels':      {self.meta['blip_labels']}.")

        if "clusters" in variables_config.keys():
            self.meta['blip_clusters'] = variables_config["clusters"]
            for ii, clusters in enumerate(self.meta['blip_clusters']):
                if clusters not in self.meta['clusters']:
                    self.logger.error(f'specified clusters "{clusters}" variable not in arrakis meta!')
        else:
            self.meta['blip_clusters'] = None
        self.logger.info(f"setting 'clusters':      {self.meta['blip_clusters']}.")

        if "hits" in variables_config.keys():
            self.meta['blip_hits'] = variables_config["hits"]
            for ii, hits in enumerate(self.meta['blip_hits']):
                if hits not in self.meta['hits']:
                    self.logger.error(f'specified hits "{hits}" variable not in arrakis meta!')
        else:
            self.meta['blip_hits'] = None
        self.logger.info(f"setting 'hits':      {self.meta['blip_hits']}.")

        if "consolidate_classes" in variables_config.keys():
            self.meta['consolidate_classes'] = variables_config["consolidate_classes"]
        else:
            self.meta['consolidate_classes'] = None
        self.logger.info(f"setting 'consolidate_classes':   {self.meta['consolidate_classes']}")

        if "classes_mask" in variables_config.keys():
            self.meta['blip_classes_mask'] = variables_config["classes_mask"]
        else:
            self.meta['blip_classes_mask'] = []
        self.logger.info(f"setting 'classes_mask':    {self.meta['blip_classes_mask']}.")

        if "labels_mask" in variables_config.keys():
            self.meta['blip_labels_mask'] = variables_config["labels_mask"]
        else:
            self.meta['blip_labels_mask'] = []
        self.logger.info(f"setting 'labels_mask':    {self.meta['blip_labels_mask']}.")

    def process_consolidate_classes(self):
        # determine if the list of class labels contains everything from the dataset list.
        # first, we check if [""] is an entry in the consolidation list, and if so, replace it with
        # the left over classes which are not mentioned.
        if self.meta['consolidate_classes'] is not None:
            for label in self.meta["classes_labels_names"]:
                all_labels = list(self.meta["classes_labels_names"][label].values())
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
                            "consolidate_classes does not contain an exhaustive" +
                            f" list for label '{label}'!  Perhaps you forgot to include" +
                            f" the ['']? Leftover classes are {all_labels}."
                        )
            # now we create a map from old indices to new
            self.consolidation_map = {
                label: {}
                for label in self.meta['blip_classes']
            }
            for label in self.meta["classes_labels_names"]:
                for key, val in self.meta["classes_labels_names"][label].items():
                    for jj, labels in enumerate(self.meta['consolidate_classes'][label]):
                        if val in labels:
                            self.consolidation_map[label][key] = jj

    def process_maps(self):
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
        except Exception as exception:
            self.logger.error('failed to get position indices from meta!')
        try:
            self.meta['blip_features_indices'] = [
                self.meta["features"][feature]
                for feature in self.meta['blip_features']
            ]
            self.meta['blip_features_indices_by_name'] = {
                feature: ii
                for ii, feature in enumerate(self.meta['blip_features'])
            }
        except Exception as exception:
            self.logger.error('failed to get feature indices from meta!')
        try:
            self.meta['blip_classes_indices'] = [
                self.meta["classes"][label]
                for label in self.meta['blip_classes']
            ]
            self.meta['blip_classes_indices_by_name'] = {
                classes: ii
                for ii, classes in enumerate(self.meta['blip_classes'])
            }
        except Exception as exception:
            self.logger.error('failed to get classes indices from meta!')
        if "clusters" in self.config['variables'].keys():
            try:
                self.meta['blip_clusters_indices'] = [
                    self.meta["clusters"][label]
                    for label in self.meta['blip_clusters']
                ]
                self.meta['blip_clusters_indices_by_name'] = {
                    clusters: ii
                    for ii, clusters in enumerate(self.meta['blip_clusters'])
                }
            except Exception as exception:
                self.logger.error('failed to get clusters indices from meta!')
        if "hits" in self.config['variables'].keys():
            try:
                self.meta['blip_hits_indices'] = [
                    self.meta["hits"][label]
                    for label in self.meta['blip_hits']
                ]
                self.meta['blip_hits_indices_by_name'] = {
                    hits: ii
                    for ii, hits in enumerate(self.meta['blip_hits'])
                }
            except Exception as exception:
                self.logger.error('failed to get classes indices from meta!')
        try:
            self.meta['blip_labels_values'] = {}
            self.meta['blip_labels_values_map'] = {}
            self.meta['blip_labels_values_inverse_map'] = {}
            for ii, classes in enumerate(self.meta['blip_classes']):
                if len(self.meta['blip_labels']) == 0:
                    self.meta['blip_labels_values'][classes] = list(
                        self.meta['classes_labels_values_by_name'][classes].values()
                    )
                else:
                    if len(self.meta['blip_labels'][ii]) == 0:
                        self.meta['blip_labels_values'][classes] = list(
                            self.meta['classes_labels_values_by_name'][classes].values()
                        )
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
        except Exception as exception:
            self.logger.error('failed to arange classes labels from meta!')

        if "clusters" in self.config['variables'].keys():
            try:
                self.meta['blip_clusters_indices'] = [
                    self.meta["clusters"][label]
                    for label in self.meta['blip_clusters']
                ]
                self.meta['blip_clusters_indices_by_name'] = {
                    cluster: ii
                    for ii, cluster in enumerate(self.meta['blip_clusters'])
                }
            except Exception as exception:
                self.logger.error('failed to get clusters indices from meta!')

        # Configure masks for classes and corresponding labels.``
        if "classes_mask" in self.config['variables'].keys():
            self.meta['blip_classes_mask_indices'] = {
                classes: self.meta['classes'][classes]
                for classes in self.meta['blip_classes_mask']
            }
        else:
            self.meta['blip_classes_mask_indices'] = {}
        if "labels_mask" in self.config['variables'].keys():
            self.meta['blip_classes_labels_mask_values'] = {
                classes: [
                    self.meta['classes_labels_values_by_name'][classes][label]
                    for label in self.meta['blip_labels_mask'][ii]
                ]
                for ii, classes in enumerate(self.meta['blip_classes_mask'])
            }
        else:
            self.meta['blip_classes_labels_indices'] = {}

    def process_clustering(self):
        if "clustering" not in self.config.keys():
            self.logger.warn('no clustering section in config!')
            return
        clustering_config = self.config["clustering"]

        self.dbscan_min_samples = clustering_config["dbscan_min_samples"]
        self.dbscan_eps = clustering_config["dbscan_eps"]
        self.meta['clustering_positions'] = clustering_config["cluster_positions"]

        if 'cluster_category_type' in clustering_config.keys():
            self.meta['cluster_category_type'] = clustering_config['cluster_category_type']
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

    # TODO: How do we use sample weights here?
    def process_weights(self):
        # set up weights
        self.meta['use_sample_weights'] = False
        self.meta['sample_weights'] = None
        self.meta['use_class_weights'] = False
        self.meta['class_weights'] = None
        self.meta['class_weight_totals'] = None
        if "weights" not in self.config.keys():
            self.logger.info('weights section not specified in config.  setting weights to None')
            return

        weights_config = self.config["weights"]

        if "sample_weights" in weights_config.keys():
            self.meta['sample_weights'] = weights_config["sample_weights"]
        else:
            self.meta['sample_weights'] = False
        self.logger.info(f"setting 'sample_weights':{self.meta['sample_weights']}.")

        if "class_weights" in weights_config.keys():
            class_weights = weights_config["class_weights"]
            for classes in class_weights:
                if f"{classes}_points" not in self.meta:
                    self.logger.error(f'{classes}_points not in meta! cant set weights properly!')
            self.meta['class_weights'] = {
                key: torch.tensor(np.sum([
                    [self.meta[ii][f"{key}_points"][jj] for jj in self.meta[ii][f"{key}_points"].keys()]
                    for ii in range(len(self.meta))
                ], axis=0), dtype=torch.float)
                for key in class_weights
            }
            self.meta['class_weight_totals'] = {
                key: float(torch.sum(value))
                for key, value in self.meta['class_weights'].items()
            }
            for key, value in self.meta['class_weights'].items():
                for ii, val in enumerate(value):
                    if val != 0:
                        self.meta['class_weights'][key][ii] = self.meta['class_weight_totals'][key] / float(len(value) * val)
        self.logger.info(f"setting 'class_weights': {self.meta['class_weights']}.")

    def process_voxelization(self):
        if "voxelization" not in self.config["variables"]:
            return
        self.meta['voxelization'] = self.config["variables"]['voxelization']
        if len(self.meta['voxelization']) != len(self.meta['blip_positions']):
            self.logger.error(
                f'specified voxelization {self.meta["voxelization"]} not ' +
                f'equal number of positions {self.meta["blip_positions"]}'
            )
        self.meta['voxelized_duplicates'] = []

    def consolidate_class(
        self,
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
        return [f'data_{ii}.pt' for ii in range(self.meta['number_of_events'])]
        ...

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def process_config(self):
        self.logger.error('"process_config" function not implemented in Dataset!')

    def process(self):
        self.logger.error('"process" function not implement in Dataset!')
