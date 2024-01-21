

import torch
import os.path as osp
import numpy   as np
from tqdm import tqdm
from dataclasses import dataclass

from torch_geometric.data import Data

from blip.dataset.generic_dataset import GenericDataset
from blip.topology.merge_tree import MergeTree
from blip.dataset.common import *


blip_dataset_config = {
    "name":             "default",
    "root":             ".",
    "transform":        None,
    "pre_transform":    None,
    "pre_filter":       None,
    "dataset_type":     "wire_view",
    "dataset_folder":   "data/",
    "dataset_files":    [""],
    "view":             2,
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
        "voxelization": [],
    },
    "weights": {
        "class_weights":    [],
        "sample_weights":   [],
    },
    "clustering": {
        "cluster_method":     "unique_physics_meso",
        "dbscan_min_samples": 10,
        "dbscan_eps":         10.0,
        "cluster_positions":  [""],
        "cluster_category_type": "classification",
    },
}


@dataclass
class BlipData:
    """
    This class holds event level data that can be passed around within
    the BlipDataset functions.  Scalars should be a dictionary with
    keys and values on an event by event basis.
    """
    positions:  torch.tensor = None
    features:   torch.tensor = None
    classes:    torch.tensor = None
    clusters:   torch.tensor = None
    hits:       torch.tensor = None
    merge_tree: None = None
    mask:       None = None
    scalars:    dict = None
    raw_path:   str = ''

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class BlipDataset(GenericDataset):
    """
    """
    def __init__(
        self,
        name:   str = "blip",
        config: dict = blip_dataset_config,
        meta:   dict = {}
    ):
        GenericDataset.__init__(
            self, name, config, meta
        )

    def process_config(self):
        self.process_dataset_type()

    def process_dataset_type(self):
        # set dataset type
        if "dataset_type" not in self.config.keys():
            self.logger.error('no dataset_type specified in config!')

        self.meta['dataset_type'] = self.config["dataset_type"]
        self.logger.info(f"setting 'dataset_type: {self.meta['dataset_type']}.")

        if self.meta['dataset_type'] == 'wire_view':
            if "view" not in self.config.keys():
                self.logger.warn('view not specified in config for wire dataset! setting view=2!')
                self.config['view'] = 2
            self.meta['view'] = self.config['view']
            self.meta['position_type'] = torch.int
            self.features_name = f'view_{self.meta["view"]}_features'
            self.classes_name = f'view_{self.meta["view"]}_classes'
            self.clusters_name = f'view_{self.meta["view"]}_clusters'
            self.hits_name = f'view_{self.meta["view"]}_hits'
        elif self.meta['dataset_type'] == 'wire_view_cluster':
            if "view" not in self.config.keys():
                self.logger.warn('view not specified in config for wire dataset! setting view=2!')
                self.config['view'] = 2
            self.meta['view'] = self.config['view']
            self.meta['position_type'] = torch.float
            self.features_name = f'view_{self.meta["view"]}_features'
            self.classes_name = f'view_{self.meta["view"]}_classes'
            self.clusters_name = f'view_{self.meta["view"]}_clusters'
            self.hits_name = f'view_{self.meta["view"]}_hits'
        elif self.meta['dataset_type'] == 'wire_view_tree':
            if "view" not in self.config.keys():
                self.logger.warn('view not specified in config for wire dataset! setting view=2!')
                self.config['view'] = 2
            self.meta['view'] = self.config['view']
            self.meta['position_type'] = torch.float
            self.features_name = f'view_{self.meta["view"]}_features'
            self.classes_name = f'view_{self.meta["view"]}_classes'
            self.clusters_name = f'view_{self.meta["view"]}_clusters'
            self.hits_name = f'view_{self.meta["view"]}_hits'
        elif self.meta['dataset_type'] == 'tpc':
            self.meta['position_type'] = torch.int
            self.features_name = 'det_features'
            self.classes_name = 'classes'
            self.clusters_name = 'clusters'
            self.hits_name = None
        elif self.meta['dataset_type'] == 'tpc_cluster':
            self.meta['position_type'] = torch.int
            self.features_name = 'det_features'
            self.classes_name = 'classes'
            self.clusters_name = 'clusters'
            self.hits_name = None
        else:
            self.meta['position_type'] = torch.float

        self.meta['feature_type'] = torch.float
        self.meta['class_type'] = torch.long
        self.meta['cluster_type'] = torch.long
        self.meta['hit_type'] = torch.float
        self.logger.info(f'setting position_type to {self.meta["position_type"]}')
        self.logger.info(f'setting feature_type to {self.meta["feature_type"]}')
        self.logger.info(f'setting class_type to {self.meta["class_type"]}')

    def apply_event_masks(
        self,
        event_data: BlipData = None,
    ):
        """
        Here we ...
        """
        mask = np.array([True for ii in range(len(event_data["features"]))])
        # Apply mask for 'labels'
        for classes in self.meta['blip_classes']:
            class_index = self.meta["classes"][classes]
            for jj, label_value in enumerate(self.meta['blip_labels_values'][classes]):
                mask |= (event_data["classes"][:, class_index] == label_value)
        if "classes_mask" in self.config["variables"]:
            # Apply 'classes_mask' and 'labels_mask'
            for classes, class_index in self.meta['blip_classes_mask_indices'].items():
                for jj, label_value in enumerate(self.meta['blip_classes_labels_mask_values'][classes]):
                    mask &= (event_data["classes"][:, class_index] == label_value)
        if self.meta['skip_undefined'] is True:
            for classes in self.meta['blip_classes']:
                class_index = self.meta["classes"][classes]
                mask &= (event_data["classes"][:, class_index] != -1)
                mask &= (event_data["classes"][:, class_index] != 0)

        # Apply masks
        event_data["features"] = event_data["features"][mask].astype(np.float)
        event_data["classes"] = event_data["classes"][mask].astype(np.int64)
        event_data["clusters"] = event_data["clusters"][mask].astype(np.int64)
        if event_data["hits"] is not None:
            event_data["hits"] = event_data["hits"][mask].astype(np.float)

        # Separate positions and features
        event_data["positions"] = event_data["features"][:, self.meta['blip_position_indices']]
        if len(self.meta['blip_features_indices']) != 0:
            event_data["features"] = event_data["features"][:, self.meta['blip_features_indices']]
        else:
            event_data["features"] = np.ones((len(event_data["features"]), 1))

        # Convert class labels to ordered list
        temp_classes = event_data["classes"].copy()
        for classes in self.meta['blip_classes']:
            class_index = self.meta["classes"][classes]
            for key, val in self.meta['blip_labels_values_map'][classes].items():
                temp_mask = (temp_classes[:, class_index] == key)
                event_data["classes"][temp_mask, class_index] = val
        event_data["classes"] = event_data["classes"][:, self.meta['blip_classes_indices']]
        event_data["clusters"] = event_data["clusters"][:, self.meta['blip_clusters_indices']]

        if event_data["hits"] is not None:
            event_data["hits"] = event_data["hits"][:, self.meta['blip_hits_indices']]

        event_data["mask"] = mask

        return event_data

    def apply_voxelization(
        self,
        event_data: BlipData = None
    ):
        voxelized_positions = np.round(event_data["positions"] * 10 / self.meta['voxelization'])
        unique_elements, inverse_indices = np.unique(voxelized_positions, return_inverse=True, axis=0)
        unique_counts = np.bincount(inverse_indices)
        duplicate_indices = np.where(unique_counts > 1)[0]
        duplicates = []
        elements_to_remove = []
        for idx in duplicate_indices:
            indices = np.where(inverse_indices == idx)[0]
            duplicates.append(indices.tolist())
            elements_to_remove.append(indices[1:])
            max_feature_index = np.argmax(event_data["features"][indices])
            event_data["features"][indices[0]] = event_data["features"][max_feature_index]
            event_data["classes"][indices[0]] = event_data["classes"][max_feature_index]
            event_data["clusters"][indices[0]] = event_data["clusters"][max_feature_index]

        self.meta['voxelized_duplicates'].append(duplicates)
        if len(duplicates) > 0:
            elements_to_remove = np.concatenate(elements_to_remove)
            mask = np.ones(len(event_data["positions"]), dtype=bool)
            mask[elements_to_remove] = False
            event_data["positions"] = voxelized_positions[mask]
            event_data["features"] = event_data["features"][mask]
            event_data["classes"] = event_data["classes"][mask]
            event_data["clusters"] = event_data["clusters"][mask]

        return event_data

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
            self.logger.info('skipping processing of data.')
            return
        self.skipped_events = []
        self.index = 0
        self.logger.info(f"processing {len(self.dataset_files)} files.")
        merge_tree = MergeTree(
            self.name,
            meta=self.meta
        )

        dataset_file_loop = tqdm(
            enumerate(self.dataset_files, 0),
            total=len(self.dataset_files),
            leave=False,
            position=0,
            colour='blue'
        )
        for jj, raw_path in dataset_file_loop:
            data = np.load(raw_path, allow_pickle=True)
            features = data[self.features_name]
            classes = data[self.classes_name]
            clusters = data[self.clusters_name]
            if self.hits_name is not None:
                hits = data[self.hits_name]
            else:
                hits = None
            for ii in range(len(features)):
                event_data = BlipData(
                    features=features[ii],
                    classes=classes[ii],
                    clusters=clusters[ii],
                    raw_path=raw_path
                )
                if hits is not None:
                    event_data["hits"] = hits[ii]

                event_data = self.apply_event_masks(event_data)
                if not np.any(event_data["mask"]):
                    self.skipped_events.append(ii)
                    continue

                if "voxelization" in self.meta.keys():
                    event_data = self.apply_voxelization(event_data)

                self.meta['event_mask'][event_data["raw_path"]].append(event_data["mask"])

                if self.meta["merge_tree"]:
                    event_data["merge_tree"] = merge_tree.create_merge_tree(event_data["positions"])
                else:
                    event_data["merge_tree"] = None

                if self.meta['dataset_type'] in ['wire_view', 'tpc', 'edep', 'segment']:
                    self.process_event(event_data)
                elif self.meta['dataset_type'] in ['wire_view_cluster', 'tpc_cluster', 'edep_cluster', 'segment_cluster']:
                    self.process_event_cluster(event_data)

            dataset_file_loop.set_description("Processing BlipDataset")
            dataset_file_loop.set_postfix_str(f"file={raw_path}")
        self.meta['number_of_events'] = self.index
        self.logger.info(f"processed {self.meta['number_of_events']} events.")

    def process_event_cluster(
        self,
        event_data: BlipData = None
    ):
        # create clusters using DBSCAN
        if self.cluster_method == "dbscan":
            cluster_labels = self.dbscan.fit(
                event_data["positions"][:, self.meta['cluster_position_indices']]
            ).labels_
            unique_labels = np.unique(cluster_labels)
        else:
            cluster_labels = event_data["clusters"][:, self.meta["blip_clusters_indices_by_name"][self.cluster_method]]
            unique_labels = np.unique(cluster_labels)

        self.meta['cluster_ids'][event_data["raw_path"]].append(cluster_labels)
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
            cluster_positions = event_data["positions"][cluster_mask]
            cluster_features = event_data["features"][cluster_mask]
            cluster_classes = event_data["classes"][cluster_mask]
            cluster_clusters = event_data["clusters"][cluster_mask]
            if self.meta['cluster_category_type'] == 'classification':
                cluster_classes = [[
                    np.bincount(cluster_classes[:, ll]).argmax()
                    for ll in range(len(self.meta['blip_classes_indices']))
                ]]

            # Normalize cluster
            min_positions = np.min(cluster_positions, axis=0)
            max_positions = np.max(cluster_positions, axis=0)
            scale = max_positions - min_positions
            scale[(scale == 0)] = max_positions[(scale == 0)]
            cluster_positions = 2 * (cluster_positions - min_positions) / scale - 1

            event = Data(
                pos=torch.tensor(cluster_positions).type(self.meta['position_type']),
                x=torch.tensor(cluster_features).type(self.meta['feature_type']),
                category=torch.tensor(cluster_classes).type(self.meta['class_type']),
                clusters=torch.tensor(cluster_clusters).type(self.meta['cluster_type']),
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

        self.meta['input_events'][event_data["raw_path"]].append(input_events)
        self.meta['cluster_indices'][event_data["raw_path"]].append(cluster_indices)

    def process_event(
        self,
        event_data: BlipData = None
    ):
        if event_data["hits"] is not None:
            event = Data(
                pos=torch.tensor(event_data["positions"]).type(self.meta['position_type']),
                x=torch.tensor(event_data["features"]).type(self.meta['feature_type']),
                category=torch.tensor(event_data["classes"]).type(self.meta['class_type']),
                clusters=torch.tensor(event_data["clusters"]).type(self.meta['cluster_type']),
                hits=torch.tensor(event_data["hits"]).type(self.meta['hit_type']),
                merge_tree=event_data["merge_tree"]
            )
        else:
            event = Data(
                pos=torch.tensor(event_data["positions"]).type(self.meta['position_type']),
                x=torch.tensor(event_data["features"]).type(self.meta['feature_type']),
                category=torch.tensor(event_data["classes"]).type(self.meta['class_type']),
                clusters=torch.tensor(event_data["clusters"]).type(self.meta['cluster_type']),
                hits=None,
                merge_tree=event_data["merge_tree"]
            )
        if self.pre_filter is not None:
            event = self.pre_filter(event)
        if self.pre_transform is not None:
            event = self.pre_transform(event)

        torch.save(event, osp.join(self.processed_dir, f'data_{self.index}.pt'))
        self.meta['input_events'][event_data["raw_path"]].append([self.index])
        self.index += 1

    def append_dataset_files(
        self,
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
            output['blip_labels_values_map'] = self.meta['blip_labels_values_map']
            output['blip_labels_values_inverse_map'] = self.meta['blip_labels_values_inverse_map']
            if self.meta['dataset_type'] == "cluster":
                output[f'{self.dbscan_eps}_{self.dbscan_min_samples}_cluster_ids'] = self.meta['cluster_ids'][raw_path]
                output[f'{self.dbscan_eps}_{self.dbscan_min_samples}_cluster_indices'] = self.meta['cluster_indices'][raw_path]
            # otherwise add the array and save
            loaded_arrays.update(output)
            # loaded_arrays.update(self.cluster_labels[raw_path])
            np.savez(
                raw_path,
                **loaded_arrays
            )
