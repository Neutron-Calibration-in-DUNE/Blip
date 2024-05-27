import torch
import h5py
import os
import glob
import numpy as np
from torch.utils.data import Dataset
from mpi4py import MPI
from tqdm import tqdm

from blip.utils.utils import profiler
from blip.utils.logger import BlipError


class BlipDataset(Dataset):
    """
    """
    def __init__(
        self,
        config: dict = {},
        meta:   dict = {}
    ):
        self.config = config
        self.meta = meta

        self.comm = MPI.COMM_WORLD

        self.label_values = {
            'topology': [0, 1, 2],
            'physics': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'vertex': [0, 1],
            'tracklette_begin': [0, 1],
            'tracklette_end': [0, 1],
            'fragment_begin': [0, 1],
            'fragment_end': [0, 1],
            'shower_begin': [0, 1],
        }

        self.parse_config()

    @profiler
    def parse_config(self):
        self.parse_folders_and_files()
        self.parse_dataset_variables()
        self.parse_dataset_parameters()

    @profiler
    def parse_folders_and_files(self):
        """Check dataset_mode.  Should be simulation or data"""
        if 'dataset_mode' not in self.config.keys():
            self.dataset_mode = 'simulation'
        else:
            self.dataset_mode = self.config['dataset_mode']

        """Check for flow_folder"""
        if 'flow_folder' not in self.config.keys():
            raise BlipError('flow_folder not specified in config!')
        else:
            self.flow_folder = self.config['flow_folder']

        """Check for arrakis_folder if dataset_mode is simulation"""
        if self.dataset_mode == 'simulation':
            if 'arrakis_folder' not in self.config.keys():
                raise BlipError('arrakis_folder not specified in config! maybe "dataset_mode" should be "data"?')
            else:
                self.arrakis_folder = self.config['arrakis_folder']
        else:
            self.arrakis_folder = None

        """Check for blip_folder"""
        if 'blip_folder' not in self.config.keys():
            raise BlipError('blip_folder not specified in config!')
        else:
            self.blip_folder = self.config['blip_folder']

        """Check for flow_files"""
        if 'flow_files' not in self.config.keys():
            raise BlipError('flow_files not specified in config!')
        else:
            self.flow_files = self.config['flow_files']

        """Check for skip_files"""
        if 'skip_files' not in self.config.keys():
            self.skip_files = []
        else:
            self.skip_files = self.config['skip_files']

        """Check that flow folder exists"""
        if not os.path.isdir(self.flow_folder):
            raise BlipError(f'specified flow_folder "{self.flow_folder}" does not exist!')

        """Check that arrakis folder exists"""
        if self.arrakis_folder is not None:
            if not os.path.isdir(self.arrakis_folder):
                raise BlipError(f'specified arrakis_folder "{self.arrakis_folder}" does not exist!')

        """Check that blip folder exists"""
        if not os.path.isdir(self.blip_folder):
            raise BlipError(f'specified blip_folder "{self.blip_folder}" does not exist!')

        """Check that flow folder has a '/' at the end"""
        if self.flow_folder[-1] != '/':
            self.flow_folder += '/'

        """Check that arrakis folder has a '/' at the end"""
        if self.arrakis_folder is not None:
            if self.arrakis_folder[-1] != '/':
                self.arrakis_folder += '/'

        """Check that blip folder has a '/' at the end"""
        if self.blip_folder[-1] != '/':
            self.blip_folder += '/'

        if isinstance(self.flow_files, list):
            """
            If the flow_files parameter is a list, look through
            the list and make sure each specified file actually exists
            in the flow_folder.
            """
            flow_files = [
                input_file for input_file in self.flow_files
                if input_file not in self.skip_files
            ]
            for flow_file in flow_files:
                if not os.path.isfile(self.flow_folder + flow_file):
                    raise BlipError(
                        f"specified file {flow_file} does not exist in directory {self.flow_folder}!"
                    )
        elif isinstance(self.flow_files, str):
            """
            If the flow_files parameter is a string, check if its
            the phrase 'all', and if so, recursively grab all h5
            flow_files in the flow_folder.

            Otherwise, assume that the flow_files parameter is a
            file extension, and search recursively for all flow_files
            with that extension.
            """
            if self.flow_files == "all":
                flow_files = [
                    os.path.basename(input_file) for input_file in glob.glob(
                        f"{self.flow_folder}*.hdf5", recursive=True
                    )
                    if 'FLOW' in input_file and input_file not in self.skip_files
                ]
            else:
                try:
                    flow_files = [
                        os.path.basename(input_file) for input_file in glob.glob(
                            f'{self.flow_folder}/*.{self.flow_files}',
                            recursive=True,
                        )
                        if input_file not in self.skip_files
                    ]
                except Exception as exception:
                    raise BlipError(
                        f'specified "files" parameter: {self.config["files"]} incompatible!'
                        + f" exception: {exception}"
                    )
        else:
            raise BlipError(
                f'specified "flow_files" parameter: {self.config["files"]} incompatible!'
            )

        """Set arrakis files from flow files"""
        if self.dataset_mode == 'simulation':
            arrakis_files = [
                flow_file.replace('FLOW', 'ARRAKIS').replace('flow', 'arrakis')
                for flow_file in flow_files
            ]
            """Check that each corresponding arrakis file exists"""
            for (flow_file, arrakis_file) in zip(flow_files, arrakis_files):
                if not os.path.isfile(self.arrakis_folder + arrakis_file):
                    flow_files.remove(flow_file)
                    arrakis_files.remove(arrakis_file)

        self.flow_files = flow_files
        if self.dataset_mode == 'simulation':
            self.arrakis_files = arrakis_files

    @profiler
    def parse_dataset_variables(self):
        """Check for dataset_name"""
        if 'dataset_name' not in self.config.keys():
            raise BlipError('dataset_name not specified in config!')
        else:
            self.dataset_name = self.config['dataset_name']

        """Check for positions"""
        if 'positions' not in self.config.keys():
            raise BlipError('positions not specified in config!')
        else:
            self.positions = self.config['positions']

        """Check for features"""
        if 'features' not in self.config.keys():
            raise BlipError('features not specified in config!')
        else:
            self.features = self.config['features']

        """Check for labels"""
        if 'labels' not in self.config.keys():
            raise BlipError('labels not specified in config!')
        else:
            self.labels = self.config['labels']

    @profiler
    def parse_dataset_parameters(self):
        """Check for voxelization"""
        if 'voxelization' not in self.config.keys():
            self.voxelization = torch.tensor([1.0 for ii in range(len(self.positions))]).reshape(1, -1)
        else:
            if len(self.config['voxelization']) != len(self.positions):
                raise BlipError('number of entries in "voxelization" does not match number of position variables!')
            else:
                self.voxelization = torch.tensor(self.config['voxelization']).reshape(1, -1)
        """Check for chunk_size"""
        if 'chunk_size' not in self.config.keys():
            self.config['chunk_size'] = 25
        self.chunk_size = self.config['chunk_size']

        """Check for class weights"""
        if 'class_weights' not in self.config.keys():
            self.config['class_weights'] = []
        self.class_weights_to_use = self.config['class_weights']

        """Check for feature normalization"""
        if 'feature_normalization' not in self.config.keys():
            self.feature_normalization = 1
        else:
            if self.config['feature_normalization'] == 'min_max':
                self.feature_normalization = 0
            else:
                self.feature_normalization = 1

    @profiler
    def create_blip_files(
        self
    ):
        """
        This function generates output Blip files
        which will contain predictions from the output of blip.
        """
        self.blip_files = []
        flow_file_loop = tqdm(
            enumerate(self.flow_files, 0),
            total=len(self.flow_files),
            leave=True,
            colour='green',
        )
        for ii, flow_file in flow_file_loop:
            flow_file_loop.set_description(
                f"Creating blip files [{ii+1}]"
            )
            blip_file = flow_file.replace('FLOW', 'BLIP').replace('flow', 'blip')
            self.blip_files.append(blip_file)
            with h5py.File(self.flow_folder + flow_file, 'r') as flow, \
                 h5py.File(self.blip_folder + blip_file, 'a') as blip:
                """Get dataset size from flow file"""

                dataset_size = len(flow[self.dataset_name])
                new_charge_data_type = np.dtype([
                    ('event_id', 'i4'),
                    ('topology', 'f4', (3, )),
                    ('physics', 'f4', (9, )),
                    ('vertex', 'f4'),
                    ('tracklette_begin', 'f4'),
                    ('tracklette_end', 'f4'),
                    ('fragment_begin', 'f4'),
                    ('fragment_end', 'f4'),
                    ('shower_begin', 'f4')
                ])
                new_charge_data = np.full(
                    dataset_size, -1, dtype=new_charge_data_type
                )
                if self.dataset_name in blip:
                    del blip[self.dataset_name]

                blip.create_dataset(self.dataset_name, data=new_charge_data)

    def apply_voxelization(
        self,
        output
    ):
        """
        This function applies a voxelization to the position variables,
        and then prunes duplicates from the other variables while also
        creating a map so that one can map values back to the original
        indices where the variables were before voxelization.  This is
        an important step for ensuring that MinkowskiEngine, and other
        algorithms play nicely with the dataset.

        The voxelization is applied by dividing by the value for each
        specified coordinate.  Fow now this is done globally for the
        detector, but in the future we will do this detector by
        detector.

        Original coordinates are in cm, so to get to mm just use
        a voxelization of 0.1.

        Args:
            output (_type_): _description_

        Returns:
            _type_: _description_
        """
        """Step 1: Apply the voxelization to positions"""
        output['positions'] = (output['positions'].clone() / self.voxelization).to(torch.int32)

        return output

    def apply_normalization(
        self,
        output
    ):
        """
        This function applies the selected feature normalization
        to the features in a batch.
        """
        for feature in self.features:
            if self.feature_normalization == 0:
                output['features'] = (
                    output['features'].clone() / (self.feature_max[feature] - self.feature_min[feature])
                )
            else:
                output['features'] = (
                    output['features'].clone() - self.feature_mean[feature]
                ) / self.feature_std[feature]

        return output

    @profiler
    def calculate_event_id_mapping(self):
        """
        Calculate the mapping of chunks based on unique event_ids in each file.
        """
        self.file_indices = []
        self.event_start_indices = []
        self.event_end_indices = []
        arrakis_file_loop = tqdm(
            enumerate(self.arrakis_files, 0),
            total=len(self.arrakis_files),
            leave=True,
            colour='green',
        )
        for ii, arrakis_file in arrakis_file_loop:
            arrakis_file_loop.set_description(
                f"Calculating event_id mapping [{ii+1}]"
            )
            with h5py.File(self.arrakis_folder + arrakis_file, 'r') as f:
                event_ids = f[self.dataset_name]['event_id'][:]
                unique_values, start_indices = np.unique(event_ids, return_index=True)
                end_indices = start_indices[1:] + [len(event_ids)]
                for jj in range(0, len(unique_values), self.chunk_size):
                    self.file_indices.append(ii)
                    self.event_start_indices.append(start_indices[jj])
                    self.event_end_indices.append(end_indices[min(jj + self.chunk_size - 1, len(end_indices) - 1)])

    @profiler
    def calculate_feature_normalization(self):
        """Calculate the totals for feature values over all files"""
        if len(self.features) == 0:
            return
        self.feature_sum = {
            feature: 0 for feature in self.features
        }
        self.feature_count = {
            feature: 0 for feature in self.features
        }
        self.feature_max = {
            feature: 0 for feature in self.features
        }
        self.feature_min = {
            feature: 0 for feature in self.features
        }
        self.feature_std = {
            feature: 0 for feature in self.features
        }
        flow_file_loop = tqdm(
            enumerate(self.flow_files, 0),
            total=len(self.flow_files),
            leave=True,
            colour='green',
        )
        for ii, flow_file in flow_file_loop:
            flow_file_loop.set_description(
                f"Calculating feature normalization [{ii+1}]"
            )
            with h5py.File(self.flow_folder + flow_file, 'r') as f:
                for feature in self.features:
                    flow_feature = f[self.dataset_name][feature]
                    self.feature_sum[feature] += np.sum(flow_feature)
                    self.feature_count[feature] += len(flow_feature)
                    self.feature_max[feature] = max(self.feature_max[feature], np.max(flow_feature))
                    self.feature_min[feature] = min(self.feature_min[feature], np.min(flow_feature))
        """Compute the mean"""
        self.feature_mean = {
            feature: self.feature_sum[feature] / self.feature_count[feature]
            for feature in self.features
        }
        """Compute the standard deviation"""
        for ii, flow_file in enumerate(self.flow_files):
            with h5py.File(self.flow_folder + flow_file, 'r') as f:
                for feature in self.features:
                    flow_feature = f[self.dataset_name][feature][:]
                    self.feature_std[feature] += sum((
                        flow_feature - self.feature_mean[feature]
                    ) ** 2 / self.feature_count[feature])
        for feature in self.features:
            self.feature_std[feature] = np.sqrt(self.feature_std[feature])

    @profiler
    def calculate_class_weights(self):
        """Set up counters for class instances"""
        self.class_instances = {
            label: [0 for ii in range(len(self.label_values[label]))]
            for label in self.label_values.keys()
        }
        """Set up initial weights of 1.0"""
        self.class_weights = {
            label: torch.tensor([1.0 for ii in range(len(self.label_values[label]))], dtype=torch.float)
            for label in self.label_values.keys()
        }
        self.total_instances = 0
        """Iterate over each file and grab instances"""
        arrakis_file_loop = tqdm(
            enumerate(self.arrakis_files, 0),
            total=len(self.arrakis_files),
            leave=True,
            colour='green',
        )
        for ii, arrakis_file in arrakis_file_loop:
            arrakis_file_loop.set_description(
                f"Calculating class weights [{ii+1}]"
            )
            with h5py.File(self.arrakis_folder + arrakis_file, 'r') as f:
                charge = f[self.dataset_name]
                for label in self.class_instances:
                    labels, counts = np.unique(charge[label], return_counts=True)
                    for ll, cc in zip(labels, counts):
                        self.class_instances[label][ll] += cc
                        self.total_instances += cc
        """Set weights for specified labels in class_weights"""
        for label in self.class_weights_to_use:
            for ll, cc in enumerate(self.class_instances[label]):
                if cc > 0:
                    self.class_instances[label][ll] = 1.0 / cc
            for ll, cc in enumerate(self.class_instances[label]):
                self.class_weights[label][ll] = self.class_instances[label][ll] / sum(self.class_instances[label])

    def __len__(self):
        """Calculate total number of chunks"""
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_idx = self.file_indices[idx]
        chunk_start_idx = self.event_start_indices[idx]
        chunk_end_idx = self.event_end_indices[idx]
        if self.dataset_mode == 'simulation':
            return self.get_simulation(file_idx, chunk_start_idx, chunk_end_idx)
        else:
            return self.get_data(file_idx, chunk_start_idx, chunk_end_idx)

    def get_simulation(
        self,
        file_idx,
        chunk_start_idx,
        chunk_end_idx
    ):
        flow_file = self.flow_files[file_idx]
        arrakis_file = self.arrakis_files[file_idx]
        output = {
            'positions': None,
            'features': None,
            'batch_id': None,
            'labels': None,
            'file_idx': file_idx,
            'chunk_start_idx': chunk_start_idx,
            'chunk_end_idx': chunk_end_idx
        }
        with h5py.File(self.flow_folder + flow_file, 'r') as flow, \
             h5py.File(self.arrakis_folder + arrakis_file, 'r') as arrakis:

            """Get charge data from flow and arrakis files"""
            charge_flow = flow[self.dataset_name]
            charge_arrakis = arrakis[self.dataset_name]

            """Get positions from charge_flow"""
            charge_positions = tuple(
                charge_flow[position][chunk_start_idx:chunk_end_idx]
                for position in self.positions
            )
            output['positions'] = torch.tensor(
                np.vstack(charge_positions),
                dtype=torch.float32
            ).transpose(0, 1)

            """Get event_id as batch_id"""
            output['batch_id'] = torch.tensor(
                charge_arrakis['event_id'][chunk_start_idx:chunk_end_idx],
                dtype=torch.int32
            ).unsqueeze(1)

            """Get features from charge_flow"""
            if self.features == []:
                output['features'] = torch.ones_like(output['batch_id'])
            else:
                charge_features = tuple(
                    charge_flow[feature][chunk_start_idx:chunk_end_idx]
                    for feature in self.features
                )
                output['features'] = torch.tensor(
                    np.vstack(charge_features),
                    dtype=torch.float32
                ).transpose(0, 1)

            """Get labels from charge_arrakis"""
            charge_labels = tuple(
                charge_arrakis[label][chunk_start_idx:chunk_end_idx]
                for label in self.labels
            )
            output['labels'] = torch.tensor(
                np.vstack(charge_labels),
                dtype=torch.int32
            ).transpose(0, 1)

            """Apply voxelization"""
            output = self.apply_voxelization(output)

            """Apply feature normalization"""
            output = self.apply_normalization(output)

        return output

    def get_data(
        self,
        file_idx,
        chunk_start_idx,
        chunk_end_idx
    ):
        flow_file = self.flow_files[file_idx]
        output = {
            'positions': None,
            'features': None,
            'batch_id': None,
            'labels': None,
            'file_idx': file_idx,
            'chunk_start_idx': chunk_start_idx,
            'chunk_end_idx': chunk_end_idx
        }
        with h5py.File(self.flow_folder + flow_file, 'r') as flow:

            """Get charge data from flow and arrakis files"""
            charge_flow = flow[self.dataset_name]

            """Get positions from charge_flow"""
            charge_positions = tuple(
                charge_flow[position][chunk_start_idx:chunk_end_idx]
                for position in self.positions
            )
            output['positions'] = torch.tensor(
                np.vstack(charge_positions),
                dtype=torch.float32
            ).transpose(0, 1)

            """Get event_id as batch_id"""
            output['batch_id'] = torch.tensor(
                charge_flow['event_id'][chunk_start_idx:chunk_end_idx],
                dtype=torch.int32
            ).unsqueeze(1)

            """Get features from charge_flow"""
            if self.features == []:
                output['features'] = torch.ones_like(output['batch_id'])
            else:
                charge_features = tuple(
                    charge_flow[feature][chunk_start_idx:chunk_end_idx]
                    for feature in self.features
                )
                output['features'] = torch.tensor(
                    np.vstack(charge_features),
                    dtype=torch.float32
                ).transpose(0, 1)

            """Apply voxelization"""
            output = self.apply_voxelization(output)

            """Apply feature normalization"""
            output = self.apply_normalization(output)

        return output

    def save_predictions(
        self,
        data,
    ):
        """Get blip file"""
        for ii, file_idx in enumerate(data['file_idx']):
            blip_file = self.blip_files[file_idx]
            with h5py.File(self.blip_folder + blip_file, 'r+', driver='mpio', comm=self.comm) as blip:
                for output in data['outputs']:
                    file_start_idx = data['chunk_start_idx'][ii]
                    file_end_idx = data['chunk_end_idx'][ii]
                    pred_start_idx = data['relative_start_idx'][ii]
                    pred_end_idx = data['relative_end_idx'][ii]

                    """Assign the predictions to the appropriate indices"""
                    output_data = data['outputs'][output][pred_start_idx:pred_end_idx]
                    if output not in ['topology', 'physics']:
                        output_data = output_data.squeeze(1)

                    existing_data = blip[self.dataset_name][output][:]
                    existing_data[file_start_idx:file_end_idx] = output_data.cpu().numpy()
                    blip[self.dataset_name][output] = existing_data
