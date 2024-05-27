"""
Generic data loader class for blip.
"""
import numpy as np
import torch
import os.path as osp
import MinkowskiEngine as ME
from torch.utils.data import Subset, random_split
from torch.utils.data import DataLoader, DistributedSampler

"""NVIDIA Dali related imports"""
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from blip.utils.common import quantization_modes, minkowski_algorithms
from blip.utils.logger import BlipError
from blip.utils.utils import profiler


def worker_init(wrk_id):
    np.random.seed(
        torch.utils.data.get_worker_info().seed % (2**32 - 1)
    )


class BlipLoader:
    """
    """
    def __init__(
        self,
        config: dict = {},
        meta:   dict = {}
    ):
        self.config = config
        self.meta = meta

        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'

        self.parse_config()

    def blip_collate(
        self,
        batch
    ):
        """Batch the incoming data"""
        positions, features, batch_ids, labels = [], [], [], []
        file_idxs, chunk_start_idxs, chunk_end_idxs = [], [], []
        relative_start_idxs, relative_end_idxs = [], []
        relative_start_idx = 0
        for sample in batch:
            relative_end_idx = relative_start_idx + len(sample['positions'])
            positions.append(sample['positions'])
            features.append(sample['features'])
            batch_ids.append(sample['batch_id'])
            labels.append(sample['labels'])
            file_idxs.append(sample['file_idx'])
            chunk_start_idxs.append(sample['chunk_start_idx'])
            chunk_end_idxs.append(sample['chunk_end_idx'])
            relative_start_idxs.append(relative_start_idx)
            relative_end_idxs.append(relative_end_idx)
            relative_start_idx = relative_end_idx + 1

        """Concatenate """
        batched_positions = torch.cat(positions)
        batched_features = torch.cat(features)
        batched_batch_ids = torch.cat(batch_ids)
        batched_labels = torch.cat(labels)
        batched_file_idxs = torch.tensor(file_idxs)
        batched_chunk_start_idxs = torch.tensor(chunk_start_idxs)
        batched_chunk_end_idxs = torch.tensor(chunk_end_idxs)
        batched_relative_start_idxs = torch.tensor(relative_start_idxs)
        batched_relative_end_idxs = torch.tensor(relative_end_idxs)
        return {
            'positions': batched_positions,
            'features': batched_features,
            'batch_id': batched_batch_ids,
            'labels': batched_labels,
            'file_idx': batched_file_idxs,
            'chunk_start_idx': batched_chunk_start_idxs,
            'chunk_end_idx': batched_chunk_end_idxs,
            'relative_start_idx': batched_relative_start_idxs,
            'relative_end_idx': batched_relative_end_idxs
        }

    @profiler
    def parse_config(self):
        self.process_meta()
        self.process_datasets()
        self.process_samplers()
        self.process_loaders()

    @profiler
    def process_meta(self):
        if "batch_size" not in self.config.keys():
            raise BlipError("'batch_size' not specified in config!")
        if not self.config["batch_size"]:
            raise BlipError("'batch_size' not specified in config!")
        if self.config["batch_size"] <= 0:
            raise BlipError(
                f"specified batch size: {self.config['batch_size']} " +
                "not allowed, must be > 0!"
            )

        """Check quantization mode"""
        if "quantization_mode" not in self.config.keys():
            self.config['quantization_mode'] = 'random_subsample'
        else:
            if self.config['quantization_mode'] not in quantization_modes:
                self.logger.warn(
                    f'specified quantization_mode {self.config["quantization_mode"]} not allowed!' +
                    'setting to "random_subsample"'
                )
                self.config['quantization_mode'] = 'random_subsample'
        self.meta['quantization_mode'] = quantization_modes[self.config['quantization_mode']]

        """Check for minkowski algorithm"""
        if "minkowski_algorithm" not in self.config.keys():
            self.config['minkowski_algorithm'] = 'speed_optimized'
        else:
            if self.config['minkowski_algorithm'] not in minkowski_algorithms:
                self.config['minkowski_algorithm'] = 'speed_optimized'
        self.meta['minkowski_algorithm'] = minkowski_algorithms[self.config['minkowski_algorithm']]

    @profiler
    def process_datasets(self):
        # setting validation split
        if "validation_split" not in self.config.keys():
            self.config["validation_split"] = 0
        if not self.config["validation_split"]:
            self.config["validation_split"] = 0
        if (self.config["validation_split"] < 0.0 or self.config["validation_split"] >= 1.0):
            raise BlipError(
                f"specified validation split: {self.config['validation_split']} " +
                "not allowed, must be 0.0 <= 'validation_split' < 1.0!"
            )

        # setting validation seed
        if "validation_seed" not in self.config.keys():
            self.config["validation_seed"] = -1
        if not self.config["validation_seed"]:
            self.config["validation_seed"] = -1
        if (self.config['validation_seed'] != -1 and self.config['validation_seed'] < 0):
            raise BlipError(
                f"specified test seed: {self.config['validation_seed']} not allowed, " +
                "must be == -1 or >= 0!"
            )
        if not isinstance(self.config['validation_seed'], int):
            raise BlipError(
                f"specified test seed: {self.config['validation_seed']} is of type " +
                f"'{type(self.config['validation_seed'])}', must be of type 'int'!"
            )

        # setting test split
        if "test_split" not in self.config.keys():
            self.config["test_split"] = 0
        if not self.config["test_split"]:
            self.config["test_split"] = 0
        if (self.config['test_split'] < 0.0 or self.config['test_split'] >= 1.0):
            raise BlipError(
                f"specified test split: {self.config['test_split']} not allowed, " +
                "must be 0.0 <= 'test_split' < 1.0!"
            )

        # setting test seed
        if "test_seed" not in self.config.keys():
            self.config["test_seed"] = -1
        if not self.config["test_seed"]:
            self.config["test_seed"] = -1
        if (self.config['test_seed'] != -1 and self.config['test_seed'] < 0):
            raise BlipError(
                f"specified test seed: {self.config['test_seed']} not allowed, " +
                "must be == -1 or >= 0!"
            )
        if not isinstance(self.config['test_seed'], int):
            raise BlipError(
                f"specified test seed: {self.config['test_seed']} is of type " +
                f"'{type(self.config['test_seed'])}', must be of type 'int'!"
            )

        # setting num_workers
        if "num_workers" not in self.config.keys():
            self.config["num_workers"] = 0
        if not self.config["num_workers"]:
            self.config["num_workers"] = 0
        if (self.config['num_workers'] != -1 and self.config['num_workers'] < 0):
            raise BlipError(
                f"specified test seed: {self.config['num_workers']} not allowed, " +
                "must be == -1 or >= 0!"
            )
        if not isinstance(self.config['num_workers'], int):
            raise BlipError(
                f"specified test seed: {self.config['num_workers']} is of type " +
                f"'{type(self.config['num_workers'])}', must be of type 'int'!"
            )

        """assign parameters"""
        self.batch_size = self.config["batch_size"]
        self.test_split = self.config["test_split"]
        self.test_seed = self.config["test_seed"]
        self.validation_split = self.config["validation_split"]
        self.validation_seed = self.config["validation_seed"]
        self.num_workers = self.config["num_workers"]

        """Determine number of all batches"""
        self.num_all_batches = len(self.meta['dataset'])

        """Determine number of training/testing samples"""
        self.num_total_train = int(len(self.meta['dataset']) * (1 - self.test_split))
        self.num_test = int(len(self.meta['dataset']) - self.num_total_train)

        """Determine number of batches for testing"""
        self.num_test_batches = int(self.num_test/self.batch_size)
        if self.num_test % self.batch_size != 0:
            self.num_test_batches += 1

        """Determine number of training/validation samples"""
        self.num_train = int(self.num_total_train * (1 - self.validation_split))
        self.num_validation = int(self.num_total_train - self.num_train)

        """Determine number of batches for training/validation"""
        self.num_train_batches = int(self.num_train/self.batch_size)
        if self.num_train % self.batch_size != 0:
            self.num_train_batches += 1
        self.num_validation_batches = int(self.num_validation/self.batch_size)
        if self.num_validation % self.batch_size != 0:
            self.num_validation_batches += 1

        """Set up the training and testing sets"""
        if self.test_seed != -1:
            self.total_train, self.test = random_split(
                dataset=self.meta['dataset'],
                lengths=[self.num_total_train, self.num_test],
                generator=torch.Generator().manual_seed(self.test_seed)
            )
            self.total_train_indices = self.total_train.indices
            self.test_indices = self.test.indices
        else:
            self.total_train_indices = range(self.num_total_train)
            self.test_indices = range(self.num_total_train, len(self.meta['dataset']))

            self.total_train = Subset(self.meta['dataset'], self.total_train_indices)
            self.test = Subset(self.meta['dataset'], self.test_indices)

        """Set up the training and validation sets"""
        if self.validation_seed != -1:
            self.train, self.validation = random_split(
                dataset=self.total_train,
                lengths=[self.num_train, self.num_validation],
                generator=torch.Generator().manual_seed(self.validation_seed)
            )
            self.train_indices = self.train.indices
            self.validation_indices = self.validation.indices
        else:
            self.train_indices = range(self.num_train)
            self.validation_indices = range(self.num_train, len(self.total_train))

            self.train = Subset(self.total_train, self.train_indices)
            self.validation = Subset(self.total_train, self.validation_indices)
        self.all_indices = range(len(self.meta['dataset']))

        """Determine number of epochs"""
        self.meta['num_epochs'] = (self.meta['num_iterations'] // len(self.train)) + 1

    @profiler
    def process_samplers(self):
        """Set up sampling if this run is distributed"""
        if self.meta["distributed"]:
            if ("num_data_shards" in self.meta) and ("data_shard_id" in self.meta):
                self.train_sampler = DistributedSampler(
                    self.train,
                    num_replicas=self.meta["num_data_shards"],
                    rank=self.meta["data_shard_id"]
                )
                self.validation_sampler = DistributedSampler(
                    self.validation,
                    num_replicas=self.meta["num_data_shards"],
                    rank=self.meta["data_shard_id"]
                )
                self.total_train_sampler = DistributedSampler(
                    self.total_train,
                    num_replicas=self.meta["num_data_shards"],
                    rank=self.meta["data_shard_id"]
                )
                self.test_sampler = DistributedSampler(
                    self.test,
                    num_replicas=self.meta["num_data_shards"],
                    rank=self.meta["data_shard_id"]
                )
                self.all_sampler = DistributedSampler(
                    self.meta['dataset'],
                    num_replicas=self.meta["num_data_shards"],
                    rank=self.meta["data_shard_id"]
                )
            else:
                self.train_sampler = DistributedSampler(self.train)
                self.validation_sampler = DistributedSampler(self.validation)
                self.total_train_sampler = DistributedSampler(self.total_train)
                self.test_sampler = DistributedSampler(self.test)
                self.all_sampler = DistributedSampler(self.meta['dataset'])
        else:
            self.train_sampler = None
            self.validation_sampler = None
            self.total_train_sampler = None
            self.test_sampler = None
            self.all_sampler = None

    @profiler
    def process_loaders(self):
        # set up dataloaders for each set
        self.train_loader = DataLoader(
            self.train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            worker_init_fn=worker_init,
            collate_fn=self.blip_collate
        )
        self.validation_loader = DataLoader(
            self.validation,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=self.validation_sampler,
            worker_init_fn=worker_init,
            collate_fn=self.blip_collate
        )
        self.total_train_loader = DataLoader(
            self.total_train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=self.total_train_sampler,
            worker_init_fn=worker_init,
            collate_fn=self.blip_collate
        )
        self.test_loader = DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=self.test_sampler,
            worker_init_fn=worker_init,
            collate_fn=self.blip_collate
        )
        self.all_loader = DataLoader(
            self.meta['dataset'],
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=self.all_sampler,
            worker_init_fn=worker_init,
            collate_fn=self.blip_collate
        )
        self.inference_loader = DataLoader(
            self.meta['dataset'],
            batch_size=1,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=self.all_sampler,
            worker_init_fn=worker_init,
            collate_fn=self.blip_collate
        )
