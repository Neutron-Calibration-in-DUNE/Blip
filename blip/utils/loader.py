"""
Generic data loader class for blip.
"""
import torch
import MinkowskiEngine as ME
from torch.utils.data import Subset, random_split
from torch_geometric.loader import DataLoader

from blip.utils.logger import Logger
from blip.dataset.generic_dataset import GenericDataset


class Loader:
    """
    """
    def __init__(
        self,
        name:   str = "",
        config: dict = {},
        meta:   dict = {}
    ):
        self.name = name + '_loader'
        self.config = config
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(self.name, output="both", file_mode="w")
        else:
            self.logger = Logger(self.name, level='warning', file_mode="w")
        self.logger.info("constructing blip loader.")

        if not issubclass(type(self.meta['dataset']), GenericDataset):
            self.logger.error(
                'dataset if not of type "GenericDataset", ' +
                f'but rather {type(self.meta["dataset"])}!'
            )
        self.process_config()

    def set_config(
        self,
        config: dict
    ):
        self.config = config
        self.logger.info("setting config.")
        self.process_config()

    def process_config(self):
        # check for parameter compatability
        self.logger.info("processing config")
        if "batch_size" not in self.config.keys():
            self.logger.error("'batch_size' not specified in config!")
        if not self.config["batch_size"]:
            self.logger.error("'batch_size' not specified in config!")
        if self.config["batch_size"] <= 0:
            self.logger.error(
                f"specified batch size: {self.config['batch_size']} " +
                "not allowed, must be > 0!"
            )

        # setting validation split
        if "validation_split" not in self.config.keys():
            self.logger.warn("'validation_split' not specified in config! setting to '0'.")
            self.config["validation_split"] = 0
        if not self.config["validation_split"]:
            self.logger.warn("'validation_split' not specified in config! setting to '0'.")
            self.config["validation_split"] = 0
        if (self.config["validation_split"] < 0.0 or self.config["validation_split"] >= 1.0):
            self.logger.error(
                f"specified validation split: {self.config['validation_split']} " +
                "not allowed, must be 0.0 <= 'validation_split' < 1.0!"
            )

        # setting validation seed
        if "validation_seed" not in self.config.keys():
            self.logger.warn("'validation_seed' not specified in config! setting to '-1'.")
            self.config["validation_seed"] = -1
        if not self.config["validation_seed"]:
            self.logger.warn("'validation_seed' not specified in config! setting to '-1'.")
            self.config["validation_seed"] = -1
        if (self.config['validation_seed'] != -1 and self.config['validation_seed'] < 0):
            self.logger.error(
                f"specified test seed: {self.config['validation_seed']} not allowed, " +
                "must be == -1 or >= 0!"
            )
        if not isinstance(self.config['validation_seed'], int):
            self.logger.error(
                f"specified test seed: {self.config['validation_seed']} is of type " +
                f"'{type(self.config['validation_seed'])}', must be of type 'int'!"
            )

        # setting test split
        if "test_split" not in self.config.keys():
            self.logger.warn("'test_split' not specified in config! setting to '0'.")
            self.config["test_split"] = 0
        if not self.config["test_split"]:
            self.logger.warn("'test_split' not specified in config! setting to '0'.")
            self.config["test_split"] = 0
        if (self.config['test_split'] < 0.0 or self.config['test_split'] >= 1.0):
            self.logger.error(
                f"specified test split: {self.config['test_split']} not allowed, " +
                "must be 0.0 <= 'test_split' < 1.0!"
            )

        # setting test seed
        if "test_seed" not in self.config.keys():
            self.logger.warn("'test_seed' not specified in config! setting to '-1'.")
            self.config["test_seed"] = -1
        if not self.config["test_seed"]:
            self.logger.warn("'test_seed' not specified in config! setting to '-1'.")
            self.config["test_seed"] = -1
        if (self.config['test_seed'] != -1 and self.config['test_seed'] < 0):
            self.logger.error(
                f"specified test seed: {self.config['test_seed']} not allowed, " +
                "must be == -1 or >= 0!"
            )
        if not isinstance(self.config['test_seed'], int):
            self.logger.error(
                f"specified test seed: {self.config['test_seed']} is of type " +
                f"'{type(self.config['test_seed'])}', must be of type 'int'!"
            )

        # setting num_workers
        if "num_workers" not in self.config.keys():
            self.logger.warn("'num_workers' not specified in config! setting to '0'.")
            self.config["num_workers"] = 0
        if not self.config["num_workers"]:
            self.logger.warn("'num_workers' not specified in config! setting to '0'.")
            self.config["num_workers"] = 0
        if (self.config['num_workers'] != -1 and self.config['num_workers'] < 0):
            self.logger.error(
                f"specified test seed: {self.config['num_workers']} not allowed, " +
                "must be == -1 or >= 0!"
            )
        if not isinstance(self.config['num_workers'], int):
            self.logger.error(
                f"specified test seed: {self.config['num_workers']} is of type " +
                f"'{type(self.config['num_workers'])}', must be of type 'int'!"
            )

        if "loader_type" not in self.config.keys():
            self.logger.warn("'loader_type' not specified in config! setting sparse to 'False'.")
            self.config["sparse"] = False
        else:
            if self.config["loader_type"] == "sparse" or self.config["loader_type"] == "minkowski":
                self.config['sparse'] = True

        # assign parameters
        self.batch_size = self.config["batch_size"]
        self.test_split = self.config["test_split"]
        self.test_seed = self.config["test_seed"]
        self.validation_split = self.config["validation_split"]
        self.validation_seed = self.config["validation_seed"]
        self.num_workers = self.config["num_workers"]
        self.sparse = self.config["sparse"]

        if self.sparse:
            self.collate_fn = ME.utils.SparseCollation()
        else:
            self.collate_fn = None

        # record values
        self.logger.info(f"batch_size:  {self.batch_size}.")
        self.logger.info(f"test_split:  {self.test_split}.")
        self.logger.info(f"test_seed:   {self.test_seed}.")
        self.logger.info(f"validation_split: {self.validation_split}.")
        self.logger.info(f"validation_seed: {self.validation_seed}.")
        self.logger.info(f"num_workers: {self.num_workers}.")
        self.logger.info(f"sparse:      {self.sparse}.")

        # determine number of all batches
        self.num_all_batches = len(self.meta['dataset'])
        self.logger.info(f"number of total samples: {len(self.meta['dataset'])}.")
        self.logger.info(f"number of all batches: {self.num_all_batches}.")

        # determine number of training/testing samples
        self.num_total_train = int(len(self.meta['dataset']) * (1 - self.test_split))
        self.num_test = int(len(self.meta['dataset']) - self.num_total_train)

        # determine number of batches for testing
        self.num_test_batches = int(self.num_test/self.batch_size)
        if self.num_test % self.batch_size != 0:
            self.num_test_batches += 1

        # determine number of training/validation samples
        self.num_train = int(self.num_total_train * (1 - self.validation_split))
        self.num_validation = int(self.num_total_train - self.num_train)

        # determine number of batches for training/validation
        self.num_train_batches = int(self.num_train/self.batch_size)
        if self.num_train % self.batch_size != 0:
            self.num_train_batches += 1
        self.num_validation_batches = int(self.num_validation/self.batch_size)
        if self.num_validation % self.batch_size != 0:
            self.num_validation_batches += 1

        self.logger.info(f"number of total training samples:    {self.num_total_train}")
        self.logger.info(f"number of test samples:              {self.num_test}")
        self.logger.info(f"number of test batches:              {self.num_test_batches}")
        self.logger.info(f"number of training samples:          {self.num_train}")
        self.logger.info(f"number of training batches per epoch:    {self.num_train_batches}")
        self.logger.info(f"number of validation samples:        {self.num_validation}")
        self.logger.info(f"number of validation batches per epoch:  {self.num_validation_batches}")

        # set up the training and testing sets
        if self.test_seed != -1:
            self.total_train, self.test = random_split(
                dataset=self.meta['dataset'],
                lengths=[self.num_total_train, self.num_test],
                generator=torch.Generator().manual_seed(self.test_seed)
            )
            self.total_train_indices = self.total_train.indices
            self.test_indices = self.test.indices
            self.logger.info(f"created train/test split with random seed: {self.test_seed}.")
        else:
            self.total_train_indices = range(self.num_total_train)
            self.test_indices = range(self.num_total_train, len(self.meta['dataset']))

            self.total_train = Subset(self.meta['dataset'], self.total_train_indices)
            self.test = Subset(self.meta['dataset'], self.test_indices)
            self.logger.info(
                f"created train/test split with first {self.num_total_train} samples " +
                f"for training and last {self.num_test} samples for testing."
            )

        # set up the training and validation sets
        if self.validation_seed != -1:
            self.train, self.validation = random_split(
                dataset=self.total_train,
                lengths=[self.num_train, self.num_validation],
                generator=torch.Generator().manual_seed(self.validation_seed)
            )
            self.train_indices = self.train.indices
            self.validation_indices = self.validation.indices
            self.logger.info(
                f"created train/validation split with random seed: {self.validation_seed}."
            )
        else:
            self.train_indices = range(self.num_train)
            self.validation_indices = range(self.num_train, len(self.total_train))

            self.train = Subset(self.total_train, self.train_indices)
            self.validation = Subset(self.total_train, self.validation_indices)
            self.logger.info(
                f"created train/validation split with first {self.num_train} samples " +
                f"for training and last {self.num_validation} samples for validation."
            )
        self.all_indices = range(len(self.meta['dataset']))

        # set up dataloaders for each set
        self.train_loader = DataLoader(
            self.train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        self.validation_loader = DataLoader(
            self.validation,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        self.test_loader = DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        self.all_loader = DataLoader(
            self.meta['dataset'],
            batch_size=1,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        self.inference_loader = DataLoader(
            self.meta['dataset'],
            batch_size=1,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
