"""
Generic module
"""
import torch
import os
import csv
import getpass
from torch import nn
from time import time
from datetime import datetime

from blip.utils.logger import Logger
from blip.utils.config import ConfigParser

from blip.dataset.wire_plane import WirePlanePointCloud
from blip.dataset.blip import BlipDataset
from blip.utils.loader import Loader
from blip.models import ModelHandler
from blip.losses import LossHandler
from blip.optimizers import Optimizer
from blip.metrics import MetricHandler
from blip.trainer import Trainer
from blip.utils.sampling import *
from blip.utils.grouping import *
from blip.utils.callbacks import CallbackHandler
from blip.utils.utils import get_files, save_model


class Module:
    """
    """
    def __init__(self,
        name:   str,
        config_file:    str="",
    ):
        self.name = name
        
        self.logger = Logger(self.name, file_mode='w')
        self.logger.info(f"configuring module.")

        self.dataset = None
        self.loader = None
        self.model = None
        self.criterion = None
        self.metrics = None
        self.optimizer = None
        self.callbacks = None
        self.trainer = None

        if config_file != "":
            self.logger.info(f"parsing config file: {config_file}.")
            self.config = ConfigParser(config_file).data
            self.parse_config()
        else:
            self.logger.info(f"no config file specified for module at constructor.")

    def set_config(self,
        config_file:    str
    ):
        self.logger.info(f"parsing config file: {config_file}.")
        self.config = ConfigParser(config_file).data
        self.parse_config()
    
    def parse_config(self):
        """
        """
        self.parse_dataset()
        self.parse_loader()
        self.parse_model()
        #self.parse_optimizer()

        training_loop = enumerate(self.loader.train_loader, 0)

        for ii, data in training_loop:
            output = self.model.model(data)
            print(output)



    def parse_dataset(self):
        """
        """
        if "dataset" not in self.config.keys():
            self.logger.warn("no dataset in config file.")
            return
        self.logger.info("configuring dataset.")
        dataset_config = self.config['dataset']

        # check for processing simulation files
        if "simulation_files" in dataset_config and dataset_config["process_simulation"]:
            for ii, simulation_file in enumerate(dataset_config["simulation_files"]):
                self.logger.info(f"processing simulation file: {simulation_file}.")
                wire_plane_dataset = WirePlanePointCloud(
                    f"{self.name}_simulation_{ii}",
                    simulation_file
                )
                wire_plane_dataset.generate_training_data()
        # check for the type of data set
        if dataset_config["dataset_type"] == "wire_plane":
            self.dataset = BlipDataset(
                name = f"{self.name}_wire_plane_dataset",
                input_files=dataset_config["dataset_files"],
                root="."
            )
    
    def parse_loader(self):
        """
        """
        if "loader" not in self.config.keys():
            self.logger.warn("no loader in config file.")
            return
        self.logger.info("configuring loader.")
        loader_config = self.config['loader']
        self.loader = Loader(
            self.dataset,
            loader_config["batch_size"],
            loader_config["test_split"],
            loader_config["test_seed"],
            loader_config["validation_split"],
            loader_config["validation_seed"],
            loader_config["num_workers"]
        )
    
    def parse_model(self):
        """
        """
        if "model" not in self.config.keys():
            self.logger.error("no model in config file!")
            return
        self.logger.info("configuring model.")
        model_config = self.config["model"]
        self.model = ModelHandler(
            "blip_model",
            model_config
        )

    def parse_optimizer(self):
        """
        """
        if "optimizer" not in self.config.keys():
            self.logger.warn("no optimizer in config file.")
            return
        self.logger.info("configuring optimizer.")
        optimizer_config = self.config['optimizer']
        self.optimizer = Optimizer(
            self.model.model,
            optimizer=optimizer_config["optimizer_type"],
            learning_rate=float(optimizer_config["learning_rate"]),
            betas=optimizer_config["betas"],
            epsilon=float(optimizer_config["epsilon"]),
            momentum=float(optimizer_config["momentum"]),
            weight_decay=float(optimizer_config["weight_decay"])
        )
        
