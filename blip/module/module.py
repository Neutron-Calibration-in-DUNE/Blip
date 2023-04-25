"""
Generic module
"""
import torch
import os
import csv
import getpass
from torch import nn
import torch.nn.functional as F
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
        self.parse_loss()
        self.parse_optimizer()
        self.parse_metrics()
        self.parse_callbacks()
        self.parse_training()

        self.trainer.train(
            self.loader,
            epochs=self.config['training']['epochs'],
            checkpoint=self.config['training']['checkpoint'],
            progress_bar=self.config['training']['progress_bar'],
            rewrite_bar=self.config['training']['rewrite_bar'],
            save_predictions=self.config['training']['save_predictions'],
            no_timing=self.config['training']['no_timing']
        )

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
                root=".",
                classes=dataset_config["classes"]
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
        self.model.model.set_device('cuda')

    def parse_loss(self):
        """
        """
        if "criterion" not in self.config.keys():
            self.logger.warn("no criterion in config file.")
            return
        self.logger.info("configuring criterion.")
        criterion_config = self.config['criterion']
        self.criterion = LossHandler(
            "blip_criterion",
            criterion_config
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
    
    def parse_metrics(self):
        """
        """
        if "metrics" not in self.config.keys():
            self.logger.warn("no metrics in config file.")
            return
        self.logger.info("configuring metrics.")
        metrics_config = self.config['metrics']
        self.metrics = MetricHandler(
            "blip_metrics",
            metrics_config
        )
    
    def parse_callbacks(self):
        """
        """
        if "callbacks" not in self.config.keys():
            self.logger.warn("no callbacks in config file.")
            return
        self.logger.info("configuring callbacks.")
        callbacks_config = self.config['callbacks']
        if "loss" in callbacks_config.keys():
            callbacks_config["loss"] = {"criterion_list": self.criterion}
        if "confusion_matrix" in callbacks_config.keys():
            callbacks_config["confusion_matrix"]["metrics_list"] = self.metrics
        self.callbacks = CallbackHandler(
            "blip_callbacks",
            callbacks_config
        )
    
    def parse_training(self):
        """
        """
        if "training" not in self.config.keys():
            self.logger.warn("no training in config file.")
            return
        self.logger.info("configuring training.")
        training_config = self.config['training']
        self.trainer = Trainer(
            self.model.model,
            self.criterion,
            self.optimizer,
            self.metrics,
            self.callbacks,
            gpu=training_config['gpu'],
            gpu_device=training_config['gpu_device'],
            seed=training_config['seed']
        )
