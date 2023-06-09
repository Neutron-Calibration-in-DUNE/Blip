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

from blip.clustering.clusterer import Clusterer
from blip.dataset.wire_plane import WirePlanePointCloud
from blip.dataset.blip import BlipDataset
from blip.utils.loader import Loader
from blip.utils.sparse_loader import SparseLoader
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

        self.analysis = None
        self.clustering = None
        self.dataset = None
        self.loader = None
        self.model = None
        self.criterion = None
        self.metrics = None
        self.optimizer = None
        self.callbacks = None
        self.topology = None
        self.trainer = None

        self.config_file = config_file
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
        self.config_file = config_file
        self.config = ConfigParser(self.config_file).data
        self.parse_config()
    
    def parse_config(self):
        """
        """
        self.check_config()
        self.parse_module()
        self.parse_dataset()
        self.parse_loader()
        self.parse_model()
        self.parse_loss()
        self.parse_optimizer()
        self.parse_metrics()
        self.parse_callbacks()
        self.parse_training()
        self.parse_cluster()

        self.run_module()

    def check_config(self):
        if "module" not in self.config.keys():
            self.logger.error(f'"module" section not specified in config!')
        if "dataset" not in self.config.keys():
            self.logger.error(f'"dataset" section not specified in config!')
        
        if self.config["module"]["module_type"] == "training":
            if "loader" not in self.config.keys():
                self.logger.error(f'"loader" section not specified in config!')
            if "model" not in self.config.keys():
                self.logger.error(f'"model" section not specified in config!')
            if "criterion" not in self.config.keys():
                self.logger.error(f'"criterion" section not specified in config!')
            if "optimizer" not in self.config.keys():
                self.logger.error(f'"optimizer" section not specified in config!')
            if "metrics" not in self.config.keys():
                self.logger.error(f'"metrics" section not specified in config!')
            if "callbacks" not in self.config.keys():
                self.logger.error(f'"callbacks" section not specified in config!')
        
            loader_config = self.config['loader']
            model_config = self.config['model']
            criterion_config = self.config['criterion']
            optimizer_config = self.config['optimizer']
            metrics_config = self.config['metrics']
            callbacks_config = self.config['callbacks']

            if metrics_config:
                for item in metrics_config.keys():
                    if item == "custom_metric_file" or item == "custom_metric_name":
                        continue
                    self.config["metrics"][item]["consolidate_classes"] = dataset_config["consolidate_classes"]
        
        dataset_config = self.config['dataset']

        if dataset_config["consolidate_classes"] is not None:
            for item in dataset_config["consolidate_classes"]:
                if item not in dataset_config["classes"]:
                    self.logger.error(
                        f"(dataset config) specified class in 'dataset: consolidate_classes' [{item}]" + 
                        f" not in 'dataset: consolidate_classes [{dataset_config['consolidate_classes']}]"
                    )
        
    
    def parse_module(self):
        # check for module type
        if "module_type" not in self.config["module"].keys():
            self.logger.error(f'"module_type" not specified in config!')
        self.module_type = self.config["module"]["module_type"]
        # check for devices
        if "gpu" not in self.config["module"].keys():
            self.logger.warn(f'"gpu" not specified in config!')
            self.gpu = None
        else:
            self.gpu = self.config["module"]["gpu"]
        if "gpu_device" not in self.config["module"].keys():
            self.logger.warn(f'"gpu_device" not specified in config!')
            self.gpu_device = None
        else:
            self.gpu_device = self.config["module"]["gpu_device"]
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA is available with devices:")
            for ii in range(torch.cuda.device_count()):
                device_properties = torch.cuda.get_device_properties(ii)
                cuda_stats = f"name: {device_properties.name}, "
                cuda_stats += f"compute: {device_properties.major}.{device_properties.minor}, "
                cuda_stats += f"memory: {device_properties.total_memory}"
                self.logger.info(f" -- device: {ii} - " + cuda_stats)

        # set gpu settings
        if self.gpu:
            if torch.cuda.is_available():
                if self.gpu_device >= torch.cuda.device_count() or self.gpu_device < 0:
                    self.logger.warn(f"desired gpu_device '{self.gpu_device}' not available, using device '0'")
                    self.gpu_device = 0
                self.device = torch.device(f"cuda:{self.gpu_device}")
                self.logger.info(
                    f"CUDA is available, using device {self.gpu_device}" + 
                    f": {torch.cuda.get_device_name(self.gpu_device)}"
                )
            else:
                self.gpu == False
                self.logger.warn(f"CUDA not available! Using the cpu")
                self.device = torch.device("cpu")
        else:
            self.logger.info(f"using cpu as device")
            self.device = torch.device("cpu")

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
                if "simulation_folder" in dataset_config:
                    simulation_file = dataset_config["simulation_folder"] + simulation_file
                self.logger.info(f"processing simulation file: {simulation_file}.")
                wire_plane_dataset = WirePlanePointCloud(
                    f"{self.name}_simulation_{ii}",
                    simulation_file
                )
                wire_plane_dataset.generate_training_data()
        dataset_config["name"] = f"{self.name}_dataset"
        dataset_config["device"] = self.device
        self.dataset = BlipDataset(dataset_config)

    def parse_loader(self):
        """
        """
        if "loader" not in self.config.keys():
            self.logger.warn("no loader in config file.")
            return
        self.logger.info("configuring loader.")
        loader_config = self.config['loader']
        if loader_config["loader_type"] == "minkowski":
            self.loader = SparseLoader(
                self.dataset,
                loader_config["batch_size"],
                loader_config["test_split"],
                loader_config["test_seed"],
                loader_config["validation_split"],
                loader_config["validation_seed"],
                loader_config["num_workers"]
            )
        else:
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
            self.logger.warn("no model in config file.")
            return
        self.logger.info("configuring model.")
        model_config = self.config["model"]
        self.model = ModelHandler(
            "blip_model",
            model_config,
            device=self.device
        )

    def parse_loss(self):
        """
        """
        if "criterion" not in self.config.keys():
            self.logger.warn("no criterion in config file.")
            return
        self.logger.info("configuring criterion.")
        criterion_config = self.config['criterion']
        # add in class weight numbers for loss functions
        self.criterion = LossHandler(
            "blip",
            criterion_config,
            device=self.device
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
            metrics_config,
            device=self.device
        )
    
    def parse_callbacks(self):
        """
        """
        if "callbacks" not in self.config.keys():
            self.logger.warn("no callbacks in config file.")
            return
        self.logger.info("configuring callbacks.")
        callbacks_config = self.config['callbacks']
        if "LossCallback" in callbacks_config.keys():
            callbacks_config["LossCallback"] = {"criterion_list": self.criterion}
        if "ConfusionMatrixCallback" in callbacks_config.keys():
            callbacks_config["ConfusionMatrixCallback"]["metrics_list"] = self.metrics
        self.callbacks = CallbackHandler(
            "blip_callbacks",
            callbacks_config,
            device=self.device
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
            device=self.device,
            gpu=self.gpu,
            seed=training_config['seed']
        )

    def parse_cluster(self):
        """
        """
        if "clustering" not in self.config.keys():
            self.logger.warn("no clustering in config file.")
            return
        self.logger.info("configuring clustering.")
        cluster_config = self.config['clustering']
        self.clustering = Clusterer(
            self.name + "_cluster", 
            device=self.device,
            gpu=self.gpu,
            seed=cluster_config['seed']
        )
    
    def run_module(self):
        """
        Once everything is configured, we run the module here.
        """
        if self.module_type == 'training':
            self.trainer.train(
                self.loader,
                epochs=self.config['training']['epochs'],
                checkpoint=self.config['training']['checkpoint'],
                progress_bar=self.config['training']['progress_bar'],
                rewrite_bar=self.config['training']['rewrite_bar'],
                save_predictions=self.config['training']['save_predictions'],
                no_timing=self.config['training']['no_timing']
            )

            # save model/data/config
            if 'run_name' in self.config['training'].keys():
                save_model(self.config['training']['run_name'], self.config_file)
            else:
                save_model(self.name, self.config_file)
                
        elif self.module_type == 'clustering':
            self.clustering.cluster(
                self.loader,
                progress_bar=self.config['clustering']['progress_bar'],
                rewrite_bar=self.config['clustering']['rewrite_bar'],
                save_predictions=self.config['clustering']['save_predictions'],
                no_timing=self.config['clustering']['no_timing']
            )
