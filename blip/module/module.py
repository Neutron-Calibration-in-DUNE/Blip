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

from blip.utils.logger import Logger, default_logger
from blip.utils.config import ConfigParser

from blip.clusterer.clusterer import Clusterer
from blip.dataset.arrakis import Arrakis
from blip.dataset.blip import BlipDataset
from blip.clustering_algorithms import ClusteringAlgorithmHandler
from blip.utils.loader import Loader
from blip.utils.sparse_loader import SparseLoader
from blip.models import ModelHandler
from blip.module.common import *
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
    The module class helps to organize meta data and objects related to different tasks
    and execute those tasks based on a configuration file.  The spirit of the 'Module' class
    is to mimic some of the functionality of LArSoft, e.g. where you can specify a chain
    of tasks to be completed, the ability to have nested config files where default parameters
    can be overwritten.
    """
    def __init__(self,
        config_file:    str="",
    ):
        self.analysis = None
        self.clusterer = None
        self.clustering_algorithms = None
        self.parameter_set = None
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
            self.config = ConfigParser(config_file).data
            if "module" not in self.config.keys():
                self.name = "blip"
            else:
                if "module_name" not in self.config["module"].keys():
                    self.config["module"]["module_name"] = "blip"
                self.name = self.config["module"]["module_name"]
            self.logger = Logger(self.name, output="both", file_mode='w')
            system_info = self.logger.get_system_info()
            for key, value in system_info.items():
                self.logger.info(f"system_info - {key}: {value}")
            self.logger.info(f"configuring module.")
        else:
            default_logger.error(f"no config file specified for module at constructor.")
        
        self.logger.info(f"parsing config file: {config_file}.")
        self.meta = {}
        self.parse_config()

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
        if "training" in self.config["module"]["module_mode"]:
            self.parse_model()
            self.parse_loss()
            self.parse_optimizer()
            self.parse_metrics()
            self.parse_callbacks()
            self.parse_training()
        if "parameter_scan" in self.config["module"]["module_mode"]:
            self.parse_clustering_algorithms()
            self.parse_metrics()
            self.parse_callbacks()
            self.parse_clusterer()
            
        self.run_module()

    def check_config(self):
        if "module" not in self.config.keys():
            self.logger.error(f'"module" section not specified in config!')
        if "dataset" not in self.config.keys():
            self.logger.error(f'"dataset" section not specified in config!')
        
        dataset_config = self.config['dataset']

        if dataset_config["consolidate_classes"] is not None:
            for item in dataset_config["consolidate_classes"]:
                if item not in dataset_config["classes"]:
                    self.logger.error(
                        f"(dataset config) specified class in 'dataset: consolidate_classes' [{item}]" + 
                        f" not in 'dataset: consolidate_classes [{dataset_config['consolidate_classes']}]"
                    )

        if self.config["module"]["module_type"] == "ml":
            if "loader" not in self.config.keys():
                self.logger.error(f'"loader" section not specified in config!')
            if self.config["module"]["module_mode"] == "dataprep":
                return
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
            if "training" not in self.config.keys():
                self.logger.error(f'"training" section not specified in config!')
        
            loader_config = self.config['loader']
            model_config = self.config['model']
            criterion_config = self.config['criterion']
            optimizer_config = self.config['optimizer']
            metrics_config = self.config['metrics']
            callbacks_config = self.config['callbacks']

        if self.config["module"]["module_type"] == "clustering":
            if "metrics" not in self.config.keys():
                self.logger.error(f'"metrics" section not specified in config!')
            if "callbacks" not in self.config.keys():
                self.logger.error(f'"callbacks" section not specified in config!')
        
    def parse_module(self):
        # First we check the "module_type" to make sure it conforms to our 
        # internal specification.
        if "module_type" not in self.config["module"].keys():
            self.logger.error(f'"module_type" not specified in config!')
        if not isinstance(self.config["module"]["module_type"], str) and not isinstance(self.config["module"]["module_type"], list):
            self.logger.error(f'"module_type" must be either a list or a str, but got {type(self.config["module"]["module_type"])}!')
        if isinstance(self.config["module"]["module_type"], str):
            self.config["module"]["module_type"] = [self.config["module"]["module_type"]]
        self.module_type = self.config["module"]["module_type"]
        for ii, module in enumerate(self.module_type):
            if not isinstance(module, str):
                self.logger.error(f'"module_type" "{module}" at index {ii} is not of type str!')
            if module not in module_types.keys():
                self.logger.error(f'"module_type" {module} at index {ii} is not an allowed type!')
        self.logger.info(f'module_type set to "{self.module_type}"')
        
        # next we check the module_mode associated to each type.
        if "module_mode" not in self.config["module"].keys():
            self.logger.error(f'"module_mode" not specified in config!')
        if not isinstance(self.config["module"]["module_mode"], str) and not isinstance(self.config["module"]["module_mode"], list):
            self.logger.error(f'"module_mode" must be either a list or a string!')
        if isinstance(self.config["module"]["module_mode"], str):
            self.config["module"]["module_mode"] = [self.config["module"]["module_mode"]]
        self.module_mode = self.config["module"]["module_mode"]
        for ii, module in enumerate(self.module_mode):
            if not isinstance(module, str):
                self.logger.error(f'"module_mode" "{module}" at index {ii} is not of mode str!')
            if module not in module_types[self.module_type[ii]]:
                self.logger.error(f'"module_mode" {module} at index {ii} is not an allowed mode for type {self.module_type[ii]}!')
        self.logger.info(f'module_mode set to "{self.module_mode}"')
        
        if len(self.module_type) != len(self.module_mode):
            self.logger.error(f'module_type and module_mode must have the same number of entries!')
        
        # Eventually we will want to check that the order of the modules makes sense,
        # and that the data products are compatible and available for the different modes.

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
                self.meta['device'] = torch.device(f"cuda:{self.gpu_device}")
                self.logger.info(
                    f"CUDA is available, using device {self.gpu_device}" + 
                    f": {torch.cuda.get_device_name(self.gpu_device)}"
                )
            else:
                self.gpu == False
                self.logger.warn(f"CUDA not available! Using the cpu")
                self.meta['device'] = torch.device("cpu")
        else:
            self.logger.info(f"using cpu as device")
            self.meta['device'] = torch.device("cpu")

    def parse_dataset(self):
        """
        """
        if "dataset" not in self.config.keys():
            self.logger.warn("no dataset in config file.")
            return
        self.logger.info("configuring dataset.")
        dataset_config = self.config['dataset']

        # default to what's in the configuration file. May decide to deprecate in the future
        if ("simulation_folder" in dataset_config):
            simulation_folder = dataset_config["simulation_folder"]
            self.logger.info(
                    f"Set simulation file folder from configuration. " +
                    f" simulation_folder : {simulation_folder}"
                    )
        elif ('BLIP_SIMULATION_PATH' in os.environ ):
            self.logger.debug(f'Found BLIP_SIMULATION_PATH in environment')
            simulation_folder = os.environ['BLIP_SIMULATION_PATH']
            self.logger.info(
                    f"Setting simulation path from Enviroment." +
                    f" BLIP_SIMULATION_PATH = {simulation_folder}"
                    )
        else:
            self.logger.error(f'No dataset_folder specified in environment or configuration file!')

        # check for processing simulation files
        if "simulation_files" in dataset_config and dataset_config["process_simulation"]:
            arrakis_dataset = Arrakis(
                self.name,
                dataset_config
            )

        dataset_config["name"] = f"{self.name}_dataset"
        dataset_config["device"] = self.meta['device']
        if self.config["module"]["module_mode"] == "dataprep":
            return
        self.dataset = BlipDataset(dataset_config)
        self.meta['dataset'] = self.dataset

    def parse_loader(self):
        """
        """
        if "loader" not in self.config.keys():
            self.logger.warn("no loader in config file.")
            return
        self.logger.info("configuring loader.")
        loader_config = self.config['loader']
        self.loader = Loader(
            self.name,
            self.dataset,
            loader_config
        )
        self.meta['loader'] = self.loader
        
    def parse_model(self):
        """
        """
        if "model" not in self.config.keys():
            self.logger.warn("no model in config file.")
            return
        self.logger.info("configuring model.")
        model_config = self.config["model"]
        self.model = ModelHandler(
            self.name,
            model_config,
            meta=self.meta
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
            self.name,
            criterion_config,
            meta=self.meta
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
            self.name,
            metrics_config,
            meta=self.meta
        )
    
    def parse_callbacks(self):
        """
        """
        if "callbacks" not in self.config.keys():
            self.logger.warn("no callbacks in config file.")
            return
        self.logger.info("configuring callbacks.")
        callbacks_config = self.config['callbacks']
        if callbacks_config == None:
            self.logger.warn("no callbacks specified.")
        else:
            for callback in callbacks_config.keys():
                if callbacks_config[callback] == None:
                    callbacks_config[callback] = {}
                callbacks_config[callback]['criterion_handler'] = self.criterion
                callbacks_config[callback]['metrics_handler'] = self.metrics
        self.callbacks = CallbackHandler(
            self.name,
            callbacks_config,
            meta=self.meta
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
            meta=self.meta,
            gpu=self.gpu,
            seed=training_config['seed']
        )

    def parse_clustering_algorithms(self):
        """
        """
        if "clustering_algorithms" not in self.config.keys():
            self.logger.warn("no clustering_algorithms in config file.")
            return
        self.logger.info("configuring clustering_algorithms.")
        clustering_algorithm_config = self.config["clustering_algorithms"]
        self.clustering_algorithms = ClusteringAlgorithmHandler(
            self.name,
            clustering_algorithm_config,
            meta=self.meta
        )

    def parse_clusterer(self):
        """
        """
        if "clusterer" not in self.config.keys():
            self.logger.warn("no clusterer in config file.")
            return
        self.logger.info("configuring clusterer.")
        cluster_config = self.config['clusterer']
        self.clusterer = Clusterer(
            self.name, 
            clustering_algorithms=self.clustering_algorithms,
            clustering_metrics=self.metrics,
            clustering_callbacks=self.callbacks,
            meta=self.meta,
            gpu=self.gpu,
            seed=cluster_config['seed']
        )
    
    def run_module(self):
        """
        Once everything is configured, we run the module here.
        """
        for ii, module_type in enumerate(self.module_type):
            if module_type == "clustering":
                self.run_clustering_module(self.module_mode[ii])
            elif module_type == "ml":
                self.run_ml_module(self.module_mode[ii])
            elif module_type == "tda":
                self.run_tda_module(self.module_mode[ii])
    
    def run_clustering_module(self,
        clustering_mode
    ):
        if clustering_mode == "parameter_scan":
            self.clusterer.cluster(
                self.loader,
                num_parameters=self.config['clusterer']['num_parameters'],
                eps_range=self.config['clusterer']['eps_range'],
                progress_bar=self.config['clusterer']['progress_bar'],
                rewrite_bar=self.config['clusterer']['rewrite_bar'],
                save_predictions=self.config['clusterer']['save_predictions'],
                no_timing=self.config['clusterer']['no_timing']
            )
    
    def run_ml_module(self,
        ml_mode
    ):
        if ml_mode == 'dataprep':
            return
        if ml_mode == 'training':
            self.trainer.train(
                self.loader,
                epochs=self.config['training']['epochs'],
                checkpoint=self.config['training']['checkpoint'],
                progress_bar=self.config['training']['progress_bar'],
                rewrite_bar=self.config['training']['rewrite_bar'],
                save_predictions=self.config['training']['save_predictions'],
                no_timing=self.config['training']['no_timing']
            )
        elif ml_mode == 'inference':
            self.trainer.inference(
                self.loader,
                progress_bar=self.config['training']['progress_bar'],
                rewrite_bar=self.config['training']['rewrite_bar']
            )

        # save model/data/config
        if 'run_name' in self.config['training'].keys():
            save_model(self.config['training']['run_name'], self.config_file)
        else:
            save_model(self.name, self.config_file)