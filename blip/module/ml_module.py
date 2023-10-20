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
from os import listdir
from os.path import isfile, join
import shutil
import copy
import random

from blip.models import ModelHandler
from blip.module.common import *
from blip.module.generic_module import GenericModule
from blip.losses import LossHandler
from blip.optimizers import Optimizer
from blip.metrics import MetricHandler
from blip.trainer import Trainer
from blip.utils.sampling import *
from blip.utils.grouping import *
from blip.utils.callbacks import CallbackHandler
from blip.utils.utils import get_files, save_model, flatten_dict, generate_combinations_from_arrays


class MachineLearningModule(GenericModule):
    """
    The module class helps to organize meta data and objects related to different tasks
    and execute those tasks based on a configuration file.  The spirit of the 'Module' class
    is to mimic some of the functionality of LArSoft, e.g. where you can specify a chain
    of tasks to be completed, the ability to have nested config files where default parameters
    can be overwritten.
    """
    def __init__(self,
        name:   str,
        config: dict={},
        mode:   str='',
        meta:   dict={}
    ):
        self.name = name + "_ml_module"
        super(MachineLearningModule, self).__init__(
            self.name, config, mode, meta
        )
    
    def parse_config(self):
        """
        """
        self.check_config()

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.metrics = None
        self.callbacks = None
        self.trainer = None

        self.parse_model()
        self.parse_loss()
        self.parse_optimizer()
        self.parse_metrics()
        self.parse_callbacks()
        self.parse_training()
        self.parse_inference()
        self.parse_hyper_parameters()
     
    def check_config(self):
        if "model" not in self.config.keys():
            self.logger.error(f'"model" section not specified in config!')
        
        if self.mode == "training":
            if "criterion" not in self.config.keys():
                self.logger.error(f'"criterion" section not specified in config!')
            if "optimizer" not in self.config.keys():
                self.logger.error(f'"optimizer" section not specified in config!')
            if "metrics" not in self.config.keys():
                self.logger.warn(f'"metrics" section not specified in config!')
            if "callbacks" not in self.config.keys():
                self.logger.warn(f'"callbacks" section not specified in config!')
            if "training" not in self.config.keys():
                self.logger.error(f'"training" section not specified in config!')
                
        if self.mode == "inference":
            if "inference" not in self.config.keys():
                self.logger.error(f'"inference" section not specified in config!')

        if self.mode == "hyper_parameter_scan":
            if "hyper_parameters" not in self.config.keys():
                self.logger.error(f'"hyper_parameters" section not specified in config!')
        
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
            self.name,
            optimizer_config,
            self.model.model
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
            seed=training_config['seed']
        )
    
    def parse_inference(self):
        if "inference" not in self.config.keys():
            self.logger.warn("no inference in config file.")
            return
        if self.trainer == None:
            self.trainer = Trainer(
                self.model.model,
                self.criterion,
                self.optimizer,
                self.metrics,
                self.callbacks,
                meta=self.meta
            )

        if "layers" in self.config["inference"].keys():
            for layer in self.config["inference"]["layers"]:
                if layer not in self.model.model.forward_views.keys():
                    self.logger.error(f"layer '{layer}' not in the model forward views!  Possible views: {self.model.model.forward_views.keys()}")
                self.module_data_product[layer] = None
        if "outputs" in self.config["inference"].keys():
            for output in self.config["inference"]["outputs"]:
                self.module_data_product[output] = None
    
    def parse_hyper_parameters(self):
        """
        """
        if "hyper_parameters" not in self.config.keys():
            self.logger.warn("no hyper_parameters in config file.")
            return
        self.logger.info("configuring hyper_parameters")
        hyper_parameters_config = self.config["hyper_parameters"]
        model_config = self.config["model"]
        if "iterations" not in hyper_parameters_config.keys():
            self.logger.error("no 'iterations' specified in hyper_parameters config!")
        if "search_type" not in hyper_parameters_config.keys():
            self.logger.error("no 'search_type' specified in hyper_parameters config!")
        if "model_parameters" not in hyper_parameters_config.keys():
            self.logger.error("no 'model_parameters' specified in hyper_parameters config!")
        self.iterations = hyper_parameters_config["iterations"]
        self.search_type = hyper_parameters_config["search_type"]
        self.hyper_parameters = {
            f'iteration_{ii}': copy.deepcopy(model_config)
            for ii in range(self.iterations)
        }
        # code for generating random hyper-parameters
        if self.search_type == 'grid':
            self.generate_grid_hyper_parameters(hyper_parameters_config)
        elif self.search_type == 'random':
            self.generate_random_hyper_parameters(hyper_parameters_config)

    def generate_grid_hyper_parameters(self,
        hyper_parameters_config
    ):
        model_parameters = hyper_parameters_config["model_parameters"]
        self.parameter_paths = flatten_dict(model_parameters)
        self.parameter_combinations = generate_combinations_from_arrays(
            {tuple(k): v for k, v in self.parameter_paths if isinstance(v, list)}
        )
        random.shuffle(self.parameter_combinations)
        if len(self.parameter_combinations) < len(self.hyper_parameters.keys()):
            self.logger.info(f'number of iterations {self.iterations} larger than possible combinations {len(self.parameter_combinations)}.  Setting number of iterations to {len(self.parameter_combinations)}.')
            self.iterations = len(self.parameter_combinations)
            self.hyper_parameters = {
                f'iteration_{ii}': copy.deepcopy(self.config["model"])
                for ii in range(self.iterations)
            }
        for ii, iteration in enumerate(self.hyper_parameters.keys()):
            for jj, path_value_pair in enumerate(self.parameter_paths):
                current_parameters = self.hyper_parameters[iteration]
                for key in path_value_pair[0][:-1]:
                    current_parameters = current_parameters[key]
                current_parameters[path_value_pair[0][-1]] = self.parameter_combinations[ii][jj]

    def generate_random_hyper_parameters(self,
        hyper_parameters_config
    ):
        model_parameters = hyper_parameters_config["model_parameters"]
        
    def run_hyper_parameter_scan(self):
        self.logger.info(f"running hyper_parameter scan over {self.iterations} iterations")
        optimizer_config = self.config['optimizer']
        training_config = self.config['training']
        if 'run_name' in self.config['training'].keys():
            now = self.config['training']['run_name'] + f"_{datetime.now()}"
        else:
            now = self.name + f"_{datetime.now()}"
        for ii, iteration in enumerate(self.hyper_parameters.keys()):
            self.model = ModelHandler(
                self.name,
                self.hyper_parameters[iteration],
                meta=self.meta
            )
            self.optimizer = Optimizer(
                self.name,
                optimizer_config,
                self.model.model
            )
            self.parse_loss()
            self.parse_metrics()
            self.parse_callbacks()
            self.trainer = Trainer(
                self.model.model,
                self.criterion,
                self.optimizer,
                self.metrics,
                self.callbacks,
                meta=self.meta,
                seed=training_config['seed']
            )
            self.trainer.train(
                epochs=self.config['training']['epochs'],
                checkpoint=self.config['training']['checkpoint'],
                progress_bar=self.config['training']['progress_bar'],
                rewrite_bar=self.config['training']['rewrite_bar'],
                save_predictions=self.config['training']['save_predictions'],
                no_timing=self.config['training']['no_timing']
            )
            os.makedirs(f"{self.meta['local_scratch']}/runs/{now}/{iteration}/")
            if os.path.isdir(f"{self.meta['local_scratch']}/predictions/"):
                shutil.copytree(f"{self.meta['local_scratch']}/predictions/", f"{self.meta['local_scratch']}/runs/{now}/{iteration}/predictions/", dirs_exist_ok=True)
            if os.path.isdir(f"{self.meta['local_scratch']}/plots/"):
                shutil.copytree(f"{self.meta['local_scratch']}/plots/", f"{self.meta['local_scratch']}/runs/{now}/{iteration}/plots/", dirs_exist_ok=True)
            if os.path.isdir(f"{self.meta['local_scratch']}/models/"):
                shutil.copytree(f"{self.meta['local_scratch']}/models/", f"{self.meta['local_scratch']}/runs/{now}/{iteration}/models/", dirs_exist_ok=True)
            shutil.copytree(f"{self.meta['local_scratch']}/.logs/", f"{self.meta['local_scratch']}/runs/{now}/{iteration}/.logs/", dirs_exist_ok=True)
            shutil.copytree(f"{self.meta['local_scratch']}/.checkpoints/", f"{self.meta['local_scratch']}/runs/{now}/{iteration}/.checkpoints/", dirs_exist_ok=True)
            shutil.copy(self.meta['config_file'], f"{self.meta['local_scratch']}/runs/{now}/{iteration}/")
            if os.path.isfile(f"{self.meta['local_scratch']}/losses.npz"):
                shutil.copy(f"{self.meta['local_scratch']}/losses.npz", f"{self.meta['local_scratch']}/runs/{now}/{iteration}/")
            if os.path.isfile(f"{self.meta['local_scratch']}/metrics.npz"):
                shutil.copy(f"{self.meta['local_scratch']}/metrics.npz", f"{self.meta['local_scratch']}/runs/{now}/{iteration}/")
            if os.path.isfile(f"{self.meta['local_scratch']}/confusion_matrix.npz"):
                shutil.copy(f"{self.meta['local_scratch']}/confusion_matrix.npz", f"{self.meta['local_scratch']}/runs/{now}/{iteration}/")
        np.savez(
            f"{self.meta['local_scratch']}/runs/{now}/hyper_parameters.npz",
            hyper_parameters=self.hyper_parameters
        )
    
    def run_bayes_hyper_parameter_scan(self):
        self.logger.info(f"running hyper_parameter scan over {self.iterations} iterations")
        optimizer_config = self.config['optimizer']
        training_config = self.config['training']
    
    def run_linear_evaluation(self):
        self.logger.info(f"running linear_evaluation protocol")

    def run_module(self):
        if self.mode == 'training':
            self.module_data_product['predictions'] = self.trainer.train(
                epochs=self.config['training']['epochs'],
                checkpoint=self.config['training']['checkpoint'],
                progress_bar=self.config['training']['progress_bar'],
                rewrite_bar=self.config['training']['rewrite_bar'],
                save_predictions=self.config['training']['save_predictions'],
                no_timing=self.config['training']['no_timing']
            )
            # save model/data/config
            if 'run_name' in self.config['training'].keys():
                now = self.config['training']['run_name'] + f"_{datetime.now()}"
            else:
                now = self.name + f"_{datetime.now()}"
            os.makedirs(f"{self.meta['local_scratch']}/runs/{now}")
            if os.path.isdir(f"{self.meta['local_scratch']}/predictions/"):
                shutil.copytree(f"{self.meta['local_scratch']}/predictions/", f"{self.meta['local_scratch']}/runs/{now}/predictions/", dirs_exist_ok=True)
            if os.path.isdir(f"{self.meta['local_scratch']}/plots/"):
                shutil.copytree(f"{self.meta['local_scratch']}/plots/", f"{self.meta['local_scratch']}/runs/{now}/plots/", dirs_exist_ok=True)
            if os.path.isdir(f"{self.meta['local_scratch']}/models/"):
                shutil.copytree(f"{self.meta['local_scratch']}/models/", f"{self.meta['local_scratch']}/runs/{now}/models/", dirs_exist_ok=True)
            shutil.copytree(f"{self.meta['local_scratch']}/.logs/", f"{self.meta['local_scratch']}/runs/{now}/.logs/", dirs_exist_ok=True)
            shutil.copytree(f"{self.meta['local_scratch']}/.checkpoints/", f"{self.meta['local_scratch']}/runs/{now}/.checkpoints/", dirs_exist_ok=True)
            shutil.copy(self.meta['config_file'], f"{self.meta['local_scratch']}/runs/{now}")
            if os.path.isfile(f"{self.meta['local_scratch']}/losses.npz"):
                shutil.copy(f"{self.meta['local_scratch']}/losses.npz", f"{self.meta['local_scratch']}/runs/{now}/")
            if os.path.isfile(f"{self.meta['local_scratch']}/metrics.npz"):
                shutil.copy(f"{self.meta['local_scratch']}/metrics.npz", f"{self.meta['local_scratch']}/runs/{now}/")
            if os.path.isfile(f"{self.meta['local_scratch']}/confusion_matrix.npz"):
                shutil.copy(f"{self.meta['local_scratch']}/confusion_matrix.npz", f"{self.meta['local_scratch']}/runs/{now}/")
        elif self.mode == 'inference':
            self.module_data_product['predictions'] = self.trainer.inference(
                layers=self.config['inference']['layers'],
                outputs=self.config['inference']['outputs'],
                progress_bar=self.config['inference']['progress_bar'],
                rewrite_bar=self.config['inference']['rewrite_bar'],
                save_predictions=self.config['inference']['save_predictions']
            )
        elif self.mode == 'hyper_parameter_scan':
            if self.search_type == 'grid' or self.search_type == 'random':
                self.run_hyper_parameter_scan()
            else:
                self.run_bayes_hyper_parameter_scan()
        elif self.mode == 'linear_evaluation':
            self.run_linear_evaluation()
        else:
            self.logger.warning(f"current mode {self.mode} not an available type!")

        