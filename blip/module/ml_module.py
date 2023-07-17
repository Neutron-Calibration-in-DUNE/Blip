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
from blip.utils.loader import Loader
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
from blip.utils.utils import get_files, save_model


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
        self.parse_model()
        self.parse_loss()
        self.parse_optimizer()
        self.parse_metrics()
        self.parse_callbacks()
        self.parse_training()
            
    def check_config(self):
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
            seed=training_config['seed']
        )
    
    def run_module(self):
        if self.mode == 'training':
            self.trainer.train(
                epochs=self.config['training']['epochs'],
                checkpoint=self.config['training']['checkpoint'],
                progress_bar=self.config['training']['progress_bar'],
                rewrite_bar=self.config['training']['rewrite_bar'],
                save_predictions=self.config['training']['save_predictions'],
                no_timing=self.config['training']['no_timing']
            )
        elif self.mode == 'inference':
            self.trainer.inference(
                progress_bar=self.config['training']['progress_bar'],
                rewrite_bar=self.config['training']['rewrite_bar']
            )

        # save model/data/config
        if 'run_name' in self.config['training'].keys():
            save_model(self.config['training']['run_name'], self.meta['config_file'])
        else:
            save_model(self.name, self.meta['config_file'])