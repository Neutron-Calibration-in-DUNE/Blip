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

from blip.analysis.model_analyzer_handler import ModelAnalyzerHandler
from blip.models import ModelHandler
from blip.models import LinearEvaluation
from blip.module.common import *
from blip.module.generic_module import GenericModule
from blip.losses import LossHandler
from blip.optimizers import Optimizer
from blip.metrics import MetricHandler
from blip.trainer import Trainer
from blip.utils.sampling import *
from blip.utils.grouping import *
from blip.utils.callbacks import CallbackHandler
from blip.utils.utils import *

class MachineLearningModule(GenericModule):
    """
    The module class helps to organize meta data and objects related to different tasks
    and execute those tasks based on a configuration file.  The spirit of the 'Module' class
    is to mimic some of the functionality of LArSoft, e.g. where you can specify a chain
    of tasks to be completed, the ability to have nested config files where default parameters
    can be overwritten.

    The ML specific module runs in several different modes, 
    """
    def __init__(self,
        name:   str,
        config: dict={},
        mode:   str='',
        meta:   dict={}
    ):
        self.name = name
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
        self.parse_linear_evaluation()
        self.parse_model_analyzer()
     
    def check_config(self):
        if "model" not in self.config.keys():
            self.logger.warning(f'"model" section not specified in config!')
        
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
        
        if self.mode == "linear_evaluation":
            if "linear_evaluation" not in self.config.keys():
                self.logger.error(f"'linear_evaluation' section not specified in config!")
        else:
            if "linear_evaluation" in self.config.keys():
                self.post_training_linear_evaluation = True
                self.logger.info(f"setting up post training linear_evaluation.")
            else:
                self.post_training_linear_evaluation = False
        
        if self.mode == "model_analyzer":
            if "model_analyzer" not in self.config.keys():
                self.logger.error(f"'model_analyzer' section not specified in config!")
        
    def parse_model(self,
        name:   str=''
    ):
        """
        """
        if self.mode == "linear_evaluation":
            return
        if "model" not in self.config.keys():
            self.logger.warn("no model in config file.")
            return
        self.logger.info("configuring model.")
        model_config = self.config["model"]
        self.model = ModelHandler(
            self.name + name,
            model_config,
            meta=self.meta
        )

    def parse_loss(self,
        name:   str=''
    ):
        """
        """
        if "criterion" not in self.config.keys():
            self.logger.warn("no criterion in config file.")
            return
        self.logger.info("configuring criterion.")
        criterion_config = self.config['criterion']
        # add in class weight numbers for loss functions
        self.criterion = LossHandler(
            self.name + name,
            criterion_config,
            meta=self.meta
        )

    def parse_optimizer(self,
        name:   str=''
    ):
        """
        """
        if self.mode == "linear_evaluation":
            return
        if "optimizer" not in self.config.keys():
            self.logger.warn("no optimizer in config file.")
            return
        self.logger.info("configuring optimizer.")
        optimizer_config = self.config['optimizer']
        self.optimizer = Optimizer(
            self.name + name,
            optimizer_config,
            self.model.model
        )
    
    def parse_metrics(self,
        name:   str=''
    ):
        """
        """
        if "metrics" not in self.config.keys():
            self.logger.warn("no metrics in config file.")
            return
        self.logger.info("configuring metrics.")
        metrics_config = self.config['metrics']
        self.metrics = MetricHandler(
            self.name + name,
            metrics_config,
            meta=self.meta
        )
    
    def parse_callbacks(self,
        name:   str=''
    ):
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
            self.name + name,
            callbacks_config,
            meta=self.meta
        )
    
    def parse_training(self,
        name:   str=''
    ):
        """
        """
        if "training" not in self.config.keys():
            self.logger.warn("no training in config file.")
            return
        self.logger.info("configuring training.")
        training_config = self.config['training']
        if "iterations" not in self.config['training'].keys():
            self.config['training']['iterations'] = 1
        if self.mode == "linear_evaluation":
            return
        self.trainer = Trainer(
            self.name + name,
            self.model.model,
            self.criterion,
            self.optimizer,
            self.metrics,
            self.callbacks,
            meta=self.meta,
            seed=training_config['seed']
        )
    
    def parse_inference(self,
        name:   str=''
    ):
        if "inference" not in self.config.keys():
            self.logger.warn("no inference in config file.")
            return
        if self.trainer == None:
            self.trainer = Trainer(
                self.name + name,
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
    
    def parse_linear_evaluation(self,
        name:   str=''
    ):
        if "linear_evaluation" not in self.config.keys():
            self.logger.warn("no linear_evaluation in config file.")
            return
        self.logger.info("configuring linear_evaluation")
        linear_evaluation_config = self.config["linear_evaluation"]
        if "model_directory" in linear_evaluation_config.keys():
            self.linear_evaluation_model_directory = linear_evaluation_config['model_directory']
            if not os.path.isdir(self.linear_evaluation_model_directory):
                self.logger.error(f"linear_evaluation model_directory: {self.linear_evaluation_model_directory} does not exist!")
            self.logger.info(f"setting linear_evaluation model_directory to {self.linear_evaluation_model_directory}.")
        else:
            self.linear_evaluation_model_directory = ''
            self.logger.info(f"linear_evaluation model_directory not specified, setting to './'.")
        if "epochs" not in linear_evaluation_config.keys():
            self.logger.warn(f"linear_evaluation: epochs not specified in config.  Setting to 50.")
            self.config["linear_evaluation"]["epochs"] = 50

    def parse_model_analyzer(self,
        name:   str=''
    ):
        """
        """
        if "model_analyzer" not in self.config.keys():
            self.logger.warn("no model_analyzer in config file.")
            self.model_analyzer = None
            return
        self.logger.info("configuring model_analyzer")
        model_analyzer_config = self.config["model_analyzer"]
        self.model_analyzer = ModelAnalyzerHandler(
            self.name + name,
            model_analyzer_config,
            meta=self.meta
        )

    # TODO: Fix this so that it doesn't need to compute all the paths, but only a subset of
    # random ones, since the number of paths is N!, which is intractable.
    def generate_grid_hyper_parameters(self,
        hyper_parameters_config
    ):
        model_parameters = hyper_parameters_config["model_parameters"]
        self.parameter_paths = flatten_dict(model_parameters)
        self.logger.info(f"generating hyper parameter combinations.")
        self.parameter_combinations = generate_combinations_from_arrays(
            {tuple(k): v for k, v in self.parameter_paths if isinstance(v, list)}
        )
        self.logger.info(f"generated {len(self.parameter_combinations)} different possible hyper parameter combinations.")
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

    def save_iteration(self,
        folder
    ):
        # save model/data/config
        # create specific folder for this iteration
        os.makedirs(f"{self.meta['local_scratch']}/runs/{folder}")
        # move predictions folder to iteration
        if os.path.isdir(f"{self.meta['local_scratch']}/predictions/"):
            shutil.move(
                f"{self.meta['local_scratch']}/predictions/", 
                f"{self.meta['local_scratch']}/runs/{folder}/predictions/"
            )
        # move plots folder to iteration
        if os.path.isdir(f"{self.meta['local_scratch']}/plots/"):
            shutil.move(
                f"{self.meta['local_scratch']}/plots/", 
                f"{self.meta['local_scratch']}/runs/{folder}/plots/"
            )
        # move models folder to iteration
        if os.path.isdir(f"{self.meta['local_scratch']}/models/"):
            shutil.move(
                f"{self.meta['local_scratch']}/models/", 
                f"{self.meta['local_scratch']}/runs/{folder}/models/"
            )
        # move checkpoints folder to iteration
        shutil.move(
            f"{self.meta['local_scratch']}/.checkpoints/", 
            f"{self.meta['local_scratch']}/runs/{folder}/.checkpoints/", 
        )
        # copy logs folder to iteration
        shutil.copytree(
            f"{self.meta['local_scratch']}/.logs/", 
            f"{self.meta['local_scratch']}/runs/{folder}/.logs/", 
            dirs_exist_ok=True
        )
        # copy config file to iteration
        shutil.copy(self.meta['config_file'], f"{self.meta['local_scratch']}/runs/{folder}")
        # copy losses, metrics and confusion matrix 
        if os.path.isfile(f"{self.meta['local_scratch']}/losses.npz"):
            shutil.copy(f"{self.meta['local_scratch']}/losses.npz", f"{self.meta['local_scratch']}/runs/{folder}/")
        if os.path.isfile(f"{self.meta['local_scratch']}/metrics.npz"):
            shutil.copy(f"{self.meta['local_scratch']}/metrics.npz", f"{self.meta['local_scratch']}/runs/{folder}/")
        if os.path.isfile(f"{self.meta['local_scratch']}/confusion_matrix.npz"):
            shutil.copy(f"{self.meta['local_scratch']}/confusion_matrix.npz", f"{self.meta['local_scratch']}/runs/{folder}/")

    def run_module(self):
        if self.mode == 'training':
            self.run_training()
        elif self.mode == 'inference':
            self.module_data_product['predictions'] = self.trainer.inference(
                dataset_type=self.config['inference']['dataset_type'],
                layers=self.config['inference']['layers'],
                outputs=self.config['inference']['outputs'],
                progress_bar=self.config['inference']['progress_bar'],
                rewrite_bar=self.config['inference']['rewrite_bar'],
                save_predictions=self.config['inference']['save_predictions']
            )
        elif self.mode == 'model_analyzer':
            self.run_model_analyzer()
        elif self.mode == 'hyper_parameter_scan':
            if self.search_type == 'grid' or self.search_type == 'random':
                self.run_hyper_parameter_scan()
            else:
                self.run_bayes_hyper_parameter_scan()
        elif self.mode == 'linear_evaluation':
            self.run_linear_evaluation()
        else:
            self.logger.warning(f"current mode {self.mode} not an available type!")
    
    def run_training(self):
        if 'run_name' in self.config['training'].keys():
            self.now = self.config['training']['run_name'] + f"_{get_datetime()}"
        else:
            self.now = self.name + f"_{get_datetime()}"
        for jj in range(self.config['training']['iterations']):
            self.parse_model(f'_{jj}')
            self.parse_optimizer(f'_{jj}')
            self.parse_loss(f'_{jj}')
            self.parse_metrics(f'_{jj}')
            self.parse_callbacks(f'_{jj}')
            self.parse_training(f'_{jj}')
            self.module_data_product[f'predictions_{jj}'] = self.trainer.train(
                epochs=self.config['training']['epochs'],
                checkpoint=self.config['training']['checkpoint'],
                progress_bar=self.config['training']['progress_bar'],
                rewrite_bar=self.config['training']['rewrite_bar'],
                save_predictions=self.config['training']['save_predictions'],
                no_timing=self.config['training']['no_timing'],
                skip_metrics=self.config['training']['skip_metrics']
            )
            if self.model_analyzer is not None:
                self.model_analyzer.analyze(self.model.model)
            self.save_iteration(f"{self.now}/iteration_{jj}")
            if self.post_training_linear_evaluation:
                    self.run_post_training_linear_evaluation(
                        name=f'_{jj}',
                        model_folder=f"{self.now}/iteration_{jj}/"
                    )
    
    def run_model_analyzer(self):
        pass
    
    def run_hyper_parameter_scan(self):
        self.logger.info(f"running hyper_parameter scan over {self.iterations} iterations")
        training_config = self.config['training']
        if 'run_name' in self.config['training'].keys():
            self.now = self.config['training']['run_name'] + f"_{get_datetime()}"
        else:
            self.now = self.name + f"_{get_datetime()}"
        for ii, iteration in enumerate(self.hyper_parameters.keys()):
            for jj in range(self.config['training']['iterations']):
                self.model = ModelHandler(
                    self.name + f"_{ii}_{jj}",
                    self.hyper_parameters[iteration],
                    meta=self.meta
                )
                self.parse_optimizer(f'_{ii}_{jj}')
                self.parse_loss(f'_{ii}_{jj}')
                self.parse_metrics(f'_{ii}_{jj}')
                self.parse_callbacks(f'_{ii}_{jj}')
                self.parse_training(f'_{ii}_{jj}')
            
                self.module_data_product[f'predictions_{ii}_{jj}'] = self.trainer.train(
                    epochs=self.config['training']['epochs'],
                    checkpoint=self.config['training']['checkpoint'],
                    progress_bar=self.config['training']['progress_bar'],
                    rewrite_bar=self.config['training']['rewrite_bar'],
                    save_predictions=self.config['training']['save_predictions'],
                    no_timing=self.config['training']['no_timing'],
                    skip_metrics=self.config['training']['skip_metrics']
                )
                if self.model_analyzer is not None:
                    self.model_analyzer.analyze(self.model.model)
                self.save_iteration(f"{self.now}/hyper_parameter_{ii}/iteration_{jj}")
                if self.post_training_linear_evaluation:
                    self.run_post_training_linear_evaluation(
                        name=f'_{ii}_{jj}',
                        model_folder=f"{self.now}/hyper_parameter_{ii}/iteration_{jj}/"
                    )
        np.savez(
            f"{self.meta['local_scratch']}/runs/{self.now}/hyper_parameters.npz",
            hyper_parameters=self.hyper_parameters
        )
    
    def run_bayes_hyper_parameter_scan(self):
        self.logger.info(f"running hyper_parameter scan over {self.iterations} iterations")
        optimizer_config = self.config['optimizer']
        training_config = self.config['training']
    
    def run_linear_evaluation(self):
        self.logger.info(f"running linear_evaluation protocol")
        if "linear_evaluation" not in self.config.keys():
            self.logger.warn("no linear_evaluation in config file.")
            return
        linear_evaluation_config = self.config['linear_evaluation']
        if 'run_name' in self.config['training'].keys():
            self.now = self.config['training']['run_name'] + f"_{get_datetime()}"
        else:
            self.now = self.name + f"_{get_datetime()}"
        for ii, model in enumerate(linear_evaluation_config['models']):
            for jj in range(self.config['training']['iterations']):
                linear_config = {
                    'model':    self.linear_evaluation_model_directory + model,
                }
                self.model = LinearEvaluation(
                    config=linear_config,
                    meta=self.meta
                )
                self.parse_optimizer(f'_{ii}_{jj}')
                self.parse_loss(f'_{ii}_{jj}')
                self.parse_metrics(f'_{ii}_{jj}')
                self.parse_callbacks(f'_{ii}_{jj}')
                self.parse_training(f'_{ii}_{jj}')
                self.module_data_product[f'predictions_{ii}_{jj}'] = self.trainer.train(
                    epochs=self.config['linear_evaluation']['epochs'],
                    checkpoint=self.config['training']['checkpoint'],
                    progress_bar=self.config['training']['progress_bar'],
                    rewrite_bar=self.config['training']['rewrite_bar'],
                    save_predictions=self.config['training']['save_predictions'],
                    no_timing=self.config['training']['no_timing'],
                    skip_metrics=self.config['training']['skip_metrics']
                )
                if self.model_analyzer is not None:
                    self.model_analyzer.analyze(self.model.model)
                self.save_iteration(f"{self.now}/linear_{ii}/iteration_{jj}")
    
    def run_post_training_linear_evaluation(self,
        name:   str='',
        model_folder:   str=''
    ):
        linear_config = {
            'model':    self.model.model,
        }
        linear_model = LinearEvaluation(
            name=f'{self.model.model.name}'+'_linear_evaluation',
            config=linear_config,
            meta=self.meta
        )
        self.model = ModelHandler(
            self.name + name,
            models=[linear_model],
            meta=self.meta
        )
        self.parse_optimizer(f'_{name}_linear_evaluation')
        self.parse_loss(f'_{name}_linear_evaluation')
        self.parse_metrics(f'_{name}_linear_evaluation')
        self.parse_callbacks(f'_{name}_linear_evaluation')
        self.parse_training(f'_{name}_linear_evaluation')
        self.module_data_product[f'predictions_{name}_linear_evaluation'] = self.trainer.train(
            epochs=self.config['linear_evaluation']['epochs'],
            checkpoint=self.config['training']['checkpoint'],
            progress_bar=self.config['training']['progress_bar'],
            rewrite_bar=self.config['training']['rewrite_bar'],
            save_predictions=self.config['training']['save_predictions'],
            no_timing=self.config['training']['no_timing'],
            skip_metrics=self.config['training']['skip_metrics']
        )
        if self.model_analyzer is not None:
            self.model_analyzer.analyze(self.model.model)
        self.save_iteration(f"{model_folder}/linear_evaluation")
