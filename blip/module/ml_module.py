"""
Generic module
"""
import os
import copy
import random
import numpy as np

from blip.analysis.model_analyzer_handler import ModelAnalyzerHandler
from blip.models import ModelHandler
from blip.models.linear_evaluation import LinearEvaluation
from blip.module.generic_module import GenericModule
from blip.losses import LossHandler
from blip.optimizers import Optimizer
from blip.metrics import MetricHandler
from blip.trainer import Trainer
from blip.utils.callbacks import CallbackHandler
from blip.utils.utils import flatten_dict, generate_combinations_from_arrays, tar_directory


class MachineLearningModule(GenericModule):
    """
    The module class helps to organize meta data and objects related to different tasks
    and execute those tasks based on a configuration file.  The spirit of the 'Module' class
    is to mimic some of the functionality of LArSoft, e.g. where you can specify a chain
    of tasks to be completed, the ability to have nested config files where default parameters
    can be overwritten.

    The ML specific module runs in several different modes,
    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        mode:   str = '',
        meta:   dict = {}
    ):
        self.name = name
        super(MachineLearningModule, self).__init__(
            self.name, config, mode, meta
        )
        self.consumes = ['dataset', 'loader']
        self.produces = ['predictions']

    def parse_config(self):
        """
        """
        self.check_config()

        self.meta['model'] = None
        self.meta['criterion'] = None
        self.meta['optimizer'] = None
        self.meta['metrics'] = None
        self.meta['callbacks'] = None
        self.meta['trainer'] = None
        self.meta['model_analyzer'] = None

        self.parse_inference()
        self.parse_hyper_parameters()
        self.parse_linear_evaluation()
        self.parse_model_analyzer()

    def check_config(self):
        if "model" not in self.config.keys():
            self.logger.warning('"model" section not specified in config!')

        if (
            self.mode == "training" or
            self.mode == "contrastive_training" or
            self.mode == "hyper_parameter_scan" or
            self.mode == "contrastive_hyper_parameter_scan"
        ):
            if "criterion" not in self.config.keys():
                self.logger.error('"criterion" section not specified in config!')
            if "optimizer" not in self.config.keys():
                self.logger.error('"optimizer" section not specified in config!')
            if "metrics" not in self.config.keys():
                self.logger.warn('"metrics" section not specified in config!')
            if "callbacks" not in self.config.keys():
                self.logger.warn('"callbacks" section not specified in config!')
            if "training" not in self.config.keys():
                self.logger.error('"training" section not specified in config!')

        if self.mode == "inference":
            if "inference" not in self.config.keys():
                self.logger.error('"inference" section not specified in config!')

        if self.mode == "hyper_parameter_scan" or self.mode == "contrastive_hyper_parameter_scan":
            if "hyper_parameters" not in self.config.keys():
                self.logger.error('"hyper_parameters" section not specified in config!')

        if (
            self.mode == "linear_evaluation" or
            self.mode == "contrastive_training" or
            self.mode == "contrastive_hyper_parameter_scan"
        ):
            if "linear_evaluation" not in self.config.keys():
                self.logger.error("'linear_evaluation' section not specified in config!")

        if self.mode == "model_analyzer":
            if "model_analyzer" not in self.config.keys():
                self.logger.error("'model_analyzer' section not specified in config!")

    def parse_model(
        self,
        name:   str = ''
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
        self.meta['model'] = ModelHandler(
            self.name + name,
            model_config,
            meta=self.meta
        )

    def parse_loss(
        self,
        name:   str = ''
    ):
        """
        """
        if "criterion" not in self.config.keys():
            self.logger.warn("no criterion in config file.")
            return
        self.logger.info("configuring criterion.")
        criterion_config = self.config['criterion']
        # add in class weight numbers for loss functions
        self.meta['criterion'] = LossHandler(
            self.name + name,
            criterion_config,
            meta=self.meta
        )

    def parse_optimizer(
        self,
        name:   str = ''
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
        self.meta['optimizer'] = Optimizer(
            self.name + name,
            optimizer_config,
            meta=self.meta,
        )

    def parse_metrics(
        self,
        name:   str = ''
    ):
        """
        """
        if "metrics" not in self.config.keys():
            self.logger.warn("no metrics in config file.")
            return
        self.logger.info("configuring metrics.")
        metrics_config = self.config['metrics']
        self.meta['metrics'] = MetricHandler(
            self.name + name,
            metrics_config,
            meta=self.meta
        )

    def parse_callbacks(
        self,
        name:   str = ''
    ):
        """
        """
        if "callbacks" not in self.config.keys():
            self.logger.warn("no callbacks in config file.")
            return
        self.logger.info("configuring callbacks.")
        callbacks_config = self.config['callbacks']
        if callbacks_config is None:
            self.logger.warn("no callbacks specified.")
        else:
            for callback in callbacks_config.keys():
                if callbacks_config[callback] is None:
                    callbacks_config[callback] = {}
                callbacks_config[callback]['criterion_handler'] = self.meta['criterion']
                callbacks_config[callback]['metrics_handler'] = self.meta['metrics']
        self.meta['callbacks'] = CallbackHandler(
            self.name + name,
            callbacks_config,
            meta=self.meta
        )

    def parse_training(
        self,
        name:   str = ''
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
        self.meta['trainer'] = Trainer(
            self.name + name,
            training_config,
            meta=self.meta,
        )

    def parse_inference(
        self,
        name:   str = ''
    ):
        if "inference" not in self.config.keys():
            self.logger.warn("no inference in config file.")
            return
        self.logger.info("configuring inference.")
        training_config = self.config['training']
        if self.meta['trainer'] is None:
            self.meta['trainer'] = Trainer(
                self.name + name,
                training_config,
                meta=self.meta
            )

        if "layers" in self.config["inference"].keys():
            for layer in self.config["inference"]["layers"]:
                if layer not in self.meta['model'].forward_views().keys():
                    self.logger.error(
                        f"layer '{layer}' not in the model forward views!" +
                        f" possible views: {self.meta['model'].forward_views().keys()}"
                    )
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

    def parse_linear_evaluation(
        self,
        name:   str = ''
    ):
        if "linear_evaluation" not in self.config.keys():
            self.logger.warn("no linear_evaluation in config file.")
            return
        self.logger.info("configuring linear_evaluation")
        linear_evaluation_config = self.config["linear_evaluation"]
        if "model_directory" in linear_evaluation_config.keys():
            self.linear_evaluation_model_directory = linear_evaluation_config['model_directory']
            if not os.path.isdir(self.linear_evaluation_model_directory):
                self.logger.error(
                    f"linear_evaluation model_directory: {self.linear_evaluation_model_directory} does not exist!"
                )
            self.logger.info(f"setting linear_evaluation model_directory to {self.linear_evaluation_model_directory}.")
        else:
            self.linear_evaluation_model_directory = ''
            self.logger.info("linear_evaluation model_directory not specified, setting to './'.")
        if "epochs" not in linear_evaluation_config.keys():
            self.logger.warn("linear_evaluation: epochs not specified in config.  Setting to 50.")
            self.config["linear_evaluation"]["epochs"] = 50

    def parse_model_analyzer(
        self,
        name:   str = ''
    ):
        """
        """
        if "model_analyzer" not in self.config.keys():
            self.logger.warn("no model_analyzer in config file.")
            self.meta['model_analyzer'] = None
            return
        self.logger.info("configuring model_analyzer")
        model_analyzer_config = self.config["model_analyzer"]
        self.meta['model_analyzer'] = ModelAnalyzerHandler(
            self.name + name,
            model_analyzer_config,
            meta=self.meta
        )

    # TODO: Fix this so that it doesn't need to compute all the paths, but only a subset of
    # random ones, since the number of paths is N!, which is intractable.
    def generate_grid_hyper_parameters(
        self,
        hyper_parameters_config
    ):
        model_parameters = hyper_parameters_config["model_parameters"]
        self.parameter_paths = flatten_dict(model_parameters)
        self.logger.info("generating hyper parameter combinations.")
        self.parameter_combinations = generate_combinations_from_arrays(
            {tuple(k): v for k, v in self.parameter_paths if isinstance(v, list)}
        )
        self.logger.info(f"generated {len(self.parameter_combinations)} different possible hyper parameter combinations.")
        random.shuffle(self.parameter_combinations)
        if len(self.parameter_combinations) < len(self.hyper_parameters.keys()):
            self.logger.info(
                f"number of iterations {self.iterations} larger than possible combinations" +
                f" {len(self.parameter_combinations)}.  Setting number of iterations to {len(self.parameter_combinations)}."
            )
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

    def generate_random_hyper_parameters(
        self,
        hyper_parameters_config
    ):
        self.logger.error("generate random hyper parameters not available yet!")

    def save_iteration(
        self,
        file_tag:   str = ''
    ):
        # save model/data/config
        # create specific folder for this iteration
        if file_tag != '':
            output_file = self.meta['run_directory'] + '/ml_module_output_' + file_tag + '.tar.gz'
        else:
            output_file = self.meta['run_directory'] + '/ml_module_output.tar.gz'
        tar_directory(f"{self.meta['local_scratch']}/.tmp/", output_file)

    def run_module(self):
        if self.mode == 'training':
            self.run_training()
        elif self.mode == 'contrastive_training':
            self.run_contrastive_training()
        elif self.mode == 'inference':
            self.run_inference()
        elif self.mode == 'model_analyzer':
            self.run_model_analyzer()
        elif self.mode == 'hyper_parameter_scan':
            self.run_hyper_parameter_scan()
        elif self.mode == 'contrastive_hyper_parameter_scan':
            self.run_contrastive_hyper_parameter_scan()
        elif self.mode == 'linear_evaluation':
            self.run_linear_evaluation()
        else:
            self.logger.warning(f"current mode {self.mode} not an available type!")

    def run_training(self):
        for jj in range(self.config['training']['iterations']):
            self.parse_model(f'_{jj}')
            self.parse_optimizer(f'_{jj}')
            self.parse_loss(f'_{jj}')
            self.parse_metrics(f'_{jj}')
            self.parse_callbacks(f'_{jj}')
            self.parse_training(f'_{jj}')
            self.module_data_product[f'predictions_{jj}'] = self.meta['trainer'].train(
                epochs=self.config['training']['epochs'],
                checkpoint=self.config['training']['checkpoint'],
                progress_bar=self.config['training']['progress_bar'],
                rewrite_bar=self.config['training']['rewrite_bar'],
                save_predictions=self.config['training']['save_predictions'],
                no_timing=self.config['training']['no_timing'],
                skip_metrics=self.config['training']['skip_metrics']
            )
            if self.meta['model_analyzer'] is not None:
                self.meta['model_analyzer'].analyze(self.meta['model'].model)
            self.save_iteration(f"iteration_{jj}")

    def run_contrastive_training(self):
        for jj in range(self.config['training']['iterations']):
            self.parse_model(f'_contrastive_{jj}')
            self.meta['model'].contrastive_learning()

            self.parse_optimizer(f'_contrastive_{jj}')
            self.parse_loss(f'_contrastive_{jj}')
            self.parse_metrics(f'_contrastive_{jj}')
            self.parse_callbacks(f'_contrastive_{jj}')
            self.parse_training(f'_contrastive_{jj}')
            self.module_data_product[f'contrastive_predictions_{jj}'] = self.meta['trainer'].train(
                epochs=self.config['training']['epochs'],
                checkpoint=self.config['training']['checkpoint'],
                progress_bar=self.config['training']['progress_bar'],
                rewrite_bar=self.config['training']['rewrite_bar'],
                save_predictions=self.config['training']['save_predictions'],
                no_timing=self.config['training']['no_timing'],
                skip_metrics=self.config['training']['skip_metrics']
            )
            if self.meta['model_analyzer'] is not None:
                self.meta['model_analyzer'].analyze(self.meta['model'].model)
            self.save_iteration(f"contrastive_iteration_{jj}")
            # linear evaluation
            self.meta['model'].linear_evaluation()
            self.parse_optimizer(f'_linear_evaluation_{jj}')
            self.parse_loss(f'_linear_evaluation_{jj}')
            # remove contrastive losses
            self.meta['criterion'].remove_loss('NTXEntropyLoss')

            self.parse_metrics(f'_linear_evaluation_{jj}')
            self.parse_callbacks(f'_linear_evaluation_{jj}')
            self.parse_training(f'_linear_evaluation_{jj}')

            self.module_data_product[f'linear_predictions_{jj}'] = self.meta['trainer'].train(
                epochs=self.config['linear_evaluation']['epochs'],
                checkpoint=self.config['training']['checkpoint'],
                progress_bar=self.config['training']['progress_bar'],
                rewrite_bar=self.config['training']['rewrite_bar'],
                save_predictions=self.config['training']['save_predictions'],
                no_timing=self.config['training']['no_timing'],
                skip_metrics=self.config['training']['skip_metrics']
            )
            self.save_iteration(f"linear_evaluation_iteration_{jj}")

    def run_inference(self):
        self.parse_model()
        self.module_data_product['predictions'] = self.meta['trainer'].inference(
            dataset_type=self.config['inference']['dataset_type'],
            layers=self.config['inference']['layers'],
            outputs=self.config['inference']['outputs'],
            progress_bar=self.config['inference']['progress_bar'],
            rewrite_bar=self.config['inference']['rewrite_bar'],
            save_predictions=self.config['inference']['save_predictions']
        )

    def run_model_analyzer(self):
        self.logger.error("running model analyzer by itself not available yet!")

    def run_hyper_parameter_scan(self):
        if self.search_type == 'grid' or self.search_type == 'random':
            self.run_grid_hyper_parameter_scan()
        else:
            self.run_bayes_hyper_parameter_scan()

    def run_grid_hyper_parameter_scan(self):
        self.logger.info(f"running hyper_parameter scan over {self.iterations} iterations")
        training_config = self.config['training']
        for ii, iteration in enumerate(self.hyper_parameters.keys()):
            for jj in range(training_config['iterations']):
                self.meta['model'] = ModelHandler(
                    self.name + f"_{ii}_{jj}",
                    self.hyper_parameters[iteration],
                    meta=self.meta
                )
                self.parse_optimizer(f'_{ii}_{jj}')
                self.parse_loss(f'_{ii}_{jj}')
                self.parse_metrics(f'_{ii}_{jj}')
                self.parse_callbacks(f'_{ii}_{jj}')
                self.parse_training(f'_{ii}_{jj}')

                self.module_data_product[f'predictions_{ii}_{jj}'] = self.meta['trainer'].train(
                    epochs=training_config['epochs'],
                    checkpoint=training_config['checkpoint'],
                    progress_bar=training_config['progress_bar'],
                    rewrite_bar=training_config['rewrite_bar'],
                    save_predictions=training_config['save_predictions'],
                    no_timing=training_config['no_timing'],
                    skip_metrics=training_config['skip_metrics']
                )
                if self.meta['model_analyzer'] is not None:
                    self.meta['model_analyzer'].analyze(self.meta['model'].model)
                self.save_iteration(f"hyper_parameter_{ii}_iteration_{jj}")
        np.savez(
            f"{self.meta['local_scratch']}/runs/{self.now}/hyper_parameters.npz",
            hyper_parameters=self.hyper_parameters
        )

    def run_bayes_hyper_parameter_scan(self):
        self.logger.error("bayes hyper parameter scan not available yet!")

    def run_contrastive_hyper_parameter_scan(self):
        if self.search_type == 'grid' or self.search_type == 'random':
            self.run_grid_contrastive_hyper_parameter_scan()
        else:
            self.run_bayes_contrastive_hyper_parameter_scan()

    def run_grid_contrastive_hyper_parameter_scan(self):
        self.logger.info(f"running hyper_parameter scan over {self.iterations} iterations")
        training_config = self.config['training']
        for ii, iteration in enumerate(self.hyper_parameters.keys()):
            for jj in range(training_config['iterations']):
                self.meta['model'] = ModelHandler(
                    self.name + f"_contrastive_{ii}_{jj}",
                    self.hyper_parameters[iteration],
                    meta=self.meta
                )
                # contrastive learning
                self.meta['model'].contrastive_learning()
                self.parse_optimizer(f'_contrastive_{ii}_{jj}')
                self.parse_loss(f'_contrastive_{ii}_{jj}')
                self.parse_metrics(f'_contrastive_{ii}_{jj}')
                self.parse_callbacks(f'_contrastive_{ii}_{jj}')
                self.parse_training(f'_contrastive_{ii}_{jj}')

                self.module_data_product[f'contrastive_predictions_{ii}_{jj}'] = self.meta['trainer'].train(
                    epochs=training_config['epochs'],
                    checkpoint=training_config['checkpoint'],
                    progress_bar=training_config['progress_bar'],
                    rewrite_bar=training_config['rewrite_bar'],
                    save_predictions=training_config['save_predictions'],
                    no_timing=training_config['no_timing'],
                    skip_metrics=training_config['skip_metrics']
                )
                if self.meta['model_analyzer'] is not None:
                    self.meta['model_analyzer'].analyze(self.meta['model'].model)
                self.save_iteration(f"contrastive_hyper_parameter_{ii}_iteration_{jj}")
                # linear evaluation
                self.meta['model'].linear_evaluation()
                self.parse_optimizer(f'_linear_evaluation_{ii}_{jj}')
                self.parse_loss(f'_linear_evaluation_{ii}_{jj}')
                # remove contrastive losses
                self.meta['criterion'].remove_loss('NTXEntropyLoss')

                self.parse_metrics(f'_linear_evaluation_{ii}_{jj}')
                self.parse_callbacks(f'_linear_evaluation_{ii}_{jj}')
                self.parse_training(f'_linear_evaluation_{ii}_{jj}')

                self.module_data_product[f'linear_predictions_{ii}_{jj}'] = self.meta['trainer'].train(
                    epochs=self.config['linear_evaluation']['epochs'],
                    checkpoint=training_config['checkpoint'],
                    progress_bar=training_config['progress_bar'],
                    rewrite_bar=training_config['rewrite_bar'],
                    save_predictions=training_config['save_predictions'],
                    no_timing=training_config['no_timing'],
                    skip_metrics=training_config['skip_metrics']
                )
                self.save_iteration(f"linear_evaluation_hyper_parameter_{ii}_iteration_{jj}")
        np.savez(
            f"{self.meta['local_scratch']}/runs/{self.now}/hyper_parameters.npz",
            hyper_parameters=self.hyper_parameters
        )

    def run_bayes_contrastive_hyper_parameter_scan(self):
        self.logger.error("bayes contrastive hyper parameter scan not available yet!")

    def run_linear_evaluation(self):
        self.logger.info("running linear_evaluation protocol")
        if "linear_evaluation" not in self.config.keys():
            self.logger.warn("no linear_evaluation in config file.")
            return
        linear_evaluation_config = self.config['linear_evaluation']
        for ii, model in enumerate(linear_evaluation_config['models']):
            for jj in range(self.config['training']['iterations']):
                linear_config = {
                    'model':    self.linear_evaluation_model_directory + model,
                }
                self.meta['model'] = LinearEvaluation(
                    config=linear_config,
                    meta=self.meta
                )
                self.parse_optimizer(f'_{ii}_{jj}')
                self.parse_loss(f'_{ii}_{jj}')
                self.parse_metrics(f'_{ii}_{jj}')
                self.parse_callbacks(f'_{ii}_{jj}')
                self.parse_training(f'_{ii}_{jj}')
                self.module_data_product[f'predictions_{ii}_{jj}'] = self.meta['trainer'].train(
                    epochs=self.config['linear_evaluation']['epochs'],
                    checkpoint=self.config['training']['checkpoint'],
                    progress_bar=self.config['training']['progress_bar'],
                    rewrite_bar=self.config['training']['rewrite_bar'],
                    save_predictions=self.config['training']['save_predictions'],
                    no_timing=self.config['training']['no_timing'],
                    skip_metrics=self.config['training']['skip_metrics']
                )
                if self.meta['model_analyzer'] is not None:
                    self.meta['model_analyzer'].analyze(self.meta['model'].model)
                self.save_iteration(f"linear_{ii}_iteration_{jj}")
