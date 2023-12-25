"""
Class for a generic model trainer.
"""
import torch
import numpy as np
import os
import importlib.util
import sys
import inspect

from blip.analysis.generic_model_analyzer import GenericModelAnalyzer
from blip.utils.logger import Logger

from blip.utils.utils import get_method_arguments


class ModelAnalyzerHandler:
    """
    """
    def __init__(
        self,
        name:   str = 'default',
        config: dict = {},
        meta:   dict = {}
    ):
        self.name = name + "_model_analyzer"
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

        self.process_config()

    def collect_model_analyzer_functions(self):
        self.available_model_analyzers = {}
        self.model_analyzer_files = [
            os.path.dirname(__file__) + '/' + file
            for file in os.listdir(path=os.path.dirname(__file__))
        ]
        self.model_analyzer_files.extend(self.meta['local_blip_files'])
        for model_analyzer_file in self.model_analyzer_files:
            if (
                ("__init__.py" in model_analyzer_file) or
                ("__pycache__.py" in model_analyzer_file) or
                ("generic_model_analyzer.py" in model_analyzer_file) or
                ("__pycache__" in model_analyzer_file) or
                (".py" not in model_analyzer_file)
            ):
                continue
            try:
                self.load_model_analyzer_function(model_analyzer_file)
            except:
                self.logger.warn(f'problem loading model_analyzer from file: {model_analyzer_file}')

    def load_model_analyzer_function(
        self,
        model_analyzer_file: str
    ):
        spec = importlib.util.spec_from_file_location(
            f'{model_analyzer_file.removesuffix(".py")}.name',
            model_analyzer_file
        )
        custom_model_analyzer_file = importlib.util.module_from_spec(spec)
        sys.modules[f'{model_analyzer_file.removesuffix(".py")}.name'] = custom_model_analyzer_file
        spec.loader.exec_module(custom_model_analyzer_file)
        for name, obj in inspect.getmembers(sys.modules[f'{model_analyzer_file.removesuffix(".py")}.name']):
            if inspect.isclass(obj):
                custom_class = getattr(custom_model_analyzer_file, name)
                if issubclass(custom_class, GenericModelAnalyzer):
                    self.available_model_analyzers[name] = custom_class

    def process_config(self):
        # list of available model_analyzers
        self.collect_model_analyzer_functions()
        # check config
        if "custom_model_analyzer_file" in self.config.keys():
            if os.path.isfile(self.config["custom_model_analyzer_file"]):
                try:
                    self.load_model_analyzer_function(self.config["custom_model_analyzer_file"])
                    self.logger.info(
                        f'added custom model analyzer function from file {self.config["custom_model_analyzer_file"]}.'
                    )
                except:
                    self.logger.error(
                        f'loading classes from file {self.config["custom_model_analyzer_file"]} failed!'
                    )
            else:
                self.logger.error(
                    f'custom_model_analyzer_file {self.config["custom_model_analyzer_file"]} not found!'
                )
        # process model analyzer functions
        for item in self.config.keys():
            if item == "custom_model_analyzer_file":
                continue
            if item == "models":
                self.models = self.config['models']
                continue
            # check that model analyzer function exists
            if item not in self.available_model_analyzers.keys():
                self.logger.error(
                    f"specified model analyzer function '{item}' is not an available type! " +
                    f"Available types:\n{self.available_model_analyzers.keys()}"
                )
            # check that function arguments are provided
            argdict = get_method_arguments(self.available_model_analyzers[item])
            for value in self.config[item].keys():
                if value not in argdict.keys():
                    self.logger.error(
                        f"specified model analyzer function value '{item}:{value}' " +
                        f"not a constructor parameter for '{item}'! " +
                        f"Constructor parameters:\n{argdict}"
                    )
            for value in argdict.keys():
                if argdict[value] is None:
                    if value not in self.config[item].keys():
                        self.logger.error(
                            f"required input parameters '{item}:{value}' " +
                            f"not specified! Constructor parameters:\n{argdict}"
                        )
        self.model_analyzers = {}
        for item in self.config.keys():
            if item == "custom_model_analyzer_file":
                continue
            if item == "models":
                continue
            if item in self.model_analyzers:
                self.logger.warn('duplicate model analyzer specified in config! Attempting to arrange by target_type.')
                if self.model_analyzers[item].target_type == self.config[item]['target_type']:
                    self.logger.error('duplicate model analyzeres with the same target_type in config!')
            self.model_analyzers[item] = self.available_model_analyzers[item](**self.config[item], meta=self.meta)
            self.logger.info(f'added model analyzer function "{item}" to ModelAnalyzerHandler.')

    def analyze(
        self,
        model
    ):
        self.logger.info('running model analyzer')
        for ii, analyzer in enumerate(self.model_analyzers.keys()):
            input = {
                "pos": [],
                "x":   [],
                "category": []
            }
            predictions = {
                layer: []
                for layer in self.model_analyzers[analyzer].layers
            }
            for output in self.model_analyzers[analyzer].outputs:
                predictions[output] = []

            if self.model_analyzers[analyzer].dataset_type == 'train':
                inference_loader = self.meta['loader'].train_loader
            elif self.model_analyzers[analyzer].dataset_type == 'validation':
                inference_loader = self.meta['loader'].validation_loader
            elif self.model_analyzers[analyzer].dataset_type == 'test':
                inference_loader = self.meta['loader'].test_loader
            else:
                inference_loader = self.meta['loader'].all_loader

            inference_loop = enumerate(inference_loader, 0)

            model.eval()
            with torch.no_grad():
                for ii, data in inference_loop:
                    # get the network output
                    input["pos"].append(data.pos.cpu().numpy())
                    input["x"].append(data.x.cpu().numpy())
                    input["category"].append(data.category.cpu().numpy())
                    model_output = model(data)
                    for jj, key in enumerate(model_output.keys()):
                        if key in predictions.keys():
                            predictions[key].append(model_output[key].cpu().numpy())
                    for jj, key in enumerate(self.model_analyzers[analyzer].layers):
                        if key in predictions.keys():
                            predictions[key].append(model.forward_views[key].cpu().numpy())
                input["pos"] = np.array(input["pos"])
                input["x"] = np.array(input["x"])
                input["category"] = np.array(np.concatenate(input["category"]))
                for key in predictions.keys():
                    predictions[key] = np.array(np.concatenate(predictions[key]))
                self.model_analyzers[analyzer].analyze(
                    input, predictions
                )
