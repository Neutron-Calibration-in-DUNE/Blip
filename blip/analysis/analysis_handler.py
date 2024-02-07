"""
Container for generic analyzers
"""
import os
import importlib.util
import sys
import inspect
import torch

from blip.utils.logger import Logger
from blip.analysis.generic_analyzer import GenericAnalyzer
from blip.utils.utils import get_method_arguments


class AnalysisHandler:
    """
    """
    def __init__(
        self,
        name:    str,
        config:  dict = {},
        meta:    dict = {}
    ):
        self.name = name + '_analyzer_handler'

        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(self.name, output="both", file_mode="w")
        else:
            self.logger = Logger(self.name, level='warning', file_mode="w")

        if bool(config):
            self.set_config(config)
        else:
            self.logger.error("handler received no config!")

    def set_config(self, config):
        self.config = config
        self.process_config()

    def collect_analyzers(self):
        self.available_analyzers = {}
        self.analyzer_files = [
            os.path.dirname(__file__) + '/' + file
            for file in os.listdir(path=os.path.dirname(__file__))
        ]
        self.analyzer_files.extend(self.meta['local_blip_files'])
        for analyzer_file in self.analyzer_files:
            if (
                ("__init__.py" in analyzer_file) or
                ("__pycache__.py" in analyzer_file) or
                ("generic_analyzer.py" in analyzer_file) or
                ("__pycache__" in analyzer_file) or
                (".py" not in analyzer_file)
            ):
                continue
            try:
                self.load_analyzer(analyzer_file)
            except:
                self.logger.warn(f'problem loading analyzer from file: {analyzer_file}')

    def load_analyzer(
        self,
        analyzer_file: str
    ):
        spec = importlib.util.spec_from_file_location(
            f'{analyzer_file.removesuffix(".py")}.name',
            analyzer_file
        )
        custom_analyzer_file = importlib.util.module_from_spec(spec)
        sys.modules[f'{analyzer_file.removesuffix(".py")}.name'] = custom_analyzer_file
        spec.loader.exec_module(custom_analyzer_file)
        for name, obj in inspect.getmembers(sys.modules[f'{analyzer_file.removesuffix(".py")}.name']):
            if inspect.isclass(obj):
                custom_class = getattr(custom_analyzer_file, name)
                if issubclass(custom_class, GenericAnalyzer):
                    self.available_analyzers[name] = custom_class

    def process_config(self):
        # list of available analyzers
        self.collect_analyzers()
        # check config
        if "custom_analyzer_file" in self.config.keys():
            if os.path.isfile(self.config["custom_analyzer_file"]):
                try:
                    self.load_analyzer(self.config["custom_analyzer_file"])
                    self.logger.info(f'added custom analyzer from file {self.config["custom_analyzer_file"]}.')
                except:
                    self.logger.error(
                        f'loading classes from file {self.config["custom_analyzer_file"]} failed!'
                    )
            else:
                self.logger.error(f'custom_analyzer_file {self.config["custom_analyzer_file"]} not found!')
        if "analyzer_type" not in self.config.keys():
            self.logger.warn('analyzer_type not specified in config! Setting to "single"!')
            self.analyzer_type = 'single'
        # process analyzers
        for item in self.config.keys():
            if item == "custom_analyzer_file" or item == "load_analyzer":
                continue
            if item == "analyzer_type":
                self.analyzer_type = self.config[item]
                continue
            # check that analyzer exists
            if item not in self.available_analyzers.keys():
                self.logger.error(
                    f"specified analyzer '{item}' is not an available type! " +
                    f"Available types:\n{self.available_analyzers.keys()}"
                )
        self.analyzers = {}
        for item in self.config.keys():
            if item == "custom_analyzer_file" or item == "load_analyzer" or item == "analyzer_type":
                continue
            self.analyzers[item] = self.available_analyzers[item](
                item, self.config[item], self.meta
            )
            self.logger.info(f'added analyzer "{item}" to AnalyzerHandler.')

    def set_device(
        self,
        device
    ):
        self.logger.info(f'setting device to "{device}".')
        for name, analyzer in self.analyzers.items():
            analyzer.set_device(device)
        self.device = device

    def add_analyzer(
        self,
        analyzer:   GenericAnalyzer
    ):
        if issubclass(type(analyzer), GenericAnalyzer):
            self.logger.info(f'added analyzer function "{analyzer}" to AnalysisHandler.')
            self.analyzers[analyzer.name] = analyzer
        else:
            self.logger.error(
                f'specified analyzer {analyzer} is not a child of "GenericAnalyzer"!' +
                ' Only analyzer functions which inherit from GenericAnalyzer can' +
                ' be used by the AnalysisHandler in BLIP.'
            )

    def remove_analyzer(
        self,
        analyzer:   str
    ):
        if analyzer in self.analyzers.keys():
            self.analyzers.pop(analyzer)
            self.logger.info(f'removed {analyzer} from analyzers.')

    def analyze_event(
        self,
        event,
    ):
        for name, analyzer in self.analyzers.items():
            analyzer.analyze_event(event)

    def analyze_events(
        self,
    ):
        for name, analyzer in self.analyzers.items():
            analyzer.analyze_events()
