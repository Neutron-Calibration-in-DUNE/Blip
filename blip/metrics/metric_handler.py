"""
Container for generic callbacks
"""
import os
import importlib.util
import sys
from blip.utils.logger import Logger
from blip.metrics import GenericMetric
from blip.metrics import AUROCMetric
from blip.metrics import ConfusionMatrixMetric
from blip.metrics import DiceScoreMetric
from blip.metrics import JaccardIndexMetric
from blip.metrics import PrecisionMetric
from blip.metrics import RecallMetric
from blip.utils.utils import get_method_arguments

class MetricHandler:
    """
    """
    def __init__(self,
        name:   str,
        config: dict={},
        metrics:list=[],
        labels: list=[],
        device: str='cpu'
    ):
        self.name = name
        self.logger = Logger(self.name, output="both", file_mode="w")
        self.device = device
        self.labels = labels

        if bool(config) and len(metrics) != 0:
            self.logger.error(f"handler received both a config and a list of metrics! The user should only provide one or the other!")
        else:
            if bool(config):
                self.config = config
                self.process_config()
            else:
                self.metrics = {metric.name: metric for metric in metrics}
    
    def process_config(self):
        # list of available criterions
        # TODO: Make this automatic
        # list of available metrics
        self.available_metrics = {
            'AUROCMetric':            AUROCMetric,
            'ConfusionMatrixMetric':  ConfusionMatrixMetric,
            'DiceScoreMetric':        DiceScoreMetric,
            'JaccardIndexMetric':     JaccardIndexMetric,
            'PrecisionMetric':        PrecisionMetric,
            'RecallMetric':           RecallMetric,
        }

        # check config
        if "custom_metric_file" in self.config.keys():
            if os.path.isfile(self.config["custom_metric_file"]):
                try:
                    spec = importlib.util.spec_from_file_location(
                        'custom_metric_module.name', self.config["custom_metric_file"]
                    )
                    custom_metric_file = importlib.util.module_from_spec(spec)
                    sys.modules['custom_metric_module.name'] = custom_metric_file
                    spec.loader.exec_module(custom_metric_file)
                    custom_metric = getattr(custom_metric_file, self.config["custom_metric_name"])
                    self.available_metrics[self.config['custom_metric_name']] = custom_metric   
                    self.logger.info(
                        f'added custom metric from file {self.config["custom_metric_file"]}' + 
                        f' with name {self.config["custom_metric_name"]}.'
                    )
                except:
                    self.logger.error(
                        f'loading class {self.config["custom_metric_name"]}' +
                        f' from file {self.config["custom_metric_file"]} failed!'
                    )
            else:
                self.logger.error(f'custom_metric_file {self.config["custom_metric_file"]} not found!')
        for item in self.config.keys():
            if item == 'classes' or item == 'class_weights' or item == 'custom_metric_file' or item == 'custom_metric_name':
                continue
            if item not in self.available_metrics.keys():
                self.logger.error(f"specified metric '{item}' is not an available type! Available types:\n{self.available_metrics}")
            argdict = get_method_arguments(self.available_metrics[item])
            for value in self.config[item].keys():
                if value == "metric":
                    continue
                if value not in argdict.keys():
                    self.logger.error(f"specified metric value '{item}:{value}' not a constructor parameter for '{item}'! Constructor parameters:\n{argdict}")
            for value in argdict.keys():
                if argdict[value] == None:
                    if value not in self.config[item].keys():
                        self.logger.error(f"required input parameters '{item}:{value}' not specified! Constructor parameters:\n{argdict}")
        
        self.metrics = {}
        for item in self.config.keys():
            if item == 'classes' or item == 'class_weights' or item == 'custom_metric_file' or item == 'custom_metric_name':
                continue
            self.metrics[item] = self.available_metrics[item](**self.config[item], device=self.device)

    def set_device(self,
        device
    ):  
        for name, metric in self.metrics.items():
            metric.set_device(device)
            metric.reset()
        self.device = device
    
    def set_shapes(self,
        input_shapes,
    ):
        pass

    def reset(self):  
        for name, metric in self.metrics.items():
            metric.reset()

    def add_metric(self,
        metric:   GenericMetric
    ):
        self.metrics.append(metric)
    
    def set_training_info(self,
        epochs: int,
        num_training_batches:   int,
        num_validation_batches:  int,
        num_test_batches:   int,
    ):
        for name, metric in self.metrics.items():
            metric.set_training_info(
                epochs,
                num_training_batches,
                num_validation_batches,
                num_test_batches
            )
    
    def update(self,
        outputs,
        data,
        train_type: str='all',
    ):
        for name, metric in self.metrics.items():
            if train_type == metric.when_to_compute or metric.when_to_compute == 'all':
                metric.update(outputs, data)
    
    def compute(self,
        outputs,
        data
    ):
        metrics = [metric.compute(outputs, data) for name, metric in self.metrics.items()]
        return 