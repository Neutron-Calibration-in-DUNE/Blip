"""
Container for generic callbacks
"""
import os
import importlib.util
import sys
import inspect

from blip.utils.logger import Logger
from blip.metrics import GenericMetric
from blip.utils.utils import get_method_arguments

class MetricHandler:
    """
    """
    def __init__(self,
        name:   str,
        config: dict={},
        metrics:list=[],
        labels: list=[],
        meta:   dict={}
    ):
        self.name = name + "_metric_handler"
        self.logger = Logger(self.name, output="both", file_mode="w")
        self.meta = meta
        if "device" in self.meta:
            self.device = self.meta['device']
        else:
            self.device = 'cpu'
        if meta['verbose']:
            self.logger = Logger(name, output="both", file_mode="w")
        else:
            self.logger = Logger(name, level='warning', file_mode="w")
        self.labels = labels
        

        if bool(config) and len(metrics) != 0:
            self.logger.error(
                f"handler received both a config and a list of metrics! " + 
                f"The user should only provide one or the other!")
        elif bool(config):
            self.set_config(config)
        else:
            if len(metrics) == 0:
                self.logger.warn(f"handler received neither a config or metrics!")
            self.metrics = {
                metric.name: metric 
                for metric in metrics
            } 

    def set_config(self, config):
        self.config = config
        self.process_config()
    
    def collect_metrics(self):
        self.available_metrics = {}
        self.metric_files = [
            os.path.dirname(__file__) + '/' + file 
            for file in os.listdir(path=os.path.dirname(__file__))
        ]
        for metric_file in self.metric_files:
            if (
                ("__init__.py" in metric_file) or 
                ("__pycache__.py" in metric_file) or 
                ("generic_metric.py" in metric_file) or 
                ("__pycache__" in metric_file)
            ):
                continue
            try:
                self.load_metric(metric_file)
            except:
                self.logger.warn(f'problem loading metric from file: {metric_file}')
    
    def load_metric(self,
        metric_file: str
    ):
        spec = importlib.util.spec_from_file_location(
            f'{metric_file.removesuffix(".py")}.name', 
            metric_file
        )
        custom_metric_file = importlib.util.module_from_spec(spec)
        sys.modules[f'{metric_file.removesuffix(".py")}.name'] = custom_metric_file
        spec.loader.exec_module(custom_metric_file)
        for name, obj in inspect.getmembers(sys.modules[f'{metric_file.removesuffix(".py")}.name']):
            if inspect.isclass(obj):
                custom_class = getattr(custom_metric_file, name)
                if issubclass(custom_class, GenericMetric):
                    self.available_metrics[name] = custom_class

    
    def process_config(self):
        # list of available criterions
        self.collect_metrics()
        # check config
        if "custom_metric_file" in self.config.keys():
            if os.path.isfile(self.config["custom_metric_file"]):
                try:
                    self.load_metric(self.config["custom_metric_file"])
                    self.logger.info(f'added custom metric function from file {self.config["custom_metric_file"]}.')
                except:
                    self.logger.error(
                        f'loading classes from file {self.config["custom_metric_file"]} failed!'
                    )
            else:
                self.logger.error(f'custom_metric_file {self.config["custom_metric_file"]} not found!')
        # process metric functions
        for item in self.config.keys():
            if item == "custom_metric_file":
                continue
            # check that metric function exists
            if item not in self.available_metrics.keys():
                self.logger.error(
                    f"specified metric function '{item}' is not an available type! " + 
                    f"Available types:\n{self.available_metrics.keys()}"
                )
            # check that function arguments are provided
            argdict = get_method_arguments(self.available_metrics[item])
            for value in self.config[item].keys():
                if value not in argdict.keys():
                    self.logger.error(
                        f"specified metric value '{item}:{value}' " + 
                        f"not a constructor parameter for '{item}'! " + 
                        f"Constructor parameters:\n{argdict}"
                    )
            for value in argdict.keys():
                if argdict[value] == None:
                    if value not in self.config[item].keys():
                        self.logger.error(
                            f"required input parameters '{item}:{value}' "+
                            f"not specified! Constructor parameters:\n{argdict}"
                        )
        self.metrics = {}
        for item in self.config.keys():
            if item == "custom_metric_file":
                continue
            self.metrics[item] = self.available_metrics[item](**self.config[item], meta=self.meta)
            self.logger.info(f'added metric function "{item}" to MetricHandler.')

    def set_device(self,
        device
    ):  
        for name, metric in self.metrics.items():
            metric.set_device(device)
            metric.reset_batch()
        self.device = device

    def reset_batch(self):  
        for name, metric in self.metrics.items():
            metric.reset_batch()

    def add_metric(self,
        metric:   GenericMetric
    ):
        if issubclass(type(metric), GenericMetric):
            self.logger.info(f'added metric function "{metric}" to MetricHandler.')
            self.metrics[metric.name] = metric
        else:
            self.logger.error(
                f'specified metric {metric} is not a child of "GenericMetric"!' + 
                f' Only metric functions which inherit from GenericMetric can' +
                f' be used by the metricHandler in BLIP.'
            )
    
    def update(self,
        outputs,
        data,
        train_type: str='all',
    ):
        for name, metric in self.metrics.items():
            if train_type == metric.when_to_compute or metric.when_to_compute == 'all':
                metric.update(outputs, data)
    
    def compute(self):
        metrics = {
            name: metric.compute() 
            for name, metric in self.metrics.items()
        }
        return metrics